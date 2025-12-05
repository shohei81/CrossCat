"""Run the original DHA CrossCat demo inside the GenJAX environment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from crosscat import constants as const, helpers

DEFAULT_QUERY_COLUMNS = [
    "TTL_MDCR_SPND",
    "MDCR_SPND_HOME",
    "MDCR_SPND_EQP",
]
DEFAULT_CONDITION_COLUMNS = [
    "MDCR_SPND_INP",
    "MDCR_SPND_OUTP",
    "MDCR_SPND_OTHER",
]
DEFAULT_IMPUTE_COLUMNS = [
    "HI_IC_BEDS",
    "INT_IC_BEDS",
    "PCT_DTHS_W_ICU",
    "QUAL_SCORE",
    "CHF_SCORE",
]
DEFAULT_RECON_ROWS = [1, 10, 100]
DEFAULT_IMPUTE_ROWS = [10, 20, 30, 40, 50, 60, 70, 80]


def _assign_model_dimensions(
    *,
    num_rows: int,
    num_cont_cols: int,
    num_views: int,
    num_clusters: int,
):
    """Mutate the constants module so the GenJAX model matches the dataset."""
    if num_rows <= 0:
        raise ValueError("Dataset must contain at least one row.")
    if num_cont_cols <= 0:
        raise ValueError("Dataset must contain at least one continuous column.")
    const.NUM_ROWS = num_rows
    const.NUM_CONT_COLS = num_cont_cols
    const.NUM_CAT_COLS = 0
    const.NUM_VIEWS = max(1, num_views)
    const.NUM_CLUSTERS = max(1, num_clusters)


def _load_dataset(path: Path, id_column: str, max_rows: int | None):
    df = pd.read_csv(path)
    if max_rows is not None:
        df = df.head(max_rows)
    df = df.reset_index(drop=True)

    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not present in {path}.")
    feature_df = df.drop(columns=[id_column])
    raw_matrix = feature_df.to_numpy(dtype=np.float32)
    means = raw_matrix.mean(axis=0)
    stds = raw_matrix.std(axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    normalized = (raw_matrix - means) / stds

    observed_cont = jnp.asarray(normalized)
    observed_cat = jnp.zeros((observed_cont.shape[0], 0), dtype=jnp.int32)
    return df, feature_df, raw_matrix, observed_cont, observed_cat, means, stds


def _column_indices(names: Sequence[str], mapping: dict[str, int]) -> list[int]:
    indices: list[int] = []
    for name in names:
        if name not in mapping:
            raise ValueError(f"Column '{name}' not available in dataset.")
        indices.append(mapping[name])
    return indices


def _format_row_values(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.2f}" for v in values) + "]"


def _predictive_checks(
    trace,
    *,
    rng: jax.Array,
    row_indices: Iterable[int],
    query_cols: Sequence[str],
    condition_cols: Sequence[str],
    normalized_matrix: jnp.ndarray,
    actual_matrix: np.ndarray,
    name_to_idx: dict[str, int],
    num_samples: int,
    means: np.ndarray,
    stds: np.ndarray,
):
    base_rows = normalized_matrix.shape[0]
    for actual_row in row_indices:
        if actual_row < 0 or actual_row >= actual_matrix.shape[0]:
            continue
        query_indices = _column_indices(query_cols, name_to_idx)
        cond_indices = _column_indices(condition_cols, name_to_idx)
        virtual_row = base_rows + actual_row
        queries = [(virtual_row, idx) for idx in query_indices]
        conditions = []
        for idx in cond_indices:
            value = float(normalized_matrix[actual_row, idx])
            conditions.append((virtual_row, idx, value))
        rng, sample_key = jax.random.split(rng)
        sample_key, draws = helpers.predictive_samples(
            trace,
            queries,
            key=sample_key,
            num_samples=num_samples,
            conditions=conditions,
        )
        rng = sample_key
        print(f"\nRow {actual_row}: predictive samples")
        for q_pos, (col_idx, name) in enumerate(zip(query_indices, query_cols)):
            actual = float(actual_matrix[actual_row, col_idx])
            std = float(stds[col_idx])
            mean = float(means[col_idx])
            column_draws = np.asarray(draws[:, q_pos]) * std + mean
            print(
                f"  {name:>20s} actual={actual:8.2f} "
                f"mean={column_draws.mean():8.2f} std={column_draws.std():8.2f}"
            )
    return rng


def _impute_rows(
    trace,
    *,
    rng: jax.Array,
    row_indices: Iterable[int],
    impute_cols: Sequence[str],
    actual_matrix: np.ndarray,
    name_to_idx: dict[str, int],
    num_samples: int,
    means: np.ndarray,
    stds: np.ndarray,
):
    impute_indices = _column_indices(impute_cols, name_to_idx)
    for row_idx in row_indices:
        if row_idx < 0 or row_idx >= actual_matrix.shape[0]:
            continue
        queries = [(row_idx, idx) for idx in impute_indices]
        rng, estimates = helpers.impute(
            trace,
            queries,
            key=rng,
            num_samples=num_samples,
        )
        actual = [float(actual_matrix[row_idx, idx]) for idx in impute_indices]
        denorm = []
        for idx, value in zip(impute_indices, estimates):
            std = float(stds[idx])
            mean = float(means[idx])
            denorm.append(float(value) * std + mean)
        print(f"\nImputation row {row_idx}")
        print(f"  actual : {_format_row_values(actual)}")
        print(f"  imputed: {_format_row_values(denorm)}")
    return rng


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the DHA CSV dataset (e.g., data/dha.csv).",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="NAME",
        help="Column to drop and treat as an identifier (default: NAME).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the number of rows to load.",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=5,
        help="Maximum number of column views (truncation level).",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=8,
        help="Maximum number of row clusters per view (truncation level).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of Gibbs sweeps to run (original num_transitions analogue).",
    )
    parser.add_argument(
        "--predictive-samples",
        type=int,
        default=10,
        help="Samples per predictive query.",
    )
    parser.add_argument(
        "--impute-samples",
        type=int,
        default=1000,
        help="Samples per imputation query.",
    )
    parser.add_argument(
        "--reconstruction-rows",
        type=int,
        nargs="+",
        default=DEFAULT_RECON_ROWS,
        help="Rows to test predictive reconstruction on.",
    )
    parser.add_argument(
        "--query-columns",
        type=str,
        nargs="+",
        default=DEFAULT_QUERY_COLUMNS,
        help="Column names to reconstruct in predictive checks.",
    )
    parser.add_argument(
        "--condition-columns",
        type=str,
        nargs="+",
        default=DEFAULT_CONDITION_COLUMNS,
        help="Column names to condition on during predictive checks.",
    )
    parser.add_argument(
        "--impute-rows",
        type=int,
        nargs="+",
        default=DEFAULT_IMPUTE_ROWS,
        help="Rows to impute values for.",
    )
    parser.add_argument(
        "--impute-columns",
        type=str,
        nargs="+",
        default=DEFAULT_IMPUTE_COLUMNS,
        help="Column names to impute.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the GenJAX RNG.",
    )
    args = parser.parse_args()

    (
        df,
        feature_df,
        raw_matrix,
        observed_cont,
        observed_cat,
        means,
        stds,
    ) = _load_dataset(
        args.csv_path, args.id_column, args.max_rows
    )
    _assign_model_dimensions(
        num_rows=observed_cont.shape[0],
        num_cont_cols=observed_cont.shape[1],
        num_views=args.num_views,
        num_clusters=args.num_clusters,
    )

    rng = jax.random.key(args.seed)
    rng, key_obs = jax.random.split(rng)
    posterior_trace = helpers.posterior_from_observations(
        observed_cont,
        observed_cat,
        key=key_obs,
    )

    rng, key_gibbs = jax.random.split(rng)
    _, gibbs_state = helpers.run_gibbs_iterations(
        posterior_trace,
        key=key_gibbs,
        num_iters=args.iterations,
    )
    trace = gibbs_state.trace

    name_to_idx = {name: idx for idx, name in enumerate(feature_df.columns)}
    print(
        f"Loaded {len(df)} rows and {len(feature_df.columns)} numeric columns "
        f"from {args.csv_path}"
    )
    rng = _predictive_checks(
        trace,
        rng=rng,
        row_indices=args.reconstruction_rows,
        query_cols=args.query_columns,
        condition_cols=args.condition_columns,
        normalized_matrix=observed_cont,
        actual_matrix=raw_matrix,
        name_to_idx=name_to_idx,
        num_samples=args.predictive_samples,
        means=means,
        stds=stds,
    )
    rng = _impute_rows(
        trace,
        rng=rng,
        row_indices=args.impute_rows,
        impute_cols=args.impute_columns,
        actual_matrix=raw_matrix,
        name_to_idx=name_to_idx,
        num_samples=args.impute_samples,
        means=means,
        stds=stds,
    )


if __name__ == "__main__":
    main()
