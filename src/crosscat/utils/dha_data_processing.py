"""Utilities for loading and normalizing the DHA dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import numpy as np
import pandas as pd


@dataclass
class PreprocessedDHAData:
    dataframe: pd.DataFrame
    observed_cont: jnp.ndarray
    observed_cat: jnp.ndarray
    column_names: Sequence[str]
    cont_means: np.ndarray
    cont_stds: np.ndarray


def _standardize(matrix: np.ndarray, eps: float = 1e-8):
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds = np.where(stds < eps, 1.0, stds)
    normalized = (matrix - means) / stds
    return normalized, means, stds


def load_dha_dataset(
    csv_path: Path | str,
    *,
    id_column: str = "NAME",
    max_rows: int | None = None,
    normalize: bool = True,
) -> PreprocessedDHAData:
    """Load and normalize the DHA dataset."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"DHA dataset not found at {path}")

    df = pd.read_csv(path)
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' missing from {path}")

    if max_rows is not None:
        df = df.head(max_rows)
    df = df.reset_index(drop=True)

    feature_df = df.drop(columns=[id_column])
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.dropna(axis=0, how="any")
    feature_df = feature_df.reset_index(drop=True)
    matrix = feature_df.to_numpy(dtype=np.float32)

    if normalize:
        normalized, means, stds = _standardize(matrix)
    else:
        normalized = matrix
        means = np.zeros(matrix.shape[1], dtype=np.float32)
        stds = np.ones(matrix.shape[1], dtype=np.float32)

    observed_cont = jnp.asarray(normalized, dtype=jnp.float32)
    observed_cat = jnp.zeros((observed_cont.shape[0], 0), dtype=jnp.int32)

    return PreprocessedDHAData(
        dataframe=feature_df,
        observed_cont=observed_cont,
        observed_cat=observed_cat,
        column_names=feature_df.columns.tolist(),
        cont_means=means,
        cont_stds=stds,
    )


def build_crosscat_rows(
    data: PreprocessedDHAData,
) -> tuple[list[list], list[str], list[str]]:
    """Prepare standard rows/metadata for the original crosscat API."""
    rows = data.dataframe.to_numpy(dtype=float).tolist()
    column_names = list(data.column_names)
    cctypes = ["continuous"] * len(column_names)
    return rows, column_names, cctypes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to data/dha.csv",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows to load.",
    )
    args = parser.parse_args()

    dataset = load_dha_dataset(args.csv_path, max_rows=args.max_rows)
    print(
        f"Loaded {dataset.observed_cont.shape[0]} rows and "
        f"{dataset.observed_cont.shape[1]} columns from {args.csv_path}"
    )
