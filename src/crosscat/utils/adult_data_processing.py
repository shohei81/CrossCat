"""Utilities for loading and preprocessing the UCI Adult dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import jax.numpy as jnp
import numpy as np
import pandas as pd

ADULT_FILES = ("adult.data", "adult.test")

ALL_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

CONTINUOUS_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

TARGET_COLUMN = "income"
LABEL_MAPPING = {"<=50K": 0, ">50K": 1}
NORMALIZATION_EPS = 1e-8


@dataclass
class PreprocessedAdultData:
    """Container describing the processed Adult dataset."""

    dataframe: pd.DataFrame
    observed_cont: jnp.ndarray
    observed_cat: jnp.ndarray
    labels: jnp.ndarray
    continuous_columns: Sequence[str]
    categorical_columns: Sequence[str]
    label_mapping: Mapping[str, int]
    cont_means: np.ndarray
    cont_stds: np.ndarray
    categorical_vocabs: Mapping[str, Mapping[str, int]]


class AdultCategoricalEncoder:
    """Simple vocabulary-based encoder to map strings -> integer ids."""

    def __init__(self, unknown_value: int = -1):
        self.unknown_value = unknown_value
        self._columns: List[str] = []
        self._vocab: Dict[str, Dict[str, int]] = {}

    @property
    def vocab(self) -> Mapping[str, Mapping[str, int]]:
        return self._vocab

    def fit(self, df: pd.DataFrame, columns: Sequence[str]):
        self._columns = list(columns)
        if not self._columns:
            self._vocab = {}
            return self
        vocab: Dict[str, Dict[str, int]] = {}
        for col in self._columns:
            categories = sorted({str(value).strip() for value in df[col].unique()})
            vocab[col] = {category: idx for idx, category in enumerate(categories)}
        self._vocab = vocab
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._columns:
            if not self._vocab:
                return np.zeros((len(df), 0), dtype=np.int32)
            raise ValueError("Encoder has not been fit.")
        encoded_columns: List[np.ndarray] = []
        for col in self._columns:
            mapping = self._vocab[col]
            values = (
                df[col]
                .astype(str)
                .str.strip()
                .map(mapping)
                .fillna(self.unknown_value)
                .to_numpy(dtype=np.int32)
            )
            encoded_columns.append(values)
        if not encoded_columns:
            return np.zeros((len(df), 0), dtype=np.int32)
        return np.stack(encoded_columns, axis=1)


class AdultScaler:
    """Stores feature-wise normalization statistics."""

    def __init__(self, eps: float = NORMALIZATION_EPS):
        self.eps = eps
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None

    def fit(self, values: np.ndarray):
        self.means = values.mean(axis=0)
        stds = values.std(axis=0)
        self.stds = np.where(stds < self.eps, 1.0, stds)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.means is None or self.stds is None:
            raise ValueError("Scaler has not been fit.")
        return (values - self.means) / self.stds


def _read_raw_adult_frames(data_dir: Path) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for filename in ADULT_FILES:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Adult dataset file missing: {path}")
        df = pd.read_csv(
            path,
            header=None,
            names=ALL_COLUMNS,
            na_values=["?"],
            skipinitialspace=True,
            comment="|",
        )
        frames.append(df)
    return frames


def _clean_dataframe(df: pd.DataFrame, *, drop_missing: bool) -> pd.DataFrame:
    clean_df = df.copy()
    clean_df[TARGET_COLUMN] = (
        clean_df[TARGET_COLUMN].astype(str).str.strip().str.replace(".", "", regex=False)
    )
    if drop_missing:
        clean_df = clean_df.dropna()
    return clean_df.reset_index(drop=True)


def load_adult_dataset(
    data_dir: Path | str,
    *,
    drop_missing: bool = True,
    normalize: bool = True,
    continuous_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
) -> PreprocessedAdultData:
    """Load, clean, and encode the Adult dataset as numpy/JAX arrays."""

    data_path = Path(data_dir)
    cont_cols = list(continuous_columns or CONTINUOUS_COLUMNS)
    cat_cols = list(categorical_columns or CATEGORICAL_COLUMNS)

    frames = _read_raw_adult_frames(data_path)
    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged = _clean_dataframe(merged, drop_missing=drop_missing)

    cont_matrix = merged[cont_cols].to_numpy(dtype=np.float32)
    scaler = AdultScaler()
    if normalize:
        scaler.fit(cont_matrix)
        normalized_cont = scaler.transform(cont_matrix)
    else:
        scaler.fit(cont_matrix)
        normalized_cont = cont_matrix

    encoder = AdultCategoricalEncoder()
    cat_df = merged[cat_cols]
    encoder.fit(cat_df, cat_cols)
    cat_matrix = encoder.transform(cat_df)

    labels = (
        merged[TARGET_COLUMN]
        .map(LABEL_MAPPING)
        .to_numpy(dtype=np.int32)
    )

    observed_cont = jnp.asarray(normalized_cont, dtype=jnp.float32)
    observed_cat = jnp.asarray(cat_matrix, dtype=jnp.int32)
    label_array = jnp.asarray(labels, dtype=jnp.int32)

    return PreprocessedAdultData(
        dataframe=merged,
        observed_cont=observed_cont,
        observed_cat=observed_cat,
        labels=label_array,
        continuous_columns=cont_cols,
        categorical_columns=cat_cols,
        label_mapping=LABEL_MAPPING,
        cont_means=scaler.means.copy(),
        cont_stds=scaler.stds.copy(),
        categorical_vocabs=encoder.vocab,
    )


def summarize_dataset(data: PreprocessedAdultData) -> str:
    """Return a short textual summary describing the processed dataset."""
    num_rows = data.dataframe.shape[0]
    cont_cols = len(data.continuous_columns)
    cat_cols = len(data.categorical_columns)
    summary = [
        f"Rows: {num_rows}",
        f"Continuous cols: {cont_cols}",
        f"Categorical cols: {cat_cols}",
        f"Observed cont shape: {tuple(data.observed_cont.shape)}",
        f"Observed cat shape: {tuple(data.observed_cat.shape)}",
        f"Label distribution: {data.dataframe[TARGET_COLUMN].value_counts().to_dict()}",
    ]
    return "\n".join(summary)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/adult"),
        help="Directory containing adult.data and adult.test files.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable standardization of continuous columns.",
    )
    args = parser.parse_args()

    dataset = load_adult_dataset(
        args.data_dir,
        drop_missing=True,
        normalize=not args.no_normalize,
    )
    print(summarize_dataset(dataset))
