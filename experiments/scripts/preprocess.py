#!/usr/bin/env python3
"""
CrossCat 実験の前処理エントリポイント。

役割:
- 各データセットの前処理/サブセット作成（欠損マスク付与を含む）。
- Adult については 100/1k/10k/50k 行のサブセットを生成。
- 共通の artifacts/processed 配下に CSV・マスク・メタデータを出力。
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "experiments" / "artifacts"
PROCESSED = ARTIFACTS / "processed"


@dataclass
class ColumnMeta:
    name: str
    kind: str  # "continuous" or "categorical"


def detect_column_meta(df: pd.DataFrame) -> List[ColumnMeta]:
    meta: List[ColumnMeta] = []
    for col in df.columns:
        kind = "categorical"
        if pd.api.types.is_numeric_dtype(df[col]):
            kind = "continuous"
        meta.append(ColumnMeta(name=col, kind=kind))
    return meta


def apply_missing_mask(df: pd.DataFrame, frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    mask = rng.random(df.shape) < frac
    masked_df = df.copy()
    masked_df = masked_df.mask(mask)
    mask_df = pd.DataFrame(mask, columns=df.columns)
    return masked_df, mask_df


def save_artifacts(
    base_name: str,
    df: pd.DataFrame,
    output_dir: pathlib.Path,
    missing_frac: float,
    seed: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / f"{base_name}.csv"
    df.to_csv(base_path, index=False)

    masked_df, mask_df = apply_missing_mask(df, missing_frac, seed)
    masked_path = output_dir / f"{base_name}_masked.csv"
    mask_path = output_dir / f"{base_name}_mask.csv"
    masked_df.to_csv(masked_path, index=False)
    mask_df.to_csv(mask_path, index=False)

    return {
        "base": str(base_path),
        "masked": str(masked_path),
        "mask": str(mask_path),
        "missing_fraction": missing_frac,
        "rows": len(df),
        "cols": df.shape[1],
    }


def preprocess_synthetic(args: argparse.Namespace) -> Dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    rows = args.synthetic_rows
    cols = args.synthetic_cols
    data = rng.normal(loc=0.0, scale=1.0, size=(rows, cols))
    col_names = [f"x{i+1}" for i in range(cols)]
    df = pd.DataFrame(data, columns=col_names)

    dataset_dir = args.output_dir / "synthetic"
    artifacts = save_artifacts(
        base_name=f"synthetic_r{rows}_c{cols}",
        df=df,
        output_dir=dataset_dir,
        missing_frac=args.missing_fraction,
        seed=args.seed,
    )

    return {
        "dataset": "synthetic",
        "rows": rows,
        "cols": cols,
        "columns": [asdict(meta) for meta in detect_column_meta(df)],
        "artifacts": [artifacts],
    }


def preprocess_dha(args: argparse.Namespace) -> Dict[str, Any]:
    source = args.dha_source or (ROOT / "data" / "dha.csv")
    df = pd.read_csv(source)
    # 欠損行は現状ドロップ（今後の欠損マスク処理は別ファイルで保持）
    df = df.dropna().reset_index(drop=True)
    dataset_dir = args.output_dir / "dha"
    artifacts = save_artifacts(
        base_name="dha_full",
        df=df,
        output_dir=dataset_dir,
        missing_frac=args.missing_fraction,
        seed=args.seed,
    )
    return {
        "dataset": "dha",
        "rows": len(df),
        "cols": df.shape[1],
        "columns": [asdict(meta) for meta in detect_column_meta(df)],
        "artifacts": [artifacts],
        "source": str(source),
    }


def preprocess_adult(args: argparse.Namespace) -> Dict[str, Any]:
    train_path = args.adult_train or (ROOT / "data" / "adult" / "adult.data")
    test_path = args.adult_test or (ROOT / "data" / "adult" / "adult.test")
    col_names = [
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

    def _load(path: pathlib.Path) -> pd.DataFrame:
        df_part = pd.read_csv(
            path,
            names=col_names,
            na_values=["?"],
            skipinitialspace=True,
            comment="#",
        )
        # income列に付く末尾ドットを削除（testファイル）
        df_part["income"] = df_part["income"].astype(str).str.replace(".", "", regex=False).str.strip()
        return df_part

    df_train = _load(train_path)
    df_test = _load(test_path)
    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df.dropna().reset_index(drop=True)
    dataset_dir = args.output_dir / "adult"
    artifacts: List[Dict[str, Any]] = []

    max_rows = len(df)
    for rows in args.adult_rows:
        if rows > max_rows:
            continue
        subset = df.sample(n=rows, random_state=args.seed, replace=False).reset_index(drop=True)
        art = save_artifacts(
            base_name=f"adult_r{rows}",
            df=subset,
            output_dir=dataset_dir,
            missing_frac=args.missing_fraction,
            seed=args.seed,
        )
        artifacts.append(art)

    return {
        "dataset": "adult",
        "rows": len(df),
        "cols": df.shape[1],
        "columns": [asdict(meta) for meta in detect_column_meta(df)],
        "artifacts": artifacts,
        "train": str(train_path),
        "test": str(test_path),
    }


def write_manifest(manifest: Dict[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess datasets for CrossCat experiments")
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "dha", "adult"],
        required=True,
        help="Which dataset to preprocess",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=PROCESSED,
        help="Directory to place processed CSVs and metadata",
    )
    parser.add_argument(
        "--adult-train",
        type=pathlib.Path,
        help="Path to adult.data (train). Defaults to data/adult/adult.data",
    )
    parser.add_argument(
        "--adult-test",
        type=pathlib.Path,
        help="Path to adult.test (test). Defaults to data/adult/adult.test",
    )
    parser.add_argument(
        "--dha-source",
        type=pathlib.Path,
        help="Path to the raw DHA dataset (CSV). Defaults to data/dha.csv",
    )
    parser.add_argument(
        "--adult-rows",
        type=int,
        nargs="+",
        default=[100, 1000, 10000, 50000],
        help="Row counts for Adult subsets (skips if larger than available rows)",
    )
    parser.add_argument(
        "--synthetic-rows",
        type=int,
        default=1000,
        help="Row count for synthetic data",
    )
    parser.add_argument(
        "--synthetic-cols",
        type=int,
        default=10,
        help="Column count for synthetic data",
    )
    parser.add_argument(
        "--missing-fraction",
        type=float,
        default=0.1,
        help="Fraction of entries to mask as missing in masked outputs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for masking and synthetic generation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.missing_fraction <= 1.0:
        raise ValueError("--missing-fraction must be in [0, 1]")
    if args.dataset == "synthetic":
        manifest = preprocess_synthetic(args)
    elif args.dataset == "dha":
        manifest = preprocess_dha(args)
    elif args.dataset == "adult":
        manifest = preprocess_adult(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    write_manifest(manifest, args.output_dir / args.dataset / "manifest.json")
    print(f"Wrote manifest: {args.output_dir / args.dataset / 'manifest.json'}")


if __name__ == "__main__":
    main()
