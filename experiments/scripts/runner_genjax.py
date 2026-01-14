#!/usr/bin/env python3
"""GenJAX experiment runner.

ロード済みの前処理アーティファクト (manifest.json) を参照し、
指定された rows/cols のデータを読み込み、CrossCat の Gibbs サンプラーを実行して
共通の JSON スキーマでログを書き出す。

速度モード(speed): 尤度計算なしで指定イテレーションを回し、時間を記録。
収束モード(convergence): 同じくイテレーションを回し、現状は尤度ログ未実装。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from crosscat import constants as const
from crosscat.gibbs import GibbsConfig, GibbsState, gibbs_step
from crosscat.inference import _extract_hyperparams_from_trace, infer_mixed_sbp_multiview_table
from crosscat.model import _default_args

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "experiments" / "artifacts" / "processed"


@dataclass
class Artifact:
    base: pathlib.Path
    masked: pathlib.Path
    mask: pathlib.Path
    rows: int
    cols: int


def load_manifest(dataset: str) -> Dict[str, Any]:
    manifest_path = PROCESSED / dataset / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(path_str: str, dataset: str) -> pathlib.Path:
    path = pathlib.Path(path_str)
    if path.exists():
        return path
    dataset_dir = PROCESSED / dataset
    candidate = dataset_dir / path.name
    if candidate.exists():
        return candidate
    candidate = PROCESSED / path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Artifact path not found: {path_str}")


def select_artifact(manifest: Dict[str, Any], rows: int, dataset: str) -> Artifact:
    artifacts = sorted(manifest["artifacts"], key=lambda a: int(a["rows"]))
    # rows 以上の最小を選ぶ。無ければ最大行のアーティファクトを使い、後でダウンサンプリング。
    for art in artifacts:
        if int(art["rows"]) >= rows:
            return Artifact(
                base=_resolve_path(art["base"], dataset),
                masked=_resolve_path(art["masked"], dataset),
                mask=_resolve_path(art["mask"], dataset),
                rows=int(art["rows"]),
                cols=int(art["cols"]),
            )
    art = artifacts[-1]
    return Artifact(
        base=_resolve_path(art["base"], dataset),
        masked=_resolve_path(art["masked"], dataset),
        mask=_resolve_path(art["mask"], dataset),
        rows=int(art["rows"]),
        cols=int(art["cols"]),
    )


def split_columns(df: pd.DataFrame, manifest: Dict[str, Any], cols_limit: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    column_meta = manifest["columns"]
    selected_cols = [meta["name"] for meta in column_meta][:cols_limit]
    df = df[selected_cols]

    cont_cols = [c for c, meta in zip(selected_cols, column_meta) if meta["kind"] == "continuous"][:cols_limit]
    cat_cols = [c for c, meta in zip(selected_cols, column_meta) if meta["kind"] == "categorical"][:cols_limit]

    df_cont = df[cont_cols] if cont_cols else pd.DataFrame(index=df.index)
    df_cat = df[cat_cols] if cat_cols else pd.DataFrame(index=df.index)
    return df_cont, df_cat


def encode_categoricals(df_cat: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, List[str]], int]:
    if df_cat.shape[1] == 0:
        return np.zeros((len(df_cat), 0), dtype=np.int32), {}, 0

    encoded_cols = []
    mapping: Dict[str, List[str]] = {}
    max_card = 0
    for col in df_cat.columns:
        cat = pd.Categorical(df_cat[col])
        codes = cat.codes.astype(np.int32)
        encoded_cols.append(codes)
        categories = cat.categories.tolist()
        mapping[col] = categories
        max_card = max(max_card, len(categories))
    encoded = np.stack(encoded_cols, axis=1)
    return encoded, mapping, max_card


def configure_constants(
    n_rows: int,
    n_cont: int,
    n_cat: int,
    num_categories: int,
    override_views: int | None,
    override_clusters: int | None,
    override_categories: int | None,
) -> None:
    const.NUM_ROWS = n_rows
    const.NUM_CONT_COLS = n_cont
    const.NUM_CAT_COLS = n_cat
    const.NUM_VIEWS = override_views or max(2, min(n_cont + n_cat, 5))
    const.NUM_CLUSTERS = override_clusters or max(3, min(n_rows, 10))
    const.NUM_CATEGORIES = override_categories or max(2, num_categories)


def build_initial_trace(
    cont_arr: np.ndarray,
    cat_arr: np.ndarray,
    *,
    seed: int,
):
    key = jax.random.key(seed)
    observed_cont = jnp.asarray(cont_arr, dtype=jnp.float32)
    observed_cat = jnp.asarray(cat_arr, dtype=jnp.int32)
    trace = infer_mixed_sbp_multiview_table(observed_cont, observed_cat, key=key)
    return trace, key


def init_state_from_trace(key: jax.Array, trace) -> GibbsState:
    mu0_vec, kappa0_vec, ng_alpha0_vec, ng_beta0_vec, alpha_cat_vec = _extract_hyperparams_from_trace(trace)
    return GibbsState(
        key,
        trace,
        const.ALPHA_VIEW,
        const.ALPHA_CLUSTER,
        alpha_cat_vec,
        mu0_vec,
        kappa0_vec,
        ng_alpha0_vec,
        ng_beta0_vec,
    )


def run_speed(state: GibbsState, iterations: int) -> Tuple[GibbsState, Dict[str, float]]:
    compiled_step = jax.jit(gibbs_step)
    t0 = time.perf_counter()
    # First call triggers compilation.
    state = compiled_step(state)
    jax.block_until_ready(state)
    t1 = time.perf_counter()
    for _ in range(iterations - 1):
        state = compiled_step(state)
    jax.block_until_ready(state)
    t2 = time.perf_counter()
    return state, {
        "compilation_time_sec": t1 - t0,
        "total_time_sec": t2 - t0,
        "avg_time_per_iter_sec": (t2 - t0) / iterations,
    }


def run_convergence(state: GibbsState, iterations: int, interval: int) -> Tuple[GibbsState, Dict[str, Any]]:
    compiled_step = jax.jit(gibbs_step)
    log_likelihood_history: List[float] = []
    t0 = time.perf_counter()
    state = compiled_step(state)
    jax.block_until_ready(state)
    t1 = time.perf_counter()
    for i in range(1, iterations):
        state = compiled_step(state)
        if (i + 1) % interval == 0:
            # TODO: implement proper marginal log-likelihood; placeholder for now.
            log_likelihood_history.append(float("nan"))
    jax.block_until_ready(state)
    t2 = time.perf_counter()
    return state, {
        "compilation_time_sec": t1 - t0,
        "total_time_sec": t2 - t0,
        "avg_time_per_iter_sec": (t2 - t0) / iterations,
        "log_likelihood_history": log_likelihood_history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GenJAX experiment runner")
    parser.add_argument("--mode", required=True, choices=["speed", "convergence"])
    parser.add_argument("--dataset", required=True, choices=["synthetic", "dha", "adult"])
    parser.add_argument("--rows", required=True, type=int)
    parser.add_argument("--cols", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--n-grid", required=True, type=int)
    parser.add_argument("--likelihood-interval", required=True, type=int)
    parser.add_argument("--log-output", required=True, type=pathlib.Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--num-views", type=int, help="Override NUM_VIEWS")
    parser.add_argument("--num-clusters", type=int, help="Override NUM_CLUSTERS")
    parser.add_argument("--num-categories", type=int, help="Override NUM_CATEGORIES")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.dataset)
    artifact = select_artifact(manifest, args.rows, args.dataset)
    df = pd.read_csv(artifact.base)

    if args.rows < artifact.rows:
        df = df.sample(n=args.rows, random_state=args.seed, replace=False).reset_index(drop=True)

    if args.cols > artifact.cols:
        raise ValueError(f"Requested cols={args.cols} exceeds available {artifact.cols}")

    df_cont, df_cat = split_columns(df, manifest, args.cols)
    cont_arr = df_cont.to_numpy(dtype=np.float32) if df_cont.shape[1] > 0 else np.zeros((len(df), 0), dtype=np.float32)
    cat_arr_np, cat_mapping, max_card = encode_categoricals(df_cat)

    configure_constants(
        len(df),
        cont_arr.shape[1],
        cat_arr_np.shape[1],
        max_card if max_card > 0 else 2,
        args.num_views,
        args.num_clusters,
        args.num_categories,
    )

    error_note = None
    stats = {}
    try:
        trace, key = build_initial_trace(cont_arr, cat_arr_np, seed=args.seed)
        state = init_state_from_trace(key, trace)
        if args.mode == "speed":
            state, stats = run_speed(state, args.iterations)
        else:
            state, stats = run_convergence(state, args.iterations, args.likelihood_interval)
    except Exception as exc:  # noqa: BLE001
        error_note = f"genjax_error: {exc}"
        stats = {"error": str(exc)}

    payload = {
        "mode": args.mode,
        "model": "genjax",
        "dataset": args.dataset,
        "rows": len(df),
        "cols": args.cols,
        "params": {
            "n_grid": args.n_grid,
            "iterations": args.iterations,
            "likelihood_interval": args.likelihood_interval,
            "seed": args.seed,
            "num_views": const.NUM_VIEWS,
            "num_clusters": const.NUM_CLUSTERS,
            "num_categories": const.NUM_CATEGORIES,
        },
        "results": stats,
        "env": {
            "platform": jax.default_backend(),
        },
        "notes": {
            "categorical_mapping": cat_mapping,
            "overrides": {
                "num_views": args.num_views,
                "num_clusters": args.num_clusters,
                "num_categories": args.num_categories,
            },
        },
    }
    if error_note:
        payload["notes"]["genjax_error"] = error_note

    args.log_output.parent.mkdir(parents=True, exist_ok=True)
    with args.log_output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Wrote log to {args.log_output}")


if __name__ == "__main__":
    main()
