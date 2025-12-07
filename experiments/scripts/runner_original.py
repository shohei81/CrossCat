#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runner for Original (Python 2) CrossCat experiments (stub).

Python 2/3 両対応の軽いスタブ。前処理済みデータを読み込み、行・列を揃えて
共通スキーマの JSON ログを書き出すだけで、まだ Python2 本体の CrossCat を呼んでいません。
コンテナ内では PYTHON_BIN=python2.7 を指定すれば Python2 でも動きます。
"""

from __future__ import print_function, division

import argparse
import json
import os
import time
import tempfile
import random

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
PROCESSED = os.path.join(ROOT, "experiments", "artifacts", "processed")


def parse_args():
    parser = argparse.ArgumentParser(description="Original CrossCat experiment runner (stub)")
    parser.add_argument("--mode", required=True, choices=["speed", "convergence"])
    parser.add_argument("--dataset", required=True, choices=["synthetic", "dha", "adult"])
    parser.add_argument("--rows", required=True, type=int)
    parser.add_argument("--cols", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--n-grid", required=True, type=int)
    parser.add_argument("--likelihood-interval", required=True, type=int)
    parser.add_argument("--log-output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def load_manifest(dataset):
    manifest_path = os.path.join(PROCESSED, dataset, "manifest.json")
    with open(manifest_path, "r") as f:
        return json.load(f)


def _resolve_path(path_str, dataset):
    if os.path.exists(path_str):
        return path_str
    dataset_dir = os.path.join(PROCESSED, dataset)
    candidate = os.path.join(dataset_dir, os.path.basename(path_str))
    if os.path.exists(candidate):
        return candidate
    candidate = os.path.join(PROCESSED, os.path.basename(path_str))
    if os.path.exists(candidate):
        return candidate
    raise IOError("Artifact path not found: %s" % path_str)


def select_artifact(manifest, rows, dataset):
    artifacts = sorted(manifest["artifacts"], key=lambda a: int(a["rows"]))
    for art in artifacts:
        if int(art["rows"]) >= rows:
            return {
                "base": _resolve_path(art["base"], dataset),
                "rows": int(art["rows"]),
                "cols": int(art["cols"]),
            }
    art = artifacts[-1]
    return {
        "base": _resolve_path(art["base"], dataset),
        "rows": int(art["rows"]),
        "cols": int(art["cols"]),
    }


def build_placeholder_log(args, rows, cols):
    results = {
        "total_time_sec": None,
        "compilation_time_sec": None,
        "avg_time_per_iter_sec": None,
    }
    if args.mode == "convergence":
        results["log_likelihood_history"] = []
    return {
        "note": "TODO: replace stub runner with actual Original implementation",
        "mode": args.mode,
        "model": "original",
        "dataset": args.dataset,
        "rows": rows,
        "cols": cols,
        "params": {
            "n_grid": args.n_grid,
            "iterations": args.iterations,
            "likelihood_interval": args.likelihood_interval,
            "seed": args.seed,
        },
        "results": results,
        "env": {
            "platform": "cpu",
        },
    }


def _run_original_crosscat(args, csv_path):
    """Attempt to call the Python2 CrossCat LocalEngine using CSV path. Returns (results, error_str)."""
    try:
        import crosscat.LocalEngine as le  # type: ignore
        import crosscat.utils.data_utils as du  # type: ignore
        import crosscat.utils.diagnostic_utils as diag  # type: ignore
    except Exception as exc:
        return None, "ImportError: %s" % exc

    try:
        T, M_r, M_c = du.read_model_data_from_csv(csv_path, gen_seed=args.seed)
    except Exception as exc:
        return None, "read_model_data_from_csv failed: %s" % exc

    engine = le.LocalEngine(args.seed)
    # CrossCat expects a separate seed generator; keep it simple with RNG.
    rng = random.Random(args.seed)
    next_seed = lambda: rng.randint(1, 2 ** 31 - 1)

    do_diag = args.mode == "convergence"
    diag_every = max(1, int(args.likelihood_interval)) if do_diag else 0
    t0 = time.time()
    try:
        X_L_list, X_D_list = engine.initialize(
            M_c, M_r, T, next_seed(), initialization="from_the_prior", n_chains=1
        )
        if do_diag:
            X_L_list, X_D_list, diagnostics = engine.analyze(
                M_c,
                T,
                X_L_list,
                X_D_list,
                next_seed(),
                n_steps=args.iterations,
                do_diagnostics=True,
                diagnostics_every_N=diag_every,
            )
        else:
            X_L_list, X_D_list = engine.analyze(
                M_c, T, X_L_list, X_D_list, next_seed(), n_steps=args.iterations
            )

        log_history = []
        if do_diag:
            # Prefer diagnostics dict emitted by analyze when do_diagnostics=True
            got_log = False
            try:
                if diagnostics and isinstance(diagnostics, dict) and "logscore" in diagnostics:
                    logscore = diagnostics["logscore"]
                    if hasattr(logscore, "__iter__"):
                        log_history = [float(x) for x in logscore]
                    else:
                        log_history = [float(logscore)]
                    got_log = True
            except Exception:
                pass
            if not got_log:
                # Try diagnostic_utils.get_logscore (list or scalar)
                try:
                    logscore = diag.get_logscore(M_c, X_L_list, X_D_list, T)
                    if hasattr(logscore, "__iter__"):
                        log_history = [float(x) for x in logscore]
                    else:
                        log_history = [float(logscore)]
                    got_log = True
                except Exception:
                    # Try single-chain call
                    try:
                        logscore = diag.get_logscore(M_c, X_L_list[0], X_D_list[0], T)
                        log_history = [float(logscore)]
                        got_log = True
                    except Exception:
                        pass
            if not got_log and hasattr(engine, "get_logscore"):
                try:
                    for chain_idx in range(len(X_L_list)):
                        logscore = engine.get_logscore(M_c, X_L_list[chain_idx], X_D_list[chain_idx], T)
                        log_history.append(float(logscore))
                    got_log = True
                except Exception:
                    pass
            if not got_log:
                # Try to read logscore stack from X_L_list if present.
                try:
                    xs = X_L_list[0].get("logscore", [])
                    if xs:
                        log_history = [float(x) for x in xs]
                        got_log = True
                except Exception:
                    pass
            if not got_log:
                log_history.append(float("nan"))
    except Exception as exc:
        return None, "analyze() failed: %s" % exc
    t1 = time.time()

    return {
        "total_time_sec": t1 - t0,
        "compilation_time_sec": None,
        "avg_time_per_iter_sec": float(t1 - t0) / float(args.iterations),
        "log_likelihood_history": log_history if args.mode == "convergence" else [],
    }, None


def main():
    args = parse_args()
    manifest = load_manifest(args.dataset)
    artifact = select_artifact(manifest, args.rows, args.dataset)
    df = pd.read_csv(artifact["base"])
    if args.rows < artifact["rows"]:
        df = df.sample(n=args.rows, random_state=args.seed, replace=False).reset_index(drop=True)
    if args.cols > artifact["cols"]:
        raise ValueError("Requested cols=%d exceeds available %d" % (args.cols, artifact["cols"]))
    df = df.iloc[:, : args.cols]

    # Write a temporary CSV (limited rows/cols) for the Py2 utilities to consume.
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False, encoding="utf-8")

    results, err = _run_original_crosscat(args, tmp.name)
    payload = build_placeholder_log(args, rows=len(df), cols=args.cols)
    if results is not None:
        payload["results"] = results
        payload["note"] = "Ran crosscat.LocalEngine.analyze successfully"
        if args.mode == "convergence":
            if results.get("log_likelihood_history") == [float("nan")]:
                payload.setdefault("notes", {})["original_error"] = "logscore unavailable; log_likelihood_history filled with nan"
            else:
                payload.setdefault("notes", {})["logscore_source"] = "diagnostic_utils/get_logscore or X_L.logscore"
    if err:
        payload.setdefault("notes", {})["original_error"] = err

    log_dir = os.path.dirname(args.log_output)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(args.log_output, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print("[stub] wrote placeholder log to %s" % args.log_output)


if __name__ == "__main__":
    main()
