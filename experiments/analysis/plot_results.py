#!/usr/bin/env python3
"""Simple log aggregator and plotter for CrossCat experiments.

Reads JSON logs under experiments/artifacts/logs and produces:
- speed_summary.csv: flattened table of avg_time_per_iter by model/dataset/rows/cols.
- convergence plots: likelihood vs iteration (using likelihood_interval) for available logs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

LOG_ROOT = pathlib.Path("experiments/artifacts/logs")
PLOT_ROOT = pathlib.Path("experiments/artifacts/plots")


def load_logs(log_root: pathlib.Path) -> List[Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []
    for path in log_root.rglob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            payload["_path"] = path
            logs.append(payload)
        except Exception:
            continue
    return logs


def ensure_output_dirs(root: pathlib.Path) -> None:
    (root / "convergence").mkdir(parents=True, exist_ok=True)
    (root / "speed").mkdir(parents=True, exist_ok=True)


def write_speed_csv(logs: List[Dict[str, Any]], out_path: pathlib.Path) -> None:
    headers = ["model", "dataset", "mode", "rows", "cols", "iterations", "avg_time_per_iter_sec", "total_time_sec", "path"]
    lines = [",".join(headers)]
    for log in logs:
        if log.get("mode") != "speed":
            continue
        res = log.get("results", {})
        avg_t = res.get("avg_time_per_iter_sec")
        total_t = res.get("total_time_sec")
        if avg_t is None:
            continue
        row = [
            str(log.get("model")),
            str(log.get("dataset")),
            str(log.get("mode")),
            str(log.get("rows")),
            str(log.get("cols")),
            str(log.get("params", {}).get("iterations")),
            str(avg_t),
            str(total_t),
            str(log.get("_path")),
        ]
        lines.append(",".join(row))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote speed summary: {out_path}")


def plot_speed(logs: List[Dict[str, Any]], out_dir: pathlib.Path) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for log in logs:
        if log.get("mode") != "speed":
            continue
        dataset = str(log.get("dataset"))
        grouped.setdefault(dataset, []).append(log)

    for dataset, entries in grouped.items():
        xs = []
        models = {}
        for log in sorted(entries, key=lambda l: (l.get("model"), int(l.get("rows", 0)))):
            res = log.get("results", {})
            avg_t = res.get("avg_time_per_iter_sec")
            if avg_t is None:
                continue
            model = str(log.get("model"))
            models.setdefault(model, {"rows": [], "avg": []})
            models[model]["rows"].append(int(log.get("rows", 0)))
            models[model]["avg"].append(float(avg_t))
        if not models:
            continue
        plt.figure()
        for model, vals in models.items():
            plt.loglog(vals["rows"], vals["avg"], marker="o", label=model)
        plt.xlabel("rows")
        plt.ylabel("avg time per iter (s)")
        plt.title(f"Speed: {dataset}")
        plt.legend()
        plt.grid(True, which="both")
        out_path = out_dir / "speed" / f"{dataset}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out_path}")


def plot_convergence(logs: List[Dict[str, Any]], out_dir: pathlib.Path) -> None:
    # Group by dataset
    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for log in logs:
        if log.get("mode") != "convergence":
            continue
        hist = log.get("results", {}).get("log_likelihood_history")
        if not hist:
            continue
        by_dataset.setdefault(str(log.get("dataset")), []).append(log)

    for dataset, entries in by_dataset.items():
        plt.figure()
        has_any = False
        for log in entries:
            hist = log.get("results", {}).get("log_likelihood_history") or []
            interval = int(log.get("params", {}).get("likelihood_interval", 1) or 1)
            xs = [interval * (i + 1) for i in range(len(hist))]
            plt.plot(xs, hist, label=f"{log.get('model')}")
            has_any = True
        if not has_any:
            plt.close()
            continue
        plt.xlabel("iteration (approx)")
        plt.ylabel("log likelihood")
        plt.title(f"Convergence: {dataset}")
        plt.legend()
        plt.grid(True)
        out_path = out_dir / "convergence" / f"{dataset}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and plot CrossCat experiment logs")
    parser.add_argument("--logs-root", type=pathlib.Path, default=LOG_ROOT, help="Root directory containing JSON logs")
    parser.add_argument("--output-dir", type=pathlib.Path, default=PLOT_ROOT, help="Directory to write plots/CSVs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs(args.output_dir)
    logs = load_logs(args.logs_root)
    if not logs:
        print(f"No logs found under {args.logs_root}")
        return
    write_speed_csv(logs, args.output_dir / "speed" / "speed_summary.csv")
    plot_speed(logs, args.output_dir)
    plot_convergence(logs, args.output_dir)


if __name__ == "__main__":
    main()
