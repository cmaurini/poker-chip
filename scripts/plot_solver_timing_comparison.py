#!/usr/bin/env python3
"""Plot solver timing comparison from results/solver_timing_comparison.json.

Usage
-----
python scripts/plot_solver_timing_comparison.py

This will read the JSON file created by scripts/solver_timing_comparison.py
and save a bar plot of solver times under results/figures/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = ROOT / "results" / "solver_timing_comparison.json"
FIG_DIR = ROOT / "results" / "figures"


def load_results(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Timing results JSON not found: {path}")

    with path.open("r") as f:
        data = json.load(f)

    return data


def plot_solver_times(results: Dict[str, Any], fig_dir: Path) -> Path:
    # Filter only successful runs with a valid solver_time
    names = []
    solver_times = []
    total_times = []
    iterations = []

    for solver_name, res in results.items():
        if not res.get("success"):
            continue
        st = res.get("solver_time")
        if st is None:
            continue
        names.append(solver_name)
        solver_times.append(st)
        total_times.append(res.get("total_time", float("nan")))
        iterations.append(res.get("iterations"))

    if not names:
        raise RuntimeError(
            "No successful solver runs with solver_time found in results. "
            "Re-run scripts/solver_timing_comparison.py after installing dependencies (e.g. dolfinx)."
        )

    fig_dir.mkdir(parents=True, exist_ok=True)

    # Basic bar plot of solver_time
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(names))
    bars = ax.bar(x, solver_times, color="tab:blue", alpha=0.8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Solver time [s]")
    ax.set_title("Solver timing comparison")

    # Annotate iterations on top of bars when available
    for xi, bar, iters in zip(x, bars, iterations):
        height = bar.get_height()
        label = f"{height:.2f}s"
        if iters is not None:
            label += f"\n({iters} iter)"
        ax.text(
            xi,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()

    out_path = fig_dir / "solver_timing_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return out_path


def main() -> None:
    try:
        results = load_results(RESULTS_JSON)
    except Exception as e:
        print(f"Error loading results: {e}")
        return

    try:
        out_path = plot_solver_times(results, FIG_DIR)
    except Exception as e:
        print(f"Error plotting solver timing comparison: {e}")
        return

    print(f"Solver timing comparison plot saved to: {out_path}")


if __name__ == "__main__":
    main()
