#!/usr/bin/env python3
"""Plot solver timing comparison from a Hydra multirun directory.

Usage
-----
python scripts/plot_solver_timing_multirun.py PATH_TO_MULTIRUN_DIR

Example
-------
# 3D timing study over solver types
python solvers/poker_chip_linear.py \
  --config-dir config --config-name config_linear --multirun \
  geometry.geometric_dimension=3 geometry.L=5.0 geometry.H=1.0 geometry.h_div=12 \
  output_name=timing_study \
  solver_type=hypre_amg,gamg,fieldsplit,ilu,auto \
  use_iterative_solver=true

Then plot with:
  python scripts/plot_solver_timing_multirun.py \
    results/timing_study/multirun_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import yaml
from petsc4py import PETSc


def describe_reason(raw_reason: Any) -> str:
    """Return a descriptive PETSc convergence reason name if available."""

    reason_str = str(raw_reason)
    try:
        code = int(raw_reason)
    except (TypeError, ValueError):
        return reason_str

    try:
        return PETSc.KSP.ConvergedReason(code).name
    except Exception:
        return f"{reason_str} (unknown PETSc reason)"


def find_run_dirs(multirun_dir: Path) -> List[Path]:
    """Return all leaf directories in a multirun folder containing JSON results."""
    if not multirun_dir.exists():
        raise FileNotFoundError(f"Multirun directory does not exist: {multirun_dir}")

    run_dirs: List[Path] = []
    for d in multirun_dir.rglob("*"):
        if not d.is_dir():
            continue
        # Skip Hydra metadata dirs
        if d.name == ".hydra":
            continue
        # Consider it a run dir if it has at least one non-Hydra JSON file
        json_files = [p for p in d.glob("*.json") if ".hydra" not in str(p.parent)]
        if json_files:
            run_dirs.append(d)

    if not run_dirs:
        raise RuntimeError(
            f"No run directories with JSON files found under {multirun_dir}"
        )

    return sorted(set(run_dirs))


def load_run_data(run_dir: Path) -> Tuple[str, float, int, str]:
    """Load solver config and timing from a single run directory.

    Returns
    -------
    label : str
        Human-readable label combining solver_type and use_iterative_solver.
    solver_time : float
        Solver wall-clock time (last load step).
    iterations : int
        KSP iteration count (last load step).
    reason : str
        KSP convergence reason (e.g. CONVERGED_RTOL).
    """

    # Load config to get solver parameters
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing Hydra config in run dir: {cfg_path}")

    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    solver_type = cfg.get("solver_type", "default")
    use_iter = cfg.get("use_iterative_solver", True)

    # Find a data JSON (assume the first one is the main history file)
    json_files = [p for p in run_dir.glob("*.json") if not p.name.startswith("config")]
    if not json_files:
        raise RuntimeError(f"No data JSON files found in {run_dir}")

    data_path = sorted(json_files)[0]
    with data_path.open("r") as f:
        data = json.load(f)

    solver_times = data.get("solver_time", [])
    iterations = data.get("iterations", [])
    reasons = data.get("ksp_reason", [])

    if not solver_times or not iterations:
        raise RuntimeError(
            f"Run {run_dir} does not contain solver_time/iterations; "
            "make sure you're using the updated solver that records these."
        )

    solver_time = float(solver_times[-1])
    iters = int(iterations[-1])

    if reasons:
        reason = describe_reason(reasons[-1])
    else:
        reason = "UNKNOWN"

    iter_str = "iterative" if use_iter else "direct"
    label = f"{solver_type} ({iter_str})"

    return label, solver_time, iters, reason


def collect_timing(
    multirun_dir: Path,
) -> Tuple[List[str], List[float], List[int], List[str]]:
    names: List[str] = []
    times: List[float] = []
    iters: List[int] = []
    reasons: List[str] = []

    for run_dir in find_run_dirs(multirun_dir):
        try:
            label, t, k, r = load_run_data(run_dir)
        except Exception as e:
            print(f"Skipping {run_dir}: {e}")
            continue

        names.append(label)
        times.append(t)
        iters.append(k)
        reasons.append(r)

    if not names:
        raise RuntimeError("No valid runs with timing data found.")

    # Deduplicate labels by averaging if the same label appears multiple times
    aggregated: Dict[str, List[Tuple[float, int, str]]] = {}
    for name, t, k, r in zip(names, times, iters, reasons):
        aggregated.setdefault(name, []).append((t, k, r))

    final_names: List[str] = []
    final_times: List[float] = []
    final_iters: List[float] = []
    final_reasons: List[str] = []

    for name, vals in aggregated.items():
        ts, ks, rs = zip(*vals)
        final_names.append(name)
        final_times.append(sum(ts) / len(ts))
        final_iters.append(sum(ks) / len(ks))
        # If multiple reasons appear, just keep the first (they should match)
        final_reasons.append(str(rs[0]))

    return final_names, final_times, [int(round(k)) for k in final_iters], final_reasons


def plot_solver_timing(multirun_dir: Path) -> Path:
    names, times, iters, reasons = collect_timing(multirun_dir)

    # Sort by time (ascending)
    order = sorted(range(len(names)), key=lambda i: times[i])
    names = [names[i] for i in order]
    times = [times[i] for i in order]
    iters = [iters[i] for i in order]
    reasons = [reasons[i] for i in order]

    # Print a plain-text table to stdout
    print("\nSolver timing (sorted by time):")
    print("-" * 96)
    print(f"{'Solver':<30} {'Time [s]':>10} {'Iterations':>12} {'Reason':>40}")
    print("-" * 96)
    for name, t, k, r in zip(names, times, iters, reasons):
        print(f"{name:<30} {t:10.3f} {k:12d} {r:>40}")
    print("-" * 96)

    # Also create a bar plot for convenience
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(names))
    bars = ax.bar(x, times, color="tab:blue", alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Solver time [s]")
    ax.set_title("Solver timing comparison (Hydra multirun)")

    for xi, bar, it in zip(x, bars, iters):
        h = bar.get_height()
        label = f"{h:.2f}s\n({it} iters)"
        ax.text(xi, h, label, ha="center", va="bottom", fontsize=8)

    fig.tight_layout()

    out_dir = multirun_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "solver_timing_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"\nSaved timing comparison plot to: {out_path}")
    return out_path


def main(argv: List[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv

    if len(argv) != 1:
        print(
            "Usage: python scripts/plot_solver_timing_multirun.py PATH_TO_MULTIRUN_DIR"
        )
        sys.exit(1)

    multirun_dir = Path(argv[0]).resolve()

    try:
        plot_solver_timing(multirun_dir)
    except Exception as e:
        print(f"Error while plotting timing comparison: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
