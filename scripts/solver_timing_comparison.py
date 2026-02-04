#!/usr/bin/env python3
"""Render the solver timing comparison table without rerunning experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def format_residual(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3e}"
    return str(value) if value is not None else "N/A"


def describe_results(results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    for solver, entry in results.items():
        normalized[solver] = entry.copy()
        normalized[solver]["residual"] = format_residual(entry.get("residual"))
    return normalized


def print_table(
    results: Dict[str, Dict[str, Any]], title: str = "Solver Timing Comparison"
) -> None:
    print(f"\n{'=' * 110}")
    print(title)
    print(f"{'=' * 110}")
    print(
        f"{'Solver':<25} {'Status':<10} {'DOFs':<10} {'Iterations':<12} {'Time':<12} {'Residual':<12} {'Total Time'}"
    )
    print(f"{'-' * 110}")

    for solver, entry in results.items():
        status = "SUCCESS" if entry.get("success") else "FAILED"
        dofs = entry.get("dofs", "N/A")
        iterations = entry.get("iterations", "N/A")
        solver_time = entry.get("solver_time")
        solver_time_str = (
            f"{solver_time:.2f}s" if isinstance(solver_time, (int, float)) else "N/A"
        )
        residual_str = entry.get("residual", "N/A")
        total_time = entry.get("total_time")
        total_time_str = (
            f"{total_time:.2f}s" if isinstance(total_time, (int, float)) else "N/A"
        )
        print(
            f"{solver:<25} {status:<10} {str(dofs):<10} {str(iterations):<12} {solver_time_str:<12} {residual_str:<12} {total_time_str}"
        )

    print(f"{'=' * 110}")

    successful = {
        name: entry for name, entry in results.items() if entry.get("success")
    }
    if successful:
        fastest = min(
            successful.items(),
            key=lambda item: item[1].get("solver_time", float("inf")),
        )
        print(
            f"\n🏆 Fastest solver: {fastest[0]} ({fastest[1].get('solver_time', 0):.2f}s)"
        )
    else:
        print("\nNo successful runs available")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the solver timing comparison table"
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("results/solver_timing_comparison.json"),
        help="JSON file containing the saved comparison table",
    )
    args = parser.parse_args()

    if not args.results_file.exists():
        raise FileNotFoundError(f"Results file not found: {args.results_file}")

    with args.results_file.open("r") as f:
        data = json.load(f)

    normalized = describe_results(data)
    print_table(normalized)


if __name__ == "__main__":
    main()
