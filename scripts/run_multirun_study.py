#!/usr/bin/env python3
"""
Simple script to run Hydra multirun parametric study.

This script demonstrates how to use Hydra's multirun feature to sweep over parameters.
"""

import subprocess
import sys
from pathlib import Path


def run_aspect_ratio_study():
    """Run parametric study varying aspect ratios."""

    # Fixed parameters
    L = 1.0  # Keep L constant
    h_div = 4  # Elements through thickness (small for faster runs)
    gdim = 2  # 2D analysis

    # Vary H to get different aspect ratios
    # Aspect ratios will be: L/H = 1.0/H
    H_values = [1.0, 0.5, 0.2, 0.1]  # Gives aspect ratios: 1, 2, 5, 10
    H_str = ",".join(map(str, H_values))

    print("=== Running Hydra Multirun Parametric Study ===")
    print(f"Fixed L = {L}")
    print(f"H values = {H_values}")
    print(f"Aspect ratios = {[L / H for H in H_values]}")
    print(f"Elements through thickness = {h_div}")
    print(f"Geometric dimension = {gdim}D")

    # Build command
    script_dir = Path(__file__).parent.parent
    solver_script = script_dir / "solvers" / "poker_chip_linear.py"
    config_dir = script_dir / "config"

    cmd = [
        sys.executable,
        str(solver_script),
        "--config-dir",
        str(config_dir),
        "--config-name",
        "config_linear",
        "--multirun",
        f"geometry.H={H_str}",
        f"geometry.L={L}",
        f"geometry.h_div={h_div}",
        f"geometry.geometric_dimension={gdim}",
        "output_name=aspect_study",
    ]

    print("\\nRunning command:")
    print(" ".join(cmd))

    try:
        result = subprocess.run(cmd, cwd=script_dir)

        if result.returncode == 0:
            print("\\n✓ Multirun completed successfully!")
            print("\\nNext steps:")
            print(
                "1. Find the multirun output directory in results/aspect_study/multirun/"
            )
            print("2. Run: python scripts/collect_and_plot_results.py [multirun_dir]")
            return True
        else:
            print("\\n✗ Multirun failed")
            return False

    except Exception as e:
        print(f"\\n✗ Error running multirun: {e}")
        return False


if __name__ == "__main__":
    success = run_aspect_ratio_study()
    sys.exit(0 if success else 1)
