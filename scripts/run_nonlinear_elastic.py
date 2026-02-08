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
    H = 1.0  # Keep H constant
    h_div = 4  # Elements through thickness (small for faster runs)
    gdim = 3  # 2D analysis

    # Vary H to get different aspect ratios
    # Aspect ratios will be: L/H = 1.0/H
    L_values = [10]  # Gives aspect
    k_values = [
        1000.0,
    ]  # [10.0, 20, 30, 50.0, 200.0, 500.0]  # Vary compressibility for 3D cases
    L_str = ",".join(map(str, L_values))
    k_str = ",".join(map(str, k_values))
    load_max = 0.4
    n_steps = 50

    print("=== Running Hydra Multirun Parametric Study ===")
    print(f"Fixed L = {L_values}")
    print(f"H values = {H}")
    print(f"Aspect ratios = {[L / H for L in L_values]}")
    print(f"Elements through thickness = {h_div}")
    print(f"Geometric dimension = {gdim}D")

    # Build command
    script_dir = Path(__file__).parent.parent
    solver_script = script_dir / "solvers" / "poker_chip.py"
    config_dir = script_dir / "config"

    cmd = [
        sys.executable,
        str(solver_script),
        "--config-dir",
        str(config_dir),
        "--config-name",
        "config_elastic_nonlinear.yaml",
        "--multirun",
        f"geometry.L={L_str}",
        f"geometry.H={H}",
        f"geometry.h_div={h_div}",
        f"geometry.geometric_dimension={gdim}",
        f"model.kappa={k_str}",
        f"load_max={load_max}",
        f"n_steps={n_steps}",
        "output_name=nonlinear_aspect_study",
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
