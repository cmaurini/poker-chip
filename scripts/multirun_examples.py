#!/usr/bin/env python3
"""
Complete example of how to run parametric studies with Hydra multirun.

This demonstrates the simple two-step approach:
1. Use Hydra's built-in multirun to sweep parameters
2. Collect and plot results afterward

This is much simpler and more reliable than trying to orchestrate everything
from a single complex script.
"""


def print_usage():
    print("=== Hydra Multirun Parametric Study - Simple Approach ===")
    print()
    print("Step 1: Run the parameter sweep using Hydra multirun")
    print("  python scripts/run_multirun_study.py")
    print()
    print("Step 2: Collect and plot results")
    print("  python scripts/collect_and_plot_results.py [multirun_dir]")
    print()
    print("Alternative: Run multirun directly with custom parameters")
    print("  python solvers/poker_chip_linear.py --multirun \\")
    print("    geometry.H=0.05,0.1,0.2,0.5 \\")
    print("    geometry.L=1.0 \\")
    print("    geometry.h_div=4 \\")
    print("    output_name=my_study")
    print()
    print("Key advantages of this approach:")
    print("- Uses Hydra's native multirun capabilities")
    print("- Each simulation runs independently")
    print("- Easy to customize parameter sweeps")
    print("- Simple to debug individual cases")
    print("- Can run in parallel with Hydra launchers")
    print()
    print("Available scripts:")
    print("- run_multirun_study.py: Predefined aspect ratio study")
    print("- collect_and_plot_results.py: Analysis and plotting")
    print()
    print("Example parameter sweeps:")
    print("  Aspect ratio study:")
    print("    geometry.H=1.0,0.5,0.2,0.1 geometry.L=1.0")
    print()
    print("  Material parameter study:")
    print("    model.mu=0.3,0.6,1.0 model.kappa=100,500,1000")
    print()
    print("  Mesh convergence study:")
    print("    geometry.h_div=2,4,8,16")


if __name__ == "__main__":
    print_usage()
