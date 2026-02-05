"""
Script to collect and plot 3D results from Hydra multirun parametric study.

Usage:
1. First run the multirun:
   python solvers/poker_chip_linear.py --config-dir config --config-name config_linear --multirun \
       geometry.H=1.0 geometry.L=1.0,3,4,5,6,7,8,9,10,15,20 \
       geometry.h_div=6 geometry.geometric_dimension=3 sym=true \
       output_name=Eeq3d-L solver_type=gamg use_iterative_solver=true

2. Then collect and plot results:
   python scripts/collect_and_plot_results_3d.py results/Eeq3d-L/multirun_YYYY-MM-DD_HH-MM-SS

"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import yaml
import argparse

# Add parent directory to path for importing reference module
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from reference import formulas_paper as formulas

# Configure matplotlib for LaTeX rendering
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)
# choose the colormap for the plots
colormap_plot = plt.cm.viridis  # RdPu

# Add paths for local imports
_script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_script_dir))

from reference import formulas_paper

try:
    from reference import gent_lindley_data as gl_data
except ImportError:
    sys.path.insert(0, str(_script_dir / "reference"))
    import gent_lindley_data as gl_data


def collect_results_from_multirun(multirun_dir: Path) -> List[Dict]:
    """
    Collect results from Hydra multirun output directories.

    Args:
        multirun_dir: Directory containing multirun results

    Returns:
        List of dictionaries, each containing all data from one run
    """
    all_runs = []

    if not multirun_dir.exists():
        print(f"Multirun directory does not exist: {multirun_dir}")
        return all_runs

    # Find all subdirectories with JSON files
    run_dirs = []
    for d in multirun_dir.rglob("*"):
        if d.is_dir() and any(d.glob("*_data.json")):
            run_dirs.append(d)

    print(f"Found {len(run_dirs)} run directories with JSON files")

    for run_dir in sorted(run_dirs):
        json_files = list(run_dir.glob("*_data.json"))
        if not json_files:
            continue

        try:
            # Read JSON data
            with open(json_files[0], "r") as f:
                data = json.load(f)

            # Get geometry from config
            config_files = list(run_dir.glob(".hydra/config.yaml"))
            if config_files:
                with open(config_files[0], "r") as f:
                    config = yaml.safe_load(f)
                H = config["geometry"]["H"]
                L = config["geometry"]["L"]
                h_div = config["geometry"].get("h_div", 10)
                mu = config["model"].get("mu", 0.59)
                kappa = config["model"].get("kappa", 500.0)
                solver_type = config.get("solver_type", "unknown")
            else:
                H = data.get("H", 0.1)
                L = data.get("L", 1.0)
                h_div = 10
                mu = 1.0
                kappa = 500.0
                solver_type = "unknown"

            # Create complete run data
            run_data = {
                "H": H,
                "L": L,
                "h_div": h_div,
                "mu": mu,
                "kappa": kappa,
                "solver_type": solver_type,
                **data,  # Include all solver results
            }

            all_runs.append(run_data)

        except Exception as e:
            print(f"Error reading {run_dir}: {e}")
            continue

    # Sort by aspect ratio
    all_runs.sort(key=lambda x: x.get("L", 1.0) / x.get("H", 1.0))

    print(f"Successfully collected {len(all_runs)} runs")
    return all_runs


def create_3d_equivalent_modulus_plot(
    all_runs: List[Dict], output_dir: Path, mu: float = 1.0, kappa: float = 500.0
):
    """Create 3D equivalent modulus plot with analytical comparison"""

    # Extract FEM data
    aspect_ratios = np.array([run["L"] / run["H"] for run in all_runs])
    equivalent_stiffness = np.array(
        [
            run.get("Equivalent_modulus", [0])[-1]
            if run.get("Equivalent_modulus")
            else 0
            for run in all_runs
        ]
    )

    # Calculate Young's modulus E from mu and kappa
    E_uniaxial_stress = formulas_paper.E_uniaxial_stress(mu=mu, kappa=kappa)
    equivalent_stiffness_normalized = equivalent_stiffness / E_uniaxial_stress

    # Create analytical curves
    aspect_ratios_theory = np.logspace(0, np.log10(50), 150)
    H_base = 1

    plt.figure(figsize=(10, 6))

    # Plot analytical curves for different kappa/mu ratios
    kappa_mu_ratios = np.array([10.0, 20, 50, 100, 200, 500, 1000])
    colors = colormap_plot(np.linspace(0.9, 0.2, len(kappa_mu_ratios)))

    # Uniaxial stress reference line
    plt.loglog(
        aspect_ratios_theory,
        np.ones_like(aspect_ratios_theory),
        color="gray",
        linewidth=1.5,
        linestyle="--",
        label=r"Uniaxial stress ($E$)",
    )

    # Plot compressible analytical solutions
    for i, kappa_mu_ratio in enumerate(kappa_mu_ratios):
        mu_val = 1
        kappa_val = kappa_mu_ratio * mu_val
        Eeq_3d_comp_ar = np.array(
            formulas_paper.equivalent_modulus(
                mu0=mu_val,
                kappa0=kappa_val,
                H=H_base,
                R=aspect_ratios_theory * H_base,
                geometry="3d",
                compressible=True,
            )
        )
        E_scaling = formulas.E_uniaxial_stress(mu=mu_val, kappa=kappa_val)

        plt.loglog(
            aspect_ratios_theory,
            Eeq_3d_comp_ar / E_scaling,
            color=colors[i],
            linewidth=2.5,
            label=f"3D, $\\kappa/\\mu=${kappa_mu_ratio:.0f}",
        )

    # Add incompressible limit
    Eeq_3d_inc_ar = np.array(
        formulas_paper.equivalent_modulus(
            mu0=mu,
            H=H_base,
            R=aspect_ratios_theory * H_base,
            geometry="3d",
            compressible=False,
        )
    )

    plt.loglog(
        aspect_ratios_theory,
        Eeq_3d_inc_ar / E_uniaxial_stress,
        color="black",
        linewidth=2.5,
        linestyle="--",
        label="3D incompressible",
    )
    print(f"Nominal Gent-Lindley mu: {gl_data.mu_GL}")
    gl_data.mu_GL *= 0.35
    print(f"Adjusted Gent-Lindley mu: {gl_data.mu_GL}")
    # Add Gent-Lindley experimental data point
    try:
        GL_aspect_ratio = gl_data.R_GL / gl_data.H_GL
        initial_points = 5
        GL_Eeq_exp = np.polyfit(
            gl_data.GL_fig2_x[:initial_points],
            gl_data.GL_fig2_y_MPa[:initial_points],
            1,
        )[0]
        GL_E_material = 3 * gl_data.mu_GL
        GL_Eeq_normalized = GL_Eeq_exp / (
            E_uniaxial_stress * (GL_E_material / E_uniaxial_stress)
        )

        plt.scatter(
            GL_aspect_ratio,
            GL_Eeq_normalized,
            marker="*",
            s=200,
            color="gold",
            edgecolor="black",
            linewidth=1.5,
            label="Gent-Lindley exp.",
            zorder=6,
        )
    except Exception as e:
        print(f"Warning: Could not add Gent-Lindley data point: {e}")

    # Overlay FEM results
    plt.loglog(
        aspect_ratios,
        equivalent_stiffness_normalized,
        "ro",
        markersize=8,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=2.0,
        label="FEM 3D (CR elements)",
        zorder=5,
    )

    plt.xlabel(r"$R/H$", fontsize=14)
    plt.ylabel(r"$E_{eq}/E$", fontsize=14)
    plt.title("3D Equivalent Modulus vs Aspect Ratio", fontsize=16)
    plt.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    plt.savefig(
        output_dir / "3d_equivalent_modulus_comparison.pdf", bbox_inches="tight"
    )
    plt.close()

    print(f"3D equivalent modulus plot saved to {output_dir}")


def plot_3d_max_pressure_comparison(
    all_runs: List[Dict], output_dir: Path, mu: float = 1.0, kappa: float = 500.0
):
    """Plot maximum pressure comparison for 3D case"""

    # Extract FEM data and material properties from runs
    aspect_ratios = np.array([run["L"] / run["H"] for run in all_runs])

    # Use pressure_center (at r≈0) if available, otherwise fall back to pressure_max
    pressure_center = np.array(
        [
            run.get("pressure_center", [0])[-1]
            if run.get("pressure_center")
            else (run.get("pressure_max", [0])[-1] if run.get("pressure_max") else 0)
            for run in all_runs
        ]
    )

    # Get material properties and Delta from first run (should be same for all)
    mu_fem = all_runs[0].get("mu", mu)
    kappa_fem = all_runs[0].get("kappa", kappa)
    H_fem = all_runs[0].get("H", 1.0)
    load_max = all_runs[0].get("load", [1.0])[-1] if all_runs[0].get("load") else 1.0
    Delta_fem = load_max * H_fem

    # Create analytical curves with same material properties
    aspect_ratios_theory = np.logspace(0, np.log10(50), 150)
    H_base = H_fem
    Delta_theory = Delta_fem
    mu_theory = mu_fem

    plt.figure(figsize=(10, 6))

    # Plot analytical curves for different kappa/mu ratios
    kappa_mu_ratios = np.array([10.0, 20, 50, 100, 200, 500, 1000])
    colors = colormap_plot(np.linspace(0.9, 0.2, len(kappa_mu_ratios)))
    print("mu_fem:", mu_fem)
    print("mu_theory:", mu_theory)
    # Plot compressible analytical solutions
    for i, kappa_mu_ratio in enumerate(kappa_mu_ratios):
        kappa_val = kappa_mu_ratio * mu_theory
        p_max_3d_comp_ar = np.array(
            formulas_paper.max_pressure(
                mu0=mu_theory,
                kappa0=kappa_val,
                Delta=Delta_theory,
                H=H_base,
                R=aspect_ratios_theory * H_base,
                geometry="3d",
                compressible=True,
            )
        )

        plt.loglog(
            aspect_ratios_theory,
            p_max_3d_comp_ar / mu_theory,
            color=colors[i],
            linewidth=2.5,
            label=f"3D, $\\kappa/\\mu=${kappa_mu_ratio:.0f}",
        )
    # Add incompressible limit
    p_max_3d_inc_ar = np.array(
        formulas_paper.max_pressure(
            mu0=mu_theory,
            Delta=Delta_theory,
            H=H_base,
            R=aspect_ratios_theory * H_base,
            geometry="3d",
            compressible=False,
        )
    )
    print("p_max_3d_inc_ar / mu_theory:", p_max_3d_inc_ar / mu_theory)
    print("pressure_center_fem / mu_fem:", pressure_center / mu_fem)
    plt.loglog(
        aspect_ratios_theory,
        p_max_3d_inc_ar / mu_theory,
        color="black",
        linewidth=2.5,
        linestyle="--",
        label="3D incompressible",
    )

    # Overlay FEM results
    plt.loglog(
        aspect_ratios,
        pressure_center / mu_fem,
        "o",
        markersize=8,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=2.0,
        label=f"FEM 3D at $r\\approx 0$ ($\\kappa/\\mu={kappa_fem / mu_fem:.0f}$)",
        zorder=5,
    )

    plt.xlabel(r"$R/H$", fontsize=14)
    plt.ylabel(r"$p(r=0)/\mu$", fontsize=14)
    plt.title(
        f"3D Maximum Pressure vs Aspect Ratio)",
        fontsize=16,
    )
    plt.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    plt.savefig(output_dir / "3d_max_pressure_comparison.pdf", bbox_inches="tight")
    plt.close()

    print(f"3D maximum pressure plot saved to {output_dir}")


def plot_solver_performance(all_runs: List[Dict], output_dir: Path):
    """Plot solver performance metrics"""

    aspect_ratios = np.array([run["L"] / run["H"] for run in all_runs])
    solver_times = np.array(
        [
            run.get("solver_time", [0])[-1] if run.get("solver_time") else 0
            for run in all_runs
        ]
    )
    iterations = np.array(
        [
            run.get("iterations", [0])[-1] if run.get("iterations") else 0
            for run in all_runs
        ]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Solver time
    ax1.semilogy(aspect_ratios, solver_times, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel(r"$R/H$", fontsize=14)
    ax1.set_ylabel("Solver Time [s]", fontsize=14)
    ax1.set_title("Solver Time vs Aspect Ratio", fontsize=16)
    ax1.grid(True, alpha=0.3)

    # Iterations
    ax2.semilogy(aspect_ratios, iterations, "ro-", linewidth=2, markersize=8)
    ax2.set_xlabel(r"$R/H$", fontsize=14)
    ax2.set_ylabel("KSP Iterations", fontsize=14)
    ax2.set_title("KSP Iterations vs Aspect Ratio", fontsize=16)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "3d_solver_performance.pdf", bbox_inches="tight")
    plt.close()

    print(f"Solver performance plot saved to {output_dir}")


def create_summary_table(all_runs: List[Dict], output_dir: Path):
    """Create a summary table of all runs"""

    with open(output_dir / "summary_table.txt", "w") as f:
        f.write("=" * 120 + "\n")
        f.write(
            f"{'R/H':>8} {'H':>8} {'L':>8} {'h_div':>8} {'Eeq':>12} {'p_max':>12} {'Iterations':>12} {'Time[s]':>12} {'Solver':>12}\n"
        )
        f.write("=" * 120 + "\n")

        for run in all_runs:
            aspect_ratio = run["L"] / run["H"]
            Eeq = (
                run.get("Equivalent_modulus", [0])[-1]
                if run.get("Equivalent_modulus")
                else 0
            )
            p_max = run.get("pressure_max", [0])[-1] if run.get("pressure_max") else 0
            its = run.get("iterations", [0])[-1] if run.get("iterations") else 0
            time = run.get("solver_time", [0])[-1] if run.get("solver_time") else 0
            solver = run.get("solver_type", "unknown")

            f.write(
                f"{aspect_ratio:8.2f} {run['H']:8.2f} {run['L']:8.2f} {run.get('h_div', 0):8.1f} "
                f"{Eeq:12.4f} {p_max:12.4f} {its:12d} {time:12.3f} {solver:>12}\n"
            )

        f.write("=" * 120 + "\n")

    print(f"Summary table saved to {output_dir / 'summary_table.txt'}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect and plot 3D poker chip simulation results"
    )
    parser.add_argument(
        "multirun_dir",
        type=str,
        nargs="?",
        help="Path to multirun directory (e.g., results/Eeq3d-L/multirun_20260204_231254)",
    )
    parser.add_argument("--mu", type=float, default=1.0, help="Shear modulus")
    parser.add_argument("--kappa", type=float, default=500.0, help="Bulk modulus")

    args = parser.parse_args()

    # Find multirun directory
    if args.multirun_dir:
        multirun_dir = Path(args.multirun_dir)
    else:
        # Search for the most recent multirun directory with Eeq3d
        results_dir = Path("results")
        if results_dir.exists():
            multirun_dirs = list(results_dir.glob("Eeq3d*/multirun_*/"))
            if multirun_dirs:
                multirun_dir = max(multirun_dirs, key=lambda p: p.stat().st_mtime)
                print(f"Using most recent multirun directory: {multirun_dir}")
            else:
                print("No Eeq3d multirun directories found in results/")
                return
        else:
            print("Results directory not found. Please specify multirun directory.")
            return

    if not multirun_dir.exists():
        print(f"Multirun directory does not exist: {multirun_dir}")
        return

    # Collect results
    print(f"Collecting results from: {multirun_dir}")
    all_runs = collect_results_from_multirun(multirun_dir)

    if not all_runs:
        print("No results found")
        return

    # Create output directory
    output_dir = multirun_dir / "analysis"
    output_dir.mkdir(exist_ok=True)

    print(f"\nCreating plots...")
    print(f"Material parameters: mu={args.mu}, kappa={args.kappa}")
    print(f"Geometric dimension: 3D")

    # Create plots
    create_3d_equivalent_modulus_plot(all_runs, output_dir, args.mu, args.kappa)
    plot_3d_max_pressure_comparison(all_runs, output_dir, args.mu, args.kappa)
    plot_solver_performance(all_runs, output_dir)
    create_summary_table(all_runs, output_dir)

    # Save complete data
    output_file = output_dir / "all_runs_data.json"
    with open(output_file, "w") as f:
        json.dump(
            all_runs,
            f,
            indent=2,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
        )

    print(f"\nResults saved to: {output_dir}")
    print(f"Complete data saved to: {output_file}")

    # Print summary
    print(f"\nSummary:")
    print(f"Number of runs: {len(all_runs)}")
    aspect_ratios = [run["L"] / run["H"] for run in all_runs]
    equivalent_stiffness = [
        run.get("Equivalent_modulus", [0])[-1] if run.get("Equivalent_modulus") else 0
        for run in all_runs
    ]
    print(f"Aspect ratios (R/H): {[f'{ar:.1f}' for ar in aspect_ratios]}")
    print(f"Equivalent moduli (Eeq): {[f'{E:.2f}' for E in equivalent_stiffness]}")


if __name__ == "__main__":
    main()
