"""
Script to collect and plot 3D results from Hydra multirun parametric study.
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

# --- Path Setup ---
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# --- Imports from local modules ---
try:
    from reference import formulas_paper
except ImportError:
    print("Warning: formulas_paper not found in reference.")
    formulas_paper = None

try:
    # Based on your correction: module is reference.gent
    from reference.gent import GentLindleyData
except ImportError:
    print("Warning: GentLindleyData could not be imported from reference.gent")
    GentLindleyData = None

# --- Configure matplotlib for LaTeX rendering ---
try:
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
except Exception:
    print("Warning: LaTeX not available, using standard fonts.")

colormap_plot = plt.cm.viridis


def collect_results_from_multirun(multirun_dir: Path) -> List[Dict]:
    """Collect results from Hydra multirun output directories."""
    all_runs = []
    if not multirun_dir.exists():
        print(f"Multirun directory does not exist: {multirun_dir}")
        return all_runs

    run_dirs = [
        d for d in multirun_dir.rglob("*") if d.is_dir() and any(d.glob("*_data.json"))
    ]
    print(f"Found {len(run_dirs)} run directories with JSON files")

    for run_dir in sorted(run_dirs):
        json_files = list(run_dir.glob("*_data.json"))
        if not json_files:
            continue
        try:
            with open(json_files[0], "r") as f:
                data = json.load(f)

            config_file = run_dir / ".hydra" / "config.yaml"
            if config_file.exists():
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)
                H = config["geometry"]["H"]
                L = config["geometry"]["L"]
                mu = config["model"].get("mu", 0.59)
                kappa = config["model"].get("kappa", 500.0)
                solver_type = config.get("solver_type", "unknown")
            else:
                H, L, mu, kappa, solver_type = 1.0, 1.0, 1.0, 500.0, "unknown"

            run_data = {
                "H": H,
                "L": L,
                "mu": mu,
                "kappa": kappa,
                "solver_type": solver_type,
                **data,
            }
            all_runs.append(run_data)
        except Exception as e:
            print(f"Error reading {run_dir}: {e}")

    all_runs.sort(key=lambda x: x.get("L", 1.0) / x.get("H", 1.0))
    return all_runs


def plot_3d_max_pressure_comparison(all_runs: List[Dict], output_dir: Path):
    """Plot maximum pressure comparison for 3D case vs analytical curves."""
    if not formulas_paper:
        return

    aspect_ratios = np.array([run["L"] / run["H"] for run in all_runs])
    pressure_center = np.array(
        [
            run.get("pressure_center", run.get("pressure_max", [0]))[-1]
            for run in all_runs
        ]
    )

    mu_fem = all_runs[0].get("mu", 1.0)
    H_fem = all_runs[0].get("H", 1.0)
    load_max = all_runs[0].get("load", [1.0])[-1]
    Delta_fem = load_max * H_fem

    aspect_ratios_theory = np.logspace(0, np.log10(50), 150)
    plt.figure(figsize=(10, 6))

    kappa_mu_ratios = [10.0, 50, 200, 1000]
    colors = colormap_plot(np.linspace(0.9, 0.2, len(kappa_mu_ratios)))

    for i, ratio in enumerate(kappa_mu_ratios):
        p_theory = formulas_paper.max_pressure(
            mu0=mu_fem,
            kappa0=ratio * mu_fem,
            Delta=Delta_fem,
            H=H_fem,
            R=aspect_ratios_theory * H_fem,
            geometry="3d",
            compressible=True,
        )
        plt.loglog(
            aspect_ratios_theory,
            np.array(p_theory) / mu_fem,
            color=colors[i],
            label=f"$\\kappa/\\mu=${ratio:.0f}",
        )

    # Incompressible limit
    p_inc = formulas_paper.max_pressure(
        mu0=mu_fem,
        Delta=Delta_fem,
        H=H_fem,
        R=aspect_ratios_theory * H_fem,
        geometry="3d",
        compressible=False,
    )
    plt.loglog(
        aspect_ratios_theory, np.array(p_inc) / mu_fem, "k--", label="3D Incompressible"
    )

    # FEM Data
    plt.loglog(
        aspect_ratios,
        pressure_center / mu_fem,
        "o",
        mfc="white",
        mec="black",
        mew=2,
        label="FEM 3D",
    )

    plt.xlabel(r"$R/H$")
    plt.ylabel(r"$p(0)/\mu$")
    plt.title("3D Maximum Pressure Comparison")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "3d_max_pressure_comparison.pdf")
    plt.close()


def plot_solver_performance(all_runs: List[Dict], output_dir: Path):
    """Plot solver time and iterations vs aspect ratio."""
    aspect_ratios = np.array([run["L"] / run["H"] for run in all_runs])
    solver_times = np.array([run.get("solver_time", [0])[-1] for run in all_runs])
    iterations = np.array([run.get("iterations", [0])[-1] for run in all_runs])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.semilogy(aspect_ratios, solver_times, "bo-")
    ax1.set_xlabel(r"$R/H$")
    ax1.set_ylabel("Time [s]")
    ax1.set_title("Solver Time")

    ax2.semilogy(aspect_ratios, iterations, "ro-")
    ax2.set_xlabel(r"$R/H$")
    ax2.set_ylabel("KSP Iterations")
    ax2.set_title("Iterations")

    plt.tight_layout()
    plt.savefig(output_dir / "3d_solver_performance.pdf")
    plt.close()


def create_summary_table(all_runs: List[Dict], output_dir: Path):
    """Create summary text table."""
    with open(output_dir / "summary_table.txt", "w") as f:
        f.write(f"{'R/H':>8} {'Eeq':>12} {'p_max':>12} {'Its':>8} {'Time':>10}\n")
        f.write("-" * 55 + "\n")
        for r in all_runs:
            ar = r["L"] / r["H"]
            eeq = r.get("Equivalent_modulus", [0])[-1]
            pmax = r.get("pressure_max", [0])[-1]
            its = r.get("iterations", [0])[-1]
            t = r.get("solver_time", [0])[-1]
            f.write(f"{ar:8.2f} {eeq:12.4f} {pmax:12.4f} {its:8d} {t:10.2f}\n")


def plot_stress_strain_curve(all_runs: List[Dict], output_dir: Path, gl_data=None):
    """Plot stress-strain curves with optional Gent-Lindley overlay."""
    plt.figure(figsize=(8, 6))

    # Overlay GL experimental branches if available

    f_I, f_II, f_III = gl_data.get_branch_functions()
    xr = gl_data.get_x_ranges()
    cols = gl_data.get_colors()
    plt.plot(xr[0], f_I(xr[0]), color=cols[0], alpha=1.0, label="GL Branch I")
    x_ = np.linspace(0, 0.5, 100)
    plt.plot(x_, gl_data.m_I * x_ + gl_data.b_I, "--", color="gray", alpha=0.5)
    plt.plot(x_, gl_data.m_Ib * x_ + gl_data.b_Ib, "--", alpha=0.5, color="gray")

    # Plot FEM data
    for run in all_runs:
        if "average_strain" in run and "average_stress" in run:
            plt.plot(
                run["average_strain"],
                run["average_stress"],
                "o-",
                label=f"FEM R/H={run['L'] / run['H']:.1f}",
            )

    plt.xlabel(r"$\Delta/H$")
    plt.ylabel(r"$F/\mu S$")
    # plt.title("Stress vs. Strain Comparison")
    plt.xlim(0, 0.4)
    plt.ylim(0, 2.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "stress_strain_curve.pdf", bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "multirun_dir", type=str, nargs="?", help="Path to multirun directory"
    )
    parser.add_argument("--keyword", type=str, default="Eeq3d", help="Filter keyword")
    args = parser.parse_args()

    # Find dir logic
    if args.multirun_dir:
        multirun_dir = Path(args.multirun_dir)
    else:
        results_dir = Path("results")
        dirs = list(results_dir.glob(f"**/*multirun*"))
        if not dirs:
            return
        multirun_dir = max(dirs, key=lambda p: p.stat().st_mtime)

    print(f"Analyzing: {multirun_dir}")
    all_runs = collect_results_from_multirun(multirun_dir)
    if not all_runs:
        return

    output_dir = multirun_dir / "analysis"
    output_dir.mkdir(exist_ok=True)

    # Initialize Gent-Lindley reference
    gl = GentLindleyData()

    # Run all analysis tasks
    print("Generating plots...")
    plot_stress_strain_curve(all_runs, output_dir, gl_data=gl)
    # plot_3d_max_pressure_comparison(all_runs, output_dir)
    # plot_solver_performance(all_runs, output_dir)
    # reate_summary_table(all_runs, output_dir)

    # gl.plot_branches_and_fits(save_path=str(output_dir / "GL_branches_and_fits.png"))

    print(f"\nAnalysis complete. Results in: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
