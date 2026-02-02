#!/usr/bin/env python3
"""
Comprehensive mesh convergence study script.
Runs simulations with two different mesh sizes and plots both equivalent modulus
and maximum pressure as functions of aspect ratio.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from reference import formulas_paper


def calculate_all_theory_formulas(aspect_ratios, H_base=1.0, mu0=1.0, kappa0=500.0):
    """Calculate all theoretical predictions for equivalent modulus."""
    results = {
        "2D_incompressible": [],
        "2D_compressible": [],
        "3D_incompressible": [],
        "3D_incompressible_exam": [],
        "3D_compressible": [],
    }

    for ar in aspect_ratios:
        L = ar * H_base
        H = H_base
        R = L / 2  # Half-length for formulas

        # 2D incompressible
        E_eq_2d_inc = formulas_paper.equivalent_modulus(
            mu0=mu0, H=H, R=R, geometry="2d", compressible=False
        )
        results["2D_incompressible"].append(E_eq_2d_inc)

        # 2D compressible
        E_eq_2d = formulas_paper.equivalent_modulus(
            mu0=mu0, kappa0=kappa0, H=H, R=R, geometry="2d", compressible=True
        )
        results["2D_compressible"].append(E_eq_2d)

        # 3D incompressible
        E_eq_3d_inc = formulas_paper.equivalent_modulus(
            mu0=mu0, H=H, R=R, geometry="3d", compressible=False
        )
        results["3D_incompressible"].append(E_eq_3d_inc)

        # 3D incompressible (exam)
        E_eq_3d_inc_exam = formulas_paper.equivalent_modulus(
            mu0=mu0, H=H, R=R, geometry="3d", compressible=False
        )  # Note: same as 3D_incompressible for now
        results["3D_incompressible_exam"].append(E_eq_3d_inc_exam)

        # 3D compressible
        E_eq_3d = formulas_paper.equivalent_modulus(
            mu0=mu0, kappa0=kappa0, H=H, R=R, geometry="3d", compressible=True
        )
        results["3D_compressible"].append(E_eq_3d)

    return {k: np.array(v) for k, v in results.items()}


@hydra.main(version_base=None, config_path="../config", config_name="config_linear")
def run_single_simulation(cfg: DictConfig) -> None:
    """Run a single simulation with given configuration."""

    # Import here to avoid issues with MPI
    from solvers.poker_chip_linear import main

    # Run the simulation - it saves results to files
    main(cfg)

    # Get output directory
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    # Look for the data file
    data_files = list(output_dir.glob("*_data.json"))

    if data_files:
        data_file = data_files[0]  # Take the first one found

        try:
            with open(data_file, "r") as f:
                data = json.load(f)

            # Calculate equivalent modulus
            force = data["F"][-1] if data["F"] else 0
            displacement = data["load"][-1] if data["load"] else 0

            # Get maximum pressure
            pressure_max = (
                data.get("pressure_max", [0])[-1] if data.get("pressure_max") else 0
            )

            # Create geometry dict from config
            geometry = {"L": cfg.geometry.L, "H": cfg.geometry.H}

            E_equiv = data.get("Equivalent_modulus", [0])[-1]

            # Save additional metrics
            metrics = {
                "aspect_ratio": geometry["L"] / geometry["H"],
                "L": geometry["L"],
                "H": geometry["H"],
                "h_div": cfg.geometry.h_div,
                "mesh_size": geometry["H"] / cfg.geometry.h_div,
                "force": force,
                "displacement": displacement,
                "E_equiv": E_equiv,
                "pressure_max": pressure_max,
                "mu": cfg.model.mu,
                "kappa": cfg.model.kappa,
            }

            with open(output_dir / "convergence_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            print(
                f"Simulation completed: AR={metrics['aspect_ratio']:.2f}, "
                f"h_div={cfg.geometry.h_div}, E_eq={E_equiv:.2e}, P_max={pressure_max:.2e}"
            )

        except Exception as e:
            print(f"Error processing results: {e}")

    else:
        print("No data files found!")


def collect_and_analyze_results():
    """Collect results from multirun and create comprehensive convergence analysis."""

    # Find all result directories from the multirun
    multirun_dirs = glob.glob("results/poker_linear/multirun/*/")
    if not multirun_dirs:
        print("No multirun results found!")
        return

    # Take the most recent multirun
    multirun_dir = Path(max(multirun_dirs, key=lambda x: Path(x).stat().st_mtime))
    print(f"Analyzing results from: {multirun_dir}")

    # Find the most recent timestamp directory
    timestamp_dirs = [d for d in multirun_dir.iterdir() if d.is_dir()]
    if not timestamp_dirs:
        print("No timestamp directories found!")
        return

    latest_timestamp = max(timestamp_dirs, key=lambda x: x.stat().st_mtime)
    print(f"Using results from: {latest_timestamp}")

    # Collect all individual run results
    results = {"coarse": [], "fine": []}

    for run_dir in latest_timestamp.glob("*"):
        if run_dir.is_dir():
            metrics_file = run_dir / "convergence_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)

                # Classify as coarse or fine mesh based on h_div
                if metrics["h_div"] == 8:
                    results["coarse"].append(metrics)
                elif metrics["h_div"] == 16:
                    results["fine"].append(metrics)

    print(f"Found {len(results['coarse'])} coarse mesh results")
    print(f"Found {len(results['fine'])} fine mesh results")

    if not results["coarse"] or not results["fine"]:
        print("Need both coarse and fine mesh results for comparison!")
        return

    # Sort by aspect ratio for plotting
    results["coarse"].sort(key=lambda x: x["aspect_ratio"])
    results["fine"].sort(key=lambda x: x["aspect_ratio"])

    # Extract data for plotting
    ar_coarse = np.array([r["aspect_ratio"] for r in results["coarse"]])
    E_coarse = np.array([r["E_equiv"] for r in results["coarse"]])
    P_coarse = np.array([r.get("pressure_max", 0) for r in results["coarse"]])

    ar_fine = np.array([r["aspect_ratio"] for r in results["fine"]])
    E_fine = np.array([r["E_equiv"] for r in results["fine"]])
    P_fine = np.array([r.get("pressure_max", 0) for r in results["fine"]])

    # Calculate theoretical results for comparison
    H_base = results["coarse"][0]["H"]
    mu = results["coarse"][0]["mu"]
    kappa = results["coarse"][0]["kappa"]

    theory_results = calculate_all_theory_formulas(
        ar_coarse, H_base=H_base, mu0=mu, kappa0=kappa
    )

    # Create comprehensive comparison plots
    plt.style.use("default")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Equivalent Modulus comparison
    ax1.loglog(
        ar_coarse,
        E_coarse,
        "o-",
        label="FEM Coarse (h_div=8)",
        color="blue",
        markersize=6,
        linewidth=2,
    )
    ax1.loglog(
        ar_fine,
        E_fine,
        "s-",
        label="FEM Fine (h_div=16)",
        color="red",
        markersize=6,
        linewidth=2,
    )

    # Add theoretical curves
    colors = ["green", "orange", "purple", "brown", "pink"]
    linestyles = ["--", ":", "-.", "--", ":"]
    theory_labels = [
        "2D incompressible",
        "2D compressible",
        "3D incompressible",
        "3D incompressible (exam)",
        "3D compressible",
    ]

    for i, (key, theory_values) in enumerate(theory_results.items()):
        ax1.loglog(
            ar_coarse,
            theory_values,
            linestyles[i],
            label=f"Theory: {theory_labels[i]}",
            color=colors[i],
            linewidth=2,
        )

    ax1.set_xlabel("Aspect Ratio (L/H)", fontsize=12)
    ax1.set_ylabel("Equivalent Modulus", fontsize=12)
    ax1.set_title("Equivalent Modulus vs Aspect Ratio", fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Maximum Pressure comparison
    ax2.loglog(
        ar_coarse,
        P_coarse,
        "o-",
        label="FEM Coarse (h_div=8)",
        color="blue",
        markersize=6,
        linewidth=2,
    )
    ax2.loglog(
        ar_fine,
        P_fine,
        "s-",
        label="FEM Fine (h_div=16)",
        color="red",
        markersize=6,
        linewidth=2,
    )

    ax2.set_xlabel("Aspect Ratio (L/H)", fontsize=12)
    ax2.set_ylabel("Maximum Pressure", fontsize=12)
    ax2.set_title("Maximum Pressure vs Aspect Ratio", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Relative difference in Equivalent Modulus
    common_ar = []
    rel_diff_E = []
    rel_diff_P = []

    for i, ar_c in enumerate(ar_coarse):
        # Find corresponding fine mesh result
        idx_fine = np.where(np.abs(ar_fine - ar_c) < 1e-2)[0]
        if len(idx_fine) > 0:
            j = idx_fine[0]
            common_ar.append(ar_c)

            # Relative difference in Equivalent Modulus
            if E_coarse[i] > 0:
                diff_E = abs(E_fine[j] - E_coarse[i]) / E_coarse[i] * 100
                rel_diff_E.append(diff_E)
            else:
                rel_diff_E.append(0)

            # Relative difference in Pressure
            if P_coarse[i] > 0:
                diff_P = abs(P_fine[j] - P_coarse[i]) / P_coarse[i] * 100
                rel_diff_P.append(diff_P)
            else:
                rel_diff_P.append(0)

    if len(common_ar) > 0:
        ax3.semilogx(
            common_ar,
            rel_diff_E,
            "o-",
            color="purple",
            markersize=6,
            linewidth=2,
            label="Equivalent Modulus",
        )
        ax3.set_xlabel("Aspect Ratio (L/H)", fontsize=12)
        ax3.set_ylabel("Relative Difference (%)", fontsize=12)
        ax3.set_title("Mesh Convergence: Equivalent Modulus", fontsize=14)
        ax3.grid(True, alpha=0.3)

        # Add horizontal lines for reference
        ax3.axhline(y=1, color="green", linestyle="--", alpha=0.7, label="1% threshold")
        ax3.axhline(
            y=5, color="orange", linestyle="--", alpha=0.7, label="5% threshold"
        )
        ax3.axhline(y=10, color="red", linestyle="--", alpha=0.7, label="10% threshold")
        ax3.legend(fontsize=10)

        # Plot 4: Relative difference in Pressure
        ax4.semilogx(
            common_ar,
            rel_diff_P,
            "o-",
            color="darkgreen",
            markersize=6,
            linewidth=2,
            label="Maximum Pressure",
        )
        ax4.set_xlabel("Aspect Ratio (L/H)", fontsize=12)
        ax4.set_ylabel("Relative Difference (%)", fontsize=12)
        ax4.set_title("Mesh Convergence: Maximum Pressure", fontsize=14)
        ax4.grid(True, alpha=0.3)

        # Add horizontal lines for reference
        ax4.axhline(y=1, color="green", linestyle="--", alpha=0.7, label="1% threshold")
        ax4.axhline(
            y=5, color="orange", linestyle="--", alpha=0.7, label="5% threshold"
        )
        ax4.axhline(y=10, color="red", linestyle="--", alpha=0.7, label="10% threshold")
        ax4.legend(fontsize=10)

    plt.tight_layout()

    # Save results
    output_dir = latest_timestamp / "convergence_analysis"
    output_dir.mkdir(exist_ok=True)

    # Save plot
    plot_file = output_dir / "mesh_convergence_study_complete.pdf"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_file}")

    # Also save as PNG for easier viewing
    png_file = output_dir / "mesh_convergence_study_complete.png"
    plt.savefig(png_file, dpi=300, bbox_inches="tight")
    print(f"Plot also saved as {png_file}")

    # Save numerical data
    convergence_data = {
        "coarse_mesh": {"h_div": 8, "results": results["coarse"]},
        "fine_mesh": {"h_div": 16, "results": results["fine"]},
        "convergence_analysis": {
            "common_aspect_ratios": common_ar,
            "relative_differences_E_percent": rel_diff_E,
            "relative_differences_P_percent": rel_diff_P,
        },
        "theory_comparison": {
            "aspect_ratios": ar_coarse.tolist(),
            "theory_results": {k: v.tolist() for k, v in theory_results.items()},
        },
    }

    data_file = output_dir / "convergence_data_complete.json"
    with open(data_file, "w") as f:
        json.dump(convergence_data, f, indent=2)
    print(f"Data saved to {data_file}")

    # Print summary statistics
    if len(rel_diff_E) > 0 and len(rel_diff_P) > 0:
        print(f"\n=== Mesh Convergence Summary ===")
        print(f"Equivalent Modulus:")
        print(f"  Average relative difference: {np.mean(rel_diff_E):.2f}%")
        print(f"  Maximum relative difference: {np.max(rel_diff_E):.2f}%")
        print(
            f"  Results with < 5% difference: {np.sum(np.array(rel_diff_E) < 5.0)}/{len(rel_diff_E)}"
        )

        print(f"\nMaximum Pressure:")
        print(f"  Average relative difference: {np.mean(rel_diff_P):.2f}%")
        print(f"  Maximum relative difference: {np.max(rel_diff_P):.2f}%")
        print(
            f"  Results with < 5% difference: {np.sum(np.array(rel_diff_P) < 5.0)}/{len(rel_diff_P)}"
        )

        # Print detailed comparison table
        print(f"\n=== Detailed Results Table ===")
        print(
            f"{'AR':>6} {'E_coarse':>10} {'E_fine':>10} {'P_coarse':>10} {'P_fine':>10} {'E_diff%':>8} {'P_diff%':>8}"
        )
        print("-" * 70)

        for i, ar in enumerate(common_ar):
            idx_c = np.where(np.abs(ar_coarse - ar) < 1e-2)[0][0]
            idx_f = np.where(np.abs(ar_fine - ar) < 1e-2)[0][0]

            print(
                f"{ar:6.2f} {E_coarse[idx_c]:10.1f} {E_fine[idx_f]:10.1f} "
                f"{P_coarse[idx_c]:10.2e} {P_fine[idx_f]:10.2e} "
                f"{rel_diff_E[i]:8.2f} {rel_diff_P[i]:8.2f}"
            )

    plt.show()


if __name__ == "__main__":
    # Check if we're running analysis mode
    if "--analyze" in sys.argv:
        collect_and_analyze_results()
    else:
        # Remove our custom arguments before Hydra processes them
        sys.argv = [arg for arg in sys.argv if arg != "--analyze"]

        # This will run the simulation - multirun will be handled by Hydra configuration
        run_single_simulation()
        collect_and_analyze_results()
