"""
Script to collect and plot results from Hydra multirun parametric study.

Usage:
1. First run the multirun:
   python solvers/poker_chip_linear.py --multirun geometry.H=0.05,0.1,0.2,0.25,0.5 geometry.L=1.0
   python solvers/poker_chip_linear.py --config-dir config --config-name config_linear --multirun geometry.H=1.0 geometry.L=1.0,3,4,5,6,7,8,9,10,20   geometry.h_div=10 geometry.geometric_dimension=2 output_name=aspect_study

2. Then collect and plot results:
   python scripts/collect_and_plot_results.py results/poker/multirun/YYYY-MM-DD/HH-MM-SS


"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import yaml

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

# Add paths for local imports
_script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_script_dir))

from reference import formulas_paper
from reference.formulas_paper import (
    plot_equivalent_modulus_comparison,
    plot_equivalent_modulus_vs_aspect_ratio,
)

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
        if d.is_dir() and any(d.glob("*.json")):
            run_dirs.append(d)

    print(f"Found {len(run_dirs)} run directories with JSON files")

    for run_dir in sorted(run_dirs):
        json_files = list(run_dir.glob("*.json"))
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
            else:
                H = data.get("H", 0.1)
                L = data.get("L", 1.0)

            # Create complete run data
            run_data = {
                "H": H,
                "L": L,
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


def create_enhanced_equivalent_modulus_plot(
    all_runs: List[Dict], output_dir: Path, mu: float = 1.0, kappa: float = 500.0
):
    """Create equivalent modulus plot using reference function and overlay FEM data"""

    # Extract FEM data for overlay
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
    E = 9 * kappa * mu / (3 * kappa + mu)
    equivalent_stiffness_normalized = equivalent_stiffness / E

    # Create a new plot with FEM data overlay on top of the analytical results
    aspect_ratios_theory = np.logspace(-1, np.log10(500), 150)

    plt.figure(figsize=(10, 6))

    # Plot analytical curves using the same style as reference function
    try:
        from reference import gent_lindley_data as gl_data

        mu_ref = gl_data.mu_GL
    except:
        mu_ref = mu

    H_base = 1
    kappa_mu_ratios = np.array(
        [
            10.0,
            20,
            50,
            100,
            200,
            1000,
            kappa,
        ]
    )
    kappa_mu_ratios = np.sort(kappa_mu_ratios)
    colors = plt.cm.viridis(np.linspace(0.9, 0.2, len(kappa_mu_ratios)))

    plt.loglog(
        aspect_ratios_theory,
        np.ones_like(aspect_ratios_theory),
        color="gray",
        linewidth=1.5,
        linestyle="--",
        label=f"Uniaxial",
    )

    for i, kappa_mu_ratio in enumerate(kappa_mu_ratios):
        mu = 1
        kappa_val = kappa_mu_ratio * mu
        Eeq_2d_comp_ar = np.array(
            [
                formulas_paper.equivalent_modulus(
                    mu0=mu,
                    kappa0=kappa_val,
                    H=H_base,
                    R=ar * H_base,
                    geometry="2d",
                    compressible=True,
                )
                for ar in aspect_ratios_theory
            ]
        )
        E_uniaxial_stress = 9 * mu * kappa_val / (3 * kappa_val + mu)
        E_uniaxial_strain = kappa_val + (4 / 3) * mu

        plt.loglog(
            aspect_ratios_theory,
            Eeq_2d_comp_ar / E_uniaxial_stress,
            color=colors[i],
            linewidth=2.5,
            label=f"2D, $\\kappa/\\mu=${kappa_mu_ratio:.1f}",
        )

        # Gent-Lindley paper analytical formula

        Eeq_2d_ratio_comp_gent_paper_an = (1 + (aspect_ratios_theory**2)) * 4 / 3
        Einf = E_uniaxial_strain
        Eeq_2d_ratio_comp_gent_paper_an_comp = (
            Eeq_2d_ratio_comp_gent_paper_an
            * (Einf / E_uniaxial_stress)
            / (Einf / E_uniaxial_stress + Eeq_2d_ratio_comp_gent_paper_an)
        )
        plt.loglog(
            aspect_ratios_theory,
            Eeq_2d_ratio_comp_gent_paper_an_comp,
            color=colors[i],
            linewidth=2.5,
            linestyle=":",
            label=f"GL, $\\kappa/\\mu=${kappa_mu_ratio:.1f}",
        )

    # Add incompressible limit
    Eeq_2d_inc_ar = np.array(
        [
            formulas_paper.equivalent_modulus(
                mu0=mu_ref, H=H_base, R=ar * H_base, geometry="2d", compressible=False
            )
            for ar in aspect_ratios_theory
        ]
    )
    E_incompressible = 3 * mu_ref
    plt.loglog(
        aspect_ratios_theory,
        Eeq_2d_inc_ar / E_incompressible,
        color="black",
        linewidth=2.5,
        linestyle="--",
        label="2D incompressible",
    )

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
        GL_Eeq_normalized = GL_Eeq_exp / GL_E_material

        plt.scatter(
            GL_aspect_ratio,
            GL_Eeq_normalized,
            marker="*",
            s=150,
            color="red",
            edgecolor="black",
            linewidth=1,
            label="Gent-Lindley exp.",
            zorder=6,
        )
    except:
        pass  # Skip GL data if not available

    # Overlay FEM results
    plt.loglog(
        aspect_ratios,
        equivalent_stiffness_normalized,
        "ro",
        markersize=8,
        markerfacecolor="red",
        markeredgecolor="darkred",
        markeredgewidth=2,
        label="FEM (CR elements)",
        zorder=5,
    )

    plt.xlabel(r"$R/H$", fontsize=14)
    plt.ylabel(r"$E_{eq}/E$", fontsize=14)
    plt.title("2D Equivalent Modulus vs Aspect Ratio (with FEM overlay)", fontsize=16)
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "enhanced_equivalent_stiffness_comparison.pdf", bbox_inches="tight"
    )
    plt.close()

    print(f"Enhanced equivalent stiffness plot saved to {output_dir}")


def plot_analytical_pressure_fields(
    output_dir: Path, gdim: int = 2, mu: float = 1.0, kappa: float = 500.0
):
    """Plot analytical pressure field comparisons for different kappa values"""

    H, R, Delta = 1.0, 1.0, 1.0

    if gdim == 2:
        x = np.linspace(0, R, 400)

        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.9, 0.2, 4))

        for i, k in enumerate([0.1, 1, 10, 100]):
            plt.plot(
                x,
                formulas_paper.pressure(
                    x,
                    mu0=mu,
                    kappa0=k * mu,
                    Delta=Delta,
                    H=H,
                    R=R,
                    geometry="2d",
                    compressible=True,
                ),
                label=rf"$\kappa/\mu={k}$",
                color=colors[i],
                linewidth=2.5,
            )
        plt.plot(
            x,
            formulas_paper.pressure(
                x, mu0=mu, Delta=Delta, H=H, R=R, geometry="2d", compressible=False
            ),
            linestyle="--",
            color="black",
            linewidth=2.5,
            label=r"$\kappa/\mu\to\infty$ (incompressible)",
        )

        plt.xlabel(r"$x/H$", fontsize=14)
        plt.ylabel(r"$p/(\mu_0 \Delta R/H^2)$", fontsize=14)
        plt.title("2D Analytical Pressure Distribution", fontsize=16)
        plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "analytical_pressure_2d.pdf", bbox_inches="tight")
        plt.close()

    else:  # 3D
        r = np.linspace(0, R, 400)

        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.9, 0.2, 4))

        for i, k in enumerate([0.1, 1, 10, 100]):
            plt.plot(
                r,
                formulas_paper.pressure(
                    r,
                    mu0=mu,
                    kappa0=k * mu,
                    Delta=Delta,
                    H=H,
                    R=R,
                    geometry="3d",
                    compressible=True,
                ),
                label=rf"$\kappa/\mu={k}$",
                color=colors[i],
                linewidth=2.5,
            )
        plt.plot(
            r,
            formulas_paper.pressure(
                r, mu0=mu, Delta=Delta, H=H, R=R, geometry="3d", compressible=False
            ),
            linestyle="--",
            color="black",
            linewidth=2.5,
            label=r"$\kappa/\mu\to\infty$ (incompressible)",
        )

        plt.xlabel(r"$r/H$", fontsize=14)
        plt.ylabel(r"$p/(\mu_0 \Delta R/H^2)$", fontsize=14)
        plt.title("3D Analytical Pressure Distribution", fontsize=16)
        plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "analytical_pressure_3d.pdf", bbox_inches="tight")
        plt.close()

    print(f"Analytical pressure field plots saved to {output_dir}")


def plot_analytical_shear_stress_fields(
    output_dir: Path, gdim: int = 2, mu: float = 1.0, kappa: float = 500.0
):
    """Plot analytical shear stress field comparisons for different kappa values"""

    H, R, Delta = 1.0, 1.0, 1.0

    if gdim == 2:
        x = np.linspace(0, R, 400)
        y = H  # Mid-height

        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.9, 0.2, 4))

        for i, k in enumerate([0.1, 1, 10, 100]):
            plt.plot(
                x,
                formulas_paper.shear_stress(
                    x,
                    y,
                    mu0=mu,
                    kappa0=k * mu,
                    Delta=Delta,
                    H=H,
                    R=R,
                    geometry="2d",
                    compressible=True,
                ),
                label=rf"$\kappa/\mu={k}$",
                color=colors[i],
                linewidth=2.5,
            )
        plt.plot(
            x,
            formulas_paper.shear_stress(
                x, y, mu0=mu, Delta=Delta, H=H, R=R, geometry="2d", compressible=False
            ),
            linestyle="--",
            color="black",
            linewidth=2.5,
            label=r"$\kappa/\mu\to\infty$ (incompressible)",
        )

        plt.xlabel(r"$x/H$", fontsize=14)
        plt.ylabel(r"$\tau/(\mu_0 \Delta R/H^2)$", fontsize=14)
        plt.title(
            f"2D Analytical Shear Stress Distribution (at $y/H=0.5$)", fontsize=16
        )
        plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "analytical_shear_2d.pdf", bbox_inches="tight")
        plt.close()

    else:  # 3D
        r = np.linspace(0, R, 400)
        z = H  # Mid-height

        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.9, 0.2, 4))

        for i, k in enumerate([0.1, 1, 10, 100]):
            plt.plot(
                r,
                formulas_paper.shear_stress(
                    r,
                    z,
                    mu0=mu,
                    kappa0=k * mu,
                    Delta=Delta,
                    H=H,
                    R=R,
                    geometry="3d",
                    compressible=True,
                ),
                label=rf"$\kappa/\mu={k}$",
                color=colors[i],
                linewidth=2.5,
            )
        plt.plot(
            r,
            formulas_paper.shear_stress(
                r, z, mu0=mu, Delta=Delta, H=H, R=R, geometry="3d", compressible=False
            ),
            linestyle="--",
            color="black",
            linewidth=2.5,
            label=r"$\kappa/\mu\to\infty$ (incompressible)",
        )

        plt.xlabel(r"$r/H$", fontsize=14)
        plt.ylabel(r"$\tau/(\mu_0 \Delta R/H^2)$", fontsize=14)
        plt.title(
            f"3D Analytical Shear Stress Distribution (at $z/H=0.5$)", fontsize=16
        )
        plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "analytical_shear_3d.pdf", bbox_inches="tight")
        plt.close()

    print(f"Analytical shear stress field plots saved to {output_dir}")


def plot_analytical_max_quantities_vs_eta(output_dir: Path, mu: float = 1.0):
    """Plot maximum pressure and shear stress vs normalized parameter eta"""

    H, R, Delta = 1.0, 1.0, 1.0
    eta_ = np.logspace(-1, 1, 100)
    kappa_range = 3 * mu / ((eta_ * H / R) ** 2)

    # Maximum pressure comparison
    plt.figure(figsize=(12, 8))

    # 2D pressure
    p_max_2d_comp = np.array(
        [
            formulas_paper.max_pressure(
                mu0=mu,
                kappa0=k,
                Delta=Delta,
                H=H,
                R=R,
                geometry="2d",
                compressible=True,
            )
            for k in kappa_range
        ]
    )
    p_max_2d_inc_val = formulas_paper.max_pressure(
        mu0=mu, Delta=Delta, H=H, R=R, geometry="2d", compressible=False
    )

    # 3D pressure
    p_max_3d_comp = np.array(
        [
            formulas_paper.max_pressure(
                mu0=mu,
                kappa0=k,
                Delta=Delta,
                H=H,
                R=R,
                geometry="3d",
                compressible=True,
            )
            for k in kappa_range
        ]
    )
    p_max_3d_inc_val = formulas_paper.max_pressure(
        mu0=mu, Delta=Delta, H=H, R=R, geometry="3d", compressible=False
    )

    plt.subplot(1, 2, 1)
    viridis_colors = plt.cm.viridis([0.2, 0.75])

    plt.semilogx(
        eta_,
        p_max_2d_comp / (mu * Delta * R / H**2),
        label="2D",
        color=viridis_colors[0],
        linewidth=2.5,
    )
    plt.axhline(
        p_max_2d_inc_val / (mu * Delta * R / H**2),
        linestyle="--",
        color=viridis_colors[0],
        label=r"2D incompressible",
        linewidth=2,
    )

    plt.semilogx(
        eta_,
        p_max_3d_comp / (mu * Delta * R / H**2),
        label="3D",
        color=viridis_colors[1],
        linewidth=2.5,
    )
    plt.axhline(
        p_max_3d_inc_val / (mu * Delta * R / H**2),
        linestyle="--",
        color=viridis_colors[1],
        label=r"3D incompressible",
        linewidth=2,
    )

    plt.xlabel(r"$\eta = (R/H)\sqrt{3\mu_0/\kappa_0}$", fontsize=12)
    plt.ylabel(r"$p_{\max}/(\mu_0\Delta R/H^2)$", fontsize=12)
    plt.title("Maximum Pressure", fontsize=14)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    # Maximum shear stress comparison
    plt.subplot(1, 2, 2)

    # 2D shear stress
    tau_max_2d_comp = np.array(
        [
            formulas_paper.max_shear_stress(
                mu0=mu,
                kappa0=k,
                Delta=Delta,
                H=H,
                R=R,
                geometry="2d",
                compressible=True,
            )
            for k in kappa_range
        ]
    )
    tau_max_2d_inc_val = formulas_paper.max_shear_stress(
        mu0=mu, Delta=Delta, H=H, R=R, geometry="2d", compressible=False
    )

    # 3D shear stress
    tau_max_3d_comp = np.array(
        [
            formulas_paper.max_shear_stress(
                mu0=mu,
                kappa0=k,
                Delta=Delta,
                H=H,
                R=R,
                geometry="3d",
                compressible=True,
            )
            for k in kappa_range
        ]
    )
    tau_max_3d_inc_val = formulas_paper.max_shear_stress(
        mu0=mu, Delta=Delta, H=H, R=R, geometry="3d", compressible=False
    )

    plt.semilogx(
        eta_,
        tau_max_2d_comp / (mu * Delta * R / H**2),
        label="2D",
        color=viridis_colors[0],
        linewidth=2.5,
    )
    plt.axhline(
        tau_max_2d_inc_val / (mu * Delta * R / H**2),
        linestyle="--",
        color=viridis_colors[0],
        label=r"2D incompressible",
        linewidth=2,
    )

    plt.semilogx(
        eta_,
        tau_max_3d_comp / (mu * Delta * R / H**2),
        label="3D",
        color=viridis_colors[1],
        linewidth=2.5,
    )
    plt.axhline(
        tau_max_3d_inc_val / (mu * Delta * R / H**2),
        linestyle="--",
        color=viridis_colors[1],
        label=r"3D incompressible",
        linewidth=2,
    )

    plt.xlabel(r"$\eta = (R/H)\sqrt{3\mu_0/\kappa_0}$", fontsize=12)
    plt.ylabel(r"$\tau_{\max}/(\mu_0\Delta R/H^2)$", fontsize=12)
    plt.title("Maximum Shear Stress", fontsize=14)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "analytical_max_quantities_vs_eta.pdf", bbox_inches="tight"
    )
    plt.close()

    print(f"Analytical maximum quantities vs eta plots saved to {output_dir}")


def plot_max_pressure_comparison(
    all_runs: List[Dict],
    output_dir: Path,
    mu: float = 1.0,
    kappa: float = 500.0,
    gdim: int = 2,
    delta_reference: float = 0.1,
):
    """Plot maximum pressure vs aspect ratio with theoretical comparisons."""
    if not all_runs:
        print("No results to plot")
        return

    # Extract data for plotting
    aspect_ratios = np.array([run["L"] / run["H"] for run in all_runs])
    max_pressure = np.array(
        [
            run.get("pressure_max", [0])[-1] if run.get("pressure_max") else 0
            for run in all_runs
        ]
    )

    # Create theoretical predictions
    aspect_ratios_theory = np.logspace(
        np.log10(min(aspect_ratios)), np.log10(max(aspect_ratios)), 100
    )

    H_base = 1.0  # Use normalized height
    Delta = delta_reference  # Reference displacement

    if gdim == 2:
        theory_inc = np.array(
            [
                formulas_paper.max_pressure(
                    mu0=mu,
                    Delta=Delta,
                    H=H_base,
                    R=ar * H_base,
                    geometry="2d",
                    compressible=False,
                )
                for ar in aspect_ratios_theory
            ]
        )
        theory_comp = np.array(
            [
                formulas_paper.max_pressure(
                    mu0=mu,
                    kappa0=kappa,
                    Delta=Delta,
                    H=H_base,
                    R=ar * H_base,
                    geometry="2d",
                    compressible=True,
                )
                for ar in aspect_ratios_theory
            ]
        )
    else:  # 3D
        theory_inc = np.array(
            [
                formulas_paper.max_pressure(
                    mu0=mu,
                    Delta=Delta,
                    H=H_base,
                    R=ar * H_base,
                    geometry="3d",
                    compressible=False,
                )
                for ar in aspect_ratios_theory
            ]
        )
        theory_comp = np.array(
            [
                formulas_paper.max_pressure(
                    mu0=mu,
                    kappa0=kappa,
                    Delta=Delta,
                    H=H_base,
                    R=ar * H_base,
                    geometry="3d",
                    compressible=True,
                )
                for ar in aspect_ratios_theory
            ]
        )

    plt.figure(figsize=(12, 8))

    # Plot FEM results
    plt.loglog(
        aspect_ratios,
        max_pressure,
        "bs-",
        linewidth=3,
        markersize=8,
        label=r"FEM (CR elements)",
        zorder=3,
    )

    # Plot theoretical predictions
    plt.loglog(
        aspect_ratios_theory,
        theory_inc,
        "--",
        linewidth=2,
        alpha=0.8,
        label=rf"{gdim}D incompressible theory",
    )
    plt.loglog(
        aspect_ratios_theory,
        theory_comp,
        "-.",
        linewidth=2,
        alpha=0.8,
        label=rf"{gdim}D compressible theory",
    )

    plt.xlabel(r"Aspect Ratio ($L/H$)", fontsize=14)
    plt.ylabel(r"Maximum Pressure", fontsize=14)
    plt.title(
        rf"{gdim}D Maximum Pressure vs Aspect Ratio"
        + "\n"
        + rf"($\mu={mu:.2f}$, $\kappa={kappa:.0f}$, $\Delta={delta_reference}$)",
        fontsize=16,
        pad=20,
    )
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    plt.savefig(output_dir / f"max_pressure_comparison.pdf")
    plt.close()
    print(f"Maximum pressure plot saved to {output_dir}")


def main():
    """Main function to collect and plot multirun results."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect and plot multirun results")
    parser.add_argument(
        "multirun_dir",
        type=str,
        nargs="?",
        help="Path to multirun directory (optional, will search if not provided)",
    )
    parser.add_argument(
        "--gdim",
        type=int,
        default=2,
        choices=[2, 3],
        help="Geometric dimension (2 or 3)",
    )
    parser.add_argument("--mu", type=float, default=1.0, help="Shear modulus")
    parser.add_argument("--kappa", type=float, default=500.0, help="Bulk modulus")
    parser.add_argument(
        "--analytical-only",
        action="store_true",
        help="Generate only analytical plots (no FEM data required)",
    )

    args = parser.parse_args()

    # Handle analytical-only mode
    if args.analytical_only:
        print("Generating analytical plots only...")
        output_dir = Path("analytical_plots")
        output_dir.mkdir(exist_ok=True)

        print(f"Material parameters: mu={args.mu}, kappa={args.kappa}")
        print(f"Geometric dimension: {args.gdim}D")

        plot_analytical_pressure_fields(output_dir, args.gdim, args.mu, args.kappa)
        plot_analytical_shear_stress_fields(output_dir, args.gdim, args.mu, args.kappa)
        plot_analytical_max_quantities_vs_eta(output_dir, args.mu)

        print(f"\nAnalytical plots saved to: {output_dir}")
        return

    # Find multirun directory
    if args.multirun_dir:
        multirun_dir = Path(args.multirun_dir)
    else:
        # Search for the most recent multirun directory
        results_dir = Path("results")
        if results_dir.exists():
            multirun_dirs = list(results_dir.glob("**/multirun/*/"))
            if multirun_dirs:
                multirun_dir = max(multirun_dirs, key=lambda p: p.stat().st_mtime)
                print(f"Using most recent multirun directory: {multirun_dir}")
            else:
                print("No multirun directories found in results/")
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

    print(f"\\nCreating plots...")
    print(f"Material parameters: mu={args.mu}, kappa={args.kappa}")
    print(f"Geometric dimension: {args.gdim}D")

    # Create plots
    create_enhanced_equivalent_modulus_plot(all_runs, output_dir, args.mu, args.kappa)
    plot_max_pressure_comparison(all_runs, output_dir, args.mu, args.kappa, args.gdim)

    # Create analytical plots using existing functions
    plot_analytical_pressure_fields(output_dir, args.gdim, args.mu, args.kappa)
    plot_analytical_shear_stress_fields(output_dir, args.gdim, args.mu, args.kappa)
    plot_analytical_max_quantities_vs_eta(output_dir, args.mu)

    # Save complete data
    output_file = output_dir / "all_runs_data.json"
    with open(output_file, "w") as f:
        json.dump(
            all_runs,
            f,
            indent=2,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
        )

    print(f"\\nResults saved to: {output_dir}")
    print(f"Complete data saved to: {output_file}")

    # Print summary
    print(f"\\nSummary:")
    print(f"Number of runs: {len(all_runs)}")
    aspect_ratios = [run["L"] / run["H"] for run in all_runs]
    equivalent_stiffness = [
        run.get("Equivalent_modulus", [0])[-1] if run.get("Equivalent_modulus") else 0
        for run in all_runs
    ]
    print(f"Aspect ratios: {aspect_ratios}")
    print(f"Equivalent stiffness: {equivalent_stiffness}")


if __name__ == "__main__":
    main()
