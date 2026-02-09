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
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.signal import savgol_filter
from IPython import embed

from pathlib import Path
import sys

# --- Path Setup ---
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# --- Imports from local modules ---

from reference.formulas_paper import (
    equivalent_modulus,
    max_pressure,
    uniaxial_stress_strain,
    uniaxial_elastic_energy,
    critical_loading_analytical,
)
from reference import formulas_paper
from reference.gent import GentLindleyData

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
            "figure.figsize": (8, 6),
        }
    )
except Exception:
    print("Warning: LaTeX not available, using standard fonts.")

colormap_plot = plt.cm.viridis


def create_smoothed_interpolant(x, y, window_size=11, poly_order=2, tol=1e-10):
    """
    Fixed version: Handles 2D input arrays and numerical noise.
    """
    # 1. Force inputs to 1D arrays and clean non-finite values
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if len(x) != len(y):
        raise ValueError(f"Length mismatch: x({len(x)}) and y({len(y)}) must be equal.")

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 2:
        return None  # Or handle as appropriate for your plotting loop

    # 2. Snap to grid to fix numerical noise in FEM coordinates
    if tol is not None:
        x = np.round(x / tol) * tol

    # 3. Sort and Average Duplicates (The Bincount Method)
    sort_idx = np.argsort(x)
    ux, indices = np.unique(x[sort_idx], return_inverse=True)
    uy = np.bincount(indices, weights=y[sort_idx]) / np.bincount(indices)

    # 4. Window Size Safety Checks
    n_points = len(ux)
    w = window_size

    # Ensure window is odd and fits within the data
    if w % 2 == 0:
        w += 1

    # Clip window to data size
    if w > n_points:
        w = n_points if n_points % 2 != 0 else max(3, n_points - 1)

    # 5. Smooth and Interpolate
    # Only smooth if we have enough points for the polynomial order
    if n_points > poly_order + 1 and w > poly_order:
        try:
            # mode='interp' handles boundaries without artificial dips
            uy_smooth = savgol_filter(uy, w, poly_order, mode="interp")
        except Exception:
            uy_smooth = uy
    else:
        uy_smooth = uy

    return PchipInterpolator(ux, uy_smooth)


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


def plot_nonlinear_zone_length(
    all_runs: List[Dict], output_dir: Path, p_threshold=2.5, eps=1e-2
):
    """Plot, as a function of loading, the length of the zone where p > p_threshold - eps."""
    import matplotlib.pyplot as plt

    colors = colormap_plot(np.linspace(0.9, 0.2, len(all_runs)))
    plt.figure()
    max_load = 0

    for i, run in enumerate(all_runs):
        loads = np.array(run["load"])
        max_load = max(max_load, loads.max())
        try:
            points_x = np.array(run["x_points"])
        except Exception:
            points_x = np.linspace(0, run["L"], len(run["p_x"][0]))
        pressure_fem = np.array(run["p_x"])
        lengths = []
        load_c = critical_loading_analytical(
            mu0=run["mu"],
            H=run["H"],
            R=run["L"],
            p_c=5 / 2 * run["mu"],
            kappa0=run["kappa"],
            geometry=(
                "2d" if run["parameters"]["geometry"]["geometric_dimension"] else "3d"
            ),
            compressible=True,
        )

        for p in pressure_fem:
            idx = np.where(p > (p_threshold - eps))[0]
            if len(idx) > 0:
                length = points_x[idx[-1]] - points_x[idx[0]]
            else:
                length = 0
            lengths.append(length)

        plt.plot(
            loads,
            lengths,
            "-",
            linewidth=3.0,
            color=colors[i],
            alpha=0.9,
            label=rf"$\frac{{R}}{{H}}={run['L'] / run['H']:.1f}, \frac{{\kappa}}{{\mu}}={run['kappa'] / run['mu']:.1f}$",
        )

        plt.axvline(x=load_c, color=colors[i], linestyle="--", linewidth=2.0, alpha=0.5)

    plt.xlabel("Load ($\\Delta/H$)")
    plt.ylabel(f"Length of nonlinear zone ($L_p$)")
    plt.xlim(0, max_load)
    plt.title(
        rf"FEM {('2d' if run['parameters']['geometry']['geometric_dimension'] else '3d')} "
    )
    plt.legend(loc="lower right")
    # plt.title(
    #    f"$R/H={run['L'] / run['H']:.2f}$, $\\kappa/\\mu={run#['kappa'] / run['mu']:.2f}$"
    # )
    plt.grid(True, alpha=0.3)
    # fname = output_dir / f"Lp_L{run['L']}_kappa{run['kappa']}.pdf"
    # plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.savefig(output_dir / f"nonlinear_zone.pdf")
    plt.close()


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
    plt.figure(figsize=(8, 6))

    kappa_mu_ratios = [10.0, 50, 200, 1000]
    colors = colormap_plot(np.linspace(0.2, 0.9, len(all_runs)))

    for i, ratio in enumerate(kappa_mu_ratios):
        p_theory = formulas_paper.max_pressure(
            mu0=mu_fem,
            kappa0=ratio * mu_fem,
            Delta=Delta_fem,
            H=H_fem,
            R=aspect_ratios_theory * H_fem,
            geometry=(
                "2d" if run["parameters"]["geometry"]["geometric_dimension"] else "3d"
            ),
            compressible=True,
        )
        plt.loglog(
            aspect_ratios_theory,
            np.array(p_theory) / mu_fem,
            color=colors[i],
            label=rf"$\\kappa/\\mu=${ratio:.0f}",
        )

    # Incompressible limit
    p_inc = formulas_paper.max_pressure(
        mu0=mu_fem,
        Delta=Delta_fem,
        H=H_fem,
        R=aspect_ratios_theory * H_fem,
        geometry=(
            "2d"
            if all_runs[0]["parameters"]["geometry"]["geometric_dimension"]
            else "3d"
        ),
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
        label=rf"FEM{('2d' if all_runs[0]['parameters']['geometry']['geometric_dimension'] else '3d')}",
    )

    plt.xlabel(r"$R/H$")
    plt.ylabel(r"$p(0)/\mu$")
    plt.title("3D Maximum Pressure Comparison")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(
        rf"FEM {('2d' if all_runs[0]['parameters']['geometry']['geometric_dimension'] else '3d')} "
    )
    plt.legend(loc="lower right")
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


def plot_stress_strain_curve(
    all_runs: List[Dict], output_dir: Path, gl_data=None, mu_factor=None, GL=False
):
    """Plot stress-strain curves with optional Gent-Lindley overlay."""
    plt.figure()
    # Overlay GL experimental branches if available
    if "GL" == True:
        f_I, f_II, f_III = gl_data.get_branch_functions()
        xr = gl_data.get_x_ranges()
        cols = gl_data.get_colors()
        if mu_factor == None:
            mu_factor = 1 / gl_data.mu_GL
        # mu_factor = 1.0 / gl_data.mu_GL /* 0.8
        print("Plotting GL branches with mu_factor =", mu_factor)
        plt.plot(
            xr[0],
            mu_factor * f_I(xr[0]),
            color="lightgray",
            linewidth="3.0",
            label="GL Branch I",
            alpha=1.0,
        )
        x_ = np.linspace(0, 0.5, 100)
        plt.plot(
            x_,
            mu_factor * (gl_data.m_I * x_ + gl_data.b_I),
            "--",
            color="gray",
            alpha=0.5,
        )
        plt.plot(
            x_,
            mu_factor * (gl_data.m_Ib * x_ + gl_data.b_Ib),
            "--",
            alpha=0.5,
            color="gray",
        )
    # Plot FEM data
    k_ratios = np.array([run.get("kappa") / run.get("mu") for run in all_runs])
    aspect_ratios = np.array([run["L"] / run["H"] for run in all_runs])
    k_range = k_ratios.max() - k_ratios.min()
    aspect_ratio_range = aspect_ratios.max() - aspect_ratios.min()
    k_ratios = np.array([run.get("kappa") / run.get("mu") for run in all_runs])
    aspect_ratios = np.array([run["L"] / run["H"] for run in all_runs])
    norm_k = lambda k: (k - k_ratios.min()) / (k_ratios.max() - k_ratios.min())
    norm_aspect = lambda a: (
        (a - aspect_ratios.min()) / (aspect_ratios.max() - aspect_ratios.min())
    )
    xs = np.linspace(0, 1, 200)
    delta_k = k_ratios.max() - k_ratios.min()
    delta_ar = aspect_ratios.max() - aspect_ratios.min()
    colors = colormap_plot(np.linspace(0.9, 0.2, len(all_runs)))
    # Sort all_runs by increasing kappa and then L
    all_run = sorted(all_runs, key=lambda r: (r["kappa"], r["L"]))
    for i, run in enumerate(all_run):
        gdim = run["parameters"]["geometry"]["geometric_dimension"]

        # Normalize kappa/mu to [0, 1] for colormap
        if delta_k > 0:
            norm_value = (run["kappa"] / run["mu"] - k_ratios.min()) / (delta_k)
            print("entering kappa/mu normalization for color mapping", k_ratios)
        elif delta_ar > 0:
            norm_value = (run["L"] / run["H"] - aspect_ratios.min()) / (delta_ar)
        else:
            norm_value = 0.5
        print(
            rf"Plotting run with L/H={run['L'] / run['H']:.2f}, kappa/mu={run['kappa'] / run['mu']:.2f}, norm_value={norm_value:.2f}"
        )
        plt.plot(
            run["average_strain"],
            run["average_stress"],
            "o-",
            label=rf"$R/H={run['L'] / run['H']:.1f}$, $\kappa/\mu={run['kappa'] / run['mu']:.0f}$",
            color=colors[i],
            alpha=0.8,
        )

        # Plot theoretical Equivalent Modulus
        # NOTE: This depends on external import 'equivalent_modulus'
        E_eq = equivalent_modulus(
            mu0=run["mu"],
            H=run["H"],
            R=run["L"],
            geometry=("2d" if gdim == 2 else "3d"),
            kappa0=run["kappa"] / run["mu"],
            compressible=True,
        )

        # E_uniaxial_stress = formulas_paper.E_uniaxial_stress(
        #    mu=run["mu"], kappa=run["kappa"]
        # )
        # E_eq = E_eq + E_uniaxial_stress
        # E_eq = (3 / 8) * run["mu"] * (run["L"] / run["H"]) ** 2
        plt.plot(
            xs,
            E_eq * xs,
            color=colors[i],
            # label=rf"Analytical $E_{{eq}}$",
            linewidth=1,
            alpha=1,
            linestyle="--",
        )
        # Plot Uniaxial Strain
        plt.plot(
            xs,
            uniaxial_stress_strain(
                xs, p_c=5 / 2 * run["mu"], mu=run["mu"], dimension=gdim
            ),
            color="gray",
            # label="Uniaxial strain",
            linewidth=1,
            alpha=0.4,
            linestyle="--",
        )
        load_c = critical_loading_analytical(
            mu0=run["mu"],
            H=run["H"],
            R=run["L"],
            p_c=5 / 2 * run["mu"],
            kappa0=run["kappa"],
            geometry=(
                "2d" if run["parameters"]["geometry"]["geometric_dimension"] else "3d"
            ),
            compressible=True,
        )

        plt.axvline(
            x=load_c / run["H"],
            color=colors[i],
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
            # label=rf"Critical load $R/H={run['L'] / run['H']:.1f}$",
        )

    plt.xlabel(r"$\Delta/H$")
    plt.ylabel(r"$F/\mu S$")

    # plt.title("Stress vs. Strain Comparison")
    plt.xlim(0, all_runs[0]["average_strain"][-1])
    plt.ylim(0, 3.0)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(
        rf"FEM {('2d' if all_runs[0]['parameters']['geometry']['geometric_dimension'] else '3d')} "
    )
    plt.legend(loc="lower right")
    # plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "stress_strain_curve.pdf", bbox_inches="tight")
    plt.close()


def plot_pressure_distribution(all_runs: List[Dict], output_dir: Path):
    """Plot pressure distribution across the radius for 3D case."""
    if not formulas_paper:
        return

    import matplotlib as mpl

    for run in all_runs:
        plt.figure()
        loads = np.array(run["load"])
        try:
            points_x = np.array(run["x_points"])
        except:
            points_x = np.linspace(0, run["L"], len(run["p_x"][0]))
        pressure_fem = np.array(run["p_x"])

        print(
            f"Plotting pressure distribution for L/H={run['L'] / run['H']:.2f}, kappa/mu={run['kappa'] / run['mu']:.2f}"
        )
        r = np.linspace(0, run["L"], 100)

        load_c = critical_loading_analytical(
            mu0=run["mu"],
            H=run["H"],
            R=run["L"],
            p_c=5 / 2 * run["mu"],
            kappa0=run["kappa"],
            geometry=(
                "2d" if run["parameters"]["geometry"]["geometric_dimension"] else "3d"
            ),
            compressible=True,
        )

        # Setup color normalization and colormap for t (load)
        norm = mpl.colors.Normalize(vmin=loads.min(), vmax=loads.max())
        cmap = plt.cm.viridis
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        for i, load in enumerate(loads):
            color = cmap(norm(load))
            window_size = int(len(points_x) * run["H"] / run["L"])
            pressure_fem_interp = create_smoothed_interpolant(
                points_x, pressure_fem[i], window_size=window_size, poly_order=2
            )
            plt.plot(
                points_x / run["H"],
                pressure_fem_interp(points_x) / run["mu"],
                linestyle="-",
                color=color,
                alpha=0.8,
                linewidth=2.0,
            )
            # if load <= 1.25 * load_c:
            #    p = formulas_paper.pressure(
            #        r,
            #        mu0=1,
            #        kappa0=run["kappa"] / run["mu"],
            #        Delta=load,
            #        H=1,
            #        R=run["L"] / run["H"],
            #        geometry=("2d" if run["parameters"]["geometry"]["geometric_dimension"] else "3d"),
            #        compressible=True,
            #    )
            #    plt.plot(
            #        r / run["H"],
            #        p / run["mu"],
            #        linestyle="--",
            #        color=color,
            #        linewidth=1.0,
            #    )

        plt.gca().figure.colorbar(sm, ax=plt.gca(), label=r"$\Delta/\mu$")
        plt.xlabel(r"$r/H$")
        plt.ylabel(r"$p/\mu$")
        plt.ylim(0.0, 3.0)
        plt.title("$p(r)$")
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f"pressure_L{run['L']}_kappa{run['kappa']}.pdf")
        # plt.savefig(output_dir / f"pressure_L{run['L']}_kappa{run    ['kappa']}.pdf")
        plt.close()


def plot_energy(all_runs: List[Dict], output_dir: Path):
    "Plot energy vs load for all runs."
    plt.figure()
    colors = colormap_plot(np.linspace(0.9, 0.2, len(all_runs)))
    for i, run in enumerate(all_runs):
        loads = np.array(run["load"])
        energy = np.array(run["elastic_energy"])
        L = run["L"]
        H = run["H"]
        energy_normalized = energy / (run["mu"] * L)
        plt.plot(
            loads,
            energy_normalized,
            "o-",
            linewidth=2.0,
            color=colors[i],
            label=rf"$R/H={run['L'] / run['H']:.1f}$, $\kappa/\mu={run['kappa'] / run['mu']:.1f}$",
        )
        plt.plot(
            loads,
            uniaxial_elastic_energy(
                loads,
                p_c=5 / 2 * run["mu"],
                mu=run["mu"],
                dimension=run["parameters"]["geometry"]["geometric_dimension"],
            )
            / (run["mu"]),
            color="gray",
            # label="Uniaxial strain",
            linewidth=1,
            alpha=0.4,
            linestyle="--",
        )
        load_c = critical_loading_analytical(
            mu0=run["mu"],
            H=run["H"],
            R=run["L"],
            p_c=5 / 2 * run["mu"],
            kappa0=run["kappa"],
            geometry=(
                "2d" if run["parameters"]["geometry"]["geometric_dimension"] else "3d"
            ),
            compressible=True,
        )

        plt.axvline(
            x=load_c / run["H"],
            color=colors[i],
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
            # label=rf"Critical load $R/H={run['L'] / run['H']:.1f}$",
        )
    plt.xlabel(r"Load ($\Delta/H$)")
    plt.ylabel(r"Elastic energy/$\mu S$")
    plt.title(
        rf"FEM {('2d' if all_runs[0]['parameters']['geometry']['geometric_dimension'] else '3d')} "
    )
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "energy.pdf")
    plt.close()


def plot_3d_max_pressure(all_runs: List[Dict], output_dir: Path):
    """Plot pressure distribution across the radius for 3D case."""
    if not formulas_paper:
        return

    import matplotlib as mpl

    colors = colormap_plot(np.linspace(0.9, 0.2, len(all_runs)))
    plt.figure()
    for i, run in enumerate(all_runs):
        loads = np.array(run["load"])
        pressure_fem = np.array(run["p_x"])
        pressure_fem_max = np.max(pressure_fem, axis=1)

        load_c = critical_loading_analytical(
            mu0=run["mu"],
            H=run["H"],
            R=run["L"],
            p_c=5 / 2 * run["mu"],
            kappa0=run["kappa"],
            geometry=(
                "2d" if run["parameters"]["geometry"]["geometric_dimension"] else "3d"
            ),
            compressible=True,
        )

        plt.plot(
            loads,
            pressure_fem_max,
            "o-",
            linewidth=2.0,
            color=colors[i],
            label=rf"$R/H={run['L'] / run['H']:.1f}$, $\kappa/\mu={run['kappa'] / run['mu']:.1f}$",
        )  # Max pressure at each load step

        plt.axvline(
            x=load_c,
            color=colors[i],
            linestyle="--",
            linewidth=2.0,
            alpha=0.5,
        )

    plt.axhline(
        y=2.5,  # ,run["p_c"] / run["mu"],
        color="gray",
        linestyle="--",
        linewidth=2.0,
        alpha=0.5,
    )
    plt.xlabel(r"Load ($\Delta/H$)")
    plt.ylabel(r"$p_\mathrm{max}/\mu$")
    # plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.title(
        rf"FEM {('2d' if all_runs[0]['parameters']['geometry']['geometric_dimension'] else '3d')} "
    )
    plt.legend(loc="lower right")
    # plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "p_max.pdf")
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
    plot_pressure_distribution(all_runs, output_dir)
    # plot_high_pressure_zone_length(all_runs, output_dir)
    plot_3d_max_pressure(all_runs, output_dir)
    plot_nonlinear_zone_length(all_runs, output_dir)
    plot_energy(all_runs, output_dir)
    # plot_solver_performance(all_runs, output_dir)
    # create_summary_table(all_runs, output_dir)

    # gl.plot_branches_and_fits(save_path=str(output_dir / "GL_branches_and_fits.png"))

    print(f"\nAnalysis complete. Results in: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
