import numpy as np
import matplotlib.pyplot as plt


# Dedicated simple plot functions
def plot_energies(history_data, outpath):
    """Plot energies and alt_min_it."""
    from solvers.poker_chip import plot_energies2

    plt.figure()
    plot_energies2(history_data)
    plt.plot(history_data["load"], history_data["alt_min_it"], "o")
    plt.savefig(outpath)
    plt.close()


def plot_force(
    history_data,
    outpath,
    mu,
    S,
    scaling=None,
    p_cav=None,
    gdim=None,
    loads=None,
    gl_data=None,
    formulas=None,
    L=None,
    H=None,
    k=None,
    elastic=None,
    e_c_dim_k=None,
    sliding=None,
):
    """Plot force vs strain, optionally with reference data."""
    if scaling is None:
        scaling = mu
    if loads is None:
        loads = np.array(history_data["load"])
    plt.figure()
    plt.plot(
        np.array(history_data["load"]),
        np.array(history_data["F"]) / scaling / float(S),
        ".-",
        label=f"FE / {scaling}",
    )
    if p_cav is not None and gdim is not None:
        plt.plot(
            history_data["load"],
            p_cav / mu + np.array(history_data["load"]) / gdim,
            ":",
            color="gray",
            label="Linear",
        )
    if gdim == 3 and L is not None and H is not None:
        plt.axvline(x=float(5 / 3) * (H / L) ** 2, color="gray", linestyle="--")
    if elastic == 0 and e_c_dim_k is not None:
        plt.axvline(x=float(e_c_dim_k), color="gray", linestyle="--")
    plt.xlabel(r"$\\Delta/H$")
    plt.ylabel(r"$F/\\mu S$")
    plt.xlim(loads[0], loads[-1])
    plt.grid(True)
    if sliding == 0 and gl_data is not None:
        plt.plot(
            gl_data.x_all,
            gl_data.y_mpa / scaling,
            "gray",
            label=f"Gent and Lindley (exp) / {scaling}",
            lw=2,
        )
        f_I, f_II, f_III = gl_data.get_branch_functions()
        x_ranges = gl_data.get_x_ranges()
        colors = gl_data.get_colors()
        plt.plot(
            x_ranges[0],
            f_I(x_ranges[0]) / scaling,
            "--",
            color=colors[0],
            label="GL Branch I",
            lw=1,
        )
        plt.plot(
            x_ranges[1],
            f_II(x_ranges[1]) / scaling,
            "--",
            color=colors[1],
            label="GL Branch II",
            lw=1,
        )
        plt.plot(
            x_ranges[2],
            f_III(x_ranges[2]) / scaling,
            "--",
            color=colors[2],
            label="GL Branch III",
            lw=1,
        )
    if (
        formulas is not None
        and mu is not None
        and k is not None
        and L is not None
        and H is not None
    ):
        Ea_3d_inc = formulas.equivalent_modulus(
            mu0=mu, H=H, R=L, geometry="3d", compressible=False
        )
        plt.plot(
            np.array([0, 0.07]),
            np.array([0, 0.07]) * Ea_3d_inc / mu,
            "-.",
            color="orange",
            label="Incompressible model",
            lw=1,
        )
    plt.tight_layout()
    plt.legend()
    plt.savefig(outpath)
    plt.close()


def plot_alpha_x(x_points, alpha_x, outpath, color="b"):
    plt.figure()
    plt.plot(x_points, alpha_x, color=color, linewidth=3, label="Finite Element")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\\alpha$")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def plot_alpha_y(y_points, alpha_y, outpath, color="b"):
    plt.figure()
    plt.plot(y_points, alpha_y, color=color, linewidth=3, label="Finite Element")
    plt.xlabel(r"$y$")
    plt.ylabel(r"$\\alpha$")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def plot_eps_nl_x(x_points, eps_nl_x, outpath, color="b"):
    plt.figure()
    plt.plot(x_points, eps_nl_x, color=color, linewidth=3, label="Finite Element")
    plt.xlabel(r"$x$")
    plt.ylabel("nonlinear deformation")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def plot_eps_nl_y(y_points, eps_nl_y, outpath, color="b"):
    plt.figure()
    plt.plot(y_points, eps_nl_y, color=color, linewidth=3, label="Finite Element")
    plt.xlabel(r"$y$")
    plt.ylabel(r"$\\varepsilon_\\mathrm{NL}$")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def plot_p_y(y_points, p_val_y, outpath, color="b"):
    plt.figure()
    plt.plot(
        y_points, p_val_y, color=color, marker=".", linewidth=3, label="Finite Element"
    )
    plt.xlabel(r"$y$")
    plt.ylabel(r"$p$")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def plot_p_x(
    x_points,
    p_val_x,
    outpath,
    mu=1.0,
    color="b",
    formulas=None,
    Delta=None,
    H=None,
    L=None,
    k=None,
    gdim=None,
):
    plt.figure()
    plt.plot(
        x_points,
        p_val_x / mu,
        color=color,
        marker="o",
        linewidth=3,
        label="Finite Element",
    )
    if formulas is not None and Delta is not None and H is not None and L is not None:
        plt.plot(
            x_points,
            formulas.pressure(
                x_points,
                mu0=mu,
                Delta=Delta,
                H=H,
                R=L,
                geometry="3d",
                compressible=False,
            )
            / mu,
            color="orange",
            linewidth=1,
            label="Asymptotic",
        )
        if k is not None:
            plt.plot(
                x_points,
                formulas.pressure(
                    x_points,
                    mu0=mu,
                    kappa0=k,
                    Delta=Delta,
                    H=H,
                    R=L,
                    geometry="3d",
                    compressible=True,
                )
                / mu,
                color="red",
                linewidth=1,
                label="Asymptotic Compressible",
            )
    plt.xlabel(r"$r/R$")
    plt.ylabel(r"$p/\mu$")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


"""Postprocess elastic-2d results."""

import argparse
import json
import sys
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from solvers.plots import plot_energies2
from reference import formulas_paper as formulas

sys.path.append(str(Path(__file__).resolve().parents[1]))


def plot_energies_pdf(history_data, outpath):
    """Plot alternate minimization iterations and save to PDF (reuse energies plot)."""
    plt.figure(0)
    plot_energies2(history_data)
    plt.savefig(outpath)
    plt.close(0)


def plot_force_pdf(params, history_data, outpath):
    """Plot force vs load and save to PDF."""
    mu = params["model"]["mu"]
    kappa = params["model"]["kappa"]  # Default to incompressible if not specified
    gdim = params["geometry"]["geometric_dimension"]
    p_c = params["model"]["p_cav"]
    # Use 'kappa' from params instead of computing from 'lambda'
    h_val = params["geometry"]["H"]
    l_val = params["geometry"]["L"]
    print(
        f"Using parameters: mu={mu}, kappa={kappa}, H={h_val}, L={l_val}, gdim={gdim}"
    )
    # gdim = params["geometry"]["geometric_dimension"]
    plt.figure(0)
    loads = np.array(history_data["load"])
    plt.plot(
        loads,
        np.array(history_data["average_stress"]) / float(mu),
        ".-",
    )
    plt.plot(
        loads,
        formulas.equivalent_modulus(
            mu0=1.0,
            kappa0=kappa / mu,
            H=h_val,
            R=l_val,
            geometry="3d",
            compressible=True,
        )
        * loads,
        ".-",
    )
    plt.plot(
        history_data["load"],
        params["model"]["p_cav"] / params["model"]["mu"] + loads * 2 / gdim,
        ":",
        color="gray",
    )
    plt.axhline(
        y=float(5 / 2),
        color="gray",
        linestyle="--",
    )
    plt.xlabel(r"$\Delta/H$")
    plt.ylabel(r"$F/\mu S$")
    plt.xlim(
        history_data["load"][0],
        history_data["load"][-1],
    )
    plt.grid(False)
    plt.ylim(
        0,
        max(history_data["F"]) / float(mu),
    )
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close(0)


def main():
    """Main entry point for postprocessing results."""
    parser = argparse.ArgumentParser(
        description=("Postprocess elastic-2d results for Snakemake workflow.")
    )
    parser.add_argument(
        "--dir",
        required=True,
        help=(
            "Directory containing simulation results "
            "(expects one '*_data.json' and one "
            "'*-parameters.yml')"
        ),
    )
    args = parser.parse_args()

    result_dir = Path(args.dir)

    data_files = list(result_dir.glob("*_data.json"))
    if len(data_files) == 0:
        print(f"Error: No '*_data.json' file found in '{result_dir}'")
        sys.exit(1)
    if len(data_files) > 1:
        print(
            "Error: Multiple '*_data.json' files found in '"
            f"{result_dir}'. "
            "Please specify a directory with a single result."
        )
        sys.exit(1)
    data_file = data_files[0]

    params_files = list(result_dir.glob("*-parameters.yml"))
    if len(params_files) == 0:
        print(f"Error: No '*-parameters.yml' file found in '{result_dir}'")
        sys.exit(1)
    if len(params_files) > 1:
        print(
            "Error: Multiple '*-parameters.yml' files found in '"
            f"{result_dir}'. "
            "Please specify a directory with a single result."
        )
        sys.exit(1)
    params_file = params_files[0]

    slug = data_file.name.replace("_data.json", "")

    with open(data_file, "r", encoding="utf-8") as f:
        history_data = json.load(f)
    with open(params_file, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    plot_energies_pdf(history_data, result_dir / f"{slug}_energies.pdf")
    plot_force_pdf(params, history_data, result_dir / f"{slug}_force.pdf")


if __name__ == "__main__":
    main()
