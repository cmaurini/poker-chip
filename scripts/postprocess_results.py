import argparse
import json
from matplotlib import pyplot as plt
import numpy as np
import yaml
from poker_chip.models import formulas
from poker_chip.solvers.plots import plot_energies2
from reference import formulas_paper as formulas


def plot_energies_pdf(params, history_data, outpath):
    """Plot alternate minimization iterations and save to PDF (reuse energies plot)."""
    plt.figure(0)
    plot_energies2(history_data)
    plt.savefig(outpath)
    plt.close(0)


def plot_force_pdf(params, history_data, outpath):
    """Plot force vs load and save to PDF."""
    mu = params["model"]["mu"]
    lmbda = params["model"]["lambda"]
    k_2d = lmbda + mu
    H = params["geometry"]["H"]
    L = params["geometry"]["L"]
    # gdim = params["geometry"]["geometric_dimension"]
    plt.figure(0)
    loads = np.array(history_data["load"])
    plt.plot(
        loads,
        np.array(history_data["F"]) / float(mu),
        ".-",
    )
    plt.plot(
        loads,
        formulas.equivalent_modulus(
            mu0=mu, kappa0=k_2d, H=H, R=L, geometry="2d", compressible=True
        )
        * loads,
        ".-",
    )
    plt.plot(
        history_data["load"],
        params["model"]["p_cav"] / params["model"]["mu"] + loads,
        ":",
        color="gray",
    )
    plt.axhline(y=float(5 / 2), color="gray", linestyle="--")
    plt.xlabel(r"$\Delta/H$")
    plt.ylabel(r"$F/\mu S$")
    plt.xlim(history_data["load"][0], history_data["load"][-1])
    plt.grid(False)
    plt.ylim(0, max(history_data["F"]) / float(mu))
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close(0)


def main():
    parser = argparse.ArgumentParser(
        description="Postprocess elastic-2d results for Snakemake workflow."
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing simulation results (expects <slug>_data.json and <slug>-parameters.yml)",
    )
    args = parser.parse_args()

    import os

    # Find slug from directory name
    result_dir = args.dir.rstrip("/")
    slug = os.path.basename(result_dir)
    data_file = os.path.join(result_dir, f"{slug}_data.json")
    params_file = os.path.join(result_dir, f"{slug}-parameters.yml")

    with open(data_file, "r") as f:
        history_data = json.load(f)
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)

    plot_energies_pdf(
        params, history_data, os.path.join(result_dir, f"{slug}_energies.pdf")
    )
    plot_force_pdf(params, history_data, os.path.join(result_dir, f"{slug}_force.pdf"))


if __name__ == "__main__":
    main()
