import numpy as np
import matplotlib.pyplot


def plot_energies(history_data, title="", file=None):
    fig, ax1 = matplotlib.pyplot.subplots()

    if title is not None:
        ax1.set_title(title, fontsize=12)

    ax1.set_xlabel(r"Load", fontsize=12)
    ax1.set_ylabel(r"Energies", fontsize=12)
    ax1.grid(linewidth=0.25)
    fig.tight_layout()

    t = np.array(history_data["load"])
    e_e = np.array(history_data["elastic_energy"])
    e_d = np.array(history_data["dissipated_energy"])

    # stress-strain curve
    ax1.plot(
        t,
        e_e,
        color="tab:blue",
        linestyle="-",
        linewidth=1.0,
        markersize=4.0,
        marker="o",
        label=r"Elastic",
    )
    ax1.plot(
        t,
        e_d,
        color="tab:red",
        linestyle="-",
        linewidth=1.0,
        markersize=4.0,
        marker="^",
        label=r"Dissipated",
    )
    ax1.plot(
        t,
        e_d + e_e,
        color="black",
        linestyle="-",
        linewidth=1.0,
        label=r"Total",
    )

    ax1.legend(loc="upper left")
    if file is not None:
        fig.savefig(file)
    return fig, ax1


def plot_energies2(history_data, title="Damage model evolution", file=None):
    fig, ax1 = matplotlib.pyplot.subplots()

    if title is not None:
        ax1.set_title(title, fontsize=12)

    ax1.set_xlabel(r"Load", fontsize=12)
    ax1.set_ylabel(r"Energies", fontsize=12)
    ax1.grid(linewidth=0.25)
    fig.tight_layout()

    t = np.array(history_data["load"])
    e_e = np.array(history_data["elastic_energy"])
    e_d_e = np.array(history_data["elastic_deviatoric_energy"])
    e_v_e = np.array(history_data["elastic_volumetric_energy"])
    e_d = np.array(history_data["dissipated_energy"])

    # stress-strain curve
    ax1.plot(
        t,
        e_e,
        # color="tab:blue",
        linestyle="-",
        linewidth=1.0,
        markersize=4.0,
        marker="o",
        label=r"Elastic",
    )
    ax1.plot(
        t,
        e_d_e,
        # color="tab:green",
        linestyle="-",
        linewidth=1.0,
        markersize=4.0,
        marker="o",
        label=r"Deviatoric",
    )
    ax1.plot(
        t,
        e_v_e,
        # color="yellow",
        linestyle="-",
        linewidth=1.0,
        markersize=4.0,
        marker="o",
        label=r"Volumetric",
    )
    ax1.plot(
        t,
        e_d,
        # color="tab:red",
        linestyle="-",
        linewidth=1.0,
        markersize=4.0,
        marker="^",
        label=r"Dissipated",
    )
    ax1.plot(
        t,
        e_d + e_e,
        color="black",
        linestyle="-",
        linewidth=1.0,
        label=r"Total",
    )

    ax1.legend(loc="upper left")
    if file is not None:
        fig.savefig(file)
        matplotlib.pyplot.close()
    return fig, ax1


def plot_dissipated_energy(
    history_data, title="Dissipated energy evolution", file=None
):
    fig, ax1 = matplotlib.pyplot.subplots()

    if title is not None:
        ax1.set_title(title, fontsize=12)

    ax1.set_xlabel(r"Load", fontsize=12)
    ax1.set_ylabel(r"Dissipated energy", fontsize=12)
    ax1.grid(linewidth=0.25)
    fig.tight_layout()

    t = np.array(history_data["load"])
    e_d = np.array(history_data["dissipated_energy"])

    ax1.plot(
        t,
        e_d,
        color="tab:red",
        linestyle="-",
        linewidth=1.0,
        markersize=4.0,
        marker="^",
    )

    if file is not None:
        fig.savefig(file)
    return fig, ax1


def plot_AMit_load(history_data, title="AM max it - Load", file=None):
    fig, ax1 = matplotlib.pyplot.subplots()

    if title is not None:
        ax1.set_title(title, fontsize=12)

    ax1.set_xlabel(r"Load", fontsize=12)
    ax1.set_ylabel(r"AM max it", fontsize=12)
    ax1.grid(linewidth=0.25)
    fig.tight_layout()

    t = np.array(history_data["load"])
    it = np.zeros_like(t)
    for i, load in enumerate(t):
        it[i] = np.array(history_data["solver_data"][i]["iteration"][-1])

    # stress-strain curve
    ax1.plot(
        t,
        it,
        color="tab:red",
        linestyle="-",
        linewidth=1.0,
        markersize=2.0,
        marker="o",
    )

    if file is not None:
        fig.savefig(file)
    return fig, ax1


def plot_force_displacement(
    history_data, title="Force - Displacement", file=None
):
    fig, ax1 = matplotlib.pyplot.subplots()

    if title is not None:
        ax1.set_title(title, fontsize=12)

    ax1.set_xlabel(r"Load", fontsize=12)
    ax1.set_ylabel(r"Force", fontsize=12)
    ax1.grid(linewidth=0.25)
    fig.tight_layout()

    t = np.array(history_data["load"])
    F = np.array(history_data["F"])

    # stress-strain curve
    ax1.plot(
        t,
        F,
        color="tab:blue",
        linestyle="-",
        linewidth=1.0,
        markersize=4.0,
        marker="o",
    )

    if file is not None:
        fig.savefig(file)
    return fig, ax1


def plot_residual_AMit(data, criterion, title="Residual - AM it", file=None):
    fig, ax1 = matplotlib.pyplot.subplots()

    """if title is not None:
        ax1.set_title(title, fontsize=12)"""

    ax1.set_xlabel(r"$i$", fontsize=25)
    ax1.set_ylabel(r"$||\alpha_{i-1}-\alpha_{i}||$", fontsize=25)
    ax1.grid(linewidth=0.25)
    fig.tight_layout()

    it = np.array(data["iteration"])
    if criterion == "residual_u":
        R = np.array(data["error_residual_u"])
    if criterion == "alpha_H1":
        R = np.array(data["error_alpha_H1"])

    # stress-strain curve
    ax1.plot(
        it,
        R,
        color="tab:orange",
        linestyle="-",
        linewidth=1.0,
        markersize=1.0,
        marker="o",
    )

    if file is not None:
        fig.savefig(file)
    return fig, ax1


def plot_energy_AMit(data, title="Total energy - AM it", file=None):
    fig, ax1 = matplotlib.pyplot.subplots()

    """if title is not None:
        ax1.set_title(title, fontsize=12)"""

    ax1.set_xlabel(r"$i$", fontsize=25)
    ax1.set_ylabel(r"$\cal{E}$", fontsize=5)
    ax1.grid(linewidth=0.25)
    fig.tight_layout()

    it = np.array(data["iteration"])
    E = np.array(data["total_energy"])

    # stress-strain curve
    ax1.plot(
        it,
        E,
        color="tab:orange",
        linestyle="-",
        linewidth=1.0,
        markersize=1.0,
        marker="o",
    )

    if file is not None:
        fig.savefig(file)
    return fig, ax1
