"""
Shear stress, and equivalent modulus models derived from asymptotic analysis

All functions are vectorized (NumPy-compatible).
"""

import numpy as np
from scipy.special import jv

import scipy.special
import numpy as np
import scienceplots
import matplotlib.pyplot as plt
import gent_lindley_data as gl_data

plt.style.use("science")
# =========================
# 2D INCOMPRESSIBLE
# =========================


def p_2d_inc(x, *, mu0, Delta, H, R):
    """
    2D incompressible pressure
    """
    x_ = x / H
    delta_ = Delta / H
    R_ = R / H
    return (3 / 2) * mu0 * delta_ * (R_**2 - x_**2)


def tau_2d_inc(x, y, *, mu0, Delta, H):
    """
    2D incompressible shear stress
    """
    return 3 * mu0 * (Delta / H) * (x * y / H**2)


def Eeq_2d_inc(*, mu0, H, R):
    """
    Equivalent modulus (2D incompressible)
    """
    return mu0 * R**2 / (2 * H**2)


def p_max_2d_inc(*, mu0, Delta, H, R):
    """
    Maximum pressure (2D incompressible) - occurs at x=0
    """
    delta_ = Delta / H
    R_ = R / H
    return (3 / 2) * mu0 * delta_ * R_**2


def tau_max_2d_inc(*, mu0, Delta, H, R):
    """
    Maximum shear stress (2D incompressible) - occurs at x=R, y=H
    """
    delta_ = Delta / H
    R_ = R / H
    return 3 * mu0 * delta_ * R_


# =========================
# 2D COMPRESSIBLE
# =========================


def p_2d(x, *, mu0, kappa0, Delta, H, R):
    """
    2D compressible pressure
    """
    a_ = np.sqrt(3 * mu0 / (kappa0))
    x_ = x / H
    R_ = R / H
    eta = a_ * R_
    delta_ = Delta / H
    return kappa0 * delta_ * (1 - np.cosh(eta * x_ / R_) / np.cosh(eta))


def tau_2d(x, y, *, mu0, kappa0, Delta, H, R):
    """
    2D compressible shear stress
    """
    a_ = np.sqrt(3 * mu0 / (kappa0))
    x_ = x / H
    R_ = R / H
    delta_ = Delta / H
    y_ = y / H
    return (
        np.sqrt(3 * kappa0 * mu0) * delta_ * y_ * (np.sinh(a_ * x_) / np.cosh(a_ * R_))
    )


def Eeq_2d(*, mu0, kappa0, H, R):
    """
    Equivalent modulus (2D compressible)
    """
    a_ = np.sqrt(3 * mu0 / (kappa0))
    R_ = R / H
    return (kappa0 / 2) * (1 - (1 / (a_ * R_)) * np.tanh(a_ * R_))


def p_max_2d(*, mu0, kappa0, Delta, H, R):
    """
    Maximum pressure (2D compressible) - occurs at x=0
    """
    a_ = np.sqrt(3 * mu0 / (kappa0))
    R_ = R / H
    delta_ = Delta / H
    eta_ = a_ * R_
    return kappa0 * delta_ * (1 - 1 / np.cosh(eta_))


def tau_max_2d(*, mu0, kappa0, Delta, H, R):
    """
    Maximum shear stress (2D compressible) - occurs at x=R, y=H
    """
    a_ = np.sqrt(3 * mu0 / (kappa0))
    R_ = R / H
    delta_ = Delta / H
    return np.sqrt(3 * kappa0 * mu0) * delta_ * np.tanh(a_ * R_)


# =========================
# 3D INCOMPRESSIBLE
# =========================


def p_3d_inc(r, *, mu0, Delta, H, R):
    """
    3D incompressible pressure
    """
    return (3 * mu0 / 4) * (R**2 * Delta / H**3) * (1 - r**2 / R**2)


def tau_3d_inc(r, z, *, mu0, Delta, H):
    """
    3D incompressible shear stress
    """
    return (3 * mu0 / 2) * (Delta / H) * (r * z / H**2)


def Eeq_3d_inc(*, mu0, H, R):
    """
    Equivalent modulus (3D incompressible)
    """
    return (3 / 8) * mu0 * (R / H) ** 2


def Eeq_3d_inc_exam(*, mu0, H, R):
    """
    Equivalent modulus (3D incompressible)
    """
    r = R / H
    f = 2 * np.sqrt(6) / r
    return 3 * mu0 / (1 - np.tanh(f) / f)


def p_max_3d_inc(*, mu0, Delta, H, R):
    """
    Maximum pressure (3D incompressible) - occurs at r=0
    """
    return (3 * mu0 / 4) * (R**2 * Delta / H**3)


def tau_max_3d_inc(*, mu0, Delta, H, R):
    """
    Maximum shear stress (3D incompressible) - occurs at r=R, z=H
    """
    delta_ = Delta / H
    R_ = R / H
    return (3 * mu0 / 2) * delta_ * R_


# =========================
# 3D COMPRESSIBLE
# =========================


def p_3d(r, *, mu0, kappa0, Delta, H, R):
    """
    3D compressible pressure
    """
    a_ = np.sqrt(3 * mu0 / (kappa0))
    r_ = r / H
    R_ = R / H
    delta_ = Delta / H
    return kappa0 * delta_ * (1 - jv(0, 1j * a_ * r_) / jv(0, 1j * a_ * R_)).real


def tau_3d(r, z, *, mu0, kappa0, Delta, H, R):
    """
    3D compressible shear stress
    """
    a_ = np.sqrt(3 * mu0 / (kappa0))
    r_ = r / H
    R_ = R / H
    delta_ = Delta / H
    z_ = z / H
    return (
        -1j
        * np.sqrt(3 * kappa0 * mu0)
        * delta_
        * z_
        * jv(1, 1j * a_ * r_)
        / jv(0, 1j * a_ * R_)
    ).real


def Eeq_3d(*, mu0, kappa0, H, R):
    """
    Equivalent modulus (3D compressible)
    """
    a_ = np.sqrt(3 * mu0 / (kappa0))
    R_ = R / H
    return -kappa0 * (jv(2, 1j * a_ * R_) / jv(0, 1j * a_ * R_)).real


def p_max_3d(*, mu0, kappa0, Delta, H, R):
    """
    Maximum pressure (3D compressible) - occurs at r=0
    """
    a_ = np.sqrt(3 * mu0 / (kappa0))
    R_ = R / H
    delta_ = Delta / H
    return kappa0 * delta_ * (1 - jv(0, 0) / jv(0, 1j * a_ * R_)).real


def tau_max_3d(*, mu0, kappa0, Delta, H, R):
    """
    Maximum shear stress (3D compressible) - occurs at r=R, z=H
    """
    a_ = np.sqrt(3 * mu0 / (kappa0))
    R_ = R / H
    delta_ = Delta / H
    return (
        -1j
        * np.sqrt(3 * kappa0 * mu0)
        * delta_
        * jv(1, 1j * a_ * R_)
        / jv(0, 1j * a_ * R_)
    ).real


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    output_dir = "figures/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # -------------------------
    # Configure matplotlib for LaTeX-style plots
    # -------------------------
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.figsize": (6, 4),
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "lines.linewidth": 2.5,
        }
    )

    # -------------------------
    # Parameters
    # -------------------------
    mu0 = 1.0
    kappa0 = 1.0
    Delta = 1.0
    H = 1.0
    R = 1.0

    # -------------------------
    # Domains
    # -------------------------
    x = np.linspace(0, R, 400)  # Use symmetry: x from 0 to R
    r = np.linspace(0, R, 400)

    # -------------------------
    # 2D pressure comparison
    # -------------------------
    plt.figure()
    colors = plt.cm.viridis(np.linspace(0.9, 0.2, 4))
    for i, k in enumerate([0.1, 1, 10, 100]):
        plt.plot(
            x,
            p_2d(x, mu0=mu0, kappa0=k, Delta=Delta, H=H, R=R),
            label=rf"$\kappa={k}$",
            color=colors[i],
        )
    plt.plot(
        x,
        p_2d_inc(x, mu0=mu0, Delta=Delta, H=H, R=R),
        linestyle="--",
        color="black",
        label=r"$\kappa\to\infty$",
    )
    plt.xlabel(r"$x$")
    plt.ylabel(r"$p$")
    plt.legend()
    plt.title("2D")
    plt.savefig(f"{output_dir}p_2d_comparison.pdf")
    plt.close()

    # -------------------------
    # 2D shear stress comparison
    # -------------------------
    y = H / 2  # Evaluate at mid-height
    plt.figure()
    colors = plt.cm.viridis(np.linspace(0.9, 0.2, 4))
    for i, k in enumerate([0.1, 1, 10, 100]):
        plt.plot(
            x,
            tau_2d(x, y, mu0=mu0, kappa0=k, Delta=Delta, H=H, R=R),
            label=rf"$\kappa={k}$",
            color=colors[i],
        )
    plt.plot(
        x,
        tau_2d_inc(x, y, mu0=mu0, Delta=Delta, H=H),
        linestyle="--",
        color="black",
        label=r"$\kappa\to\infty$",
    )
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\tau$")
    plt.legend()
    plt.title("2D")
    plt.savefig(f"{output_dir}tau_2d_x_comparison.pdf")
    plt.close()

    # -------------------------
    # 2D shear stress vs y
    # -------------------------
    x_fixed = R / 2  # Evaluate at mid-radius
    y_points = np.linspace(0, H, 400)
    plt.figure()
    colors = plt.cm.viridis(np.linspace(0.9, 0.2, 4))
    for i, k in enumerate([0.1, 1, 10, 100]):
        plt.plot(
            y_points,
            tau_2d(x_fixed, y_points, mu0=mu0, kappa0=k, Delta=Delta, H=H, R=R),
            label=rf"$\kappa={k}$",
            color=colors[i],
        )
    plt.plot(
        y_points,
        tau_2d_inc(x_fixed, y_points, mu0=mu0, Delta=Delta, H=H),
        linestyle="--",
        color="black",
        label=r"$\kappa\to\infty$",
    )
    plt.xlabel(r"$y$")
    plt.ylabel(r"$\tau$")
    plt.legend()
    plt.title("2D")
    plt.savefig(f"{output_dir}tau_2d_y_comparison.pdf")
    plt.close()

    # -------------------------
    # 3D pressure comparison
    # -------------------------
    plt.figure()
    colors = plt.cm.viridis(np.linspace(0.9, 0.2, 4))
    for i, k in enumerate([0.1, 1, 10, 100]):
        plt.plot(
            r,
            p_3d(r, mu0=mu0, kappa0=k, Delta=Delta, H=H, R=R),
            label=rf"$\kappa={k}$",
            color=colors[i],
        )
    plt.plot(
        r,
        p_3d_inc(r, mu0=mu0, Delta=Delta, H=H, R=R),
        linestyle="--",
        color="black",
        label=r"$\kappa\to\infty$",
    )
    plt.xlabel(r"$r$")
    plt.ylabel(r"$p$")
    plt.legend()
    plt.title("3D")
    plt.savefig(f"{output_dir}p_3d_comparison.pdf")
    plt.close()

    # -------------------------
    # 3D shear stress comparison
    # -------------------------
    z = H / 2  # Evaluate at mid-height
    plt.figure()
    colors = plt.cm.viridis(np.linspace(0.9, 0.2, 4))
    for i, k in enumerate([0.1, 1, 10, 100]):
        plt.plot(
            r,
            tau_3d(r, z, mu0=mu0, kappa0=k, Delta=Delta, H=H, R=R),
            label=rf"$\kappa={k}$",
            color=colors[i],
        )
    plt.plot(
        r,
        tau_3d_inc(r, z, mu0=mu0, Delta=Delta, H=H),
        linestyle="--",
        color="black",
        label=r"$\kappa\to\infty$",
    )
    plt.xlabel(r"$r$")
    plt.ylabel(r"$\tau$")
    plt.legend()
    plt.title("3D")
    plt.savefig(f"{output_dir}tau_3d_r_comparison.pdf")
    plt.close()

    # -------------------------
    # 3D shear stress vs z
    # -------------------------
    r_fixed = R / 2  # Evaluate at mid-radius
    z_points = np.linspace(0, H, 400)
    plt.figure()
    colors = plt.cm.viridis(np.linspace(0.9, 0.2, 4))
    for i, k in enumerate([0.1, 1, 10, 100]):
        plt.plot(
            z_points,
            tau_3d(r_fixed, z_points, mu0=mu0, kappa0=k, Delta=Delta, H=H, R=R),
            label=rf"$\kappa={k}$",
            color=colors[i],
        )
    plt.plot(
        z_points,
        tau_3d_inc(r_fixed, z_points, mu0=mu0, Delta=Delta, H=H),
        linestyle="--",
        color="black",
        label=r"$\kappa\to\infty$",
    )
    plt.xlabel(r"$z$")
    plt.ylabel(r"$\tau$")
    plt.legend()
    plt.title("3D")
    plt.savefig(f"{output_dir}tau_3d_z_comparison.pdf")
    plt.close()

    # -------------------------
    # Maximum pressure vs normalized parameter
    # -------------------------
    # eta_ = sqrt(3 * mu0 / kappa) * R / H, so for eta_ in [0.1, 10] with mu0=R=H=1:
    # kappa ranges from 3/0.01 = 300 (for eta_=0.1) to 3/100 = 0.03 (for eta_=10)
    eta_ = np.logspace(-1, 1, 100)
    # eta_ = np.sqrt(3 * mu0 / kappa_range) * R / H
    kappa_range = 3 * mu0 / ((eta_ * H / R) ** 2)  # Rearranged to get kappa from eta
    p_0 = mu0 * (Delta * R / H) ** 2
    # 2D cases
    p_max_2d_comp = p_max_2d(mu0=mu0, kappa0=kappa_range, Delta=Delta, H=H, R=R)
    p_max_2d_inc_val = p_max_2d_inc(mu0=mu0, Delta=Delta, H=H, R=R)

    # 3D cases
    p_max_3d_comp = p_max_3d(mu0=mu0, kappa0=kappa_range, Delta=Delta, H=H, R=R)
    p_max_3d_inc_val = p_max_3d_inc(mu0=mu0, Delta=Delta, H=H, R=R)

    plt.figure()
    viridis_colors = plt.cm.viridis([0.2, 0.75])

    plt.plot(eta_, p_max_2d_comp / p_0, label="2D", color=viridis_colors[0])
    plt.axhline(
        p_max_2d_inc_val / p_0,
        linestyle="--",
        color=viridis_colors[0],
        label=r"2D, $\kappa\to\infty$",
        linewidth=1.5,
    )
    plt.plot(eta_, p_max_3d_comp / mu0, label="3D", color=viridis_colors[1])
    plt.axhline(
        p_max_3d_inc_val / mu0,
        linestyle="--",
        color=viridis_colors[1],
        linewidth=1.5,
        label=r"3D, $\kappa\to\infty$",
    )
    plt.xlabel(r"$\eta = (R/H)\sqrt{3\mu_0/\kappa_0}$")
    plt.ylabel(r"$p_{\max}/(\mu_0\Delta R/H^2)$")
    plt.xscale("log")

    # Add yticks at incompressible values with formula labels
    inc_values = [p_max_2d_inc_val / mu0, p_max_3d_inc_val / mu0]
    ytick_labels = [r"$\frac{3}{2}$", r"$\frac{3}{4}$"]
    plt.yticks(inc_values, ytick_labels)
    plt.gca().yaxis.tick_right()
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}p_max_normalized.pdf")
    plt.close()

    # -------------------------
    # Maximum shear stress vs normalized parameter
    # -------------------------
    # 2D cases
    tau_max_2d_comp = np.array(
        [tau_max_2d(mu0=mu0, kappa0=k, Delta=Delta, H=H, R=R) for k in kappa_range]
    )
    tau_max_2d_inc_val = tau_max_2d_inc(mu0=mu0, Delta=Delta, H=H, R=R)

    # 3D cases
    tau_max_3d_comp = np.array(
        [tau_max_3d(mu0=mu0, kappa0=k, Delta=Delta, H=H, R=R) for k in kappa_range]
    )
    tau_max_3d_inc_val = tau_max_3d_inc(mu0=mu0, Delta=Delta, H=H, R=R)

    plt.figure()
    plt.plot(
        eta_, tau_max_2d_comp / mu0, label="2D", color=viridis_colors[0], linewidth=2.5
    )
    plt.axhline(
        tau_max_2d_inc_val / mu0,
        linestyle="--",
        color=viridis_colors[0],
        linewidth=2.5,
        label=r"2D, $\kappa\to\infty$",
    )
    plt.plot(
        eta_, tau_max_3d_comp / mu0, label="3D", color=viridis_colors[1], linewidth=2.5
    )
    plt.axhline(
        tau_max_3d_inc_val / mu0,
        linestyle="--",
        color=viridis_colors[1],
        linewidth=2.5,
        label=r"3D, $\kappa\to\infty$",
    )
    plt.xlabel(r"$\eta = (R/H)\sqrt{3\mu_0/\kappa_0}$")
    plt.ylabel(r"$\tau_{\max}/(\mu_0\Delta R/H)$")
    plt.xscale("log")
    plt.xlim(0.1, 10)
    # Add yticks at incompressible values with formula labels
    inc_values = [tau_max_2d_inc_val / mu0, tau_max_3d_inc_val / mu0]
    ytick_labels = [r"$3$", r"$\frac{3}{2}$"]
    plt.yticks(inc_values, ytick_labels)
    plt.gca().yaxis.tick_right()
    # Add xticks at min and max
    current_xticks = list(plt.xticks()[0])
    all_xticks = sorted(set(current_xticks + [eta_.min(), eta_.max()]))
    plt.xticks(all_xticks)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}tau_max_normalized.pdf")
    plt.close()

    # Plot Force-displacement curves for verification
    plt.figure()
    plt.plot(x, p_2d_inc(x, mu0=mu0, Delta=Delta, H=H, R=R), label="2D incompressible")
    plt.plot(r, p_3d_inc(r, mu0=mu0, Delta=Delta, H=H, R=R), label="3D incompressible")
    plt.plot(
        x, p_2d(x, mu0=mu0, kappa0=1, Delta=Delta, H=H, R=R), label="2D compressible"
    )
    plt.plot(
        r, p_3d(r, mu0=mu0, kappa0=1, Delta=Delta, H=H, R=R), label="3D compressible"
    )
    plt.xlabel("Displacement")
    plt.ylabel("Force")
    plt.legend()
    plt.title("Force-displacement curves")
    plt.savefig(f"{output_dir}force_displacement_comparison.pdf")
    plt.close()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    plt.figure()
    # Force-displacement curve
    fig, ax1 = plt.subplots()
    stiffness_GL = gl_data.GL_fig2_y_MPa[4] / gl_data.GL_fig2_x[4]

    # correction term due to the machine compliance
    # surface = np.pi * (D / 2) ** 2
    # stiffness_KT_oscar = 1930.0
    # ET = stiffness_KT_oscar / surface * H
    # GL_fig2_x_corrected = GL_fig2_x - GL_fig2_y_MPa / ET

    # Gent and Lindley slopes for the first and second part of the curve
    x = [0, 0.1267, 0.2119, 0.3861]
    y = [
        -0.02849,
        9.9715,
        12.3932,
        14.2735,
    ]

    E2 = (y[-1] - y[0]) / (x[-1] - x[0]) * gl_data.Kgcm2_to_MPa
    E1 = (y[1] - y[0]) / (x[1] - x[0]) * gl_data.Kgcm2_to_MPa
    print(f"- Gent-Lindley slopes in MPa are {E1:2.3f} and {E2:2.3f} MPa")
    # oscar
    x_o = [0.0, 0.02475, 0.02475]
    y_o = [0.0, 0.1951, 0.5854]
    EoT = y_o[1] / x_o[1]
    Eo = y_o[2] / x_o[2]
    print(f"- Equivalent modulus in Oscar paper: {EoT:2.3f} - {Eo:2.3f} MPa")
    # ax1.plot(x_o, y_o, "k-")

    #    ax1.plot([0, 0.2], [0, 0.2 * stiffness_GL], "r--")
    ax1.plot(
        gl_data.GL_fig2_x,
        gl_data.GL_fig2_y_MPa,
        "black",
        label="Gent and Lindley fig 2 data",
        lw=2,
    )
    ax1.plot(
        gl_data.GL_fig2_x_corrected,
        gl_data.GL_fig2_y_MPa,
        "gray",
        label="Gent and Lindley corrected",
        lw=2,
        linestyle=":",
    )
    # plt.plot([0,1],[0,force(delta_u=H)], 'k--')
    ax1.set_xlabel("Extension")
    ax1.set_ylabel(r"$F/S\, (\mathrm{MPa})$")
    ax2 = ax1.twinx()
    mn, mx = ax1.get_ylim()
    ax2.set_ylim(mn / gl_data.Kgcm2_to_MPa, mx / gl_data.Kgcm2_to_MPa)
    ax2.set_ylabel(r"$F/S\, (\mathrm{Kg}/\mathrm{cm}^2)$")
    # plt.plot([0,1],[0,force(delta_u=H)], 'k--')
    #

    xs = np.array([0, 0.2])
    ax1.plot(xs, xs * EoT, "y:", label="Kumar's corrected result")
    # ax1.plot(xs, xs * Eo, "r .", label="Kumar's stiff result")
    ax1.plot(
        xs,
        xs * Eeq_3d(mu0=gl_data.mu_GL, kappa0=20, H=gl_data.H_GL, R=gl_data.D_GL / 2),
        "g--",
        label="Weakly compressible model k=50 MPa",
    )
    ax1.plot(
        xs,
        xs
        * Eeq_3d(
            mu0=gl_data.mu_GL,
            kappa0=gl_data.kappa_GL,
            H=gl_data.H_GL,
            R=gl_data.D_GL / 2,
        ),
        "g-.",
        label="Weakly compressible model",
    )
    ax1.plot(
        xs,
        xs * Eeq_3d_inc(mu0=gl_data.mu_GL, H=gl_data.H_GL, R=gl_data.D_GL / 2),
        "r:",
        label="Incompressible model",
    )
    ax1.set_ylim(0, 1.7)
    # add an horizontal line at 5/2*mu
    ax1.axhline(y=5 / 2 * gl_data.mu_GL, color="lightgray", linestyle=":")
    # ax1.axhline(y=5 / 6 * Y, color="gray", linestyle="--")
    ax1.legend()
    fig.savefig(f"{output_dir}GL-fig2.pdf")
    plt.close()

    # pond figure 6
    plt.figure()
    thickness_pond = 3.2  # mm
    H_pond = thickness_pond / 2  # mm
    R_pond = 25.0
    E0 = 2.6  # Young's Modulus in MN/m^2 (from Table 1)
    mu0 = E0 / 3.0  # Derived Shear Modulus (~0.867 MN/m^2)
    mu_pond = 0.45  # MPa, fitted value from Pond data
    kappa_pond = 1000.0  # MPa, assumed value
    xs = np.array([0, 0.2])

    # Experimental Result from Figure 6
    # Peak stress ~1.4 MN/m^2 at ~6.25% strain
    Eeq_pond_exp = 22.4  # Observed equivalent stiffness in MN/m^2
    x = np.linspace(1, 1000, 1000)  # kappa values in MPa
    plt.plot(
        x,
        Eeq_3d(mu0=mu_pond, kappa0=x, H=H_pond, R=R_pond),
        "g--",
        label="Weakly compressible model",
    )

    plt.axhline(y=Eeq_pond_exp, color="r", linestyle=":", label="Pond et al. data")
    plt.ylabel(r"$E_{eq}$ (MN/m$^2$)")
    plt.xlabel("kappa (MPa)")
    plt.savefig(f"{output_dir}pond_figure6.pdf")
    plt.close()
