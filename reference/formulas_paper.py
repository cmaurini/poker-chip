"""
Shear stress, and equivalent modulus models derived from asymptotic analysis

All functions are vectorized (NumPy-compatible).
"""

import numpy as np
from scipy.special import jv, i0

import scipy.special
import numpy as np
import scienceplots
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize_scalar

try:
    from . import gent_lindley_data as gl_data
except ImportError:
    import reference.gent_lindley_data as gl_data
from scipy.optimize import root_scalar

plt.style.use("science")


def E_uniaxial_stress(*, mu, kappa, compressible=True):
    """
    Young's modulus (uniaxial stress condition).
    sigma_x != 0, sigma_y = sigma_z = 0
    kappa = bulk modulus (3D)
    mu = shear modulus (3D)
    """
    if compressible:
        return 9.0 * mu * kappa / (3.0 * kappa + mu)
    else:
        # Incompressible case: E is determined by shear response since volumetric response is infinitely stiff
        return 3.0 * mu


def E_uniaxial_strain(*, mu, kappa):
    """
    Uniaxial strain (constrained / P-wave modulus).
    epsilon_y = epsilon_z = 0
    kappa = bulk modulus (3D)
    mu = shear modulus (3D)
    """
    return kappa + (4.0 / 3.0) * mu


def E_plane_strain(*, mu, kappa):
    """
    Effective modulus under plane strain.
    epsilon_z = 0
    kappa = bulk modulus (3D)
    mu = shear modulus (3D)
    """
    return 9.0 * mu * kappa / (kappa + 3.0 * mu)


def mu_lame_from_kappa_mu(*, kappa, mu, model="3D"):
    """
    Shear Lamé parameter for different elasticity models.

    Parameters
    ----------
    mu : float
        Shear modulus
    model : str
        "3D", "plane_strain", "plane_stress", "pure_2D"

    Returns
    -------
    mu_lame : float
    """
    model = model.lower()
    if model in ("3d", "plane_strain", "plane_stress", "pure_2d"):
        return mu
    else:
        raise ValueError(f"Unknown model: {model}")


def lambda_lame_from_kappa_mu(*, kappa, mu, model="3D"):
    """
    First Lamé parameter for different elasticity models.

    Parameters
    ----------
    kappa : float
        Bulk modulus
    mu : float
        Shear modulus
    model : str
        "3D", "plane_strain", "plane_stress", "pure_2D"

    Returns
    -------
    lambda_lame : float
    """
    model = model.lower()

    if model in ("3d", "plane_strain"):
        return kappa - (2.0 / 3.0) * mu

    elif model == "plane_stress":
        lam_3d = kappa - (2.0 / 3.0) * mu
        return (2.0 * mu * lam_3d) / (lam_3d + 2.0 * mu)

    elif model == "pure_2d":
        return kappa - mu

    else:
        raise ValueError(f"Unknown model: {model}")


def E_nu_from_kappa_mu(kappa, mu, model="3D"):
    """
    Return (E, nu) from (kappa, mu) for different elasticity models.

    Parameters
    ----------
    kappa : float
        Bulk modulus (3D for 3D/plane models, intrinsic 2D for pure_2D)
    mu : float
        Shear modulus
    model : str
        "3D", "plane_strain", "plane_stress", "pure_2D"

    Returns
    -------
    E : float
        Young's modulus (effective, model-dependent)
    nu : float
        Poisson's ratio (effective, model-dependent)
    """
    model = model.lower()

    # --- 3D and plane strain ---
    if model in ("3d", "plane_strain"):
        E = 9.0 * kappa * mu / (3.0 * kappa + mu)
        nu = (3.0 * kappa - 2.0 * mu) / (2.0 * (3.0 * kappa + mu))

    # --- plane stress (effective in-plane constants) ---
    elif model == "plane_stress":
        # E is the same as 3D
        E = 9.0 * kappa * mu / (3.0 * kappa + mu)
        # effective Poisson ratio after eliminating sigma_zz
        nu = (3.0 * kappa - 2.0 * mu) / (3.0 * kappa + 4.0 * mu)

    # --- pure 2D elasticity ---
    elif model == "pure_2d":
        # intrinsic 2D relations
        E = 4.0 * kappa * mu / (kappa + mu)
        nu = (kappa - mu) / (kappa + mu)

    else:
        raise ValueError(
            "model must be one of: '3D', 'plane_strain', 'plane_stress', 'pure_2D'"
        )

    return E, nu


# =========================
# UNIFIED FUNCTIONS
# =========================


def pressure(
    coord, *, mu0, Delta, H, R, kappa0=None, geometry="2d", compressible=False
):
    """
    Unified pressure function for different geometries, compressible/incompressible cases

    Parameters:
    -----------
    coord : array-like
        x for 2D cases, r for 3D
    mu0 : float
        Shear modulus
    Delta : float
        Displacement
    H : float
        Height
    R : float
        Radius/half-width
    kappa0 : float, optional
        Bulk modulus (required if compressible=True)
    geometry : str, default="2d"
        Geometry type: "2d", "3d", "2d_plane_stress", "2d_plane_strain"
    compressible : bool, default=False
        Whether to use compressible formulation
    """
    E_0 = E_uniaxial_stress(
        mu=mu0, kappa=kappa0, compressible=compressible
    )  # Base stiffness from uniaxial stress condition
    gdim = 2 if geometry in ["2d", "2d_plane_stress", "2d_plane_strain"] else 3
    p_0 = E_0 * (Delta / H) / gdim  # Base pressure from uniaxial stress condition
    if not compressible:  # Incompressible case
        if geometry == "2d":
            # Pure 2D incompressible pressure
            x = coord
            x_ = x / H
            delta_ = Delta / H
            R_ = R / H
            return (3 / 2) * mu0 * delta_ * (R_**2 - x_**2) + p_0
        elif geometry == "2d_plane_stress":
            raise NotImplementedError("Plane stress formulation not implemented")
        elif geometry == "2d_plane_strain":
            # 2D plane strain incompressible pressure
            x = coord
            x_ = x / H
            delta_ = Delta / H
            R_ = R / H
            return (3 / 2) * mu0 * delta_ * (R_**2 - x_**2) + p_0
        elif geometry == "3d":
            # 3D incompressible pressure
            r = coord
            return (3 * mu0 / 4) * (R**2 * Delta / H**3) * (1 - r**2 / R**2) + p_0
        else:
            raise ValueError(f"Unknown geometry: {geometry}")
    else:  # Compressible case
        if kappa0 is None:
            raise ValueError("kappa0 must be provided when compressible=True")
        if geometry == "2d":
            # 2D compressible pressure
            x = coord
            a_ = np.sqrt(3 * mu0 / (kappa0))
            x_ = x / H
            R_ = R / H
            eta = a_ * R_
            delta_ = Delta / H
            return kappa0 * delta_ * (1 - np.cosh(eta * x_ / R_) / np.cosh(eta)) + p_0
        elif geometry == "2d_plane_stress":
            raise NotImplementedError("Plane stress formulation not implemented")
        elif geometry == "2d_plane_strain":
            # 2D plane strain compressible pressure
            x = coord
            a_ = np.sqrt(3 * mu0 / (kappa0))
            x_ = x / H
            R_ = R / H
            eta = a_ * R_
            delta_ = Delta / H
            return kappa0 * delta_ * (1 - np.cosh(eta * x_ / R_) / np.cosh(eta)) + p_0
        elif geometry == "3d":
            # 3D compressible pressure
            r = coord
            a_ = np.sqrt(3 * mu0 / (kappa0))
            r_ = r / H
            R_ = R / H
            delta_ = Delta / H
            return (
                kappa0 * delta_ * (1 - jv(0, 1j * a_ * r_) / jv(0, 1j * a_ * R_)).real
                + p_0
            )
        else:
            raise ValueError(f"Unknown geometry: {geometry}")


def shear_stress(
    coord1, coord2, *, mu0, Delta, H, R, kappa0=None, geometry="2d", compressible=False
):
    """
    Unified shear stress function for 2D/3D, compressible/incompressible cases

    Parameters:
    -----------
    coord1 : array-like
        x for 2D, r for 3D
    coord2 : array-like
        y for 2D, z for 3D
    mu0 : float
        Shear modulus
    Delta : float
        Displacement
    H : float
        Height
    R : float
        Radius/half-width (only used for compressible case)
    kappa0 : float, optional
        Bulk modulus (required if compressible=True)
    geometry : str, default="2d"
        Geometry type: "2d", "3d", "2d_plane_stress", "2d_plane_strain"
    compressible : bool, default=False
        Whether to use compressible formulation
    """
    if not compressible:  # Incompressible case
        if geometry == "2d":
            # 2D incompressible shear stress (pure 2D formulation)
            x, y = coord1, coord2
            return 3 * mu0 * (Delta / H) * (x * y / H**2)
        elif geometry == "2d_plane_stress":
            raise NotImplementedError("Plane stress formulation not implemented")
        elif geometry == "2d_plane_strain":
            # 2D plane strain incompressible shear stress
            x, y = coord1, coord2
            return 3 * mu0 * (Delta / H) * (x * y / H**2)
        elif geometry == "3d":
            # 3D incompressible shear stress
            r, z = coord1, coord2
            return (3 * mu0 / 2) * (Delta / H) * (r * z / H**2)
        else:
            raise ValueError(f"Unknown geometry: {geometry}")
    else:  # Compressible case
        if kappa0 is None:
            raise ValueError("kappa0 must be provided when compressible=True")
        if geometry in ["2d", "2d_plane_stress", "2d_plane_strain"]:
            # 2D compressible shear stress
            x, y = coord1, coord2
            a_ = np.sqrt(3 * mu0 / (kappa0))
            x_ = x / H
            R_ = R / H
            delta_ = Delta / H
            y_ = y / H
            return (
                np.sqrt(3 * kappa0 * mu0)
                * delta_
                * y_
                * (np.sinh(a_ * x_) / np.cosh(a_ * R_))
            )
        elif geometry == "3d":
            # 3D compressible shear stress
            r, z = coord1, coord2
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
        else:
            raise ValueError(f"Unknown geometry: {geometry}")


def equivalent_modulus(*, mu0, H, R, kappa0=None, geometry="2d", compressible=False):
    """
    Unified equivalent modulus function for 2D/3D, compressible/incompressible cases

    Parameters:
    -----------
    mu0 : float
        Shear modulus
    H : float
        Height
    R : float
        Radius/half-width
    kappa0 : float, optional
        Bulk modulus (required if compressible=True)
    geometry : str, default="2d"
        Geometry type: "2d", "3d", "2d_plane_stress", "2d_plane_strain"
    compressible : bool, default=False
        Whether to use compressible formulation
    """
    E_0 = E_uniaxial_stress(
        mu=mu0, kappa=kappa0, compressible=compressible
    )  # Base stiffness from uniaxial stress condition
    if not compressible:  # Incompressible case
        if geometry == "2d":
            # Pure 2D incompressible equivalent modulus
            R_ = R / H
            return mu0 * R_**2 + E_0
        elif geometry == "2d_plane_stress":
            # 2D plane stress incompressible equivalent modulus
            raise NotImplementedError("Plane stress formulation not implemented")
        elif geometry == "2d_plane_strain":
            # 2D plane strain incompressible equivalent modulus
            R_ = R / H
            return mu0 * R_**2 + E_0
        elif geometry == "3d":
            # 3D incompressible equivalent modulus
            return (3 / 8) * mu0 * (R / H) ** 2 + E_0
        else:
            raise ValueError(f"Unknown geometry: {geometry}")
    else:  # Compressible case
        if kappa0 is None:
            raise ValueError("kappa0 must be provided when compressible=True")
        if geometry in ["2d", "2d_plane_strain"]:
            # 2D compressible equivalent modulus
            a_ = np.sqrt(3 * mu0 / (kappa0))
            R_ = R / H
            return 2 * (kappa0 / 2) * (1 - (1 / (a_ * R_)) * np.tanh(a_ * R_)) + E_0
        elif geometry == "2d_plane_stress":
            # 2D plane stress compressible equivalent modulus
            raise NotImplementedError("Plane stress formulation not implemented")
        elif geometry == "3d":
            # 3D compressible equivalent modulus
            a_ = np.sqrt(3 * mu0 / (kappa0))
            R_ = R / H
            return -kappa0 * (jv(2, 1j * a_ * R_) / jv(0, 1j * a_ * R_)).real + E_0
        else:
            raise ValueError(f"Unknown geometry: {geometry}")


def uniaxial_stress_strain(xs, *, p_c, mu, dimension):
    uniaxial_stress = p_c + (2 / dimension) * mu * xs
    return uniaxial_stress


def uniaxial_elastic_energy(xs, *, p_c, mu, dimension):
    uniaxial_stress = p_c * xs + (2 / dimension) * mu * xs**2 / 2
    return uniaxial_stress


def max_pressure(*, mu0, Delta, H, R, kappa0=None, geometry="2d", compressible=False):
    """
    Unified maximum pressure function for 2D/3D, compressible/incompressible cases

    Parameters:
    -----------
    mu0 : float
        Shear modulus
    Delta : float
        Displacement
    H : float
        Height
    R : float
        Radius/half-width
    kappa0 : float, optional
        Bulk modulus (required if compressible=True)
    geometry : str, default="2d"
        Geometry type: "2d", "3d", "2d_plane_stress", "2d_plane_strain"
    compressible : bool, default=False
        Whether to use compressible formulation
    """
    E_0 = E_uniaxial_stress(
        mu=mu0, kappa=kappa0, compressible=compressible
    )  # Base stiffness from uniaxial stress condition
    gdim = 2 if geometry in ["2d", "2d_plane_stress", "2d_plane_strain"] else 3
    p_0 = E_0 * (Delta / H) / gdim  # Base pressure from uniaxial stress condition
    if not compressible:  # Incompressible case
        if geometry == "2d":
            # 2D incompressible max pressure (at x=0)
            delta_ = Delta / H
            R_ = R / H
            return (3 / 2) * mu0 * delta_ * R_**2 + p_0
        elif geometry == "2d_plane_stress":
            raise NotImplementedError("Plane stress formulation not implemented")
        elif geometry == "2d_plane_strain":
            # 2D plane strain incompressible max pressure (at x=0)
            delta_ = Delta / H
            R_ = R / H
            return (3 / 2) * mu0 * delta_ * R_**2 + p_0
        elif geometry == "3d":
            # 3D incompressible max pressure (at r=0)
            return (3 * mu0 / 4) * (R**2 * Delta / H**3)
        else:
            raise ValueError(f"Unknown geometry: {geometry}")
    else:  # Compressible case
        if kappa0 is None:
            raise ValueError("kappa0 must be provided when compressible=True")
        if geometry == "2d":
            # 2D compressible max pressure (at x=0)
            a_ = np.sqrt(3 * mu0 / (kappa0))
            R_ = R / H
            delta_ = Delta / H
            eta_ = a_ * R_
            return kappa0 * delta_ * (1 - 1 / np.cosh(eta_)) + p_0
        elif geometry == "2d_plane_stress":
            raise NotImplementedError("Plane stress formulation not implemented")
        elif geometry == "2d_plane_strain":
            # 2D plane strain compressible max pressure (at x=0)
            a_ = np.sqrt(3 * mu0 / (kappa0))
            R_ = R / H
            delta_ = Delta / H
            eta_ = a_ * R_
            return kappa0 * delta_ * (1 - 1 / np.cosh(eta_)) + p_0
        elif geometry == "3d":
            # 3D compressible max pressure (at r=0)
            a_ = np.sqrt(3 * mu0 / (kappa0))
            R_ = R / H
            delta_ = Delta / H
            return kappa0 * delta_ * (1 - jv(0, 0) / jv(0, 1j * a_ * R_)).real + p_0
        else:
            raise ValueError(f"Unknown geometry: {geometry}")


def critical_loading_analytical(
    *, mu0, H, R, p_c, kappa0=None, geometry="3d", compressible=False
):
    """
    Returns the critical displacement Delta such that p_max = p_c.

    Parameters:
    -----------
    p_c : float
        Target maximum pressure
    ... (other parameters same as max_pressure)
    """

    # We calculate the pressure for a reference displacement Delta = 1.0
    # and use the linear relationship: p_max = Slope * Delta
    # Therefore: Delta_c = p_c / Slope

    ref_delta = 1.0

    # Calculate base stiffness (E_0) and base pressure (p_0) for Delta = 1.0
    E_0 = E_uniaxial_stress(mu=mu0, kappa=kappa0, compressible=compressible)
    gdim = 2 if geometry in ["2d", "2d_plane_stress", "2d_plane_strain"] else 3

    # Base pressure slope (p_0 / Delta)
    p0_slope = (E_0 / H) / gdim

    R_ = R / H

    if not compressible:
        if geometry in ["2d", "2d_plane_strain"]:
            # p_max = (3/2 * mu0 * R^2 / H^3 + p0_slope) * Delta
            slope = (3 / 2) * mu0 * (R**2 / H**3) + p0_slope
        elif geometry == "3d":
            # Note: Your 3D incompressible code does not add p_0
            slope = (3 * mu0 / 4) * (R**2 / H**3) + p0_slope
        else:
            raise ValueError(f"Unknown geometry: {geometry}")

    else:  # Compressible case
        if kappa0 is None:
            raise ValueError("kappa0 must be provided when compressible=True")

        a_ = np.sqrt(3 * mu0 / kappa0)
        eta_ = a_ * R_

        if geometry in ["2d", "2d_plane_strain"]:
            # p_max = (kappa0/H * (1 - 1/cosh(eta)) + p0_slope) * Delta
            slope = (kappa0 / H) * (1 - 1 / np.cosh(eta_)) + p0_slope
        elif geometry == "3d":
            # p_max = (kappa0/H * (1 - J0(0)/J0(i*a*R/H)) + p0_slope) * Delta
            # J0(0) = 1.0
            term = (1 - 1 / jv(0, 1j * a_ * R_)).real
            slope = (kappa0 / H) * term + p0_slope
        else:
            raise ValueError(f"Unknown geometry: {geometry}")

    return p_c / slope


def max_shear_stress(
    *, mu0, Delta, H, R, kappa0=None, geometry="2d", compressible=False
):
    """
    Unified maximum shear stress function for 2D/3D, compressible/incompressible cases

    Parameters:
    -----------
    mu0 : float
        Shear modulus
    Delta : float
        Displacement
    H : float
        Height
    R : float
        Radius/half-width
    kappa0 : float, optional
        Bulk modulus (required if compressible=True)
    geometry : str, default="2d"
        Geometry type: "2d", "3d", "2d_plane_stress", "2d_plane_strain"
    compressible : bool, default=False
        Whether to use compressible formulation
    """
    if not compressible:  # Incompressible case
        if geometry == "2d":
            # 2D incompressible max shear stress (at x=R, y=H)
            delta_ = Delta / H
            R_ = R / H
            return 3 * mu0 * delta_ * R_
        elif geometry == "2d_plane_stress":
            raise NotImplementedError("Plane stress formulation not implemented")
        elif geometry == "2d_plane_strain":
            # 2D plane strain incompressible max shear stress (at x=R, y=H)
            delta_ = Delta / H
            R_ = R / H
            return 3 * mu0 * delta_ * R_
        elif geometry == "3d":
            # 3D incompressible max shear stress (at r=R, z=H)
            delta_ = Delta / H
            R_ = R / H
            return (3 * mu0 / 2) * delta_ * R_
        else:
            raise ValueError(f"Unknown geometry: {geometry}")
    else:  # Compressible case
        if kappa0 is None:
            raise ValueError("kappa0 must be provided when compressible=True")
        if geometry == "2d":
            # 2D compressible max shear stress
            a_ = np.sqrt(3 * mu0 / (kappa0))
            R_ = R / H
            delta_ = Delta / H
            return np.sqrt(3 * kappa0 * mu0) * delta_ * np.tanh(a_ * R_)
        elif geometry == "2d_plane_stress":
            raise NotImplementedError("Plane stress formulation not implemented")
        elif geometry == "2d_plane_strain":
            # 2D plane strain compressible max shear stress
            a_ = np.sqrt(3 * mu0 / (kappa0))
            R_ = R / H
            delta_ = Delta / H
            return np.sqrt(3 * kappa0 * mu0) * delta_ * np.tanh(a_ * R_)
        elif geometry == "3d":
            # 3D compressible max shear stress (at r=R, z=H)
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
        else:
            raise ValueError(f"Unknown geometry: {geometry}")


def plot_pressure_comparison_2d(output_dir="figures/"):
    """Plot 2D pressure comparison for different kappa values"""
    plt.figure()
    colors = plt.cm.viridis(np.linspace(0.9, 0.2, 4))
    mu0, Delta, H, R = 1.0, 1.0, 1.0, 1.0
    x = np.linspace(0, R, 400)

    for i, k in enumerate([0.1, 1, 10, 100]):
        plt.plot(
            x,
            pressure(
                x,
                mu0=mu0,
                kappa0=k,
                Delta=Delta,
                H=H,
                R=R,
                geometry="2d",
                compressible=True,
            ),
            label=rf"$\kappa={k}$",
            color=colors[i],
        )
    plt.plot(
        x,
        pressure(x, mu0=mu0, Delta=Delta, H=H, R=R, geometry="2d", compressible=False),
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


def plot_shear_stress_comparison_2d(output_dir="figures/"):
    """Plot 2D shear stress comparison for different kappa values"""
    mu0, Delta, H, R = 1.0, 1.0, 1.0, 1.0
    x = np.linspace(0, R, 400)

    # vs x
    y = H / 2  # Evaluate at mid-height
    plt.figure()
    colors = plt.cm.viridis(np.linspace(0.9, 0.2, 4))
    for i, k in enumerate([0.1, 1, 10, 100]):
        plt.plot(
            x,
            shear_stress(
                x,
                y,
                mu0=mu0,
                kappa0=k,
                Delta=Delta,
                H=H,
                R=R,
                geometry="2d",
                compressible=True,
            ),
            label=rf"$\kappa={k}$",
            color=colors[i],
        )
    plt.plot(
        x,
        shear_stress(
            x, y, mu0=mu0, Delta=Delta, H=H, R=R, geometry="2d", compressible=False
        ),
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

    # vs y
    x_fixed = R / 2  # Evaluate at mid-radius
    y_points = np.linspace(0, H, 400)
    plt.figure()
    colors = plt.cm.viridis(np.linspace(0.9, 0.2, 4))
    for i, k in enumerate([0.1, 1, 10, 100]):
        plt.plot(
            y_points,
            shear_stress(
                x_fixed,
                y_points,
                mu0=mu0,
                kappa0=k,
                Delta=Delta,
                H=H,
                R=R,
                geometry="2d",
                compressible=True,
            ),
            label=rf"$\kappa={k}$",
            color=colors[i],
        )
    plt.plot(
        y_points,
        shear_stress(
            x_fixed,
            y_points,
            mu0=mu0,
            Delta=Delta,
            H=H,
            R=R,
            geometry="2d",
            compressible=False,
        ),
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


def plot_pressure_comparison_3d(output_dir="figures/"):
    """Plot 3D pressure comparison for different kappa values"""
    plt.figure()
    colors = plt.cm.viridis(np.linspace(0.9, 0.2, 4))
    mu0, Delta, H, R = 1.0, 1.0, 1.0, 1.0
    r = np.linspace(0, R, 400)

    for i, k in enumerate([0.1, 1, 10, 100]):
        plt.plot(
            r,
            pressure(
                r,
                mu0=mu0,
                kappa0=k,
                Delta=Delta,
                H=H,
                R=R,
                geometry="3d",
                compressible=True,
            ),
            label=rf"$\kappa={k}$",
            color=colors[i],
        )
    plt.plot(
        r,
        pressure(r, mu0=mu0, Delta=Delta, H=H, R=R, geometry="3d", compressible=False),
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


def plot_shear_stress_comparison_3d(output_dir="figures/"):
    """Plot 3D shear stress comparison for different kappa values"""
    mu0, Delta, H, R = 1.0, 1.0, 1.0, 1.0
    r = np.linspace(0, R, 400)

    # vs r
    z = H / 2  # Evaluate at mid-height
    plt.figure()
    colors = plt.cm.viridis(np.linspace(0.9, 0.2, 4))
    for i, k in enumerate([0.1, 1, 10, 100]):
        plt.plot(
            r,
            shear_stress(
                r,
                z,
                mu0=mu0,
                kappa0=k,
                Delta=Delta,
                H=H,
                R=R,
                geometry="3d",
                compressible=True,
            ),
            label=rf"$\kappa={k}$",
            color=colors[i],
        )
    plt.plot(
        r,
        shear_stress(
            r, z, mu0=mu0, Delta=Delta, H=H, R=R, geometry="3d", compressible=False
        ),
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

    # vs z
    r_fixed = R / 2  # Evaluate at mid-radius
    z_points = np.linspace(0, H, 400)
    plt.figure()
    colors = plt.cm.viridis(np.linspace(0.9, 0.2, 4))
    for i, k in enumerate([0.1, 1, 10, 100]):
        plt.plot(
            z_points,
            shear_stress(
                r_fixed,
                z_points,
                mu0=mu0,
                kappa0=k,
                Delta=Delta,
                H=H,
                R=R,
                geometry="3d",
                compressible=True,
            ),
            label=rf"$\kappa={k}$",
            color=colors[i],
        )
    plt.plot(
        z_points,
        shear_stress(
            r_fixed,
            z_points,
            mu0=mu0,
            Delta=Delta,
            H=H,
            R=R,
            geometry="3d",
            compressible=False,
        ),
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


def plot_equivalent_modulus_comparison(output_dir="figures/"):
    """Plot equivalent modulus vs kappa comparison"""
    plt.figure()
    colors = plt.cm.viridis(np.linspace(0.9, 0.2, 4))
    mu0, H, R = 1.0, 1.0, 1.0

    # 2D equivalent modulus
    Eeq_2d_comp = np.array(
        [
            equivalent_modulus(
                mu0=mu0, kappa0=k, H=H, R=R, geometry="2d", compressible=True
            )
            for k in [0.1, 1, 10, 100]
        ]
    )
    Eeq_2d_inc_val = equivalent_modulus(
        mu0=mu0, H=H, R=R, geometry="2d", compressible=False
    )

    # 3D equivalent modulus
    Eeq_3d_comp = np.array(
        [
            equivalent_modulus(
                mu0=mu0, kappa0=k, H=H, R=R, geometry="3d", compressible=True
            )
            for k in [0.1, 1, 10, 100]
        ]
    )
    Eeq_3d_inc_val = equivalent_modulus(
        mu0=mu0, H=H, R=R, geometry="3d", compressible=False
    )

    kappa_vals = [0.1, 1, 10, 100]
    for i, k in enumerate(kappa_vals):
        if i == 0:  # Add labels only for the first plot
            plt.scatter(
                k,
                Eeq_2d_comp[i],
                color="blue",
                marker="o",
                s=60,
                label="2D compressible",
                alpha=0.8,
            )
            plt.scatter(
                k,
                Eeq_3d_comp[i],
                color="red",
                marker="s",
                s=60,
                label="3D compressible",
                alpha=0.8,
            )
        else:
            plt.scatter(k, Eeq_2d_comp[i], color="blue", marker="o", s=60, alpha=0.8)
            plt.scatter(k, Eeq_3d_comp[i], color="red", marker="s", s=60, alpha=0.8)

    # Add incompressible limits as horizontal lines
    plt.axhline(
        y=Eeq_2d_inc_val,
        color="blue",
        linestyle="--",
        linewidth=2,
        label="2D incompressible",
    )
    plt.axhline(
        y=Eeq_3d_inc_val,
        color="red",
        linestyle="--",
        linewidth=2,
        label="3D incompressible",
    )

    plt.xlabel(r"$\kappa_0$")
    plt.ylabel(r"$E_{eq}$")
    plt.xscale("log")
    plt.legend()
    plt.title("Equivalent Modulus")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}Eeq_comparison.pdf")
    plt.close()


def plot_equivalent_modulus_vs_aspect_ratio(
    output_dir="figures/", max_aspect_ratio=50, factor=1.0
):
    """Plot equivalent modulus vs aspect ratio with GL data fitting"""
    aspect_ratios = np.logspace(-1, np.log10(max_aspect_ratio), 150)
    kappa_mu_ratios = [0.1, 1.0, 10.0, 100.0]
    colors = plt.cm.viridis(np.linspace(0.9, 0.2, len(kappa_mu_ratios)))

    # Calculate Gent-Lindley experimental data
    GL_aspect_ratio = gl_data.R_GL / gl_data.H_GL
    initial_points = 5
    GL_Eeq_exp = np.polyfit(
        gl_data.GL_fig2_x[:initial_points],
        gl_data.GL_fig2_y_MPa[:initial_points],
        1,
    )[0]
    GL_E_material = 3 * gl_data.mu_GL
    GL_Eeq_normalized = GL_Eeq_exp / GL_E_material

    # Find kappa values that fit GL point
    def objective_2d(kappa_val):
        Eeq_model = factor * equivalent_modulus(
            mu0=gl_data.mu_GL,
            kappa0=kappa_val,
            H=gl_data.H_GL,
            R=gl_data.R_GL,
            geometry="2d",
            compressible=True,
        )
        E_model = 9 * gl_data.mu_GL * kappa_val / (3 * kappa_val + gl_data.mu_GL)
        return abs(Eeq_model / E_model - GL_Eeq_normalized)

    def objective_3d(kappa_val):
        Eeq_model = factor * equivalent_modulus(
            mu0=gl_data.mu_GL,
            kappa0=kappa_val,
            H=gl_data.H_GL,
            R=gl_data.R_GL,
            geometry="3d",
            compressible=True,
        )
        E_model = 9 * gl_data.mu_GL * kappa_val / (3 * kappa_val + gl_data.mu_GL)
        return abs(Eeq_model / E_model - GL_Eeq_normalized)

    result_2d = minimize_scalar(objective_2d, bounds=(0.1, 10000), method="bounded")
    kappa_GL_fit_2d = result_2d.x
    kappa_mu_GL_fit_2d = round(kappa_GL_fit_2d / gl_data.mu_GL)

    result_3d = minimize_scalar(objective_3d, bounds=(0.1, 10000), method="bounded")
    kappa_GL_fit_3d = result_3d.x
    kappa_mu_GL_fit_3d = round(kappa_GL_fit_3d / gl_data.mu_GL)

    # Plot 2D cases
    plt.figure()
    for i, kappa_mu_ratio in enumerate(kappa_mu_ratios):
        kappa_val = kappa_mu_ratio * gl_data.mu_GL
        Eeq_2d_comp_ar = np.array(
            [
                equivalent_modulus(
                    mu0=gl_data.mu_GL,
                    kappa0=kappa_val,
                    H=gl_data.H_GL,
                    R=ar * gl_data.H_GL,
                    geometry="2d",
                    compressible=True,
                )
                for ar in aspect_ratios
            ]
        )
        E_uniaxial = 9 * gl_data.mu_GL * kappa_val / (3 * kappa_val + gl_data.mu_GL)
        plt.loglog(
            aspect_ratios,
            factor * Eeq_2d_comp_ar / E_uniaxial,
            color=colors[i],
            linewidth=2.5,
            label=f"2D, $\\kappa/\\mu=${kappa_mu_ratio:.1f}",
        )

    # Add GL fit curve for 2D
    Eeq_2d_GL_fit = np.array(
        [
            equivalent_modulus(
                mu0=gl_data.mu_GL,
                kappa0=kappa_GL_fit_2d,
                H=gl_data.H_GL,
                R=ar * gl_data.H_GL,
                geometry="2d",
                compressible=True,
            )
            for ar in aspect_ratios
        ]
    )
    E_GL_fit = (
        9 * gl_data.mu_GL * kappa_GL_fit_2d / (3 * kappa_GL_fit_2d + gl_data.mu_GL)
    )
    plt.loglog(
        aspect_ratios,
        factor * Eeq_2d_GL_fit / E_GL_fit,
        color="orange",
        linewidth=3,
        linestyle="-.",
        label=f"2D, $\\kappa/\\mu=${kappa_mu_GL_fit_2d:.0f} (GL fit)",
    )

    # Add incompressible limit for 2D
    mu0, H = 1.0, 1.0
    Eeq_2d_inc_ar = np.array(
        [
            equivalent_modulus(
                mu0=mu0, H=H, R=ar * H, geometry="2d", compressible=False
            )
            for ar in aspect_ratios
        ]
    )
    E_incompressible = 3 * mu0
    plt.loglog(
        aspect_ratios,
        factor * Eeq_2d_inc_ar / E_incompressible,
        color="black",
        linewidth=2.5,
        linestyle="--",
        label="2D incompressible",
    )

    # Add GL experimental point
    plt.scatter(
        GL_aspect_ratio,
        GL_Eeq_normalized,
        marker="*",
        s=150,
        color="red",
        edgecolor="black",
        linewidth=1,
        label="Gent-Lindley exp.",
        zorder=5,
    )

    plt.xlabel(r"$R/H$")
    plt.ylabel(r"$E_{eq}/E$")
    plt.legend()
    plt.title("2D Equivalent Modulus vs Aspect Ratio")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}Eeq_2D_aspect_ratio.pdf")
    plt.close()

    # Plot 3D cases
    plt.figure()
    for i, kappa_mu_ratio in enumerate(kappa_mu_ratios):
        kappa_val = kappa_mu_ratio * gl_data.mu_GL
        Eeq_3d_comp_ar = np.array(
            [
                equivalent_modulus(
                    mu0=gl_data.mu_GL,
                    kappa0=kappa_val,
                    H=gl_data.H_GL,
                    R=ar * gl_data.H_GL,
                    geometry="3d",
                    compressible=True,
                )
                for ar in aspect_ratios
            ]
        )
        E_uniaxial = 9 * gl_data.mu_GL * kappa_val / (3 * kappa_val + gl_data.mu_GL)
        plt.loglog(
            aspect_ratios,
            factor * Eeq_3d_comp_ar / E_uniaxial,
            color=colors[i],
            linewidth=2.5,
            label=f"3D, $\\kappa/\\mu=${kappa_mu_ratio:.1f}",
        )

    # Add GL fit curve for 3D
    Eeq_3d_GL_fit = np.array(
        [
            equivalent_modulus(
                mu0=gl_data.mu_GL,
                kappa0=kappa_GL_fit_3d,
                H=gl_data.H_GL,
                R=ar * gl_data.H_GL,
                geometry="3d",
                compressible=True,
            )
            for ar in aspect_ratios
        ]
    )
    E_GL_fit_3d = (
        9 * gl_data.mu_GL * kappa_GL_fit_3d / (3 * kappa_GL_fit_3d + gl_data.mu_GL)
    )
    plt.loglog(
        aspect_ratios,
        factor * Eeq_3d_GL_fit / E_GL_fit_3d,
        color="orange",
        linewidth=3,
        linestyle="-.",
        label=f"3D, $\\kappa/\\mu=${kappa_mu_GL_fit_3d:.0f} (GL fit)",
    )

    # Add incompressible limits for 3D
    Eeq_3d_inc_ar = np.array(
        [
            equivalent_modulus(
                mu0=gl_data.mu_GL,
                H=gl_data.H_GL,
                R=ar * gl_data.H_GL,
                geometry="3d",
                compressible=False,
            )
            for ar in aspect_ratios
        ]
    )
    E_incompressible = 3 * gl_data.mu_GL
    plt.loglog(
        aspect_ratios,
        factor * Eeq_3d_inc_ar / E_incompressible,
        color="black",
        linewidth=2.5,
        linestyle="--",
        label="3D incompressible",
    )

    # Add theoretical scaling line
    theoretical_3d_inc = (3 / 8) * gl_data.mu_GL * aspect_ratios**2 / E_incompressible
    plt.loglog(
        aspect_ratios,
        factor * theoretical_3d_inc,
        "k:",
        alpha=0.7,
        label=r"$\frac{1}{8}(R/H)^2$",
    )

    # Add GL experimental point
    plt.scatter(
        GL_aspect_ratio,
        GL_Eeq_normalized,
        marker="*",
        s=150,
        color="red",
        edgecolor="black",
        linewidth=1,
        label="Gent-Lindley exp.",
        zorder=5,
    )

    plt.xlabel(r"$R/H$")
    plt.ylabel(r"$E_{eq}/E$")
    plt.legend()
    plt.title("3D Equivalent Modulus vs Aspect Ratio")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}Eeq_3D_aspect_ratio.pdf")
    plt.close()


def plot_max_pressure_vs_eta(output_dir="figures/"):
    """Plot maximum pressure vs normalized parameter eta"""
    mu0, Delta, H, R = 1.0, 1.0, 1.0, 1.0
    eta_ = np.logspace(-1, 1, 100)
    kappa_range = 3 * mu0 / ((eta_ * H / R) ** 2)
    p_0 = mu0 * (Delta * R / H) ** 2

    # 2D cases
    p_max_2d_comp = np.array(
        [
            max_pressure(
                mu0=mu0,
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
    p_max_2d_inc_val = max_pressure(
        mu0=mu0, Delta=Delta, H=H, R=R, geometry="2d", compressible=False
    )

    # 3D cases
    p_max_3d_comp = np.array(
        [
            max_pressure(
                mu0=mu0,
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
    p_max_3d_inc_val = max_pressure(
        mu0=mu0, Delta=Delta, H=H, R=R, geometry="3d", compressible=False
    )

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


def plot_max_shear_vs_eta(output_dir="figures/"):
    """Plot maximum shear stress vs normalized parameter eta"""
    mu0, Delta, H, R = 1.0, 1.0, 1.0, 1.0
    eta_ = np.logspace(-1, 1, 100)
    kappa_range = 3 * mu0 / ((eta_ * H / R) ** 2)

    # 2D cases
    tau_max_2d_comp = np.array(
        [
            max_shear_stress(
                mu0=mu0,
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
    tau_max_2d_inc_val = max_shear_stress(
        mu0=mu0, Delta=Delta, H=H, R=R, geometry="2d", compressible=False
    )

    # 3D cases
    tau_max_3d_comp = np.array(
        [
            max_shear_stress(
                mu0=mu0,
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
    tau_max_3d_inc_val = max_shear_stress(
        mu0=mu0, Delta=Delta, H=H, R=R, geometry="3d", compressible=False
    )

    plt.figure()
    viridis_colors = plt.cm.viridis([0.2, 0.75])
    plt.plot(
        eta_,
        tau_max_2d_comp / mu0,
        label="2D",
        color=viridis_colors[0],
        linewidth=2.5,
    )
    plt.axhline(
        tau_max_2d_inc_val / mu0,
        linestyle="--",
        color=viridis_colors[0],
        linewidth=2.5,
        label=r"2D, $\kappa\to\infty$",
    )
    plt.plot(
        eta_,
        tau_max_3d_comp / mu0,
        label="3D",
        color=viridis_colors[1],
        linewidth=2.5,
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


def plot_gent_lindley_comparison(output_dir="figures/"):
    """Plot Gent-Lindley comparison with theoretical models"""
    plt.figure()
    fig, ax1 = plt.subplots()

    # Gent and Lindley slopes for the first and second part of the curve
    x = [0, 0.1267, 0.2119, 0.3861]
    y = [-0.02849, 9.9715, 12.3932, 14.2735]

    E2 = (y[-1] - y[0]) / (x[-1] - x[0]) * gl_data.Kgcm2_to_MPa
    E1 = (y[1] - y[0]) / (x[1] - x[0]) * gl_data.Kgcm2_to_MPa
    print(f"- Gent-Lindley slopes in MPa are {E1:2.3f} and {E2:2.3f} MPa")

    # oscar
    x_o = [0.0, 0.02475, 0.02475]
    y_o = [0.0, 0.1951, 0.5854]
    EoT = y_o[1] / x_o[1]
    Eo = y_o[2] / x_o[2]
    print(f"- Equivalent modulus in Oscar paper: {EoT:2.3f} - {Eo:2.3f} MPa")

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

    ax1.set_xlabel("Extension")
    ax1.set_ylabel(r"$F/S\, (\mathrm{MPa})$")
    ax2 = ax1.twinx()
    mn, mx = ax1.get_ylim()
    ax2.set_ylim(mn / gl_data.Kgcm2_to_MPa, mx / gl_data.Kgcm2_to_MPa)
    ax2.set_ylabel(r"$F/S\, (\mathrm{Kg}/\mathrm{cm}^2)$")

    xs = np.array([0, 0.2])
    ax1.plot(xs, xs * EoT, "y:", label="Kumar's corrected result")
    ax1.plot(
        xs,
        xs
        * equivalent_modulus(
            mu0=gl_data.mu_GL,
            kappa0=20,
            H=gl_data.H_GL,
            R=gl_data.D_GL / 2,
            geometry="3d",
            compressible=True,
        ),
        "g--",
        label="Weakly compressible model k=50 MPa",
    )
    ax1.plot(
        xs,
        xs
        * equivalent_modulus(
            mu0=gl_data.mu_GL,
            kappa0=gl_data.kappa_GL,
            H=gl_data.H_GL,
            R=gl_data.D_GL / 2,
            geometry="3d",
            compressible=True,
        ),
        "g-.",
        label="Weakly compressible model",
    )
    ax1.plot(
        xs,
        xs
        * equivalent_modulus(
            mu0=gl_data.mu_GL,
            H=gl_data.H_GL,
            R=gl_data.D_GL / 2,
            geometry="3d",
            compressible=False,
        ),
        "r:",
        label="Incompressible model",
    )
    ax1.set_ylim(0, 1.7)
    # add an horizontal line at 5/2*mu
    ax1.axhline(y=5 / 2 * gl_data.mu_GL, color="lightgray", linestyle=":")
    ax1.legend()
    fig.savefig(f"{output_dir}GL-fig2.pdf")
    plt.close()


def plot_pond_comparison(output_dir="figures/"):
    """Plot Pond et al. comparison"""
    plt.figure()
    thickness_pond = 3.2  # mm
    H_pond = thickness_pond / 2  # mm
    R_pond = 25.0
    E0 = 2.6  # Young's Modulus in MN/m^2 (from Table 1)
    mu0 = E0 / 3.0  # Derived Shear Modulus (~0.867 MN/m^2)
    mu_pond = 0.45  # MPa, fitted value from Pond data
    kappa_pond = 1000.0  # MPa, assumed value

    # Experimental Result from Figure 6
    # Peak stress ~1.4 MN/m^2 at ~6.25% strain
    Eeq_pond_exp = 22.4  # Observed equivalent stiffness in MN/m^2
    x = np.linspace(1, 1000, 1000)  # kappa values in MPa
    plt.plot(
        x,
        [
            equivalent_modulus(
                mu0=mu_pond,
                kappa0=k,
                H=H_pond,
                R=R_pond,
                geometry="3d",
                compressible=True,
            )
            for k in x
        ],
        "g--",
        label="Weakly compressible model",
    )

    plt.axhline(y=Eeq_pond_exp, color="r", linestyle=":", label="Pond et al. data")
    plt.ylabel(r"$E_{eq}$ (MN/m$^2$)")
    plt.xlabel("kappa (MPa)")
    plt.savefig(f"{output_dir}pond_figure6.pdf")
    plt.close()


if __name__ == "__main__":
    output_dir = "results/figures/"
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
            pressure(
                x,
                mu0=mu0,
                kappa0=k,
                Delta=Delta,
                H=H,
                R=R,
                geometry="2d",
                compressible=True,
            ),
            label=rf"$\kappa={k}$",
            color=colors[i],
        )
    plt.plot(
        x,
        pressure(x, mu0=mu0, Delta=Delta, H=H, R=R, geometry="2d", compressible=False),
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
            shear_stress(
                x,
                y,
                mu0=mu0,
                kappa0=k,
                Delta=Delta,
                H=H,
                R=R,
                geometry="2d",
                compressible=True,
            ),
            label=rf"$\kappa={k}$",
            color=colors[i],
        )
    plt.plot(
        x,
        shear_stress(
            x, y, mu0=mu0, Delta=Delta, H=H, R=R, geometry="2d", compressible=False
        ),
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
            shear_stress(
                x_fixed,
                y_points,
                mu0=mu0,
                kappa0=k,
                Delta=Delta,
                H=H,
                R=R,
                geometry="2d",
                compressible=True,
            ),
            label=rf"$\kappa={k}$",
            color=colors[i],
        )
    plt.plot(
        y_points,
        shear_stress(
            x_fixed,
            y_points,
            mu0=mu0,
            Delta=Delta,
            H=H,
            R=R,
            geometry="2d",
            compressible=False,
        ),
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
            pressure(
                r,
                mu0=mu0,
                kappa0=k,
                Delta=Delta,
                H=H,
                R=R,
                geometry="3d",
                compressible=True,
            ),
            label=rf"$\kappa={k}$",
            color=colors[i],
        )
    plt.plot(
        r,
        pressure(r, mu0=mu0, Delta=Delta, H=H, R=R, geometry="3d", compressible=False),
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
            shear_stress(
                r,
                z,
                mu0=mu0,
                kappa0=k,
                Delta=Delta,
                H=H,
                R=R,
                geometry="3d",
                compressible=True,
            ),
            label=rf"$\kappa={k}$",
            color=colors[i],
        )
    plt.plot(
        r,
        shear_stress(
            r, z, mu0=mu0, Delta=Delta, H=H, R=R, geometry="3d", compressible=False
        ),
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
            shear_stress(
                r_fixed,
                z_points,
                mu0=mu0,
                kappa0=k,
                Delta=Delta,
                H=H,
                R=R,
                geometry="3d",
                compressible=True,
            ),
            label=rf"$\kappa={k}$",
            color=colors[i],
        )
    plt.plot(
        z_points,
        shear_stress(
            r_fixed,
            z_points,
            mu0=mu0,
            Delta=Delta,
            H=H,
            R=R,
            geometry="3d",
            compressible=False,
        ),
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
    # Generate all plots using the modular functions
    print(f"Generating 2D pressure comparison in {output_dir}...")
    plot_pressure_comparison_2d(output_dir)

    print(f"Generating 2D shear stress comparison in {output_dir}...")
    plot_shear_stress_comparison_2d(output_dir)

    print(f"Generating 3D pressure comparison in {output_dir}...")
    plot_pressure_comparison_3d(output_dir)

    print(f"Generating 3D shear stress comparison in {output_dir}...")
    plot_shear_stress_comparison_3d(output_dir)

    print(f"Generating equivalent modulus comparison in {output_dir}...")
    plot_equivalent_modulus_comparison(output_dir)

    print(f"Generating equivalent modulus vs aspect ratio in {output_dir}...")
    plot_equivalent_modulus_vs_aspect_ratio(output_dir, max_aspect_ratio=50, factor=1.0)

    print(f"Generating maximum pressure vs eta in {output_dir}...")
    plot_max_pressure_vs_eta(output_dir)

    print(f"Generating maximum shear stress vs eta in {output_dir}...")
    plot_max_shear_vs_eta(output_dir)

    print(f"Generating Gent-Lindley comparison in {output_dir}...")
    plot_gent_lindley_comparison(output_dir)

    print(f"Generating Pond comparison in {output_dir}...")
    plot_pond_comparison(output_dir)

    print(f"All plots saved to {output_dir}")
