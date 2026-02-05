"""
Test consistency between pressure and stiffness analytical formulas.

Verifies that the integral of pressure over the contact surface
equals the force predicted by the equivalent stiffness.
"""

import pytest
import numpy as np
from scipy.integrate import quad
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reference import formulas_paper


def test_2d_incompressible_pressure_stiffness_consistency():
    """
    Test 2D incompressible: integrate p(x) over [-R, R] should equal F from Eeq.

    Force from stiffness: F = Eeq * Delta/H * (2*R)  [force per unit depth]
    Force from pressure: F = 2 * integral_0^R p(x) dx  [factor 2 for symmetry]
    """
    # Test parameters
    mu0 = 1.0
    H = 1.0
    R = 5.0
    Delta = 0.1

    # Get equivalent stiffness
    Eeq = formulas_paper.equivalent_modulus(
        mu0=mu0, H=H, R=R, geometry="2d", compressible=False
    )

    # Force from stiffness (per unit depth)
    strain = Delta / H
    F_stiffness = Eeq * strain * (2 * R)

    # Force from integrating pressure (use symmetry: 2 * integral from 0 to R)
    def pressure_2d(x):
        return formulas_paper.pressure(
            x, mu0=mu0, Delta=Delta, H=H, R=R, geometry="2d", compressible=False
        )

    F_pressure, _ = quad(pressure_2d, 0, R)
    F_pressure *= 2  # Account for both sides [-R, R]

    # Check relative error
    rel_error = abs(F_stiffness - F_pressure) / abs(F_stiffness)
    print(
        f"2D incompressible: F_stiffness = {F_stiffness:.6e}, F_pressure = {F_pressure:.6e}, rel_error = {rel_error:.6e}"
    )

    assert rel_error < 1e-3, (
        f"2D incompressible: Force mismatch, relative error = {rel_error}"
    )


def test_2d_compressible_pressure_stiffness_consistency():
    """
    Test 2D compressible: integrate p(x) over [-R, R] should equal F from Eeq.
    """
    # Test parameters
    mu0 = 1.0
    kappa0 = 100.0
    H = 1.0
    R = 5.0
    Delta = 0.1

    # Get equivalent stiffness
    Eeq = formulas_paper.equivalent_modulus(
        mu0=mu0, kappa0=kappa0, H=H, R=R, geometry="2d", compressible=True
    )

    # Force from stiffness (per unit depth)
    strain = Delta / H
    F_stiffness = Eeq * strain * (2 * R)

    # Force from integrating pressure (use symmetry: 2 * integral from 0 to R)
    def pressure_2d(x):
        return formulas_paper.pressure(
            x,
            mu0=mu0,
            kappa0=kappa0,
            Delta=Delta,
            H=H,
            R=R,
            geometry="2d",
            compressible=True,
        )

    F_pressure, _ = quad(pressure_2d, 0, R)
    F_pressure *= 2  # Account for both sides [-R, R]

    # Check relative error
    rel_error = abs(F_stiffness - F_pressure) / abs(F_stiffness)
    print(
        f"2D compressible: F_stiffness = {F_stiffness:.6e}, F_pressure = {F_pressure:.6e}, rel_error = {rel_error:.6e}"
    )

    assert rel_error < 1e-3, (
        f"2D compressible: Force mismatch, relative error = {rel_error}"
    )


def test_3d_incompressible_pressure_stiffness_consistency():
    """
    Test 3D incompressible: integrate p(r) over circle should equal F from Eeq.

    Force from stiffness: F = Eeq * Delta/H * (pi*R^2)
    Force from pressure: F = 2*pi * integral_0^R p(r) * r dr
    """
    # Test parameters
    mu0 = 1.0
    H = 1.0
    R = 5.0
    Delta = 0.1

    # Get equivalent stiffness
    Eeq = formulas_paper.equivalent_modulus(
        mu0=mu0, H=H, R=R, geometry="3d", compressible=False
    )

    # Force from stiffness
    strain = Delta / H
    area = np.pi * R**2
    F_stiffness = Eeq * strain * area

    # Force from integrating pressure over circular surface
    # F = integral p dA = integral_0^R p(r) * 2*pi*r dr
    def integrand_3d(r):
        p = formulas_paper.pressure(
            r, mu0=mu0, Delta=Delta, H=H, R=R, geometry="3d", compressible=False
        )
        return p * 2 * np.pi * r

    F_pressure, _ = quad(integrand_3d, 0, R)

    # Check relative error
    rel_error = abs(F_stiffness - F_pressure) / abs(F_stiffness)
    print(
        f"3D incompressible: F_stiffness = {F_stiffness:.6e}, F_pressure = {F_pressure:.6e}, rel_error = {rel_error:.6e}"
    )

    assert rel_error < 1e-3, (
        f"3D incompressible: Force mismatch, relative error = {rel_error}"
    )


def test_3d_compressible_pressure_stiffness_consistency():
    """
    Test 3D compressible: integrate p(r) over circle should equal F from Eeq.
    """
    # Test parameters
    mu0 = 1.0
    kappa0 = 100.0
    H = 1.0
    R = 5.0
    Delta = 0.1

    # Get equivalent stiffness
    Eeq = formulas_paper.equivalent_modulus(
        mu0=mu0, kappa0=kappa0, H=H, R=R, geometry="3d", compressible=True
    )

    # Force from stiffness
    strain = Delta / H
    area = np.pi * R**2
    F_stiffness = Eeq * strain * area

    # Force from integrating pressure over circular surface
    # F = integral p dA = integral_0^R p(r) * 2*pi*r dr
    def integrand_3d(r):
        p = formulas_paper.pressure(
            r,
            mu0=mu0,
            kappa0=kappa0,
            Delta=Delta,
            H=H,
            R=R,
            geometry="3d",
            compressible=True,
        )
        return p * 2 * np.pi * r

    F_pressure, _ = quad(integrand_3d, 0, R)

    # Check relative error
    rel_error = abs(F_stiffness - F_pressure) / abs(F_stiffness)
    print(
        f"3D compressible: F_stiffness = {F_stiffness:.6e}, F_pressure = {F_pressure:.6e}, rel_error = {rel_error:.6e}"
    )

    assert rel_error < 1e-3, (
        f"3D compressible: Force mismatch, relative error = {rel_error}"
    )


@pytest.mark.parametrize("R", [1.0, 3.0, 5.0, 10.0])
def test_3d_compressible_multiple_aspect_ratios(R):
    """Test consistency across different aspect ratios for 3D compressible case."""
    mu0 = 1.0
    kappa0 = 100.0
    H = 1.0
    Delta = 0.1

    # Get equivalent stiffness
    Eeq = formulas_paper.equivalent_modulus(
        mu0=mu0, kappa0=kappa0, H=H, R=R, geometry="3d", compressible=True
    )

    # Force from stiffness
    strain = Delta / H
    area = np.pi * R**2
    F_stiffness = Eeq * strain * area

    # Force from integrating pressure
    def integrand_3d(r):
        p = formulas_paper.pressure(
            r,
            mu0=mu0,
            kappa0=kappa0,
            Delta=Delta,
            H=H,
            R=R,
            geometry="3d",
            compressible=True,
        )
        return p * 2 * np.pi * r

    F_pressure, _ = quad(integrand_3d, 0, R)

    # Check relative error
    rel_error = abs(F_stiffness - F_pressure) / abs(F_stiffness)

    assert rel_error < 1e-3, (
        f"R/H = {R / H:.1f}: Force mismatch, relative error = {rel_error}"
    )


if __name__ == "__main__":
    # Run tests with verbose output
    print("Testing analytical formula consistency...\n")

    test_2d_incompressible_pressure_stiffness_consistency()
    test_2d_compressible_pressure_stiffness_consistency()
    test_3d_incompressible_pressure_stiffness_consistency()
    test_3d_compressible_pressure_stiffness_consistency()

    print("\nTesting multiple aspect ratios...")
    for R in [1.0, 3.0, 5.0, 10.0]:
        test_3d_compressible_multiple_aspect_ratios(R)
        print(f"  R/H = {R:.1f}: PASSED")

    print("\n✓ All consistency tests passed!")
