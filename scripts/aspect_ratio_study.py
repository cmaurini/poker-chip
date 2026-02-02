"""Parametric study of equivalent modulus vs aspect ratio.

Runs multiple linear elastic simulations with varying L/H ratios
and plots the equivalent modulus as a function of aspect ratio.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpi4py import MPI
import subprocess
import tempfile
import yaml

# Import poker_chip package
_script_dir = Path(__file__).parent.parent  # Go up to project root
sys.path.insert(0, str(_script_dir))

# Import Gent and Lindley formulas from reference directory
from reference import formulas_paper
from reference import gent_lindley_data as gl_data

# Extract the functions we need
Eeq_2d = formulas_paper.Eeq_2d
Eeq_2d_inc = formulas_paper.Eeq_2d_inc
Eeq_3d = formulas_paper.Eeq_3d
Eeq_3d_inc = formulas_paper.Eeq_3d_inc
Eeq_3d_inc_exam = formulas_paper.Eeq_3d_inc_exam

comm = MPI.COMM_WORLD


def calculate_all_theory_formulas(aspect_ratios, L_base=1.0, mu0=0.59, kappa0=500.0):
    """Calculate theoretical equivalent modulus using all available formulas."""
    results = {
        "2D_incompressible": [],
        "2D_compressible": [],
        "3D_incompressible": [],
        "3D_incompressible_exam": [],
        "3D_compressible": [],
    }

    for ar in aspect_ratios:
        H = L_base / ar
        R = L_base  # For chip geometry, R ≈ L

        # 2D formulas
        results["2D_incompressible"].append(Eeq_2d_inc(mu0=mu0, H=H, R=R))
        results["2D_compressible"].append(Eeq_2d(mu0=mu0, kappa0=kappa0, H=H, R=R))

        # 3D formulas
        results["3D_incompressible"].append(Eeq_3d_inc(mu0=mu0, H=H, R=R))
        results["3D_incompressible_exam"].append(Eeq_3d_inc_exam(mu0=mu0, H=H, R=R))
        results["3D_compressible"].append(Eeq_3d(mu0=mu0, kappa0=kappa0, H=H, R=R))

    # Convert to numpy arrays
    for key in results:
        results[key] = np.array(results[key])

    return results


def create_temp_config(aspect_ratio, h_div=8, H_base=1.0):
    """Create temporary configuration files for given aspect ratio."""
    H = H_base
    L = H_base * aspect_ratio

    # Create temporary config
    temp_config = {
        "geometry": {
            "L": L,
            "H": H,
            "h_div": h_div,
            "geometric_dimension": 2,
            "geometry_type": "chip",
        },
        "model": {"mu": 1, "kappa": 500.0},
        "fem": {"degree_u": 1},
        "loading": {"body_force": [0.0, 0.0, 0.0], "loading_steps": [1.0]},
        "sliding": 0,
        "solvers": {
            "elasticity": {
                "snes": {
                    "snes_type": "ksponly",
                    "snes_linesearch_type": "basic",
                    "snes_rtol": 1e-8,
                    "snes_atol": 1e-8,
                    "snes_max_it": 50,
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                },
                "prefix": "elasticity",
            }
        },
        "io": {"results_folder": f"results/aspect_ratio_study/AR_{aspect_ratio:.2f}"},
    }

    return temp_config


def calculate_equivalent_modulus(force, displacement, geometry):
    """Calculate equivalent modulus from force-displacement data."""
    L = geometry["L"]
    H = geometry["H"]

    # The 'force' from the simulation is actually stress (force/top_surface)
    # The 'displacement' from the simulation is the load parameter (0.1)
    # Actual applied displacement is: Delta = displacement * H
    # Total displacement between top and bottom is: 2 * Delta = 2 * displacement * H

    stress = force  # This is already stress (force/area) from the simulation
    strain = (
        displacement  # The loading parameter is the strain applied (displacement/H)
    )

    # Simplifies to: strain = 2 * displacement = 2 * 0.1 = 0.2
    strain = 2 * displacement

    if strain > 0:
        E_equiv = stress / strain
    else:
        E_equiv = 0.0

    return E_equiv


def run_single_simulation(aspect_ratio, h_div=8):
    """Run a single simulation for given aspect ratio."""

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create config files
        config = create_temp_config(aspect_ratio, h_div)
        config_file = temp_path / "temp_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config, f)

        # Run simulation using hydra override
        script_path = Path(__file__).parent.parent / "solvers" / "poker_chip_linear.py"

        # Build hydra overrides
        overrides = [
            f"geometry.L={config['geometry']['L']}",
            f"geometry.H={config['geometry']['H']}",
            f"geometry.h_div={config['geometry']['h_div']}",
            f"hydra.run.dir={temp_path}",
            "output_name=temp_result",
        ]

        cmd = [
            sys.executable,
            str(script_path),
            "-cd",
            str(Path(__file__).parent.parent / "config"),
            "-cn",
            "config_linear",
            f"geometry.L={config['geometry']['L']}",
            f"geometry.H={config['geometry']['H']}",
            f"geometry.h_div={config['geometry']['h_div']}",
            f"hydra.run.dir={temp_path}",
            "output_name=temp_result",
        ]

        print(f"Running simulation for aspect ratio {aspect_ratio:.2f}...")
        print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_path)
            if result.returncode != 0:
                print(f"Error running simulation: {result.stderr}")
                return None

            # Load results
            result_file = temp_path / "temp_result_data.json"
            if result_file.exists():
                with open(result_file, "r") as f:
                    data = json.load(f)
                return data, config["geometry"]
            else:
                print(f"Result file not found: {result_file}")
                return None

        except Exception as e:
            print(f"Exception running simulation: {e}")
            return None


def main():
    """Main function to run parametric study."""

    # Only run on rank 0
    if comm.rank != 0:
        return

    # Define aspect ratios to study
    aspect_ratios = np.logspace(np.log10(1.0), np.log10(20.0), 10)  # From 1 to 20

    results = []

    for ar in aspect_ratios:
        result = run_single_simulation(ar, h_div=8)
        if result is not None:
            data, geometry = result

            # Get final values
            force = data["F"][-1] if data["F"] else 0
            displacement = data["load"][-1] if data["load"] else 0

            # Calculate equivalent modulus
            E_equiv = calculate_equivalent_modulus(force, displacement, geometry)

            results.append(
                {
                    "aspect_ratio": ar,
                    "L": geometry["L"],
                    "H": geometry["H"],
                    "force": force,
                    "displacement": displacement,
                    "E_equiv": E_equiv,
                    "elastic_energy": data["elastic_energy"][-1]
                    if data["elastic_energy"]
                    else 0,
                    "elastic_energy_vol": data["elastic_energy_vol"][-1]
                    if data["elastic_energy_vol"]
                    else 0,
                    "elastic_energy_dev": data["elastic_energy_dev"][-1]
                    if data["elastic_energy_dev"]
                    else 0,
                }
            )

            print(
                f"AR={ar:.2f}: L={geometry['L']:.3f}, H={geometry['H']:.3f}, "
                f"F={force:.2e}, δ={displacement:.2e}, E_eq={E_equiv:.2e}"
            )
        else:
            print(f"Failed to get results for aspect ratio {ar:.2f}")

    # Save results
    output_dir = Path("results/aspect_ratio_study")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "aspect_ratio_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot results with Gent and Lindley theory comparison
    if results:
        aspect_ratios_data = np.array([r["aspect_ratio"] for r in results])
        E_equiv_data = np.array([r["E_equiv"] for r in results])

        # Calculate all theoretical values
        theory_results = calculate_all_theory_formulas(
            aspect_ratios_data, L_base=1.0, mu0=0.59, kappa0=500.0
        )

        # Create comprehensive comparison plot with all theories
        plt.figure(figsize=(14, 8))

        # Plot numerical results
        plt.semilogx(
            aspect_ratios_data,
            E_equiv_data,
            "o-",
            linewidth=3,
            markersize=8,
            label="FEM (CR elements)",
            color="red",
        )

        # Plot all theoretical predictions with different line styles
        colors = ["blue", "green", "purple", "orange", "brown"]
        linestyles = ["--", "-.", ":", "-", "--"]

        theory_labels = [
            "2D incompressible",
            "2D compressible",
            "3D incompressible",
            "3D incompressible (exam)",
            "3D compressible",
        ]

        for i, (key, theory_values) in enumerate(theory_results.items()):
            plt.semilogx(
                aspect_ratios_data,
                theory_values,
                linestyles[i],
                linewidth=2,
                label=f"Theory: {theory_labels[i]}",
                color=colors[i],
            )
        plt.xlabel("Aspect Ratio (L/H)", fontsize=12)
        plt.ylabel("Equivalent Modulus (E_equiv)", fontsize=12)
        plt.title(
            "Equivalent Modulus vs Aspect Ratio: FEM vs All Theoretical Formulas\n(8 elements through thickness)",
            fontsize=14,
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            output_dir / "equivalent_modulus_all_theories.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            output_dir / "equivalent_modulus_all_theories.pdf", bbox_inches="tight"
        )

        # Create error analysis plot for all theories
        plt.figure(figsize=(14, 8))

        for i, (key, theory_values) in enumerate(theory_results.items()):
            error = 100 * (E_equiv_data - theory_values) / theory_values
            plt.semilogx(
                aspect_ratios_data,
                error,
                linestyles[i] if i < len(linestyles) else "-",
                linewidth=2,
                label=f"{theory_labels[i]}",
                color=colors[i],
                marker="o" if i < 3 else "s",
            )

        plt.axhline(y=0, color="k", linestyle=":", alpha=0.5)

        plt.xlabel("Aspect Ratio (L/H)", fontsize=12)
        plt.ylabel("Relative Error (%)", fontsize=12)
        plt.title(
            "FEM vs All Theoretical Formulas: Relative Error Analysis", fontsize=14
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            output_dir / "theory_comparison_all_errors.png",
            dpi=300,
            bbox_inches="tight",
        )

        # Additional plot: Energy components
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        vol_energy = [r["elastic_energy_vol"] for r in results]
        dev_energy = [r["elastic_energy_dev"] for r in results]
        plt.semilogx(
            aspect_ratios_data, vol_energy, "o-", label="Volumetric", linewidth=2
        )
        plt.semilogx(
            aspect_ratios_data, dev_energy, "s-", label="Deviatoric", linewidth=2
        )
        plt.xlabel("Aspect Ratio (L/H)")
        plt.ylabel("Elastic Energy")
        plt.title("Energy Components vs Aspect Ratio")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        forces = [r["force"] for r in results]
        displacements = [r["displacement"] for r in results]
        plt.semilogx(aspect_ratios_data, forces, "o-", label="Force", linewidth=2)
        plt.xlabel("Aspect Ratio (L/H)")
        plt.ylabel("Force")
        plt.title("Force vs Aspect Ratio")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "energy_and_force_vs_aspect_ratio.png", dpi=300)

        print(f"\nResults saved to {output_dir}")
        print(f"Total successful simulations: {len(results)}")

        # Print comprehensive theory comparison summary
        print(f"\n=== All Theoretical Formulas Comparison ===")

        # Calculate errors for all theories
        all_errors = {}
        theory_labels = [
            "2D incompressible",
            "2D compressible",
            "3D incompressible",
            "3D incompressible (exam)",
            "3D compressible",
        ]

        for i, (key, theory_values) in enumerate(theory_results.items()):
            error = 100 * (E_equiv_data - theory_values) / theory_values
            all_errors[theory_labels[i]] = error

        # Print header
        header = f"{'AR':>6} {'FEM':>8}"
        for label in theory_labels:
            header += f" {label.replace(' ', '_')[:10]:>10}"
        print(header)
        print("-" * (8 + 6 + 10 * len(theory_labels)))

        # Print data rows
        for i, ar in enumerate(aspect_ratios_data):
            row = f"{ar:6.2f} {E_equiv_data[i]:8.1f}"
            for key, theory_values in theory_results.items():
                row += f" {theory_values[i]:10.1f}"
            print(row)

        print("\n=== Relative Errors (%) ===")
        header = f"{'AR':>6}"
        for label in theory_labels:
            header += f" {label.replace(' ', '_')[:10]:>10}"
        print(header)
        print("-" * (6 + 10 * len(theory_labels)))

        for i, ar in enumerate(aspect_ratios_data):
            row = f"{ar:6.2f}"
            for label in theory_labels:
                row += f" {all_errors[label][i]:10.1f}"
            print(row)

        # Find best matching theory
        print(f"\n=== Best Matching Theories (by average absolute error) ===")
        avg_errors = {}
        for label in theory_labels:
            avg_errors[label] = np.mean(np.abs(all_errors[label]))
            print(f"  {label}: {avg_errors[label]:.1f}%")

        best_theory = min(avg_errors, key=avg_errors.get)
        print(
            f"\nBest overall match: {best_theory} (avg error: {avg_errors[best_theory]:.1f}%)"
        )


if __name__ == "__main__":
    main()
