"""Simple linear elastic model for poker chip geometry.

Solves linear elasticity problems in 2D and 3D.
Simplified version without damage modeling.
"""

import os
import sys
from pathlib import Path

import numpy as np
import dolfinx
from mpi4py import MPI
import petsc4py
import ufl
import hydra
from omegaconf import DictConfig, OmegaConf

petsc4py.init(sys.argv)
from petsc4py import PETSc

from dolfinx.io import XDMFFile
from dolfinx.io.gmsh import model_to_mesh

# Import local modules
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from solvers import (
    SNESSolver,
    ColorPrint,
    assemble_scalar_reduce,
    evaluate_function,
)
from solvers.solver_configs import configure_3d_solver
from mesh import mesh_bar, mesh_chip, box_mesh, mesh_chip_eight, mesh_bar_quarter
from reference import formulas_paper

comm = MPI.COMM_WORLD


def mpi_print(*args, **kwargs):
    """Print only from rank 0 to avoid duplicate output in parallel."""
    if comm.rank == 0:
        print(*args, **kwargs)


def mpi_save_plot(filename):
    """Save plot only from rank 0."""
    if comm.rank == 0:
        import matplotlib.pyplot as plt

        plt.savefig(filename, bbox_inches="tight")
        plt.close()


def mpi_save_json(data, filename):
    """Save JSON file only from rank 0."""
    if comm.rank == 0:
        import json

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)


def plot_fields_along_lines(
    x_points, y_points, p_val_x, p_val_y, u_val_x, u_val_y, gdim, prefix
):
    """Plot pressure and displacement along x and y lines and save figures."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Pressure along x-line (y=0)
    axes[0, 0].plot(x_points, p_val_x, "b-", linewidth=2)
    axes[0, 0].set_xlabel("x position")
    axes[0, 0].set_ylabel("Pressure")
    axes[0, 0].set_title("Pressure along x-direction (y=0)")
    axes[0, 0].grid(True)

    # Pressure along y-line (x=0)
    axes[0, 1].plot(y_points, p_val_y, "r-", linewidth=2)
    axes[0, 1].set_xlabel("y position")
    axes[0, 1].set_ylabel("Pressure")
    axes[0, 1].set_title("Pressure along y-direction (x=0)")
    axes[0, 1].grid(True)

    # Displacement along x-line (y=0)
    if gdim == 2:
        axes[1, 0].plot(x_points, u_val_x[:, 0], "b-", linewidth=2, label="u_x")
        axes[1, 0].plot(x_points, u_val_x[:, 1], "g-", linewidth=2, label="u_y")
    else:  # gdim == 3
        axes[1, 0].plot(x_points, u_val_x[:, 0], "b-", linewidth=2, label="u_x")
        axes[1, 0].plot(x_points, u_val_x[:, 1], "g-", linewidth=2, label="u_y")
        axes[1, 0].plot(x_points, u_val_x[:, 2], "m-", linewidth=2, label="u_z")
    axes[1, 0].set_xlabel("x position")
    axes[1, 0].set_ylabel("Displacement")
    axes[1, 0].set_title("Displacement along x-direction (y=0)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Displacement along y-line (x=0)
    if gdim == 2:
        axes[1, 1].plot(y_points, u_val_y[:, 0], "b-", linewidth=2, label="u_x")
        axes[1, 1].plot(y_points, u_val_y[:, 1], "g-", linewidth=2, label="u_y")
    else:  # gdim == 3
        axes[1, 1].plot(y_points, u_val_y[:, 0], "b-", linewidth=2, label="u_x")
        axes[1, 1].plot(y_points, u_val_y[:, 1], "g-", linewidth=2, label="u_y")
        axes[1, 1].plot(y_points, u_val_y[:, 2], "m-", linewidth=2, label="u_z")
    axes[1, 1].set_xlabel("y position")
    axes[1, 1].set_ylabel("Displacement")
    axes[1, 1].set_title("Displacement along y-direction (x=0)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    mpi_save_plot(f"{prefix}_fields.pdf")


# Determine config path relative to script location
config_path = str(Path(__file__).parent.parent / "config")


@hydra.main(version_base=None, config_path=config_path, config_name="config_linear")
def main(cfg: DictConfig):
    parameters = cfg

    # Material parameters
    mu = parameters.model.get("mu", 0.59)
    k = parameters.model.get("kappa", 500.0)
    elasticity_model = parameters.model.get("elasticity_model", "pure_2d")

    # Geometry parameters
    gdim = parameters.geometry.get("geometric_dimension", 3)
    L = parameters.geometry.get("L", 1.0)
    H = parameters.geometry.get("H", 0.1)
    h_div = parameters.geometry.get("h_div", 3.0)
    degree_u = parameters.fem.get("degree_u", 1)  # Use standard linear elements
    lc = min(H, L) / h_div  # Simplified mesh size calculation
    sym = parameters.get("sym", True)  # Symmetry boundary conditions

    # Derived quantities - use plane strain (3D-like) expressions for 2D, 3D expressions for 3D
    mu = formulas_paper.mu_lame_from_kappa_mu(kappa=k, mu=mu)
    if gdim == 3:
        lmbda = formulas_paper.lambda_lame_from_kappa_mu(kappa=k, mu=mu, model="3D")
        E, nu = formulas_paper.E_nu_from_kappa_mu(kappa=k, mu=mu, model="3D")
    elif gdim == 2:
        lmbda = formulas_paper.lambda_lame_from_kappa_mu(
            kappa=k, mu=mu, model=elasticity_model
        )
        E, nu = formulas_paper.E_nu_from_kappa_mu(
            kappa=k, mu=mu, model=elasticity_model
        )
    else:
        raise ValueError("Invalid geometric dimension specified.")

    ColorPrint.print_info(f"Material properties: mu = {mu:.3f}, k = {k:.1f}")
    ColorPrint.print_info(f"Derived: E = {E:.3f}, nu = {nu:.3f}, lambda = {lmbda:.3f}")

    # Load parameters
    output_name = parameters.get("output_name", "poker_linear")
    load_max = parameters.get("load_max", 1.0)  # Single load for linear case
    sliding = parameters.get("sliding", 0)
    sym = parameters.get("sym", True)  # Symmetry boundary conditions
    outdir_arg = parameters.get("outdir", None)

    # For linear elasticity, single step is sufficient
    loads = [load_max]

    # Create the mesh

    if gdim == 2:
        if sym:
            gmsh_model, tdim, tag_names = mesh_bar_quarter(
                L, H, lc, gdim, verbose=False
            )
        elif sliding == 0:
            gmsh_model, tdim, tag_names = mesh_bar(L, H, lc, gdim, verbose=False)
        elif sliding == 1:
            gmsh_model, tdim, tag_names = mesh_bar(L, H, lc, gdim, verbose=False)
    elif gdim == 3:
        if sym:
            gmsh_model, tdim, tag_names = mesh_chip_eight(L, H, lc, gdim)
        elif sliding == 1:
            gmsh_model, tdim, tag_names = box_mesh(L, H, L, lc, gdim)
        else:
            gmsh_model, tdim, tag_names = box_mesh(L, H, L, lc, gdim)
    else:
        raise ValueError("Invalid geometric dimension specified.")

    mesh_data = model_to_mesh(
        gmsh_model,
        comm,
        0,  # model_rank
        gdim=gdim,
        partitioner=dolfinx.mesh.create_cell_partitioner(
            dolfinx.mesh.GhostMode.shared_facet
        ),
    )
    mesh, cell_tags, facet_tags = (
        mesh_data.mesh,
        mesh_data.cell_tags,
        mesh_data.facet_tags,
    )
    interfaces_keys = tag_names["facets"]

    # Output file setup
    if outdir_arg is not None:
        prefix = os.path.join(outdir_arg, output_name)
    else:
        prefix = output_name

    ColorPrint.print_info(f"Output prefix: {prefix}")

    # Define function spaces (using Crouzeix-Raviart elements)
    element_type = parameters.fem.get("element", "CR")
    if element_type == "CR":
        V_u = dolfinx.fem.functionspace(mesh, ("CR", degree_u, (mesh.geometry.dim,)))
    else:
        V_u = dolfinx.fem.functionspace(
            mesh, ("Lagrange", degree_u, (mesh.geometry.dim,))
        )
    V_scalar = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    # For output visualization (CR functions can't be directly written to XDMF)
    V_u_output = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))

    # Define functions
    u = dolfinx.fem.Function(V_u, name="displacement")
    u_bcs = dolfinx.fem.Function(V_u, name="boundary_displacement")

    # Function for output visualization
    u_output = dolfinx.fem.Function(V_u_output, name="displacement")

    # Functions for equivalent measures
    stress_func = dolfinx.fem.Function(V_scalar, name="stress")
    strain_func = dolfinx.fem.Function(V_scalar, name="strain")
    pressure_func = dolfinx.fem.Function(V_scalar, name="pressure")
    tau_func = dolfinx.fem.Function(V_scalar, name="equivalent_stress")

    # Define test and trial functions
    v = ufl.TestFunction(V_u)
    du = ufl.TrialFunction(V_u)

    # Define boundary conditions
    bottom_facets = facet_tags.indices[
        facet_tags.values == tag_names["facets"]["bottom"]
    ]
    top_facets = facet_tags.indices[facet_tags.values == tag_names["facets"]["top"]]
    dofs_u_top = dolfinx.fem.locate_dofs_topological(
        V_u, tdim - 1, np.array(top_facets)
    )
    dofs_u_bottom = dolfinx.fem.locate_dofs_topological(
        V_u, tdim - 1, np.array(bottom_facets)
    )

    dofs_u_bottom_normal = dolfinx.fem.locate_dofs_topological(
        V_u.sub(1), tdim - 1, facet_tags.find(tag_names["facets"]["bottom"])
    )

    # Robust facet tag lookup
    def get_facet_indices(tag_key):
        if tag_key in tag_names["facets"]:
            return facet_tags.indices[facet_tags.values == tag_names["facets"][tag_key]]
        else:
            print(f"Warning: facet tag '{tag_key}' not found in tag_names['facets'].")
            return np.array([])

    def get_facet_dofs(sub, tag_key):
        if tag_key in tag_names["facets"]:
            return dolfinx.fem.locate_dofs_topological(
                V_u.sub(sub), tdim - 1, facet_tags.find(tag_names["facets"][tag_key])
            )
        else:
            print(f"Warning: facet tag '{tag_key}' not found in tag_names['facets'].")
            return np.array([])

    left_facets = get_facet_indices("left")
    dofs_u_left_normal = get_facet_dofs(0, "left")
    right_facets = get_facet_indices("right")
    dofs_u_right_normal = get_facet_dofs(0, "right")
    front_facets = get_facet_indices("front")
    dofs_u_front_normal = get_facet_dofs(2, "front")
    back_facets = get_facet_indices("back")
    dofs_u_back_normal = get_facet_dofs(2, "back")

    # Define boundary conditions based on geometry and sliding type
    if gdim == 2:
        if sym:
            # 2D quarter domain with symmetry: constrain x-displacement on left, y-displacement on bottom
            bcs_u = [
                dolfinx.fem.dirichletbc(
                    0.0, dofs_u_bottom_normal, V_u.sub(1)
                ),  # bottom: u_y = 0 (blocked)
                dolfinx.fem.dirichletbc(
                    u_bcs, dofs_u_top
                ),  # top: prescribed displacement
                dolfinx.fem.dirichletbc(
                    0.0, dofs_u_left_normal, V_u.sub(0)
                ),  # left: u_x = 0 (symmetry)
            ]
        elif sliding == 1:
            # 2D with sliding: constrain x-displacement on left/right faces
            bcs_u = [
                dolfinx.fem.dirichletbc(
                    u_bcs, dofs_u_bottom
                ),  # bottom: prescribed displacement
                dolfinx.fem.dirichletbc(
                    u_bcs, dofs_u_top
                ),  # top: prescribed displacement
                dolfinx.fem.dirichletbc(
                    0.0, dofs_u_left_normal, V_u.sub(0)
                ),  # left: u_x = 0
                dolfinx.fem.dirichletbc(
                    0.0, dofs_u_right_normal, V_u.sub(0)
                ),  # right: u_x = 0
            ]
        else:
            # Default case: prescribed displacement on both top and bottom
            bcs_u = [
                dolfinx.fem.dirichletbc(
                    u_bcs, dofs_u_bottom
                ),  # bottom: prescribed displacement
                dolfinx.fem.dirichletbc(
                    u_bcs, dofs_u_top
                ),  # top: prescribed displacement
            ]
    elif gdim == 3:
        if sliding == 1:
            # 3D with sliding: constrain displacements on all side faces
            bcs_u = [
                dolfinx.fem.dirichletbc(
                    u_bcs, dofs_u_bottom
                ),  # bottom: prescribed displacement
                dolfinx.fem.dirichletbc(
                    u_bcs, dofs_u_top
                ),  # top: prescribed displacement
                dolfinx.fem.dirichletbc(
                    0.0, dofs_u_left_normal, V_u.sub(0)
                ),  # left: u_x = 0
                dolfinx.fem.dirichletbc(
                    0.0, dofs_u_right_normal, V_u.sub(0)
                ),  # right: u_x = 0
                dolfinx.fem.dirichletbc(
                    0.0, dofs_u_front_normal, V_u.sub(2)
                ),  # front: u_z = 0
                dolfinx.fem.dirichletbc(
                    0.0, dofs_u_back_normal, V_u.sub(2)
                ),  # back: u_z = 0
            ]

        elif sym:
            bcs_u = [
                dolfinx.fem.dirichletbc(
                    0.0, dofs_u_bottom_normal, V_u.sub(1)
                ),  # bottom: u_y = 0 (blocked)
                dolfinx.fem.dirichletbc(
                    u_bcs, dofs_u_top
                ),  # top: prescribed displacement
                dolfinx.fem.dirichletbc(
                    0.0, dofs_u_left_normal, V_u.sub(0)
                ),  # left: u_x = 0 (symmetry)
                dolfinx.fem.dirichletbc(
                    0.0, dofs_u_back_normal, V_u.sub(2)
                ),  # back: u_z = 0 (symmetry)
            ]

        else:
            # Default case: prescribed displacement on both top and bottom
            bcs_u = [
                dolfinx.fem.dirichletbc(
                    u_bcs, dofs_u_bottom
                ),  # bottom: prescribed displacement
                dolfinx.fem.dirichletbc(
                    u_bcs, dofs_u_top
                ),  # top: prescribed displacement
            ]

    k_ = dolfinx.fem.Constant(mesh, float(k))
    mu_ = dolfinx.fem.Constant(mesh, float(mu))
    lmbda_ = dolfinx.fem.Constant(mesh, lmbda)

    # Define strain and stress for linear elasticity
    def eps(u):
        return ufl.sym(ufl.grad(u))

    def eps_vol(u):
        """Volumetric strain (trace of strain tensor)."""
        return ufl.tr(eps(u))

    def eps_dev(u):
        """Deviatoric strain tensor."""
        return eps(u) - (1 / gdim) * eps_vol(u) * ufl.Identity(gdim)

    def sigma_vol(u):
        """Volumetric stress tensor."""
        return k_ * eps_vol(u) * ufl.Identity(gdim)

    def sigma_dev(u):
        """Deviatoric stress tensor."""
        return 2 * mu_ * eps_dev(u)

    def sigma(u):
        """Total linear elastic stress tensor."""
        return sigma_vol(u) + sigma_dev(u)

    def pressure(u):
        """Pressure (positive mean stress): p = tr(sigma)/gdim."""
        return ufl.tr(sigma(u)) / gdim

    def tau(u):
        """Equivalent stress (von Mises): sqrt(3/2 * dev(sigma):dev(sigma))."""
        dev_stress = sigma_dev(u)
        norm_dev = ufl.sqrt(ufl.inner(dev_stress, dev_stress))
        if gdim == 2:
            return ufl.sqrt(2) * norm_dev
        elif gdim == 3:
            return ufl.sqrt(3 / 2) * norm_dev

    # Define elastic energy components
    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 2 * degree_u})

    # Volumetric energy: (1/2) * K * (tr(ε))^2
    elastic_energy_vol = 0.5 * k_ * eps_vol(u) ** 2 * dx

    # Deviatoric energy: μ * dev(ε) : dev(ε)
    elastic_energy_dev = mu_ * ufl.inner(eps_dev(u), eps_dev(u)) * dx

    # Total elastic energy
    elastic_energy = elastic_energy_vol + elastic_energy_dev

    # Weak form for linear elasticity
    F = ufl.inner(sigma(u), eps(v)) * dx

    # ADD CR stabilization (jump stabilization on interior faces)
    if element_type == "CR":
        h_mesh = ufl.CellDiameter(mesh)
        h_avg = (h_mesh("+") + h_mesh("-")) / 2.0
        # For linear elasticity, use simple jump stabilization
        stabilizing_term = (1 / h_avg * ufl.dot(ufl.jump(u), ufl.jump(u))) * ufl.dS
        stabilizing_term_der = ufl.derivative(stabilizing_term, u, v)
        F += stabilizing_term_der

    J = ufl.derivative(F, u, du)

    # Prepare solver options
    solver_options = dict(parameters.solvers.elasticity.snes)

    # Configure 3D solver if needed
    if gdim == 3:
        solver_options = configure_3d_solver(V_u, parameters, solver_options, mpi_print)

    # Add verbose options if requested
    verbose_solver = parameters.get("verbose_solver", False)
    if verbose_solver:
        mpi_print("Enabling KSP monitor (iterations only)...")
        solver_options.update(
            {
                "ksp_monitor": "",  # Show KSP iterations only
            }
        )

    # Define solver
    solver_u = SNESSolver(
        F,
        u,
        bcs_u,
        J_form=J,
        petsc_options=solver_options,
        prefix=parameters.solvers.elasticity.prefix,
    )

    # Surface area for force calculation
    ds = ufl.Measure("ds", subdomain_data=facet_tags, domain=mesh)
    normal = ufl.FacetNormal(mesh)
    top_surface = assemble_scalar_reduce(
        ufl.dot(normal, normal) * ds(interfaces_keys["top"])
    )

    # History data (stored to JSON for post-processing and multirun analysis)
    history_data = {
        "load": [],
        "elastic_energy": [],
        "elastic_energy_vol": [],
        "elastic_energy_dev": [],
        "tau_x": [],
        "tau_y": [],
        "p_x": [],
        "p_y": [],
        "u_x": [],
        "u_y": [],
        "pressure_max": [],
        "pressure_center": [],  # Pressure at r≈0 for analytical comparison
        "tau_max": [],
        "F": [],
        "Equivalent_modulus": [],
        "imposed_strain": [],
        "average_stress": [],
        # Solver performance metrics (per load step)
        "solver_time": [],
        "iterations": [],
        "ksp_reason": [],
        "residual_norm": [],
        "parameters": OmegaConf.to_container(parameters, resolve=True),
    }

    with XDMFFile(comm, f"{prefix}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_mesh(mesh)
        file.write_meshtags(cell_tags, mesh.geometry)
        file.write_meshtags(facet_tags, mesh.geometry)

    # Apply load and solve
    ColorPrint.print_bold(f"-- Solving for load = {load_max:3.4f} --")

    # Update boundary conditions
    Delta = load_max * H
    if gdim == 2:
        u_bcs.interpolate(lambda x: (np.zeros_like(x[0]), Delta * x[1] / H))
    elif gdim == 3:
        # For other 3D cases: linear variation
        u_bcs.interpolate(
            lambda x: (
                np.zeros_like(x[0]),
                Delta * x[1] / H,
                np.zeros_like(x[2]),
            )
        )
    u_bcs.x.scatter_forward()

    # Solve the linear elastic problem
    import time

    solve_start_time = time.time()

    solver_u.solve()

    solve_time = time.time() - solve_start_time

    # Get solver statistics
    ksp = solver_u.solver.getKSP()
    its = ksp.getIterationNumber()
    reason_code = ksp.getConvergedReason()
    try:
        reason_name = PETSc.KSP.ConvergedReason(reason_code).name
    except Exception:
        reason_name = str(reason_code)
    residual_norm = ksp.getResidualNorm()

    mpi_print(
        f"Solver converged in {its} iterations (reason: {reason_name}) - Time: {solve_time:.3f}s"
    )
    mpi_print(f"Residual norm: {residual_norm:.3e}")

    # Calculate elastic energies
    elastic_energy_int = assemble_scalar_reduce(elastic_energy)
    elastic_energy_vol_int = assemble_scalar_reduce(elastic_energy_vol)
    elastic_energy_dev_int = assemble_scalar_reduce(elastic_energy_dev)

    # Calculate equivalent measures
    stress = sigma(u)
    strain = eps(u)
    press = pressure(u)

    # Project equivalent measures onto scalar function space
    V_tensor = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (gdim, gdim)))
    V_DG1 = dolfinx.fem.functionspace(mesh, ("DG", 1))  # DG1 space for pressure
    stress_expr = dolfinx.fem.Expression(stress, V_tensor.element.interpolation_points)
    strain_expr = dolfinx.fem.Expression(strain, V_tensor.element.interpolation_points)
    pressure_expr = dolfinx.fem.Expression(press, V_scalar.element.interpolation_points)
    pressure_expr_DG1 = dolfinx.fem.Expression(
        press, V_DG1.element.interpolation_points
    )
    tau_expr = dolfinx.fem.Expression(tau(u), V_scalar.element.interpolation_points)
    sigma_fun = dolfinx.fem.Function(V_tensor, name="stress_tensor")
    pressure_func_DG1 = dolfinx.fem.Function(V_DG1, name="pressure_DG1")

    #    stress_func.interpolate(stress_expr)
    #    strain_func.interpolate(strain_expr)
    pressure_func.interpolate(pressure_expr)
    pressure_func_DG1.interpolate(pressure_expr_DG1)  # Interpolate pressure on DG1
    tau_func.interpolate(tau_expr)
    sigma_fun.interpolate(stress_expr)

    # Find maximum values over the domain
    pressure_max = comm.allreduce(np.max(np.abs(pressure_func.x.array.real)))
    tau_max = comm.allreduce(np.max(np.abs(tau_func.x.array.real)))

    # Evaluate pressure at center point (r≈0) to compare with analytical formula
    # For 3D quarter cylinder: evaluate at (x=0.001, y=H/2, z=0.001) to avoid exact corner
    # For 2D: evaluate at (x=0.001, y=H/2)
    tol_center = 0.001  # Small offset to avoid exact corner/boundary
    if gdim == 3:
        center_point = np.array([[tol_center, H / 2, tol_center]], dtype=np.float64)
    else:
        center_point = np.array([[tol_center, H / 2]], dtype=np.float64)

    pressure_at_center = evaluate_function(pressure_func_DG1, center_point)
    if pressure_at_center is not None and len(pressure_at_center) > 0:
        # pressure_at_center is a 1D array with one value per point
        pressure_center = float(np.abs(pressure_at_center.flat[0]))
    else:
        pressure_center = 0.0

    # Use allreduce max to get the value from whichever rank owns the point
    pressure_center = comm.allreduce(pressure_center, op=MPI.MAX)

    # Calculate force on top surface
    top_surface = assemble_scalar_reduce(
        ufl.dot(normal, normal) * ds(interfaces_keys["top"])
    )
    # Calculate total force in y-direction on top surface
    if gdim == 2:
        force = assemble_scalar_reduce(
            ufl.dot(ufl.dot(sigma(u), normal), ufl.as_vector([0, 1]))
            * ds(interfaces_keys["top"])
        )
    else:  # gdim == 3
        force = assemble_scalar_reduce(
            ufl.dot(ufl.dot(sigma(u), normal), ufl.as_vector([0, 1, 0]))
            * ds(interfaces_keys["top"])
        )
    imposed_strain = Delta / H
    average_stress = force / top_surface
    equivalent_modulus = average_stress / imposed_strain

    # lines for saving fields
    tol = 0.0001  # Avoid hitting the outside of the domain
    npoints = 100
    x_points = np.linspace(tol, L - tol, npoints)  # Mesh from 0 to L in x-direction
    radial_line = np.zeros((3, npoints))
    radial_line[0] = x_points
    radial_line[1] = 0.0
    thickness_line = np.zeros((3, npoints))
    y_points = np.linspace(tol, H - tol, npoints)  # Mesh from 0 to H in y-direction
    thickness_line[1] = y_points
    radial_pts = np.ascontiguousarray(
        radial_line[: mesh.geometry.dim].T, dtype=np.float64
    )
    thickness_pts = np.ascontiguousarray(
        thickness_line[: mesh.geometry.dim].T, dtype=np.float64
    )

    # Evaluate fields along lines
    tau_x = evaluate_function(tau_func, radial_pts)
    tau_y = evaluate_function(tau_func, radial_pts)
    p_val_x = evaluate_function(pressure_func_DG1, radial_pts)  # Use DG1 pressure
    p_val_y = evaluate_function(pressure_func_DG1, thickness_pts)  # Use DG1 pressure

    # Evaluate displacement along lines
    u_val_x = evaluate_function(u, radial_pts)  # displacement along x-line
    u_val_y = evaluate_function(u, thickness_pts)  # displacement along y-line

    # Create and save plots (only on rank 0 and if evaluation succeeded)
    if comm.rank == 0 and tau_x is not None:
        plot_fields_along_lines(
            x_points, y_points, p_val_x, p_val_y, u_val_x, u_val_y, gdim, prefix
        )
        ColorPrint.print_info(
            f"Field plots saved to {os.path.abspath(prefix)}_fields.pdf"
        )

    # Store results (only on rank 0 or if data is available)
    history_data["load"].append(load_max)
    history_data["elastic_energy"].append(elastic_energy_int)
    history_data["elastic_energy_vol"].append(elastic_energy_vol_int)
    history_data["elastic_energy_dev"].append(elastic_energy_dev_int)
    history_data["pressure_max"].append(pressure_max)
    history_data["pressure_center"].append(pressure_center)
    history_data["tau_max"].append(tau_max)
    history_data["tau_x"].append(tau_x.tolist() if tau_x is not None else [])
    history_data["tau_y"].append(tau_y.tolist() if tau_y is not None else [])
    history_data["p_x"].append(p_val_x.tolist() if p_val_x is not None else [])
    history_data["p_y"].append(p_val_y.tolist() if p_val_y is not None else [])
    history_data["u_x"].append(u_val_x.tolist() if u_val_x is not None else [])
    history_data["u_y"].append(u_val_y.tolist() if u_val_y is not None else [])
    history_data["F"].append(force)
    history_data["imposed_strain"].append(imposed_strain)
    history_data["average_stress"].append(average_stress)
    history_data["Equivalent_modulus"].append(equivalent_modulus)
    # Store solver performance metrics for this load step
    history_data["solver_time"].append(solve_time)
    history_data["iterations"].append(its)
    history_data["ksp_reason"].append(reason_name)
    history_data["residual_norm"].append(residual_norm)
    ColorPrint.print_info(
        f"Energy: {elastic_energy_int:.6e} (Vol: {elastic_energy_vol_int:.6e}, Dev: {elastic_energy_dev_int:.6e})"
    )
    ColorPrint.print_info(f"Force: {force:.6e}, Max pressure: {pressure_max:.6e}")

    # Write solution to file
    # Project CR solution to Lagrange for visualization
    u_output.interpolate(u)

    with XDMFFile(comm, f"{prefix}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_function(u_output, 0)
        file.write_function(tau_func, 0)
        file.write_function(strain_func, 0)
        file.write_function(pressure_func, 0)
    # Save basic results
    if comm.rank == 0:
        import json

        mpi_save_json(history_data, f"{prefix}_data.json")

        ColorPrint.print_info(f"Results saved to {os.path.abspath(prefix)}_data.json")
        ColorPrint.print_info("Linear elastic computation completed successfully")

    # Return results for programmatic use
    geometry_data = {
        "L": L,
        "H": H,
        "h_div": h_div,
        "gdim": gdim,
        "mu": mu,
        "kappa": k,
        "E": E,
        "nu": nu,
    }
    # Print equivalent modulus and theoretical comparisons
    if comm.rank == 0:
        aspect_ratio = L / H  # This is the aspect ratio of the half-geometry
        # Convert mu, nu to kappa for compressible models
        kappa = k  # Use the kappa from the FEM calculation
        R_theory = L  # Radius/half-width for theoretical formulas
        H_theory = H  # Half-height for theoretical formulas
        Delta_theory = Delta  # Applied displacement
        # Calculate theoretical values using original functions with keyword arguments
        theories = {
            "FEM": equivalent_modulus,
            "2D_inc": formulas_paper.equivalent_modulus(
                mu0=mu, H=H_theory, R=R_theory, geometry="2d", compressible=False
            ),
            "2D_comp": formulas_paper.equivalent_modulus(
                mu0=mu,
                kappa0=kappa,
                H=H_theory,
                R=R_theory,
                geometry="2d",
                compressible=True,
            ),
            "3D_inc": formulas_paper.equivalent_modulus(
                mu0=mu, H=H_theory, R=R_theory, geometry="3d", compressible=False
            ),
            "3D_inc_exam": formulas_paper.equivalent_modulus(
                mu0=mu, H=H_theory, R=R_theory, geometry="3d", compressible=False
            ),  # Note: same as 3D_inc for now
            "3D_comp": formulas_paper.equivalent_modulus(
                mu0=mu,
                kappa0=kappa,
                H=H_theory,
                R=R_theory,
                geometry="3d",
                compressible=True,
            ),
            "2D_GL_analytic": (1 + (L / H) ** 2) * 4.0 / 3.0,
        }

        # Calculate theoretical pressure maxima using original functions
        p_theories = {
            "FEM": pressure_max,
            "2D_inc": formulas_paper.max_pressure(
                mu0=mu,
                Delta=Delta_theory,
                H=H_theory,
                R=R_theory,
                geometry="2d",
                compressible=False,
            ),
            "2D_comp": formulas_paper.max_pressure(
                mu0=mu,
                kappa0=kappa,
                Delta=Delta_theory,
                H=H_theory,
                R=R_theory,
                geometry="2d",
                compressible=True,
            ),
            "3D_inc": formulas_paper.max_pressure(
                mu0=mu,
                Delta=Delta_theory,
                H=H_theory,
                R=R_theory,
                geometry="3d",
                compressible=False,
            ),
            "3D_inc_exam": formulas_paper.max_pressure(
                mu0=mu,
                Delta=Delta_theory,
                H=H_theory,
                R=R_theory,
                geometry="3d",
                compressible=False,
            ),
            "3D_comp": formulas_paper.max_pressure(
                mu0=mu,
                kappa0=kappa,
                Delta=Delta_theory,
                H=H_theory,
                R=R_theory,
                geometry="3d",
                compressible=True,
            ),
        }

        ColorPrint.print_bold("=== Results Summary ===")
        ColorPrint.print_info(
            f"L/H = {aspect_ratio:.2f}, μ = {mu:.3f}, κ = {k:.1f}, E = {E:.3f}, ν = {nu:.3f}, E_plane_strain = {E / (1 - nu**2):.3f}, E_uniaxial_strain = {2 * mu + lmbda:.3f}"
        )

        ColorPrint.print_info(
            "E_equiv: " + " | ".join([f"{k}: {v:.3f}" for k, v in theories.items()])
        )
        ColorPrint.print_info(
            "p_max:   " + " | ".join([f"{k}: {v:.3f}" for k, v in p_theories.items()])
        )

    # compare computed and expected mesh volume (collective operation - must be outside rank 0 block!)
    mesh_volume = assemble_scalar_reduce(
        dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * dx)
    )
    if gdim == 2:
        expected_volume = 4 * L * H  # Rectangle: (2L) × (2H)
    else:  # gdim == 3
        expected_volume = 2 * np.pi * L**2 * H  # Cylinder: π*L²*(2H)

    mpi_print(
        f"Mesh volume (computed/expected): {mesh_volume:.6e}/{expected_volume:.6e}"
    )
    mpi_print(f"[Rank {comm.rank}] Returning from main function")

    return history_data, geometry_data


if __name__ == "__main__":
    main()
