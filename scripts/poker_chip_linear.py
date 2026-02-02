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

from dolfinx.io import XDMFFile
from dolfinx.io.gmsh import model_to_mesh

# Import poker_chip package
_script_dir = Path(__file__).parent.parent  # Go up to project root
sys.path.insert(0, str(_script_dir / "src"))

from poker_chip.solvers import (
    SNESSolver,
    ColorPrint,
    assemble_scalar_reduce,
    evaluate_function,
)
from poker_chip.mesh import mesh_bar, mesh_chip, box_mesh

comm = MPI.COMM_WORLD


@hydra.main(version_base=None, config_path="../config", config_name="config_linear")
def main(cfg: DictConfig):
    parameters = cfg

    # Material parameters
    mu = parameters.model.get("mu", 0.59)
    k = parameters.model.get("kappa", 500.0)

    # Geometry parameters
    gdim = parameters.geometry.get("geometric_dimension", 3)
    L = parameters.geometry.get("L", 1.0)
    H = parameters.geometry.get("H", 0.1)
    h_div = parameters.geometry.get("h_div", 3.0)
    degree_u = parameters.fem.get("degree_u", 1)  # Use standard linear elements
    lc = H / h_div  # Simplified mesh size calculation

    # Derived quantities
    lmbda = k - 2 / gdim * mu
    nu = (1 - 2 * mu / (gdim * k)) / (gdim - 1 + 2 * mu / (gdim * k))
    E = 2 * mu * (1 + nu)

    ColorPrint.print_info(f"Material properties: mu = {mu:.3f}, k = {k:.1f}")
    ColorPrint.print_info(f"Derived: E = {E:.3f}, nu = {nu:.3f}, lambda = {lmbda:.3f}")

    # Load parameters
    output_name = parameters.get("output_name", "poker_linear")
    load_max = parameters.get("load_max", 0.1)  # Single load for linear case
    sliding = parameters.get("sliding", 1)
    outdir_arg = parameters.get("outdir", None)

    # For linear elasticity, single step is sufficient
    loads = [load_max]

    # Create the mesh
    if gdim == 2:
        gmsh_model, tdim, tag_names = mesh_bar(L * 2, H * 2, lc, gdim)
    elif gdim == 3 and sliding == 0:
        gmsh_model, tdim, tag_names = mesh_chip(L * 2, H * 2, lc, gdim)
    elif gdim == 3 and sliding == 1:
        gmsh_model, tdim, tag_names = box_mesh(L * 2, H * 2, L * 2, lc, gdim)

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
    V_u = dolfinx.fem.functionspace(mesh, ("CR", degree_u, (mesh.geometry.dim,)))
    V_scalar = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    def CR_stabilisation(u):
        """CR stabilization term for linear elasticity (no damage)"""
        h = ufl.CellDiameter(mesh)
        h_avg = (h("+") + h("-")) / 2.0
        # For linear elasticity, use simple jump stabilization
        stabilizing_term = (1 / h_avg * ufl.dot(ufl.jump(u), ufl.jump(u))) * ufl.dS
        return stabilizing_term

    # For output visualization (CR functions can't be directly written to XDMF)
    V_u_output = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))

    # Define functions
    u = dolfinx.fem.Function(V_u, name="displacement")
    u_top = dolfinx.fem.Function(V_u, name="boundary_displacement_top")
    u_bottom = dolfinx.fem.Function(V_u, name="boundary_displacement_bottom")

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

    if gdim == 2 and sliding == 1:
        left_facets = facet_tags.indices[
            facet_tags.values == tag_names["facets"]["left"]
        ]
        right_facets = facet_tags.indices[
            facet_tags.values == tag_names["facets"]["right"]
        ]
        dofs_u_left = dolfinx.fem.locate_dofs_topological(
            V_u.sub(0), tdim - 1, np.array(left_facets)
        )
        dofs_u_right = dolfinx.fem.locate_dofs_topological(
            V_u.sub(0), tdim - 1, np.array(right_facets)
        )
        bcs_u = [
            dolfinx.fem.dirichletbc(u_bottom, dofs_u_bottom),
            dolfinx.fem.dirichletbc(u_top, dofs_u_top),
            dolfinx.fem.dirichletbc(0.0, dofs_u_left, V_u.sub(0)),
            dolfinx.fem.dirichletbc(0.0, dofs_u_right, V_u.sub(0)),
        ]
    elif gdim == 3 and sliding == 1:
        dof_u_left = dolfinx.fem.locate_dofs_topological(
            V_u.sub(0),
            tdim - 1,
            facet_tags.find(tag_names["facets"]["left"]),
        )
        dof_u_right = dolfinx.fem.locate_dofs_topological(
            V_u.sub(0),
            tdim - 1,
            facet_tags.find(tag_names["facets"]["right"]),
        )
        dof_u_front = dolfinx.fem.locate_dofs_topological(
            V_u.sub(2),
            tdim - 1,
            facet_tags.find(tag_names["facets"]["front"]),
        )
        dof_u_back = dolfinx.fem.locate_dofs_topological(
            V_u.sub(2),
            tdim - 1,
            facet_tags.find(tag_names["facets"]["back"]),
        )
        bcs_u = [
            dolfinx.fem.dirichletbc(u_bottom, dofs_u_bottom),
            dolfinx.fem.dirichletbc(u_top, dofs_u_top),
            dolfinx.fem.dirichletbc(0.0, dof_u_left, V_u.sub(0)),
            dolfinx.fem.dirichletbc(0.0, dof_u_right, V_u.sub(0)),
            dolfinx.fem.dirichletbc(0.0, dof_u_front, V_u.sub(2)),
            dolfinx.fem.dirichletbc(0.0, dof_u_back, V_u.sub(2)),
        ]
    else:
        bcs_u = [
            dolfinx.fem.dirichletbc(u_bottom, dofs_u_bottom),
            dolfinx.fem.dirichletbc(u_top, dofs_u_top),
        ]

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
        return k * eps_vol(u) * ufl.Identity(gdim)

    def sigma_dev(u):
        """Deviatoric stress tensor."""
        return 2 * mu * eps_dev(u)

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
    elastic_energy_vol = 0.5 * k * eps_vol(u) ** 2 * dx

    # Deviatoric energy: μ * dev(ε) : dev(ε)
    elastic_energy_dev = mu * ufl.inner(eps_dev(u), eps_dev(u)) * dx

    # Total elastic energy
    elastic_energy = elastic_energy_vol + elastic_energy_dev

    # Weak form for linear elasticity
    F = ufl.inner(sigma(u), eps(v)) * dx

    # ADD CR stabilization (jump stabilization on interior faces)
    if degree_u == 1:
        stabilizing_term_der = ufl.derivative(CR_stabilisation(u), u, v)
        F += stabilizing_term_der

    J = ufl.derivative(F, u, du)

    # Define solver
    solver_u = SNESSolver(
        F,
        u,
        bcs_u,
        J_form=J,
        petsc_options=parameters.solvers.elasticity.snes,
        prefix=parameters.solvers.elasticity.prefix,
    )

    # Surface area for force calculation
    ds = ufl.Measure("ds", subdomain_data=facet_tags, domain=mesh)
    normal = ufl.FacetNormal(mesh)
    top_surface = assemble_scalar_reduce(
        ufl.dot(normal, normal) * ds(interfaces_keys["top"])
    )

    # History data
    history_data = {
        "load": [],
        "elastic_energy": [],
        "elastic_energy_vol": [],
        "elastic_energy_dev": [],
        "tau_x": [],
        "tau_y": [],
        "p_x": [],
        "p_y": [],
        "pressure_max": [],
        "F": [],
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
        u_top.interpolate(lambda x: (np.zeros_like(x[0]), Delta * np.ones_like(x[1])))
        u_bottom.interpolate(
            lambda x: (np.zeros_like(x[0]), -Delta * np.ones_like(x[1]))
        )
    elif gdim == 3:
        u_top.interpolate(
            lambda x: (
                np.zeros_like(x[0]),
                Delta * np.ones_like(x[1]),
                np.zeros_like(x[2]),
            )
        )
        u_bottom.interpolate(
            lambda x: (
                np.zeros_like(x[0]),
                -Delta * np.ones_like(x[1]),
                np.zeros_like(x[2]),
            )
        )

    u_top.x.scatter_forward()
    u_bottom.x.scatter_forward()

    # Solve the linear elastic problem
    solver_u.solve()

    # Calculate elastic energies
    elastic_energy_int = assemble_scalar_reduce(elastic_energy)
    elastic_energy_vol_int = assemble_scalar_reduce(elastic_energy_vol)
    elastic_energy_dev_int = assemble_scalar_reduce(elastic_energy_dev)

    # Calculate equivalent measures
    stress = sigma(u)
    strain = eps(u)
    press = pressure(u)

    # Project equivalent measures onto scalar function space

    stress_expr = dolfinx.fem.Expression(stress, V_scalar.element.interpolation_points)
    strain_expr = dolfinx.fem.Expression(strain, V_scalar.element.interpolation_points)
    pressure_expr = dolfinx.fem.Expression(press, V_scalar.element.interpolation_points)
    tau_expr = dolfinx.fem.Expression(tau(u), V_scalar.element.interpolation_points)

    #    stress_func.interpolate(stress_expr)
    #    strain_func.interpolate(strain_expr)
    pressure_func.interpolate(pressure_expr)
    tau_func.interpolate(tau_expr)

    # Find maximum values over the domain
    pressure_max = comm.allreduce(np.max(np.abs(pressure_func.x.array.real)))

    # Calculate force on top surface
    sigma_sol = sigma(u)
    force = assemble_scalar_reduce(
        ufl.dot(sigma_sol * normal, normal) * ds(interfaces_keys["top"])
    )
    force /= top_surface

    # lines for saving fields
    tol = 0.0001  # Avoid hitting the outside of the domain
    npoints = 100
    x_points = np.linspace(-L / 2 + tol, L / 2 - tol, npoints)
    radial_line = np.zeros((3, npoints))
    radial_line[0] = x_points
    radial_line[1] = 0.0
    thickness_line = np.zeros((3, npoints))
    y_points = np.linspace(-H / 2 + tol, H / 2 - tol, npoints)
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
    p_val_x = evaluate_function(pressure_func, radial_pts)
    p_val_y = evaluate_function(pressure_func, thickness_pts)

    # Store results
    history_data["load"].append(load_max)
    history_data["elastic_energy"].append(elastic_energy_int)
    history_data["elastic_energy_vol"].append(elastic_energy_vol_int)
    history_data["elastic_energy_dev"].append(elastic_energy_dev_int)
    history_data["pressure_max"].append(pressure_max)
    history_data["tau_x"].append(tau_x.tolist())
    history_data["tau_y"].append(tau_y.tolist())
    history_data["p_x"].append(p_val_x.tolist())
    history_data["p_y"].append(p_val_y.tolist())
    history_data["F"].append(force)

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

        with open(f"{prefix}_data.json", "w") as f:
            json.dump(history_data, f)

        ColorPrint.print_info(f"Results saved to {prefix}_data.json")
        ColorPrint.print_info("Linear elastic computation completed successfully")


if __name__ == "__main__":
    main()
