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
from mesh import mesh_bar, mesh_chip, box_mesh

comm = MPI.COMM_WORLD


def mesh_bar_refined(L, H, lc_center, lc_edge, edge_width, tdim, order=1):
    """Create a refined bar mesh with smaller elements near left/right edges."""
    from mpi4py import MPI

    facet_tag_names = {"top": 14, "bottom": 12, "left": 15, "right": 13}
    tag_names = {"facets": facet_tag_names}

    if MPI.COMM_WORLD.rank == 0:
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 5)

        model = gmsh.model()
        model.add("RefinedRectangle")
        model.setCurrent("RefinedRectangle")

        # Create points with different mesh sizes
        # Corners
        p0 = model.geo.addPoint(-L / 2, -H / 2, 0, lc_edge, tag=0)
        p1 = model.geo.addPoint(L / 2, -H / 2, 0, lc_edge, tag=1)
        p2 = model.geo.addPoint(L / 2, H / 2, 0.0, lc_edge, tag=2)
        p3 = model.geo.addPoint(-L / 2, H / 2, 0, lc_edge, tag=3)

        # Transition points (where refinement transitions to coarser)
        transition_x = edge_width
        p4 = model.geo.addPoint(-L / 2 + transition_x, -H / 2, 0, lc_center, tag=4)
        p5 = model.geo.addPoint(L / 2 - transition_x, -H / 2, 0, lc_center, tag=5)
        p6 = model.geo.addPoint(L / 2 - transition_x, H / 2, 0, lc_center, tag=6)
        p7 = model.geo.addPoint(-L / 2 + transition_x, H / 2, 0, lc_center, tag=7)

        # Create lines
        bottom_left = model.geo.addLine(p0, p4, tag=12)
        bottom_center = model.geo.addLine(p4, p5, tag=121)
        bottom_right = model.geo.addLine(p5, p1, tag=122)

        right = model.geo.addLine(p1, p2, tag=13)

        top_right = model.geo.addLine(p2, p6, tag=14)
        top_center = model.geo.addLine(p6, p7, tag=141)
        top_left = model.geo.addLine(p7, p3, tag=142)

        left = model.geo.addLine(p3, p0, tag=15)

        # Interior vertical lines
        left_vertical = model.geo.addLine(p4, p7, tag=16)
        right_vertical = model.geo.addLine(p5, p6, tag=17)

        # Create curve loops and surfaces
        # Left refined region
        cloop_left = model.geo.addCurveLoop(
            [bottom_left, left_vertical, -top_left, left]
        )
        surf_left = model.geo.addPlaneSurface([cloop_left], tag=1)

        # Center region
        cloop_center = model.geo.addCurveLoop(
            [bottom_center, right_vertical, -top_center, -left_vertical]
        )
        surf_center = model.geo.addPlaneSurface([cloop_center], tag=2)

        # Right refined region
        cloop_right = model.geo.addCurveLoop(
            [bottom_right, right, top_right, -right_vertical]
        )
        surf_right = model.geo.addPlaneSurface([cloop_right], tag=3)

        model.geo.synchronize()

        # Physical groups
        surface_entities = [surf_left, surf_center, surf_right]
        model.addPhysicalGroup(tdim, surface_entities, tag=22)
        model.setPhysicalName(tdim, 22, "Rectangle surface")

        # Boundary physical groups (combine segments for each boundary)
        model.addPhysicalGroup(tdim - 1, [12, 121, 122], tag=12)  # bottom
        model.addPhysicalGroup(tdim - 1, [13], tag=13)  # right
        model.addPhysicalGroup(tdim - 1, [14, 141, 142], tag=14)  # top
        model.addPhysicalGroup(tdim - 1, [15], tag=15)  # left

        for k, v in facet_tag_names.items():
            model.setPhysicalName(tdim - 1, v, k)

        model.mesh.setOrder(order)
        model.mesh.generate(tdim)

    if MPI.COMM_WORLD.rank == 0:
        model_out = gmsh.model
    else:
        model_out = 0

    return model_out, tdim, tag_names


def box_mesh_refined(Lx, Ly, Lz, lc_center, lc_edge, edge_width, tdim=3, order=1):
    """Create a refined box mesh with smaller elements near left/right edges."""
    from mpi4py import MPI

    facet_tag_names = {
        "left": 101,
        "right": 102,
        "bottom": 103,
        "top": 104,
        "front": 105,
        "back": 106,
    }
    tag_names = {"facets": facet_tag_names}

    if MPI.COMM_WORLD.rank == 0:
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        model = gmsh.model()
        model.add("RefinedBox")
        model.setCurrent("RefinedBox")

        # Transition coordinates
        x_trans_left = -Lx / 2 + edge_width
        x_trans_right = Lx / 2 - edge_width

        # Create the refined box with three regions in X direction
        # Left refined region
        box_left = model.occ.addBox(
            -Lx / 2, -Ly / 2, -Lz / 2, edge_width, Ly, Lz, tag=1
        )

        # Center coarse region
        box_center = model.occ.addBox(
            x_trans_left, -Ly / 2, -Lz / 2, x_trans_right - x_trans_left, Ly, Lz, tag=2
        )

        # Right refined region
        box_right = model.occ.addBox(
            x_trans_right, -Ly / 2, -Lz / 2, edge_width, Ly, Lz, tag=3
        )

        model.occ.synchronize()

        # Set mesh sizes using distance fields
        # Create distance field from left and right faces
        dist_field = model.mesh.field.add("Distance")
        left_faces = []
        right_faces = []

        # Get all faces and identify left/right ones
        all_faces = model.getEntities(2)
        for _, face_id in all_faces:
            com = model.occ.getCenterOfMass(2, face_id)
            if abs(com[0] + Lx / 2) < 1e-6:  # Left face
                left_faces.append(face_id)
            elif abs(com[0] - Lx / 2) < 1e-6:  # Right face
                right_faces.append(face_id)

        model.mesh.field.setNumbers(dist_field, "FacesList", left_faces + right_faces)

        # Threshold field for mesh size
        threshold_field = model.mesh.field.add("Threshold")
        model.mesh.field.setNumber(threshold_field, "IField", dist_field)
        model.mesh.field.setNumber(threshold_field, "LcMin", lc_edge)
        model.mesh.field.setNumber(threshold_field, "LcMax", lc_center)
        model.mesh.field.setNumber(threshold_field, "DistMin", 0)
        model.mesh.field.setNumber(threshold_field, "DistMax", edge_width)

        model.mesh.field.setAsBackgroundMesh(threshold_field)

        # Physical groups
        all_volumes = model.getEntities(3)
        volume_tags = [v[1] for v in all_volumes]
        model.addPhysicalGroup(3, volume_tags, tag=1)
        model.setPhysicalName(3, 1, "Box volume")

        # Physical groups for boundaries
        for k, v in facet_tag_names.items():
            faces_for_boundary = []
            for _, face_id in all_faces:
                com = model.occ.getCenterOfMass(2, face_id)
                if k == "left" and abs(com[0] + Lx / 2) < 1e-6:
                    faces_for_boundary.append(face_id)
                elif k == "right" and abs(com[0] - Lx / 2) < 1e-6:
                    faces_for_boundary.append(face_id)
                elif k == "bottom" and abs(com[1] + Ly / 2) < 1e-6:
                    faces_for_boundary.append(face_id)
                elif k == "top" and abs(com[1] - Ly / 2) < 1e-6:
                    faces_for_boundary.append(face_id)
                elif k == "back" and abs(com[2] + Lz / 2) < 1e-6:
                    faces_for_boundary.append(face_id)
                elif k == "front" and abs(com[2] - Lz / 2) < 1e-6:
                    faces_for_boundary.append(face_id)

            if faces_for_boundary:
                model.addPhysicalGroup(2, faces_for_boundary, tag=v)
                model.setPhysicalName(2, v, k)

        model.mesh.setOrder(order)
        model.mesh.generate(tdim)

    if MPI.COMM_WORLD.rank == 0:
        model_out = gmsh.model
    else:
        model_out = 0

    return model_out, tdim, tag_names


# Determine config path relative to script location
config_path = str(Path(__file__).parent.parent / "config")


@hydra.main(version_base=None, config_path=config_path, config_name="config_linear")
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

    # Derived quantities - use plane strain (3D-like) expressions for 2D, 3D expressions for 3D
    if gdim == 2:
        # Plane strain: use 3D relationships for plane strain
        lmbda = k - 2 / 3 * mu
        nu = lmbda / (2 * (lmbda + mu))
        E = mu * (3 * lmbda + 2 * mu) / (lmbda + mu)
    elif gdim == 3:
        # 3D expressions
        lmbda = k - 2 / 3 * mu
        nu = lmbda / (2 * (lmbda + mu))
        E = mu * (3 * lmbda + 2 * mu) / (lmbda + mu)

    ColorPrint.print_info(f"Material properties: mu = {mu:.3f}, k = {k:.1f}")
    ColorPrint.print_info(f"Derived: E = {E:.3f}, nu = {nu:.3f}, lambda = {lmbda:.3f}")

    # Load parameters
    output_name = parameters.get("output_name", "poker_linear")
    load_max = parameters.get("load_max", 1.0)  # Single load for linear case
    sliding = parameters.get("sliding", 0)
    outdir_arg = parameters.get("outdir", None)

    # For linear elasticity, single step is sufficient
    loads = [load_max]

    # Create the mesh
    if gdim == 2:
        gmsh_model, tdim, tag_names = mesh_bar(L, H, lc, gdim)
    elif gdim == 3:
        gmsh_model, tdim, tag_names = box_mesh(L, H, L, lc, gdim)

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
    if element_type == "CR":
        h_mesh = ufl.CellDiameter(mesh)
        h_avg = (h_mesh("+") + h_mesh("-")) / 2.0
        # For linear elasticity, use simple jump stabilization
        stabilizing_term = (1 / h_avg * ufl.dot(ufl.jump(u), ufl.jump(u))) * ufl.dS
        stabilizing_term_der = ufl.derivative(stabilizing_term, u, v)
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
