"""Phase-field fracture model for poker chip geometry.

Solves coupled elasticity-damage problems in 2D and 3D using alternating minimization.
Supports both elastic and damage models with optional material nonlinearity.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import dolfinx
from mpi4py import MPI
from matplotlib import pyplot as plt
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
    AltMinFractureSolver as FractureSolver,
    SNESSolver,
    TAOSolver,
    plot_energies2,
    plot_AMit_load,
    ColorPrint,
    assemble_scalar_reduce,
    evaluate_function,
)
from mesh import mesh_bar, mesh_chip, box_mesh, mesh_chip_eight, mesh_bar_quarter

from reference import formulas_paper as formulas
# from reference import gent_lindley_data as gl_data
# from reference.old.GL import GentLindleyData

# Configure plotting
plt.rcParams.update({"text.usetex": True})

comm = MPI.COMM_WORLD


# Determine config path relative to script location
config_path = str(Path(__file__).parent.parent / "config")


def get_run_slug(model, geometry):
    elastic = int(model.get("elastic", 0))
    gdim = int(geometry.get("geometric_dimension", 3))
    H = float(geometry.get("H", 0.2))
    L = float(geometry.get("L", 1.0))
    ell = float(model.get("ell", 0.01))

    if elastic == 0:
        e_c = float(model.get("e_c", 0.2))
        w_1 = float(model.get("w_1", 1.0))
        gamma_mu = float(model.get("gamma_mu", 2.0))
        return f"{gdim:d}d-ec{e_c:3.2f}_ell{ell:3.3f}_w{w_1:2.3f}_gmu{gamma_mu:3.2f}_H{H:3.2f}_L{L}"
    else:
        kappa = float(model.get("kappa", 500.0))
        mu = float(model.get("mu", 0.59))
        return f"{gdim:d}d-elastic_H{H:3.2f}_L{L}_ell{ell:3.2f}_kappa_{kappa:3.1f}_mu_{mu:3.2f}"


if not OmegaConf.has_resolver("get_run_slug"):
    OmegaConf.register_new_resolver("get_run_slug", get_run_slug)


@hydra.main(version_base=None, config_path=config_path, config_name="config")
def main(cfg: DictConfig):
    parameters = cfg

    # Material parameters
    mu = float(parameters.model.get("mu", 0.59))
    k = float(parameters.model.get("kappa", 500.0))
    ell = float(parameters.model.get("ell", 0.01))
    e_c = float(parameters.model.get("e_c", 0.2))
    p_cav = float(parameters.model.get("p_cav", 1.475))
    w_1 = float(parameters.model.get("w_1", 1.0))
    gamma_mu = float(parameters.model.get("gamma_mu", 2.0))

    # Derived material parameters
    tau_c = float(np.sqrt(2 * w_1 * mu / gamma_mu))
    G_c = float(w_1 * (np.pi * ell))
    p_c = float(p_cav / np.sqrt(1 - e_c))
    e_c_dim = e_c * w_1 / p_cav
    e_c_dim_k = e_c * w_1 / p_cav + p_cav / k

    ColorPrint.print_info(
        f"e_c = {e_c:3.3f}, e_c_dim = {e_c_dim:3.3f}, e_c_dim_k = {e_c_dim_k:3.3f}"
    )
    ColorPrint.print_info(f"p_c = {p_c:3.3f}, p_cav = {p_cav:3.3f}")
    gamma_k = float(2 * w_1 * k / p_c**2)

    nonlinear_elastic = parameters.model.get("nonlinear_elastic", 1)
    elastic = parameters.model.get("elastic", 0)
    gdim = parameters.geometry.get("geometric_dimension", 3)
    model_dimension = gdim
    L = float(parameters.geometry.get("L", 1.0))
    H = float(parameters.geometry.get("H", 0.1))
    h_div = parameters.geometry.get("h_div", 3.0)
    degree_q = parameters.fem.get("degree_q", 2)
    degree_u = parameters.fem.get("degree_u", 0)
    lc = ell / h_div

    # Derived quantities
    lmbda = k - 2 / model_dimension * mu
    nu = (1 - 2 * mu / (model_dimension * k)) / (
        model_dimension - 1 + 2 * mu / (model_dimension * k)
    )
    E = 2 * mu * (1 + nu)

    # Additional parametres
    output_name = parameters.get("output_name", "poker")
    load_max = float(parameters.get("load_max", 2.0))
    n_steps = parameters.get("n_steps", 100)
    unload = parameters.get("unload", 0)
    sliding = parameters.get("sliding", 1)
    full_output = parameters.get("full_output", 0)
    outdir_arg = parameters.get("outdir", None)
    sym = parameters.get("sym", True)  # Symmetry boundary conditions

    if elastic == 0:
        ColorPrint.print_info(f"gamma_mu = {gamma_mu:3.2f}, gamma_k = {gamma_k:3.2f}")

    # define the load as average strain
    # loads = e_c_dim_k * np.linspace(0.0, load_max, n_steps)
    loads = np.linspace(0.0, load_max, n_steps)
    if unload == 1:
        loads = np.concatenate((loads, loads[::-1][1:]))

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
            gmsh_model, tdim, tag_names = box_mesh(L * 2, H * 2, L * 2, lc, gdim)
        else:
            gmsh_model, tdim, tag_names = box_mesh(2 * L, 2 * H, 2 * L, lc, gdim)
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

    if outdir_arg is not None:
        # If outdir is explicitly provided, use it directly
        prefix = os.path.join(outdir_arg, output_name)
    else:
        # Hydra mode: CWD is already unique.
        prefix = output_name

    ColorPrint.print_info(f"Output prefix: {prefix}")

    with XDMFFile(comm, f"{prefix}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_mesh(mesh)
        file.write_meshtags(cell_tags, mesh.geometry)
        file.write_meshtags(facet_tags, mesh.geometry)

    # Save parameters for reproducibility
    if comm.rank == 0:
        # Save the complete configuration (includes all command-line overrides)
        OmegaConf.save(parameters, f"{prefix}-parameters.yml")
        ColorPrint.print_info(f"Parameters saved to: {prefix}-parameters.yml")
        ColorPrint.print_info(
            f"To reproduce this run: python {__file__} --config {prefix}-parameters.yml"
        )

    # Define integration measures
    dx_quad = ufl.Measure(
        "dx",
        domain=mesh,
        metadata={"quadrature_degree": 4, "quadrature_scheme": "default"},
    )
    dx = ufl.Measure(
        "dx",
        domain=mesh,
        metadata={
            "quadrature_degree": 4,
            "quadrature_scheme": "default",
        },
    )
    ds = ufl.Measure(
        "ds",
        subdomain_data=facet_tags,
        domain=mesh,
    )

    # Define function spaces
    if degree_u == 0:
        V_u = dolfinx.fem.functionspace(mesh, ("CR", 1, (mesh.geometry.dim,)))
    else:
        V_u = dolfinx.fem.functionspace(
            mesh, ("Lagrange", degree_u, (mesh.geometry.dim,))
        )

    def CR_stabilisation(u, alpha):
        h = ufl.CellDiameter(mesh)
        h_avg = (h("+") + h("-")) / 2.0
        stabilizing_term = (
            (1 - alpha) ** 2 * (1 / h_avg * ufl.dot(ufl.jump(u), ufl.jump(u))) * ufl.dS
        )
        return stabilizing_term

    V_alpha = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    DG = dolfinx.fem.functionspace(mesh, ("DG", 0))
    DG_vec = dolfinx.fem.functionspace(mesh, ("DG", 0, (mesh.geometry.dim,)))
    V_vec_cg1 = dolfinx.fem.functionspace(mesh, ("CG", 1, (mesh.geometry.dim,)))
    V_scalar_cg1 = dolfinx.fem.functionspace(mesh, ("CG", 1))

    alpha = dolfinx.fem.Function(V_alpha, name="Damage")
    alpha_old = dolfinx.fem.Function(V_alpha)
    zero_alpha = dolfinx.fem.Function(V_alpha)

    eps_nl_ = dolfinx.fem.Function(DG, name="eps_nl")
    trace_eps_out = dolfinx.fem.Function(DG, name="trace_eps")
    tau_out = dolfinx.fem.Function(DG, name="tau")
    Y_dev_out = dolfinx.fem.Function(DG, name="Y_dev_out")
    Y_vol_out = dolfinx.fem.Function(DG, name="Y_vol_out")
    Y_tot_out = dolfinx.fem.Function(DG, name="Y_tot_out")

    # functions for output
    u_out = dolfinx.fem.Function(V_vec_cg1, name="displacement")
    eps_nl_out = dolfinx.fem.Function(DG, name="eps_nl")
    p_out = dolfinx.fem.Function(DG, name="p")

    u = dolfinx.fem.Function(V_u, name="total_displacement")
    u_old = dolfinx.fem.Function(V_u, name="old_displacement")
    u_top = dolfinx.fem.Function(V_u, name="boundary_displacement_top")
    u_bottom = dolfinx.fem.Function(V_u, name="boundary_displacement_bottom")

    state = {"u": u, "alpha": alpha}

    v = ufl.TrialFunction(V_u)
    u_ = ufl.TestFunction(V_u)

    # %% Define bounds for damage field
    alpha_lb = dolfinx.fem.Function(V_alpha, name="Lower_bound")
    alpha_ub = dolfinx.fem.Function(V_alpha, name="Upper_bound")

    alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
    alpha_lb.x.scatter_forward()
    if parameters.model.elastic == 1:
        alpha_ub.interpolate(lambda x: np.zeros_like(x[0]))
    else:
        alpha_ub.interpolate(lambda x: np.ones_like(x[0]))
    alpha_ub.x.scatter_forward()

    # %% Define boundary conditions
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

    if gdim == 2:
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
    # apply boudary conditions to box mesh
    if gdim == 3 and sliding == 1:
        # Get facet indices for each surface
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

    dofs_alpha_top = dolfinx.fem.locate_dofs_topological(
        V_alpha, tdim - 1, np.array(top_facets)
    )
    dofs_alpha_bottom = dolfinx.fem.locate_dofs_topological(
        V_alpha, tdim - 1, np.array(bottom_facets)
    )

    if gdim == 2 and sym:
        # Symmetry boundary conditions for quarter domain
        dof_u_left = dolfinx.fem.locate_dofs_topological(
            V_u.sub(0), tdim - 1, facet_tags.find(tag_names["facets"]["left"])
        )
        dof_u_bottom_normal = dolfinx.fem.locate_dofs_topological(
            V_u.sub(1), tdim - 1, facet_tags.find(tag_names["facets"]["bottom"])
        )
        bcs_u = [
            dolfinx.fem.dirichletbc(
                0.0, dof_u_bottom_normal, V_u.sub(1)
            ),  # bottom: u_y = 0 (blocked)
            dolfinx.fem.dirichletbc(u_top, dofs_u_top),  # top: prescribed displacement
            dolfinx.fem.dirichletbc(
                0.0, dof_u_left, V_u.sub(0)
            ),  # left: u_x = 0 (symmetry)
        ]
    elif gdim == 2 and sliding == 1:
        bcs_u = [
            dolfinx.fem.dirichletbc(u_bottom, dofs_u_bottom),
            dolfinx.fem.dirichletbc(u_top, dofs_u_top),
            dolfinx.fem.dirichletbc(0.0, dofs_u_left, V_u.sub(0)),
            dolfinx.fem.dirichletbc(0.0, dofs_u_right, V_u.sub(0)),
        ]
    elif gdim == 3 and sliding == 1:
        bcs_u = [
            dolfinx.fem.dirichletbc(u_bottom, dofs_u_bottom),
            dolfinx.fem.dirichletbc(u_top, dofs_u_top),
            dolfinx.fem.dirichletbc(0.0, dof_u_left, V_u.sub(0)),
            dolfinx.fem.dirichletbc(0.0, dof_u_right, V_u.sub(0)),
            dolfinx.fem.dirichletbc(0.0, dof_u_front, V_u.sub(2)),
            dolfinx.fem.dirichletbc(0.0, dof_u_back, V_u.sub(2)),
        ]
    elif gdim == 3 and sym:
        # Symmetry boundary conditions for quarter domain
        dof_u_left = dolfinx.fem.locate_dofs_topological(
            V_u.sub(0), tdim - 1, facet_tags.find(tag_names["facets"]["left"])
        )
        dof_u_back = dolfinx.fem.locate_dofs_topological(
            V_u.sub(2), tdim - 1, facet_tags.find(tag_names["facets"]["back"])
        )
        dof_u_bottom_normal = dolfinx.fem.locate_dofs_topological(
            V_u.sub(1), tdim - 1, facet_tags.find(tag_names["facets"]["bottom"])
        )
        bcs_u = [
            dolfinx.fem.dirichletbc(
                0.0, dof_u_bottom_normal, V_u.sub(1)
            ),  # bottom: u_y = 0 (blocked)
            dolfinx.fem.dirichletbc(u_top, dofs_u_top),  # top: prescribed displacement
            dolfinx.fem.dirichletbc(
                0.0, dof_u_left, V_u.sub(0)
            ),  # left: u_x = 0 (symmetry)
            dolfinx.fem.dirichletbc(
                0.0, dof_u_back, V_u.sub(2)
            ),  # back: u_z = 0 (symmetry)
        ]
    else:
        bcs_u = [
            dolfinx.fem.dirichletbc(u_bottom, dofs_u_bottom),
            dolfinx.fem.dirichletbc(u_top, dofs_u_top),
        ]

    bcs_alpha = [
        dolfinx.fem.dirichletbc(zero_alpha, dofs_alpha_bottom),
        dolfinx.fem.dirichletbc(zero_alpha, dofs_alpha_top),
    ]

    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}

    dolfinx.fem.petsc.set_bc(alpha_ub.x.petsc_vec, bcs_alpha)
    alpha_ub.x.scatter_forward()

    # %%% Define energies and variational forms
    Identity = ufl.Identity(gdim)
    kres_a = dolfinx.fem.Constant(mesh, 1.0e-6)

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def wf(alpha):
        return 1 - (1 - alpha) ** 2

    def sigma_p(alpha, k_res=kres_a):
        """Modulation of the linear elastic limit"""
        return (1 - alpha) ** 2 + k_res

    # def sigma_p(alpha, k_res=kres_a, gamma=gamma_p):
    #    """Modulation of the linear elastic limit"""
    #    return ((1 - wf(alpha)) / (1 + (gamma - 1) * wf(alpha))) + k_res

    def kappa(alpha, k0=k, k_res=kres_a):
        """Modulation of the linear elastic limit"""
        k = dolfinx.fem.Constant(mesh, k0)
        return (k * (1 - wf(alpha)) / (1 + (gamma_k - 1) * wf(alpha))) + k_res

    def muf(alpha, mu_0=mu, k_res=kres_a):
        """Modulation of the linear elastic limit"""
        mu = dolfinx.fem.Constant(mesh, mu_0)
        return (mu * (1 - wf(alpha)) / (1 + (gamma_mu - 1) * wf(alpha))) + k_res

    def eps_nl(eps, alpha):
        criterion = kappa(alpha) * ufl.tr(eps) - p_cav * sigma_p(alpha)
        zero = dolfinx.fem.Constant(mesh, 0.0)
        if parameters.model.nonlinear_elastic:
            eps_nl = ufl.conditional(
                ufl.ge(criterion, 0.0), criterion / (kappa(alpha)), zero * criterion
            )
        else:
            eps_nl = zero
        return eps_nl

    def elastic_deviatoric_energy_density(eps, alpha):
        """
        Consider a nonlinear elastic model for the deviatoric part
        """
        eps_dev = ufl.dev(eps)
        quadratic_part = muf(alpha) * ufl.inner(eps_dev, eps_dev)
        return quadratic_part

    def elastic_volumetric_energy_density(eps, eps_nl, alpha):
        """
        Consider a nonlinear elastic model for the isotropic part
        """
        p_cav_ = dolfinx.fem.Constant(mesh, p_cav) * sigma_p(alpha)
        linear_part = p_cav_ * eps_nl
        quadratic_part = 0.5 * kappa(alpha) * (ufl.tr(eps) - eps_nl) ** 2
        # return ufl.conditional(ufl.ge(criterion, 0), energy_1 + energy_2, energy_2)
        return linear_part + quadratic_part

    def elastic_energy(u, eps_nl, alpha):
        return (
            elastic_deviatoric_energy_density(eps(u), alpha) * dx
            + elastic_volumetric_energy_density(eps(u), eps_nl, alpha) * dx_quad
        )

    def damage_dissipation_density(alpha):
        grad_alpha = ufl.grad(alpha)
        w_1_ = dolfinx.fem.Constant(mesh, w_1)
        # w_1_ = dolfinx.fem.Function(V_alpha)
        # w_1_.interpolate(lambda x:  w_1 * (1-0.1* np.maximum(1-((x[0]-L/4) / (L/2))**2,0 * x[0]) ))
        ell_ = dolfinx.fem.Constant(mesh, ell)
        return w_1_ * (wf(alpha) + ell_**2 * ufl.dot(grad_alpha, grad_alpha))

    def dissipated_energy(alpha):
        """Dissipated energy due to damage."""
        return damage_dissipation_density(alpha) * dx

    def total_energy(u, eps_nl, alpha):
        """Total energy: elastic + dissipated."""
        return elastic_energy(u, eps_nl, alpha) + dissipated_energy(alpha)

    def sigma(eps, eps_nl, alpha):
        """Stress tensor."""
        return kappa(alpha) * (ufl.tr(eps) - eps_nl) * Identity + 2 * muf(
            alpha
        ) * ufl.dev(eps)

    elastic_energy_ = elastic_energy(u, eps_nl(eps(u), alpha), alpha)
    elastic_volumetric_energy_ = (
        elastic_volumetric_energy_density(eps(u), eps_nl(eps(u), alpha), alpha)
        * dx_quad
    )
    elastic_deviatoric_energy_ = elastic_deviatoric_energy_density(eps(u), alpha) * dx
    total_energy_ = total_energy(u, eps_nl(eps(u), alpha), alpha)
    dissipated_energy_ = dissipated_energy(alpha)
    alternate_minimization = FractureSolver(
        total_energy_,
        state,
        bcs,
        parameters.get("solvers"),
        bounds=(alpha_lb, alpha_ub),
    )
    residual_alpha = ufl.derivative(
        total_energy(u, eps_nl(eps(u), alpha), alpha),
        alpha,
        ufl.TestFunction(V_alpha),
    )
    v = ufl.TestFunction(V_u)
    residual_u = ufl.derivative(total_energy(u, eps_nl(eps(u), alpha), alpha), u, v)
    stabilizing_term_der = ufl.derivative(CR_stabilisation(u, alpha), u, v)

    # Compute the driving force for post-processing
    alpha_var = ufl.variable(alpha)
    Y_dev = -ufl.diff(elastic_deviatoric_energy_density(eps(u), alpha_var), alpha_var)
    Y_vol = -ufl.diff(
        elastic_volumetric_energy_density(eps(u), eps_nl(eps(u), alpha_var), alpha_var),
        alpha_var,
    )
    Y_tot = Y_dev + Y_vol
    Y_dev_expr = dolfinx.fem.Expression(Y_dev, DG.element.interpolation_points)
    Y_vol_expr = dolfinx.fem.Expression(Y_vol, DG.element.interpolation_points)
    Y_tot_expr = dolfinx.fem.Expression(Y_tot, DG.element.interpolation_points)

    # ADD CR stabilization
    if degree_u == 0:
        residual_u += stabilizing_term_der
    # residual_u = sigma(eps(u), eps_nl(eps(u),alpha), alpha) * eps(v) * dx
    J_u = ufl.derivative(residual_u, u, ufl.TrialFunction(V_u))

    # %% define solvers
    solver_u = SNESSolver(
        residual_u,
        u,
        bcs_u,
        J_form=J_u,
        petsc_options=parameters.solvers.elasticity.snes,
        prefix=parameters.solvers.elasticity.prefix,
    )

    if parameters.solvers.damage.type == "SNES":
        solver_alpha = SNESSolver(
            residual_alpha,
            alpha,
            bcs_alpha,
            bounds=(alpha_lb, alpha_ub),
            petsc_options=parameters.solvers.damage.snes,
            prefix=parameters.solvers.damage.prefix,
        )
    if parameters.solvers.damage.type == "TAO":
        solver_alpha = TAOSolver(
            total_energy_,
            alpha,
            bcs_alpha,
            bounds=(alpha_lb, alpha_ub),
            # J_form=J_alpha,
            petsc_options=parameters.solvers.damage.tao,
            prefix=parameters.solvers.damage.prefix,
        )

    # %%
    history_data = {
        "load": [],
        "elastic_energy": [],
        "dissipated_energy": [],
        "solver_data": [],
        "F": [],
        "elastic_volumetric_energy": [],
        "elastic_deviatoric_energy": [],
        "p_x": [],
        "p_y": [],
        "tau": [],
        "alpha_x": [],
        "alpha_y": [],
        "eps_nl_x": [],
        "eps_nl_y": [],
        "alt_min_it": [],
        "error_residual_u": [],
        "average_stress": [],
        "average_strain": [],
        "x_points": [],
        "y_points": [],
        "parameters": OmegaConf.to_container(parameters, resolve=True),
        "run_slug": get_run_slug(parameters.model, parameters.geometry),
    }

    eps_nl_expr = dolfinx.fem.Expression(
        eps_nl(eps(u), alpha), DG.element.interpolation_points
    )

    # lines for saving fields
    tol = 0.0001  # Avoid hitting the outside of the domain
    npoints = 100
    x_points = np.linspace(0, L, npoints)
    radial_line = np.zeros((3, npoints))
    radial_line[0] = x_points
    radial_line[1] = 0.0
    thickness_line = np.zeros((3, npoints))
    y_points = np.linspace(0, H, npoints)
    history_data["x_points"] = x_points.tolist()
    history_data["y_points"] = y_points.tolist()
    thickness_line[1] = y_points
    t_post = dolfinx.common.Timer("Post_process")
    t_u = dolfinx.common.Timer("Solver_u")
    t_alpha = dolfinx.common.Timer("Solver_alpha")
    normal = ufl.FacetNormal(mesh)

    top_surface = assemble_scalar_reduce(
        ufl.dot(normal, normal) * ds(interfaces_keys["top"])
    )
    # % initialization of the history variables
    total_energy_int_old = 0
    alpha.x.petsc_vec.set(0.0)
    # alpha.interpolate(lambda x:  .05* (1-(x[0] / (L/2))**2) )
    # alpha.interpolate(lambda x:  .05* np.maximum(1-((x[0]-L/4) / (L/2))**2,0 * x[0]) )

    # % Main load loop
    for i_t, t in enumerate(loads):
        # update boundary conditions
        Delta = t * H
        if gdim == 2:
            u_top.interpolate(
                lambda x: (
                    np.zeros_like(x[0]),
                    Delta * (np.ones_like(x[1]) + 0.0 * (1 - x[0] / (L / 2) ** 2)),
                )
            )
            u_bottom.interpolate(
                lambda x: (np.zeros_like(x[0]), -Delta * np.ones_like(x[1]))
            )
        elif gdim == 3:
            u_top.interpolate(
                lambda x: (
                    np.zeros_like(x[0]),
                    Delta * (np.ones_like(x[1])),
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

        # update the lower bound
        alpha.x.petsc_vec.copy(result=alpha_lb.x.petsc_vec)
        alpha_lb.x.scatter_forward()
        alpha_diff = dolfinx.fem.Function(alpha.function_space)
        residual = dolfinx.fem.Function(u.function_space)

        ColorPrint.print_bold(f"-- Solving for t = {t:3.4f} --")

        # % Alternating minimization loop
        alt_min_it = 0
        for iteration in range(parameters.solvers.damage_elasticity.max_it):
            alt_min_it += 1
            # solve non-linear elastoplastic problem
            t_u.start()
            solver_u.solve()
            eps_nl_.interpolate(eps_nl_expr)
            t_u.stop()

            t_alpha.start()
            (solver_alpha_it, solver_alpha_reason) = solver_alpha.solve()
            eps_nl_.interpolate(eps_nl_expr)

            alpha.x.petsc_vec.copy(alpha_diff.x.petsc_vec)
            alpha_diff.x.petsc_vec.axpy(-1, alpha_old.x.petsc_vec)
            alpha_diff.x.scatter_forward()
            t_alpha.stop()

            error_alpha_max = alpha_diff.x.petsc_vec.max()[1]

            try:
                solver_u.solver.computeFunction(u.x.petsc_vec, residual.x.petsc_vec)
            except:
                pass
            u.x.petsc_vec, residual.x.scatter_forward()
            error_residual_u = residual.x.petsc_vec.norm()
            total_energy_int = assemble_scalar_reduce(total_energy_)
            elastic_energy_int = assemble_scalar_reduce(elastic_energy_)
            elastic_deviatoric_energy_int = assemble_scalar_reduce(
                elastic_deviatoric_energy_
            )
            elastic_volumetric_energy_int = assemble_scalar_reduce(
                elastic_volumetric_energy_
            )
            dissipated_energy_int = assemble_scalar_reduce(
                dolfinx.fem.form(dissipated_energy_)
            )
            error_energy_a = abs(total_energy_int - total_energy_int_old)

            if total_energy_int_old > 0:
                error_energy_r = abs(total_energy_int / total_energy_int_old - 1)
            else:
                error_energy_r = 1.0

            total_energy_int_old = total_energy_int
            alpha.x.petsc_vec.copy(alpha_old.x.petsc_vec)
            alpha_old.x.scatter_forward()
            ColorPrint.print_info(
                f"AM - Iteration: {iteration:3d}, "
                + f"alpha_max: {alpha.x.petsc_vec.max()[1]:3.3e}, "
                + f"eps_nl_min: {eps_nl_.x.petsc_vec.min()[1]:3.3e}, "
                + f"eps_nl_max: {eps_nl_.x.petsc_vec.max()[1]:3.3e}, "
                + f"Error_alpha_max: {error_alpha_max:3.3e}, "
                + f"Error_energy: {error_energy_r:3.2e}, "
                + f"Error_residual_u: {error_residual_u:3.3e}"
            )

            # Check convergence based on specified criterion
            solver_params = parameters.solvers.damage_elasticity
            criterion = solver_params.criterion

            converged = False
            if criterion == "residual_u":
                converged = error_residual_u <= solver_params.residual_u_tol
            elif criterion == "alpha_max":
                converged = error_alpha_max <= solver_params.alpha_tol
            elif criterion == "energy":
                converged = (
                    error_energy_r <= solver_params.energy_rtol
                    or error_energy_a <= solver_params.energy_atol
                )

            if converged:
                break
        else:
            if not parameters.solvers.damage_elasticity.error_on_nonconvergence:
                ColorPrint.print_warn(
                    (
                        f"Could not converge after {iteration:3d} iterations,"
                        + f"error_u {error_residual_u:3.4e},"
                        + f"error_alpha_max {error_alpha_max:3.4e},"
                        + f"error_energy_r {error_energy_r:3.4e}"
                    )
                )

            else:
                raise RuntimeError(
                    f"Could not converge after {iteration:3d} iterations, error {error_residual_u:3.4e}"
                )

        t_post.start()
        alpha.x.petsc_vec.copy(result=alpha_lb.x.petsc_vec)
        alpha_lb.x.scatter_forward()
        sigma_sol = sigma(eps(u), eps_nl(eps(u), alpha), alpha)
        p_expr = dolfinx.fem.Expression(
            ufl.tr(sigma_sol) / gdim,
            p_out.function_space.element.interpolation_points,
        )
        tau_expr = dolfinx.fem.Expression(
            np.sqrt(2)
            * ufl.sqrt(
                ufl.inner(
                    ufl.dev(sigma_sol),
                    ufl.dev(sigma_sol),
                )
            ),
            DG.element.interpolation_points,
        )

        p_out.interpolate(p_expr)
        p_out.x.scatter_forward()
        tau_out.interpolate(tau_expr)

        Y_dev_out.interpolate(Y_dev_expr)
        Y_vol_out.interpolate(Y_vol_expr)
        Y_tot_out.interpolate(Y_tot_expr)

        force = assemble_scalar_reduce(
            ufl.dot(sigma_sol * normal, normal) * ds(interfaces_keys["top"])
        )

        history_data["load"].append(t)
        history_data["alt_min_it"].append(alt_min_it)
        history_data["dissipated_energy"].append(dissipated_energy_int)
        history_data["elastic_energy"].append(elastic_energy_int)
        history_data["elastic_volumetric_energy"].append(elastic_volumetric_energy_int)
        history_data["elastic_deviatoric_energy"].append(elastic_deviatoric_energy_int)
        history_data["F"].append(force)

        history_data["average_strain"].append(Delta / H)
        history_data["average_stress"].append(force / top_surface)
        # Only pass the required spatial dimensions for the mesh (2D or 3D)
        radial_pts = np.ascontiguousarray(
            radial_line[: mesh.geometry.dim].T, dtype=np.float64
        )
        thickness_pts = np.ascontiguousarray(
            thickness_line[: mesh.geometry.dim].T, dtype=np.float64
        )
        tau_val = evaluate_function(tau_out, radial_pts)
        p_val_x = evaluate_function(p_out, radial_pts)
        p_val_y = evaluate_function(p_out, thickness_pts)
        alpha_x = evaluate_function(alpha, radial_pts)
        alpha_y = evaluate_function(alpha, thickness_pts)
        eps_nl_x = evaluate_function(eps_nl_, radial_pts)
        eps_nl_y = evaluate_function(eps_nl_, thickness_pts)
        if comm.rank == 0:
            history_data["p_x"].append(p_val_x.tolist())
            history_data["p_y"].append(p_val_y.tolist())
            history_data["tau"].append(tau_val.tolist())
            history_data["alpha_x"].append(alpha_x.tolist())
            history_data["alpha_y"].append(alpha_y.tolist())
            history_data["eps_nl_x"].append(eps_nl_x.tolist())
            history_data["eps_nl_y"].append(eps_nl_y.tolist())

        with XDMFFile(
            comm, f"{prefix}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5
        ) as file:
            u_out.interpolate(u)
            eps_nl_out.interpolate(eps_nl_)
            file.write_function(u_out, i_t)
            file.write_function(alpha, i_t)
            file.write_function(p_out, i_t)
            file.write_function(tau_out, i_t)
            file.write_function(eps_nl_out, i_t)
            if full_output == 1:
                file.write_function(Y_vol_out, i_t)
                file.write_function(Y_dev_out, i_t)
                file.write_function(Y_tot_out, i_t)

        if comm.rank == 0:
            a_file = open(f"{prefix}_data.json", "w")
            json.dump(history_data, a_file)
            a_file.close()

            ColorPrint.print_info(f"Results saved to {prefix}_data.json")


if __name__ == "__main__":
    main()
