"""Phase-field fracture model for poker chip geometry.

Solves coupled elasticity-damage problems in 2D and 3D using alternating minimization.
Supports both elastic and damage models with optional material nonlinearity.
"""

import json
import os
import sys
from pathlib import Path
import argparse

import numpy as np
import dolfinx
from mpi4py import MPI
from matplotlib import pyplot as plt
import petsc4py
import ufl
import yaml

petsc4py.init(sys.argv)

from dolfinx.io import XDMFFile
from dolfinx.io.gmsh import model_to_mesh

# Add subdirectories to path for imports
_script_dir = Path(__file__).parent.parent  # Go up to project root
sys.path.insert(0, str(_script_dir / "damage"))
sys.path.insert(0, str(_script_dir / "meshes"))
sys.path.insert(0, str(_script_dir))

# Import damage mechanics modules
from alternate_minimization import AltMinFractureSolver as FractureSolver
from scifem import evaluate_function
from plots import plot_energies2, plot_AMit_load
from utils import ColorPrint, assemble_scalar_reduce
from petsc_solvers import SNESSolver, TAOSolver

# Import mesh generation modules
from mesh_bar import mesh_bar
from mesh_chip import mesh_chip
from mesh_box import box_mesh

import formulas

# Configure plotting
plt.rcParams.update({"text.usetex": True})

comm = MPI.COMM_WORLD


# Load and parse configuration
with open(_script_dir / "parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_name",
    default="poker",
    type=str,
    dest="output_name",
    help="output_name",
)
parser.add_argument("--ell", default=0.01, type=float, dest="ell", help="ell")
parser.add_argument("--L", default=1.0, type=float, dest="L", help="L")
parser.add_argument(
    "--full_output",
    default=0,
    type=int,
    dest="full_output",
    help="saving extra fields",
)
parser.add_argument("--H", default=0.2, type=float, dest="H", help="H")
parser.add_argument(
    "--sliding",
    default=1,
    type=int,
    dest="sliding",
    help="if 1 slinding boundaries otherwise free boundaries",
)
parser.add_argument(
    "--unload",
    default=0,
    type=int,
    dest="unload",
    help="if 1 unload after reaching the max load",
)

parser.add_argument(
    "--kappa", default=500.0, type=float, dest="kappa", help="kappa"
)
parser.add_argument(
    "--p_cav",
    default=1.475,
    type=float,
    dest="p_cav",
    help="p_cav, default value to 5/2 mu = 1.475",
)
parser.add_argument("--e_c", default=0.2, type=float, dest="e_c", help="e_c")
parser.add_argument(
    "--gamma_mu", default=2.0, type=float, dest="gamma_mu", help="gamma_mu"
)
parser.add_argument(
    "--w_1",
    default=1.0,
    type=float,
    dest="w_1",
    help="w_1 can be set to 1.0 by dimensional analysis, it is a trivial rescaling of the deformations: c_0=u_0/L= sqrt(w_1/mu)",
)

parser.add_argument(
    "--mu",
    default=0.59,
    type=float,
    dest="mu_0",
    help="mu_0, 0.59 for GL, but can use 1.0, it is only a rescaling of the stress",
)

parser.add_argument(
    "--loadmax", default=2.0, type=float, dest="load_max", help="load_max"
)
parser.add_argument(
    "--nsteps", default=100, type=int, dest="n_steps", help="n_steps"
)
parser.add_argument(
    "--hdiv", default=3.0, type=float, dest="h_div", help="h_div"
)
parser.add_argument(
    "--degree_u",
    default=0,
    type=int,
    dest="degree_u",
    help="degree_u. If degree_u=0 then use CR, if degree_u>0 then use Lagrange of order degree_u",
)
parser.add_argument(
    "--degree_q", default=2, type=int, dest="degree_q", help="degree_q"
)
parser.add_argument("--elastic", default=0, type=int, dest="elastic")
parser.add_argument(
    "--nonlinear_elastic", default=1, type=int, dest="nonlinear_elastic"
)
parser.add_argument(
    "--model_dimension", default=3, type=int, dest="model_dimension"
)
parser.add_argument("--gdim", default=3, type=int, dest="gdim")
parser.add_argument("--outdir", default=None, type=str, dest="outdir")

args = parser.parse_args()

# Material parameters
mu = args.mu_0
k = args.kappa
ell = args.ell
e_c = args.e_c
p_cav = args.p_cav
w_1 = args.w_1
gamma_mu = args.gamma_mu

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


parameters["model"]["mu"] = mu
parameters["model"]["kappa"] = k
parameters["model"]["w_1"] = w_1
parameters["model"]["ell"] = ell
parameters["model"]["G_c"] = G_c
parameters["model"]["p_cav"] = p_cav
# parameters["model"]["gamma_p"] = gamma_p
parameters["model"]["e_c"] = e_c
parameters["model"]["p_c"] = p_c
parameters["model"]["tau_c"] = tau_c
parameters["model"]["gamma_k"] = gamma_k
parameters["model"]["gamma_mu"] = gamma_mu
# parameters["model"]["gamma_p"] = gamma_p


model_dimension = args.model_dimension
nonlinear_elastic = args.nonlinear_elastic
elastic = args.elastic
gdim = args.gdim
L = args.L
H = args.H
h_div = args.h_div
degree_q = args.degree_q
degree_u = args.degree_u
lc = ell / h_div

parameters["model"]["model_dimension"] = model_dimension
parameters["model"]["nonlinear_elastic"] = nonlinear_elastic
parameters["model"]["elastic"] = elastic
parameters["geometry"]["geometric_dimension"] = gdim
parameters["geometry"]["L"] = L
parameters["geometry"]["H"] = H
parameters["geometry"]["h_div"] = h_div
parameters["geometry"]["lc"] = h_div
parameters["fem"]["degree_u"] = degree_u
parameters["fem"]["degree_q"] = degree_q

# Derived quantities
lmbda = k - 2 / model_dimension * mu
parameters["model"]["lambda"] = lmbda
nu = (1 - 2 * mu / (model_dimension * k)) / (
    model_dimension - 1 + 2 * mu / (model_dimension * k)
)
parameters["model"]["nu"] = nu
E = 2 * mu * (1 + nu)
parameters["model"]["E"] = E

# Additional parametres
output_name = args.output_name

if elastic == 0:
    ColorPrint.print_info(
        f"gamma_mu = {gamma_mu:3.2f}, gamma_k = {gamma_k:3.2f}"
    )

# define the load as average strain
# loads = e_c_dim_k * np.linspace(0.0, args.load_max, args.n_steps)
loads = np.linspace(0.0, args.load_max, args.n_steps)
if args.unload == 1:
    loads = np.concatenate((loads, loads[::-1][1:]))

# Create the mesh of the specimen with given dimensions
if gdim == 2:
    gmsh_model, tdim, tag_names = mesh_bar(L, H, lc, gdim)
elif gdim == 3 and args.sliding == 0:
    gmsh_model, tdim, tag_names = mesh_chip(L / 2, H, lc, gdim)
elif gdim == 3 and args.sliding == 1:
    gmsh_model, tdim, tag_names = box_mesh(L, H, L, lc, gdim)
model_rank = 0
mesh_comm = MPI.COMM_WORLD
partitioner = dolfinx.mesh.create_cell_partitioner(
    dolfinx.mesh.GhostMode.shared_facet
)
mesh_data = model_to_mesh(
    gmsh_model,
    mesh_comm,
    model_rank,
    gdim=gdim,
    partitioner=partitioner,
)
mesh, cell_tags, facet_tags = mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags
interfaces_keys = tag_names["facets"]
# Define output directory based on problem type
if elastic == 0:
    outdir = f"{gdim:d}d-ec{e_c:3.2f}_ell{ell:3.3f}_w{w_1:2.3f}_gmu{gamma_mu:3.2f}_H{H:3.2f}_L{L}_{output_name}"
else:
    outdir = f"{gdim:d}d-elastic_H{H:3.2f}_L{L}_ell{ell:3.2f}_kappa_{k:3.1f}_mu_{mu:3.2f}_{output_name}"

prefix = os.path.join(outdir, output_name)
if args.outdir is not None:
    prefix = os.path.join(args.outdir, prefix)

ColorPrint.print_info(f"Output directory: {prefix}")


with XDMFFile(
    comm, f"{prefix}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5
) as file:
    file.write_mesh(mesh)
    file.write_meshtags(cell_tags, mesh.geometry)
    file.write_meshtags(facet_tags, mesh.geometry)

if comm.rank == 0:
    with open(f"{prefix}-parameters.yml", "w") as f:
        yaml.dump(parameters, f)

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
        (1 - alpha) ** 2
        * (1 / h_avg * ufl.dot(ufl.jump(u), ufl.jump(u)))
        * ufl.dS
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

# need upper/lower bound for the damage field
alpha_lb = dolfinx.fem.Function(V_alpha, name="Lower_bound")
alpha_ub = dolfinx.fem.Function(V_alpha, name="Upper_bound")

alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
alpha_lb.x.scatter_forward()
if parameters["model"]["elastic"] == 1:
    alpha_ub.interpolate(lambda x: np.zeros_like(x[0]))
else:
    alpha_ub.interpolate(lambda x: np.ones_like(x[0]))
alpha_ub.x.scatter_forward()


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
if gdim == 3 and args.sliding == 1:
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

if gdim == 2 and args.sliding == 1:
    bcs_u = [
        dolfinx.fem.dirichletbc(u_bottom, dofs_u_bottom),
        dolfinx.fem.dirichletbc(u_top, dofs_u_top),
        dolfinx.fem.dirichletbc(0.0, dofs_u_left, V_u.sub(0)),
        dolfinx.fem.dirichletbc(0.0, dofs_u_right, V_u.sub(0)),
    ]
elif gdim == 3 and args.sliding == 1:
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

bcs_alpha = [
    dolfinx.fem.dirichletbc(zero_alpha, dofs_alpha_bottom),
    dolfinx.fem.dirichletbc(zero_alpha, dofs_alpha_top),
]

bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}

dolfinx.fem.petsc.set_bc(alpha_ub.x.petsc_vec, bcs_alpha)
alpha_ub.x.scatter_forward()

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


def kappa(alpha, k_res=kres_a):
    """Modulation of the linear elastic limit"""
    return (k * (1 - wf(alpha)) / (1 + (gamma_k - 1) * wf(alpha))) + k_res


def muf(alpha, k_res=kres_a):
    """Modulation of the linear elastic limit"""
    return (mu * (1 - wf(alpha)) / (1 + (gamma_mu - 1) * wf(alpha))) + k_res


def eps_nl(eps, alpha):
    criterion = kappa(alpha) * ufl.tr(eps) - p_cav * sigma_p(alpha)
    zero = dolfinx.fem.Constant(mesh, 0.0)
    if parameters.get("model").get("nonlinear_elastic"):
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
elastic_deviatoric_energy_ = (
    elastic_deviatoric_energy_density(eps(u), alpha) * dx
)
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
Y_dev = -ufl.diff(
    elastic_deviatoric_energy_density(eps(u), alpha_var), alpha_var
)
Y_vol = -ufl.diff(
    elastic_volumetric_energy_density(
        eps(u), eps_nl(eps(u), alpha_var), alpha_var
    ),
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

solver_u = SNESSolver(
    residual_u,
    u,
    bcs_u,
    J_form=J_u,
    petsc_options=parameters["solvers"]["elasticity"]["snes"],
    prefix=parameters["solvers"]["elasticity"]["prefix"],
)


if parameters.get("solvers").get("damage").get("type") == "SNES":
    solver_alpha = SNESSolver(
        residual_alpha,
        alpha,
        bcs_alpha,
        bounds=(alpha_lb, alpha_ub),
        petsc_options=parameters.get("solvers").get("damage").get("snes"),
        prefix=parameters.get("solvers").get("damage").get("prefix"),
    )
if parameters.get("solvers").get("damage").get("type") == "TAO":
    solver_alpha = TAOSolver(
        total_energy_,
        alpha,
        bcs_alpha,
        bounds=(alpha_lb, alpha_ub),
        # J_form=J_alpha,
        petsc_options=parameters.get("solvers").get("damage").get("tao"),
        prefix=parameters.get("solvers").get("damage").get("prefix"),
    )

# %%
alpha.x.petsc_vec.set(0.0)
# alpha.interpolate(lambda x:  .05* (1-(x[0] / (L/2))**2) )
# alpha.interpolate(lambda x:  .05* np.maximum(1-((x[0]-L/4) / (L/2))**2,0 * x[0]) )

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
}

total_energy_int_old = 0
eps_nl_expr = dolfinx.fem.Expression(
    eps_nl(eps(u), alpha), DG.element.interpolation_points
)

# lines for saving fields
tol = 0.0001  # Avoid hitting the outside of the domain
npoints = 100
x_points = np.linspace(-L / 2 + tol, L / 2 - tol, npoints)
radial_line = np.zeros((3, npoints))
radial_line[0] = x_points
radial_line[1] = H / 2
thickness_line = np.zeros((3, npoints))
y_points = np.linspace(tol, H - tol, npoints)
thickness_line[1] = np.linspace(tol, H - tol, npoints)
t_post = dolfinx.common.Timer("Post_process")
t_u = dolfinx.common.Timer("Solver_u")
t_alpha = dolfinx.common.Timer("Solver_alpha")
normal = ufl.FacetNormal(mesh)

top_surface = assemble_scalar_reduce(
    ufl.dot(normal, normal) * ds(interfaces_keys["top"])
)
for i_t, t in enumerate(loads):
    u.x.scatter_forward()
    if gdim == 2:
        u_top.interpolate(
            lambda x: (
                np.zeros_like(x[0]),
                t
                * H
                / 2
                * (np.ones_like(x[1]) + 0.0 * (1 - x[0] / (L / 2) ** 2)),
            )
        )
        u_bottom.interpolate(
            lambda x: (np.zeros_like(x[0]), -t * H / 2 * np.ones_like(x[1]))
        )
    elif gdim == 3:
        u_top.interpolate(
            lambda x: (
                np.zeros_like(x[0]),
                t * (H / 2) * (np.ones_like(x[1])),
                np.zeros_like(x[2]),
            )
        )
        u_bottom.interpolate(
            lambda x: (
                np.zeros_like(x[0]),
                -t * (H / 2) * np.ones_like(x[1]),
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

    alt_min_it = 0
    for iteration in range(
        parameters.get("solvers").get("damage_elasticity").get("max_it")
    ):
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

        """error_alpha_H1 = norm_H1(alpha_diff)
        error_alpha_L2 = norm_L2(alpha_diff)"""
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
        solver_params = parameters["solvers"]["damage_elasticity"]
        criterion = solver_params.get("criterion")
        
        converged = False
        if criterion == "residual_u":
            converged = error_residual_u <= solver_params.get("residual_u_tol")
        elif criterion == "alpha_max":
            converged = error_alpha_max <= solver_params.get("alpha_tol")
        elif criterion == "energy":
            converged = (error_energy_r <= solver_params.get("energy_rtol") or 
                        error_energy_a <= solver_params.get("energy_atol"))
        
        if converged:
            break
    else:
        if (
            not parameters["solvers"]
            .get("damage_elasticity")
            .get("error_on_nonconvergence")
        ):
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

    force /= top_surface

    history_data["load"].append(t)
    history_data["alt_min_it"].append(alt_min_it)
    history_data["dissipated_energy"].append(dissipated_energy_int)
    history_data["elastic_energy"].append(elastic_energy_int)
    history_data["elastic_volumetric_energy"].append(
        elastic_volumetric_energy_int
    )
    history_data["elastic_deviatoric_energy"].append(
        elastic_deviatoric_energy_int
    )
    history_data["F"].append(force)
    # Only pass the required spatial dimensions for the mesh (2D or 3D)
    radial_pts = np.ascontiguousarray(radial_line[:mesh.geometry.dim].T, dtype=np.float64)
    thickness_pts = np.ascontiguousarray(thickness_line[:mesh.geometry.dim].T, dtype=np.float64)
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
        if args.full_output == 1:
            file.write_function(Y_vol_out, i_t)
            file.write_function(Y_dev_out, i_t)
            file.write_function(Y_tot_out, i_t)

    if comm.rank == 0:
        a_file = open(f"{prefix}_data.json", "w")
        json.dump(history_data, a_file)
        a_file.close()
    if comm.rank == 0:
        # plot energies
        plt.figure(0)
        plot_energies2(history_data, file=f"{prefix}_energies.pdf")
        # plot altmin iterations
        plt.plot(
            history_data["load"],
            history_data["alt_min_it"],
            "o",
        )
        plt.savefig(f"{prefix}_alt_min_it.pdf")
        plt.close(0)
        # plot force
        plt.figure(0)
        plt.plot(
            np.array(history_data["load"]),
            np.array(history_data["F"]) / float(mu),
            ".-",
            # color="black",
        )
        if args.sliding == 1:
            plt.plot(
                history_data["load"],
                p_cav / mu + np.array(history_data["load"]),
                ":",
                color="gray",
            )
        # add gridline in t_lim
        plt.axhline(y=float(5 / 2), color="gray", linestyle="--")
        # plt.axvline(x=float(e_c_dim), color="gray", linestyle="--")
        if elastic == 0:
            plt.axvline(x=float(e_c_dim_k), color="gray", linestyle="--")
            # plt.xlabel(r"$e/e_c$")
        plt.xlabel(r"$\Delta/H$")
        plt.ylabel(r"$F/\mu S$")
        plt.xlim(loads[0], loads[-1])
        plt.grid(True)
        if args.sliding == 0:
            plt.plot(
                formulas.GL_fig2_x_corrected,
                formulas.GL_fig2_y_MPa / mu,
                "gray",
                label="Gent and Lindley",
                lw=2,
            )
            Ea_3D_wc = formulas.equivalent_modulus_3d(lmbda, mu, H / 2, L / 2)
            Ea_3D_inc = formulas.equivalent_modulus_3d_incompressible(
                mu, H / 2, L / 2
            )
            plt.plot(
                np.array([0, 0.07]),
                np.array([0, 0.07]) * Ea_3D_inc / mu,
                "-.",
                color="orange",
                label="Incompressible model",
                lw=1,
            )
            # plt.plot(
            #    loads,
            #    loads * Ea_3D_wc / mu,
            #    "r:",
            #    label="Compressible model",
            # )
            # plt.plot(
            #    loads,
            #    loads * 3 / 8 * mu * (L / H) ** 2 / mu,
            # )
        plt.ylim(0, max(3, max(history_data["F"]) / float(mu)))
        plt.xlim(0, max(loads))
        plt.tight_layout()
        plt.savefig(f"{prefix}_force.pdf")
        plt.close(0)

        color = (0.7 * (1 - (i_t / len(loads))),) * 3

        fig_5 = plt.figure(7)
        plt.plot(
            x_points,
            alpha_x,
            color=color,
            # x color="#E63946",
            # marker="o",
            linewidth=3,
            label="Finite Element",
        )
        plt.xlabel(r"$x$")
        plt.ylabel(r"$\alpha$")
        plt.savefig(f"{prefix}_alpha_x.pdf", bbox_inches="tight")

        fig_5 = plt.figure(6)
        plt.plot(
            y_points,
            alpha_y,
            color=color,
            # x color="#E63946",
            # marker="o",
            linewidth=3,
            label="Finite Element",
        )
        plt.xlabel(r"$y$")
        plt.ylabel(r"$\alpha$")
        plt.savefig(f"{prefix}_alpha_y.pdf", bbox_inches="tight")

        fig_5 = plt.figure(5)
        plt.plot(
            x_points,
            eps_nl_x,
            color=color,
            # x color="#E63946",
            # marker="o",
            linewidth=3,
            label="Finite Element",
        )
        plt.xlabel(r"$x$")
        plt.ylabel("nonlinear deformation")
        plt.savefig(f"{prefix}_eps_nl_x.pdf", bbox_inches="tight")

        fig_4 = plt.figure(4)
        plt.plot(
            y_points,
            eps_nl_y,
            color=color,
            # x color="#E63946",
            # marker="o",
            linewidth=3,
            label="Finite Element",
        )
        plt.xlabel(r"$y$")
        plt.ylabel(r"$\varepsilon_\mathrm{NL}$")
        plt.savefig(f"{prefix}_eps_nl_y.pdf", bbox_inches="tight")

        fig_3 = plt.figure(3)
        plt.plot(
            y_points,
            p_val_y,
            color=color,
            # x color="#E63946",
            marker=".",
            linewidth=3,
            label="Finite Element",
        )
        plt.xlabel(r"$y$")
        plt.ylabel(r"$p$")
        plt.savefig(f"{prefix}_p_y.pdf", bbox_inches="tight")

        fig_2 = plt.figure(2)
        plt.plot(
            x_points,
            p_val_x / mu,
            color=color,
            # x color="#E63946",
            marker="o",
            linewidth=3,
            label="Finite Element",
        )
        if args.sliding == 0:
            if gdim == 3:
                plt.plot(
                    x_points,
                    formulas.pressure_3d(
                        lmbda, mu, H / 2, L / 2, t * H / 2, r=x_points
                    )
                    / mu,
                    color="orange",
                    linewidth=1,
                    label="Asymptotic",
                )
            plt.plot(
                x_points,
                formulas.pressure_3d_incompressible(
                    mu, H / 2, L / 2, t * H / 2, r=x_points
                )
                / mu,
                color="red",
                linewidth=1,
                label="Asymptotic",
            )
        # plt.grid(True)
        # plt.legend(loc=(0.2, -0.47), frame alpha=1, fontsize=20)
        plt.xlabel(r"$r/R$")
        plt.ylabel(r"$p/\mu$")

        plt.savefig(f"{prefix}_p_x.pdf", bbox_inches="tight")
        t_post.stop()

        # list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
