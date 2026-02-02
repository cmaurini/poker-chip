from mpi4py import MPI
import sys
import os
from pathlib import Path
from petsc4py import PETSc
import petsc4py
import logging
import yaml
import ufl
import dolfinx
import dolfinx.common

# Handle imports for both package and direct execution modes
try:
    from .petsc_solvers import SNESSolver, TAOSolver
    from .utils import ColorPrint, norm_H1, norm_L2
except ImportError:
    # Fallback for sys.path mode
    from petsc_solvers import SNESSolver, TAOSolver
    from utils import ColorPrint, norm_H1, norm_L2

petsc4py.init(sys.argv)
dir_path = os.path.dirname(os.path.realpath(__file__))

with open(f"{dir_path}/default_parameters.yml") as f:
    default_parameters = yaml.load(f, Loader=yaml.FullLoader)


class AltMinFractureSolver:
    def __init__(
        self,
        total_energy,
        state,
        bcs,
        solver_parameters={},
        bounds=(dolfinx.fem.function.Function, dolfinx.fem.function.Function),
        variable_names=["u", "alpha"],
        stabilizing_term=None,
        monitor=None,
        comm=MPI.COMM_WORLD,
        residual_u=None,
        residual_alpha=None,
        J_u=None,
        J_alpha=None,
    ):
        self.comm = comm
        self.variable_names = variable_names
        self.stabilizing_term = stabilizing_term
        self.u = state[self.variable_names[0]]
        self.alpha = state[self.variable_names[1]]
        self.bcs = bcs
        self.alpha_old = dolfinx.fem.Function(self.alpha.function_space)
        self.alpha.x.petsc_vec.copy(result=self.alpha_old.x.petsc_vec)
        self.alpha_old.x.scatter_forward()
        self.u_old = dolfinx.fem.Function(self.u.function_space)
        self.u.x.petsc_vec.copy(result=self.u_old.x.petsc_vec)
        self.u_old.x.scatter_forward()

        self.total_energy = total_energy

        self.state = state
        self.alpha_lb = bounds[0]
        self.alpha_ub = bounds[1]

        self.solver_parameters = default_parameters["solvers"]
        if solver_parameters:
            self.solver_parameters.update(solver_parameters)

        self.monitor = monitor
        V_u = self.u.function_space
        V_alpha = self.alpha.function_space

        if residual_u is None:
            energy_u = ufl.derivative(
                self.total_energy, self.u, ufl.TestFunction(V_u)
            )
        else:
            energy_u = residual_u
        if stabilizing_term:
            energy_u += ufl.derivative(
                self.stabilizing_term, self.u, ufl.TestFunction(V_u)
            )

        if residual_alpha is None:
            energy_alpha = ufl.derivative(
                self.total_energy, self.alpha, ufl.TestFunction(V_alpha)
            )
        else:
            energy_alpha = residual_alpha

        if J_u is None:
            J_u = ufl.derivative(energy_u, self.u, ufl.TrialFunction(V_u))

        if J_alpha is None:
            J_alpha = ufl.derivative(
                energy_alpha, self.alpha, ufl.TrialFunction(V_alpha)
            )

        self.F = [energy_u, energy_alpha]

        if self.solver_parameters.get("damage").get("type") == "SNES":
            self.elasticity = SNESSolver(
                energy_u,
                self.u,
                bcs.get("bcs_u"),
                bounds=None,
                J_form=J_u,
                petsc_options=self.solver_parameters.get("elasticity").get(
                    "snes"
                ),
                prefix=self.solver_parameters.get("elasticity").get("prefix"),
            )

        if self.solver_parameters.get("elasticity").get("type") == "TAO":
            self.elasticity = TAOSolver(
                total_energy,
                self.u,
                bcs.get("bcs_u"),
                bounds=None,
                J_form=J_u,
                petsc_options=self.solver_parameters.get("elasticity").get(
                    "tao"
                ),
                prefix=self.solver_parameters.get("elasticity").get("prefix"),
            )

        if self.solver_parameters.get("damage").get("type") == "SNES":
            self.damage = SNESSolver(
                energy_alpha,
                self.alpha,
                bcs.get("bcs_alpha"),
                J_form=J_alpha,
                bounds=(self.alpha_lb, self.alpha_ub),
                petsc_options=self.solver_parameters.get("damage").get("snes"),
                prefix=self.solver_parameters.get("damage").get("prefix"),
            )

        if self.solver_parameters.get("damage").get("type") == "TAO":
            self.damage = TAOSolver(
                total_energy,
                self.alpha,
                bcs.get("bcs_alpha"),
                bounds=(self.alpha_lb, self.alpha_ub),
                petsc_options=self.solver_parameters.get("damage").get("tao"),
                prefix=self.solver_parameters.get("damage").get("prefix"),
            )

    def compute_residual_u(self):
        """Return the residual norm of the u-problem"""

        V_u = self.u.function_space
        residual = dolfinx.fem.Function(V_u)
        self.elasticity.solver.computeFunction(
            self.u.x.petsc_vec, residual.x.petsc_vec
        )
        self.u.x.petsc_vec, residual.x.scatter_forward()
        return residual.x.petsc_vec.norm()

    def stop_criterion(self, parameters=None):
        """Stop criterion for alternate minimization"""

        if parameters is None:
            parameters = self.solver_parameters.get("damage_elasticity")

        criterion = parameters.get("criterion")
        criteria = ["residual_u", "alpha_H1", "alpha_L2", "alpha_max", "energy"]

        if criterion in criteria:
            if criterion == "residual_u":
                criterion_residual_u = self.error_residual_u <= parameters.get(
                    "residual_u_tol"
                )
                return criterion_residual_u

            if criterion == "alpha_H1":
                criterion_alpha_H1 = self.error_alpha_H1 <= parameters.get(
                    "alpha_tol"
                )
                return criterion_alpha_H1

            if criterion == "alpha_L2":
                criterion_alpha_L2 = self.error_alpha_L2 <= parameters.get(
                    "alpha_tol"
                )
                return criterion_alpha_L2

            if criterion == "alpha_max":
                criterion_alpha_max = self.error_alpha_max <= parameters.get(
                    "alpha_tol"
                )
                return criterion_alpha_max

            if criterion == "energy":
                criterion_error_energy_r = (
                    self.error_energy_r <= parameters.get("energy_rtol")
                )
                criterion_error_energy_a = (
                    self.error_energy_a <= parameters.get("energy_atol")
                )
                return criterion_error_energy_r or criterion_error_energy_a
        else:
            raise RuntimeError(f"{criterion} is not in {criteria}")

    def append_data(self, iteration):
        self.data["iteration"].append(iteration)
        self.data["error_alpha_L2"].append(self.error_alpha_L2)
        self.data["error_alpha_H1"].append(self.error_alpha_H1)
        self.data["error_alpha_max"].append(self.error_alpha_max)
        self.data["error_residual_u"].append(self.error_residual_u)
        self.data["solver_alpha_it"].append(self.solver_alpha_it)
        self.data["solver_alpha_reason"].append(self.solver_alpha_reason)
        self.data["solver_u_reason"].append(self.solver_u_reason)
        self.data["solver_u_it"].append(self.solver_u_it)
        self.data["total_energy"].append(self.total_energy_int)

    def initialize_data(self):
        data = {
            "iteration": [],
            "error_alpha_L2": [],
            "error_alpha_H1": [],
            "error_alpha_max": [],
            "error_residual_u": [],
            "solver_alpha_reason": [],
            "solver_alpha_it": [],
            "solver_u_reason": [],
            "solver_u_it": [],
            "total_energy": [],
        }
        return data

    def solve(self, monitor=None, parameters=None):
        if parameters is None:
            parameters = self.solver_parameters.get("damage_elasticity")

        self.max_it = parameters.get("max_it")

        alpha_diff = dolfinx.fem.Function(self.alpha.function_space)

        self.total_energy_int_old = 0

        self.data = self.initialize_data()

        if monitor is not None:
            self.alpha.x.petsc_vec.copy(self.alpha_old.x.petsc_vec)
            self.alpha_old.x.scatter_forward()

        for iteration in range(self.max_it):
            with dolfinx.common.Timer(
                "~Alternate Minimization : Elastic solver"
            ):
                ColorPrint.print_info("elastic solve")
                (self.solver_u_it, self.solver_u_reason) = (
                    self.elasticity.solve()
                )

            with dolfinx.common.Timer(
                "~Alternate Minimization : Damage solver"
            ):
                ColorPrint.print_info("damage solve")
                (self.solver_alpha_it, self.solver_alpha_reason) = (
                    self.damage.solve()
                )

            # compute errors
            self.alpha.x.petsc_vec.copy(alpha_diff.x.petsc_vec)
            alpha_diff.x.petsc_vec.axpy(-1, self.alpha_old.x.petsc_vec)
            alpha_diff.x.scatter_forward()

            self.error_alpha_H1 = norm_H1(alpha_diff)
            self.error_alpha_L2 = norm_L2(alpha_diff)
            self.error_alpha_max = alpha_diff.x.petsc_vec.max()[1]
            self.error_residual_u = self.compute_residual_u()
            self.total_energy_int = self.comm.allreduce(
                dolfinx.fem.assemble_scalar(
                    dolfinx.fem.form(self.total_energy)
                ),
                op=MPI.SUM,
            )
            self.error_energy_a = abs(
                self.total_energy_int - self.total_energy_int_old
            )

            if self.total_energy_int_old > 0:
                self.error_energy_r = abs(
                    self.total_energy_int / self.total_energy_int_old - 1
                )
            else:
                self.error_energy_r = 1.0

            # update
            self.append_data(iteration)
            self.alpha.x.petsc_vec.copy(result=self.alpha_old.x.petsc_vec)
            self.alpha_old.x.scatter_forward()
            self.u.x.petsc_vec.copy(result=self.u_old.x.petsc_vec)
            self.u_old.x.scatter_forward()
            self.total_energy_int_old = self.total_energy_int

            # monitors
            if self.monitor is not None:
                self.monitor()

            if monitor is not None:
                monitor(iteration=iteration, data=self.data)

            ColorPrint.print_info(
                f"AM - Iteration: {iteration:3d}, "
                + f"alpha_max: {self.alpha.x.petsc_vec.max()[1]:3.4e}, "
                + f"Error_alpha_max: {self.error_alpha_max:3.4e}, "
                + f"Error_energy: {self.error_energy_r:3.4e}, "
                + f"Error_residual_u: {self.error_residual_u:3.4e}"
            )
            # check convergence
            if self.stop_criterion(parameters=parameters):
                break

        else:
            if not parameters.get("error_on_nonconvergence"):
                ColorPrint.print_warn(
                    (
                        f"Could not converge after {iteration:3d} iterations,"
                        + f"error_u {self.error_residual_u:3.4e},"
                        + f"error_alpha_max {self.error_alpha_max:3.4e},"
                        + f"error_energy_r {self.error_energy_r:3.4e}"
                    )
                )
            else:
                raise RuntimeError(
                    f"Could not converge after {iteration:3d} iterations, error {self.error_residual_u:3.4e}"
                )
