"""Damage mechanics and phase-field fracture models"""

from .alternate_minimization import AltMinFractureSolver
from .models import DamageElasticityModel, ElasticityModel
from .plots import plot_energies2, plot_AMit_load, plot_energies
from .utils import ColorPrint, assemble_scalar_reduce
from .petsc_solvers import SNESSolver, TAOSolver
from scifem import evaluate_function

__all__ = [
    "AltMinFractureSolver",
    "DamageElasticityModel",
    "ElasticityModel",
    "plot_energies2",
    "plot_AMit_load",
    "plot_energies",
    "ColorPrint",
    "assemble_scalar_reduce",
    "SNESSolver",
    "TAOSolver",
    "evaluate_function",
]
