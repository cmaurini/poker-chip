name"""Solver configurations for 3D elasticity problems.

This module contains preconfigured solver options for various iterative
and direct solvers optimized for elasticity problems.
"""


def configure_3d_solver(V_u, parameters, base_solver_options, mpi_print):
    """
    Configure optimized iterative solver options for 3D elasticity problems.

    Args:
        V_u: Function space for displacement
        parameters: Configuration parameters
        base_solver_options: Base solver options dictionary
        mpi_print: Function for MPI-aware printing

    Returns:
        dict: Updated solver options for 3D problems
    """
    solver_options = dict(base_solver_options)

    if not parameters.get("use_iterative_solver", True):
        return solver_options

    mpi_print("Using iterative solver configuration for 3D problem...")

    # Get problem size for adaptive solver selection
    num_dofs = V_u.dofmap.index_map.size_global * V_u.dofmap.index_map_bs
    mpi_print(f"  Problem size: {num_dofs} DOFs")

    # Adaptive solver selection based on problem size
    solver_type = parameters.get("solver_type", "gamg")

    if solver_type == "hypre_amg":
        # BoomerAMG optimized for elasticity
        solver_options.update(
            {"ksp_type": "cg", "pc_type": "hypre", "pc_hypre_type": "boomeramg"}
        )
    elif solver_type == "gamg_chebyshev":
        # PETSc GAMG optimized for elasticity (rigid-body near-nullspace attached in solver_setup)
        # Note: For nearly incompressible problems (high κ), the volumetric mode is stiff.
        # The near-nullspace helps GAMG handle both rigid modes and volumetric constraints.
        solver_options.update(
            {
                "ksp_type": "cg",
                "pc_type": "gamg",
                "pc_gamg_type": "agg",  # Aggressive coarsening
                "pc_gamg_agg_nsmooths": 1,  # Smoothing on coarse grids
                "pc_gamg_threshold": 0.01,  # Lower threshold for better handling of stiff modes
                "pc_gamg_square_graph": 2,  # Improve coarsening quality
                "mg_levels_ksp_type": "chebyshev",  # Chebyshev smoother on levels
                "mg_levels_pc_type": "jacobi",  # Jacobi for smoothing
                "mg_levels_ksp_max_it": 3,  # More smoothing for stiff volumetric modes
                "pc_gamg_reuse_interpolation": True,  # Reuse interpolation operators
                "ksp_rtol": 1.0e-8,
                "ksp_atol": 1.0e-10,
                "ksp_max_it": 2000,
            }
        )
    elif solver_type == "gamg":
        # Classical GAMG configuration inspired by FEniCS elasticity demo
        # Relies heavily on near-nullspace (rigid body modes) for elasticity
        # Using FGMRES instead of CG for robustness with variable preconditioning
        solver_options.update(
            {
                "ksp_type": "fgmres",  # Flexible GMRES for variable preconditioner
                "ksp_gmres_restart": 100,
                "pc_type": "gamg",
                "pc_gamg_type": "agg",
                # Parallel-specific settings
                "pc_gamg_process_eq_limit": 50,  # Min equations per process before stopping coarsening
                "pc_gamg_repartition": True,  # Repartition coarse grids in parallel
                "pc_gamg_reuse_interpolation": True,  # Reuse interpolation operators
                # Smoothers
                "mg_levels_ksp_type": "richardson",  # Simple Richardson on levels
                "mg_levels_pc_type": "sor",  # SOR smoother
                "mg_levels_ksp_max_it": 2,  # Smoothing iterations
                # Coarse grid solver (critical for parallel)
                "mg_coarse_ksp_type": "preonly",
                "mg_coarse_pc_type": "bjacobi",  # Block Jacobi on coarse grid
                "mg_coarse_sub_pc_type": "lu",  # LU on each block
                # Tolerances
                "ksp_rtol": 1.0e-8,
                "ksp_atol": 1.0e-10,
                "ksp_max_it": 2000,
            }
        )

    elif solver_type == "richardson_hypre":
        # Richardson iteration with hypre AMG - simple but robust
        solver_options.update(
            {
                "ksp_type": "richardson",
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
                "ksp_rtol": 1.0e-8,
                "ksp_atol": 1.0e-10,
                "ksp_max_it": 2000,
            }
        )
    elif solver_type == "sor":
        # Block Jacobi with SOR sub-preconditioner for better vector problem handling
        solver_options.update(
            {
                "ksp_type": "gmres",
                "ksp_gmres_restart": 100,
                "pc_type": "sor",
            }
        )
    elif solver_type == "direct":
        # Direct solver via MUMPS
        solver_options.update(
            {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        )
    elif solver_type == "ilu":
        # ILU preconditioner with added fill levels for robustness (MUMPS cannot factorize ILU for seqaij)
        solver_options.update(
            {
                "ksp_type": "cg",
                "pc_type": "ilu",
                "pc_factor_levels": 3,
                "pc_factor_fill": 2.0,
            }
        )

    mpi_print(f"  Solver type: {solver_type}")
    mpi_print(f"  KSP type: {solver_options.get('ksp_type', 'default')}")
    mpi_print(f"  PC type: {solver_options.get('pc_type', 'default')}")

    return solver_options
