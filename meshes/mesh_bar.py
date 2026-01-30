#!/usr/bin/env python3

from mpi4py import MPI


def mesh_bar(L, H, lc, tdim, order=1, msh_file=None, comm=MPI.COMM_WORLD):
    facet_tag_names = {"top": 14, "bottom": 12, "left": 15, "right": 13}

    tag_names = {"facets": facet_tag_names}

    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        gmsh.option.setNumber("Mesh.Algorithm", 5)
        gmsh.model.mesh.optimize("Netgen")
        model = gmsh.model()
        model.add("Rectangle")
        model.setCurrent("Rectangle")

        p0 = model.geo.addPoint(-L / 2, 0.0, 0, lc, tag=0)
        p1 = model.geo.addPoint(L / 2, 0.0, 0, lc, tag=1)
        p2 = model.geo.addPoint(L / 2, H, 0.0, lc, tag=2)
        p3 = model.geo.addPoint(-L / 2, H, 0, lc, tag=3)

        bottom = model.geo.addLine(p0, p1, tag=12)
        right = model.geo.addLine(p1, p2, tag=13)
        top = model.geo.addLine(p2, p3, tag=14)
        left = model.geo.addLine(p3, p0, tag=15)

        cloop1 = model.geo.addCurveLoop(
            [
                bottom,
                right,
                top,
                left,
            ]
        )

        model.geo.addPlaneSurface([cloop1])
        model.geo.synchronize()
        surface_entities = [model[1] for model in model.getEntities(tdim)]
        model.addPhysicalGroup(tdim, surface_entities, tag=22)
        model.setPhysicalName(tdim, 22, "Rectangle surface")

        # Set geometric order of mesh cells
        gmsh.model.mesh.setOrder(order)

        for k, v in facet_tag_names.items():
            gmsh.model.addPhysicalGroup(tdim - 1, [v], tag=v)
            gmsh.model.setPhysicalName(tdim - 1, v, k)

        model.mesh.generate(tdim)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    if comm.rank == 0:
        model_out = gmsh.model
    else:
        model_out = 0

    return model_out, tdim, tag_names
