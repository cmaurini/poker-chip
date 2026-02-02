from mpi4py import MPI


def mesh_chip(
    R, H, lc, tdim, order=1, msh_file=None, comm=MPI.COMM_WORLD, verbose=True
):
    facet_tag_names = {"top": 17, "bottom": 16}

    tag_names = {"facets": facet_tag_names}

    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        if not verbose:
            gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.option.setNumber("General.Terminal", 1)

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        # gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.model.mesh.optimize("Netgen")
        model = gmsh.model()
        model.add("Circle")
        model.setCurrent("Circle")

        y0 = -H / 2
        p0 = model.geo.addPoint(0.0, y0, 0, lc, tag=0)
        p1 = model.geo.addPoint(R, y0, 0, lc, tag=1)
        p2 = model.geo.addPoint(0, y0, R, lc, tag=2)
        p3 = model.geo.addPoint(-R, y0, 0, lc, tag=3)
        p4 = model.geo.addPoint(0, y0, -R, lc, tag=4)

        bottom_right_up = model.geo.addCircleArc(p1, p0, p2)
        bottom_left_up = model.geo.addCircleArc(p2, p0, p3)
        bottom_left_down = model.geo.addCircleArc(p3, p0, p4)
        bottom_right_down = model.geo.addCircleArc(p4, p0, p1)

        model.geo.addCurveLoop(
            [
                bottom_right_up,
                bottom_left_up,
                bottom_left_down,
                bottom_right_down,
            ],
            5,
        )

        s1 = model.geo.addPlaneSurface([5], 5)
        model.geo.synchronize()
        model.addPhysicalGroup(tdim - 1, [5], 16)
        model.setPhysicalName(tdim - 1, 16, "bottom")

        # Extrude by H in positive y-direction from y=-H/2 to y=H/2
        v1 = gmsh.model.geo.extrude([(tdim - 1, s1)], 0.0, H, 0.0)
        model.geo.synchronize()
        model.addPhysicalGroup(tdim - 1, [v1[0][1]], 17)
        model.setPhysicalName(tdim - 1, 17, "top")
        volume_entities = [model[1] for model in model.getEntities(tdim)]
        model.addPhysicalGroup(tdim, volume_entities, tag=18)
        model.setPhysicalName(tdim, 18, "Volume")

        # Set geometric order of mesh cells
        gmsh.model.mesh.setOrder(order)

        # gmsh.option.setNumber("Mesh.RecombineAll", 1)

        model.mesh.generate(tdim)

        """phys_grps = gmsh.model.getPhysicalGroups()
        for dim, tag in phys_grps:
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)

            for entity in entities:
                element_data = gmsh.model.mesh.getElements(dim, tag=entity)
                element_types, element_tags, node_tags = element_data
                print(element_types,dim)
        print(element_data)"""
        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    if comm.rank == 0:
        model_out = gmsh.model
    else:
        model_out = 0

    return model_out, tdim, tag_names


def mesh_chip_eight(R, H, lc, tdim, order=1, msh_file=None, comm=MPI.COMM_WORLD):
    facet_tag_names = {
        "top": 17,
        "bottom": 16,
        "left": 18,
        "back": 19,
        "curved": 20,
    }

    tag_names = {"facets": facet_tag_names}

    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.optimize("Netgen")
        model = gmsh.model()
        model.add("SectorCylinder")
        model.setCurrent("SectorCylinder")

        y0 = -H / 2
        p0 = model.geo.addPoint(0.0, y0, 0, lc, tag=0)
        p1 = model.geo.addPoint(R, y0, 0, lc, tag=1)
        p2 = model.geo.addPoint(0, y0, R, lc, tag=2)

        bottom_right = model.geo.addLine(p0, p1)
        bottom_left = model.geo.addLine(p2, p0)
        bottom_arc = model.geo.addCircleArc(p1, p0, p2)

        model.geo.addCurveLoop([bottom_right, bottom_arc, bottom_left], 5)
        s1 = model.geo.addPlaneSurface([5], 5)
        model.geo.synchronize()
        model.addPhysicalGroup(tdim - 1, [5], 16)
        model.setPhysicalName(tdim - 1, 16, "bottom")

        # Extrude by H in positive y-direction from y=-H/2 to y=H/2
        v1 = gmsh.model.geo.extrude([(tdim - 1, s1)], 0.0, H, 0.0)
        model.geo.synchronize()
        model.addPhysicalGroup(tdim - 1, [v1[0][1]], 17)
        model.setPhysicalName(tdim - 1, 17, "top")
        model.addPhysicalGroup(tdim - 1, [v1[4][1]], 18)
        model.setPhysicalName(tdim - 1, 18, "left")
        model.addPhysicalGroup(tdim - 1, [v1[2][1]], 19)
        model.setPhysicalName(tdim - 1, 19, "back")
        model.addPhysicalGroup(tdim - 1, [v1[3][1]], 20)
        model.setPhysicalName(tdim - 1, 20, "curved")

        volume_entities = [model[1] for model in model.getEntities(tdim)]
        model.addPhysicalGroup(tdim, volume_entities, tag=21)
        model.setPhysicalName(tdim, 21, "Volume")

        # Set geometric order of mesh cells
        gmsh.model.mesh.setOrder(order)

        model.mesh.generate(tdim)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    if comm.rank == 0:
        model_out = gmsh.model
    else:
        model_out = 0

    return model_out, tdim, tag_names


def test_mesh_chip():
    from dolfinx.io import XDMFFile
    from dolfinx.io.gmsh import model_to_mesh
    import dolfinx

    R = 1.0
    H = 0.5
    lc = 0.1
    tdim = 3
    order = 1
    msh_file = None
    comm = MPI.COMM_WORLD
    model_rank = 0

    gmsh_model, tdim, tag_names = mesh_chip(R, H, lc, tdim, order, msh_file, comm)

    partitioner = dolfinx.mesh.create_cell_partitioner(
        dolfinx.mesh.GhostMode.shared_facet
    )
    mesh_data = model_to_mesh(
        gmsh_model,
        comm,
        model_rank,
        gdim=tdim,
        partitioner=partitioner,
    )
    mesh, cell_tags, facet_tags = (
        mesh_data.mesh,
        mesh_data.cell_tags,
        mesh_data.facet_tags,
    )
    interfaces_keys = tag_names["facets"]
    with XDMFFile(
        comm,
        f"meshes/mesh-3D-chip.xdmf",
        "w",
        encoding=XDMFFile.Encoding.HDF5,
    ) as file:
        file.write_mesh(mesh)
        file.write_meshtags(cell_tags, mesh.geometry)
        file.write_meshtags(facet_tags, mesh.geometry)
    # %%
    if comm.rank == 0:
        print("The mesh has been created.")
        print(f"Mesh has {mesh.geometry.x.shape[0]} vertices.")
        print(f"Mesh has {mesh.topology.index_map(tdim).size_local} cells.")
        print("\nFacet tags:")
        for name, tag in interfaces_keys.items():
            print(f"  {name}: {tag}")


def test_mesh_chip_eight():
    from dolfinx.io import XDMFFile
    from dolfinx.io.gmsh import model_to_mesh
    import dolfinx

    R = 1.0
    H = 0.5
    lc = 0.1
    tdim = 3
    order = 1
    msh_file = None
    comm = MPI.COMM_WORLD
    model_rank = 0

    gmsh_model, tdim, tag_names = mesh_chip_eight(R, H, lc, tdim, order, msh_file, comm)

    partitioner = dolfinx.mesh.create_cell_partitioner(
        dolfinx.mesh.GhostMode.shared_facet
    )
    mesh_data = model_to_mesh(
        gmsh_model,
        comm,
        model_rank,
        gdim=tdim,
        partitioner=partitioner,
    )
    mesh, cell_tags, facet_tags = (
        mesh_data.mesh,
        mesh_data.cell_tags,
        mesh_data.facet_tags,
    )
    interfaces_keys = tag_names["facets"]
    with XDMFFile(
        comm,
        f"meshes/mesh-3D-chip-eight.xdmf",
        "w",
        encoding=XDMFFile.Encoding.HDF5,
    ) as file:
        file.write_mesh(mesh)
        file.write_meshtags(cell_tags, mesh.geometry)
        file.write_meshtags(facet_tags, mesh.geometry)
    # %%
    if comm.rank == 0:
        print("The mesh has been created.")
        print(f"Mesh has {mesh.geometry.x.shape[0]} vertices.")
        print(f"Mesh has {mesh.topology.index_map(tdim).size_local} cells.")
        print("\nFacet tags:")
        for name, tag in interfaces_keys.items():
            print(f"  {name}: {tag}, number of facets {len(facet_tags.find(tag))}")


if __name__ == "__main__":
    test_mesh_chip()
    test_mesh_chip_eight()
