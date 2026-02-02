# %%
from mpi4py import MPI
import gmsh


def box_mesh(
    Lx, Ly, Lz, lc, tdim=3, order=1, msh_file=None, comm=MPI.COMM_WORLD, verbose=True
):
    """Create a box mesh using GMSH with specified dimensions and mesh parameters.
    This function creates a rectangular box mesh centered at the origin using GMSH. The mesh includes
    tagged boundaries for each face of the box and allows specification of mesh characteristics.
    Parameters
    ----------
    Lx : float
        Length of box in x-direction
    Ly : float
        Length of box in y-direction
    Lz : float
        Length of box in z-direction
    lc : float
        Characteristic length of mesh elements
    tdim : int, optional
        Topological dimension (default=3)
    order : int, optional
        Order of geometric elements (default=1)
    msh_file : str, optional
        Path to save .msh file (default=None)
    comm : MPI.Comm, optional
        MPI communicator (default=MPI.COMM_WORLD)
    Returns
    -------
    tuple
        Returns (model, tdim, tag_names) where:
        - model: GMSH model object
        - tdim: topological dimension
        - tag_names: dictionary containing boundary tags
    Notes
    -----
    The box is centered at the origin with faces tagged as:
    - left (101)
    - right (102)
    - front (103)
    - back (104)
    - bottom (105)
    - top (106)
    The mesh is generated only on rank 0 of the MPI communicator.
    """

    facet_tag_names = {
        "left": 101,
        "right": 102,
        "bottom": 103,
        "top": 106,
        "back": 104,
        "front": 105,
    }

    tag_names = {"facets": facet_tag_names}

    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        if not verbose:
            gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

        # gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.model.mesh.optimize("Netgen")
        model = gmsh.model()

        model.add("box")
        model.setCurrent("box")
        # add a box mesh os size .1, .2, .3 with gmsh model

        box = model.occ.addBox(-Lx / 2.0, -Ly / 2.0, -Lz / 2.0, Lx, Ly, Lz, tag=100)
        model.occ.synchronize()

        # Add physical groups
        surfaces = gmsh.model.occ.getEntities(dim=2)
        surface_tags = [s[1] for s in surfaces]

        # Add physical groups for each surface
        for i, (name, tag) in enumerate(facet_tag_names.items()):
            com = gmsh.model.occ.getCenterOfMass(2, surface_tags[i])
            print(f"Surface {name} barycenter coordinates: {com}, tag: {tag}")
            gmsh.model.addPhysicalGroup(2, [surface_tags[i]], tag)
            gmsh.model.setPhysicalName(2, tag, name)

        # Add physical group for the volume
        volume_entities = [model[1] for model in gmsh.model.getEntities(tdim)]
        gmsh.model.addPhysicalGroup(tdim, volume_entities, tag=18)
        gmsh.model.setPhysicalName(tdim, 18, "Volume")

        # Set geometric order of mesh cells
        order = 1
        gmsh.model.mesh.setOrder(order)

        # Generate the mesh
        gmsh.model.mesh.generate(tdim)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    if comm.rank == 0:
        model_out = gmsh.model
    else:
        model_out = 0

    return model_out, tdim, tag_names


if __name__ == "__main__":
    Lx, Ly, Lz = 1.0, 0.3, 1.0
    lc = 0.03
    gmsh_model, tdim, tag_names = box_mesh(Lx, Ly, Lz, lc)
    # %%
    import dolfinx
    from dolfinx.io import XDMFFile
    from dolfinx.io.gmsh import model_to_mesh

    model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    partitioner = dolfinx.mesh.create_cell_partitioner(
        dolfinx.mesh.GhostMode.shared_facet
    )
    mesh_data = model_to_mesh(
        gmsh_model,
        mesh_comm,
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
        mesh_comm,
        f"meshes/mesh-3D-box.xdmf",
        "w",
        encoding=XDMFFile.Encoding.HDF5,
    ) as file:
        file.write_mesh(mesh)
        file.write_meshtags(cell_tags, mesh.geometry)
        file.write_meshtags(facet_tags, mesh.geometry)
    # %%
    if mesh_comm.rank == 0:
        print("The mesh has been created.")
        print(f"Mesh has {mesh.geometry.x.shape[0]} vertices.")
        print(f"Mesh has {mesh.topology.index_map(tdim).size_local} cells.")
        print("\nFacet tags:")
        for name, tag in interfaces_keys.items():
            print(f"  {name}: {tag}")

# %%
