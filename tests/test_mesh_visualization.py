#!/usr/bin/env python3
"""
Test script for mesh generators with visualization
Generates mesh figures using PyVista for each mesh type
"""

import pytest
import pyvista as pv
from pathlib import Path
from mpi4py import MPI
import dolfinx
from dolfinx.io.gmsh import model_to_mesh
import sys
import os

# Add parent directory to path for importing modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import mesh generators
from mesh.mesh_bar import mesh_bar
from mesh.mesh_box import box_mesh
from mesh.mesh_chip import mesh_chip, mesh_chip_eight


# Output directory for figures
OUTPUT_DIR = Path(__file__).parent / "mesh_figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_mesh_figure(mesh, title, filename, facet_tags=None):
    """Save a figure of the mesh using PyVista, optionally coloring boundary facets by tag."""
    import numpy as np

    pv.set_plot_theme("document")

    # Create PyVista mesh from dolfinx mesh
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(mesh)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(
        grid,
        show_edges=True,
        color="lightblue",
        edge_color="black",
        line_width=1,
        opacity=0.2,
    )
    plotter.add_title(title, font_size=12)

    # Add boundary facets as colored wireframe by tag
    legend_entries = []
    tag_name_map = None
    if facet_tags is not None:
        # Get boundary facet indices and tag values
        indices = facet_tags.indices
        values = facet_tags.values
        # For each facet, get its vertex coordinates
        mesh_dim = mesh.topology.dim
        facet_dim = mesh_dim - 1
        conn = mesh.topology.connectivity(facet_dim, 0)
        if conn is None:
            mesh.topology.create_connectivity(facet_dim, 0)
            conn = mesh.topology.connectivity(facet_dim, 0)
        faces = []
        for i in indices:
            dofs = conn.links(i)
            faces.append(dofs)
        faces = [np.array(f, dtype=np.int32) for f in faces]
        # Build PyVista PolyData for the boundary
        points = mesh.geometry.x
        # PyVista expects faces in a flat format: [npts, v0, v1, v2, ...]
        faces_pv = []
        for f in faces:
            faces_pv.append(np.concatenate([[len(f)], f]))
        faces_pv = np.concatenate(faces_pv)
        surf = pv.PolyData(points, faces_pv)
        surf.cell_data["facet_tag"] = values
        # Try to get tag name mapping from mesh attribute if present
        if hasattr(mesh, "facet_tag_names"):
            tag_name_map = mesh.facet_tag_names
        # Use matplotlib to get the tab10 colormap
        import matplotlib.pyplot as mplplt

        cmap = mplplt.get_cmap("tab10")
        unique_tags = np.unique(values)
        color_map = {}
        for i, tag in enumerate(unique_tags):
            color = tuple((np.array(cmap(i % 10)[:3]) * 255).astype(int))
            color_map[tag] = color
        # Add mesh with custom colors
        for i, tag in enumerate(unique_tags):
            mask = values == tag
            tag_faces = surf.extract_cells(np.where(mask)[0])
            color = color_map[tag]
            plotter.add_mesh(
                tag_faces,
                style="wireframe",
                line_width=8,
                color=color,
                label=f"{tag}",
            )
            # Prepare legend entry: always show label and number
            if tag_name_map and tag in tag_name_map:
                tag_label = f"{tag_name_map[tag]} (tag {tag})"
            else:
                tag_label = f"tag {tag}"
            legend_entries.append((tag_label, color))
        # Add custom legend with color swatch, tag name, and number
        if legend_entries:
            legend_labels = []
            for lbl, col in legend_entries:
                color_hex = f"#{col[0]:02x}{col[1]:02x}{col[2]:02x}"
                legend_labels.append((lbl, color_hex))
            plotter.add_legend(
                legend_labels,
                bcolor="white",
                border=True,
                size=(0.28, 0.18),
                loc="upper right",
            )

    # Add bounding box
    plotter.add_bounding_box(color="black", line_width=2)
    plotter.show_grid()
    # Set view based on mesh dimension
    if mesh.topology.dim == 2:
        # XY view for 2D meshes
        plotter.view_xy()
    else:
        # Isometric view for 3D meshes
        plotter.view_isometric()

    plotter.show_axes()

    # Save figure
    plotter.screenshot(OUTPUT_DIR / filename, scale=2)
    plotter.close()

    print(f"Saved figure: {OUTPUT_DIR / filename}")


def test_mesh_bar():
    """Test bar mesh generator (2D rectangle)"""
    comm = MPI.COMM_WORLD

    # Parameters
    L = 2.0  # Length
    H = 0.5  # Height
    lc = 0.1  # Characteristic length
    tdim = 2  # Topological dimension
    order = 1

    # Generate mesh
    gmsh_model, tdim, tag_names = mesh_bar(L, H, lc, tdim, order=order, comm=comm)

    if comm.rank == 0:
        # Convert to dolfinx mesh
        mesh_data = model_to_mesh(gmsh_model, comm, rank=0, gdim=tdim)
        mesh = mesh_data.mesh
        facet_tags = mesh_data.facet_tags
        # Attach facet tag label mapping to mesh for legend
        mesh.facet_tag_names = {v: k for k, v in tag_names["facets"].items()}

        # Save figure
        title = f"Bar Mesh\nL={L}, H={H}, lc={lc}"
        save_mesh_figure(mesh, title, "mesh_bar.png", facet_tags=facet_tags)

        import gmsh

        gmsh.finalize()

    assert True


def test_mesh_box():
    """Test box mesh generator (3D box)"""
    comm = MPI.COMM_WORLD

    # Parameters
    Lx = 1.0  # Length in x
    Ly = 0.5  # Length in y
    Lz = 0.3  # Length in z
    lc = 0.1  # Characteristic length
    tdim = 3  # Topological dimension
    order = 1

    # Generate mesh
    gmsh_model, tdim, tag_names = box_mesh(
        Lx, Ly, Lz, lc, tdim=tdim, order=order, comm=comm
    )

    if comm.rank == 0:
        # Convert to dolfinx mesh
        mesh_data = model_to_mesh(gmsh_model, comm, rank=0, gdim=tdim)
        mesh = mesh_data.mesh
        facet_tags = mesh_data.facet_tags
        mesh.facet_tag_names = {v: k for k, v in tag_names["facets"].items()}

        # Save figure
        title = f"Box Mesh\nLx={Lx}, Ly={Ly}, Lz={Lz}, lc={lc}"
        save_mesh_figure(mesh, title, "mesh_box.png", facet_tags=facet_tags)

        import gmsh

        gmsh.finalize()

    assert True


def test_mesh_chip():
    """Test chip mesh generator (3D cylinder)"""
    comm = MPI.COMM_WORLD

    # Parameters
    R = 0.5  # Radius
    H = 0.2  # Height
    lc = 0.05  # Characteristic length
    tdim = 3  # Topological dimension
    order = 1

    # Generate mesh
    gmsh_model, tdim, tag_names = mesh_chip(R, H, lc, tdim, order=order, comm=comm)

    if comm.rank == 0:
        # Convert to dolfinx mesh
        mesh_data = model_to_mesh(gmsh_model, comm, rank=0, gdim=tdim)
        mesh = mesh_data.mesh
        facet_tags = mesh_data.facet_tags
        mesh.facet_tag_names = {v: k for k, v in tag_names["facets"].items()}

        # Save figure
        title = f"Chip Mesh (Full Cylinder)\nR={R}, H={H}, lc={lc}"
        save_mesh_figure(mesh, title, "mesh_chip.png", facet_tags=facet_tags)

        import gmsh

        gmsh.finalize()

    assert True


def test_mesh_chip_eight():
    """Test chip_eight mesh generator (3D cylinder eighth)"""
    comm = MPI.COMM_WORLD

    # Parameters
    R = 0.5  # Radius
    H = 0.2  # Height
    lc = 0.05  # Characteristic length
    tdim = 3  # Topological dimension
    order = 1

    # Generate mesh
    gmsh_model, tdim, tag_names = mesh_chip_eight(
        R, H, lc, tdim, order=order, comm=comm
    )

    if comm.rank == 0:
        # Convert to dolfinx mesh
        mesh_data = model_to_mesh(gmsh_model, comm, rank=0, gdim=tdim)
        mesh = mesh_data.mesh
        facet_tags = mesh_data.facet_tags
        mesh.facet_tag_names = {v: k for k, v in tag_names["facets"].items()}

        # Save figure
        title = f"Chip Mesh (Eighth)\nR={R}, H={H}, lc={lc}"
        save_mesh_figure(mesh, title, "mesh_chip_eight.png", facet_tags=facet_tags)

        import gmsh

        gmsh.finalize()

    assert True


if __name__ == "__main__":
    print("Testing mesh generators with visualization...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 60)

    test_mesh_bar()
    print("-" * 60)

    test_mesh_box()
    print("-" * 60)

    test_mesh_chip()
    print("-" * 60)

    test_mesh_chip_eight()
    print("-" * 60)

    print("\nAll tests completed successfully!")
    print(f"Figures saved to: {OUTPUT_DIR}")
