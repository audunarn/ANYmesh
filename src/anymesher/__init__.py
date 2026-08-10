"""Structured meshing for shell and beam models described by ANYgeometry.

The package owns seeding and refinement, mapped-face preparation, a mapped
(transfinite Coons) mesher, direct mesh primitives and an optional Gmsh backend.
What it produces is a neutral mesh -- nodes, quadrilaterals, triangles, beams,
coupling records, and the association back to the ANYgeometry entities that made
them. Historical geometry imports below are exact owner aliases during migration.

Elements, materials, assembly and solution live elsewhere.  A mesh here knows
that a beam node stands off the plating by an eccentricity and which shell
element it lands in with what shape weights; it does not know what an MPC is.
The consuming solver turns that record into a constraint.

``generate_mesh`` exported here dispatches on ``backend=`` and defaults to the
built-in mapped mesher.  ``anymesher.mapped.generate_mesh`` is that mesher
directly, for a caller that wants no dispatch at all.

The package deliberately does not import ANYsolver.  Meshes are produced here
and consumed there, never the other way round, so the dependency stays acyclic
and a mesh can be generated and inspected without a solver present.

``anymesher.gui`` is not imported here, so importing the package never requires a
display or a tkinter build.
"""

from __future__ import annotations

from .backends import (
    DEFAULT_BACKEND,
    MeshBackend,
    available_backends,
    generate_mesh,
    resolve_backend,
)
from .coupling import (
    StructuredShellGrid,
    build_structured_shell_grid,
    locate_shell_element_at_xy,
    shape_functions_4node,
    shape_functions_8node,
)
from .errors import GeometryError, MeshError
from .geometry import (
    Arc,
    ArcFrame,
    CurveShape,
    DegenerateArcError,
    Edge,
    EntityKind,
    EntityRef,
    Face,
    GeometryModel,
    MappabilityReport,
    OrientedEdge,
    Spline,
    Straight,
    Vertex,
    arc_frame,
    chain_breaks,
    chain_point,
    check_mappable,
    punch_circular_hole,
    sample_chain,
    split_face_at,
    split_face_between,
    strip_face,
    surface_point,
    triangle_to_quads,
)
from .mapped import ELEMENT_ORDERS, coons_grid, nodal_normals
from .mesh import Coupling, Mesh
from .primitives import (
    PANEL_EDGE_IDS,
    PANEL_FACE_ID,
    STIFFENER_EDGE_ID_BASE,
    PanelMeshConfig,
    StiffenedPanel,
    StiffenerCrossSection,
    beam_mesh,
    panel_edge_nodes,
    simple_panel_mesh,
    stiffened_panel_mesh,
)
from .quality import ASPECT_RATIO_LIMIT, WARP_LIMIT, MeshQuality, verify_mesh_quality
from .refinement import Refinement, SizeField, refine_around, refine_at
from .seeding import Seeding, SeedingConflict, edge_demand, edge_distribution, solve_seeding
from .serialize import load_mesh, mesh_from_dict, mesh_to_dict, save_mesh

__version__ = "0.1.0"

__all__ = [
    "ASPECT_RATIO_LIMIT",
    "Arc",
    "ArcFrame",
    "Coupling",
    "CurveShape",
    "DEFAULT_BACKEND",
    "DegenerateArcError",
    "ELEMENT_ORDERS",
    "Edge",
    "EntityKind",
    "EntityRef",
    "Face",
    "GeometryError",
    "GeometryModel",
    "MappabilityReport",
    "Mesh",
    "MeshBackend",
    "MeshError",
    "MeshQuality",
    "OrientedEdge",
    "PANEL_EDGE_IDS",
    "PANEL_FACE_ID",
    "PanelMeshConfig",
    "Refinement",
    "STIFFENER_EDGE_ID_BASE",
    "Seeding",
    "SeedingConflict",
    "SizeField",
    "StiffenedPanel",
    "Spline",
    "StiffenerCrossSection",
    "Straight",
    "StructuredShellGrid",
    "Vertex",
    "WARP_LIMIT",
    "arc_frame",
    "available_backends",
    "beam_mesh",
    "build_structured_shell_grid",
    "chain_breaks",
    "chain_point",
    "check_mappable",
    "coons_grid",
    "edge_demand",
    "edge_distribution",
    "generate_mesh",
    "load_mesh",
    "locate_shell_element_at_xy",
    "mesh_from_dict",
    "mesh_to_dict",
    "nodal_normals",
    "panel_edge_nodes",
    "punch_circular_hole",
    "refine_around",
    "refine_at",
    "resolve_backend",
    "sample_chain",
    "save_mesh",
    "shape_functions_4node",
    "shape_functions_8node",
    "simple_panel_mesh",
    "solve_seeding",
    "split_face_at",
    "split_face_between",
    "stiffened_panel_mesh",
    "strip_face",
    "surface_point",
    "triangle_to_quads",
    "verify_mesh_quality",
]
