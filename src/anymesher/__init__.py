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
native hybrid selector. ``backend='mapped'`` and
``anymesher.mapped.generate_mesh`` preserve the mapped-only compatibility path.

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
from .hybrid import (
    CertificationMode,
    HybridMeshResult,
    MeshingStrategy,
    generate_hybrid_mesh,
    generate_hybrid_mesh_result,
)
from .mesh import Coupling, Mesh
from .preparation import (
    StructuralPreparationOptions,
    StructuralPreparationReport,
    prepare_structural_closure,
)
from .charts import (
    ChartProjection,
    FaceChart,
    MetricField,
    metric_edge_lengths,
    metric_tensors,
)
from .optimization import (
    EdgeFlipResult,
    SmoothingResult,
    constrained_smoothing,
    local_edge_flip,
)
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
from .s3_quality import (
    DEFAULT_S3_QUALITY_POLICY,
    S3_QUALITY_CONTRACT_ID,
    S3AdmissionReport,
    S3ElementQuality,
    S3QualityError,
    S3QualityPolicy,
    assert_s3_admissible,
    evaluate_s3_admission,
)
from .s3_repair import (
    DEFAULT_S3_REPAIR_POLICY,
    S3_REPAIR_CONTRACT_ID,
    S3RepairAttempt,
    S3RepairError,
    S3RepairPolicy,
    S3RepairResult,
    repair_s3_admission,
)
from .refinement import Refinement, SizeField, refine_around, refine_at
from .seeding import Seeding, SeedingConflict, edge_demand, edge_distribution, solve_seeding
from .structured import (
    MeshQualityPolicy,
    SeedEquation,
    StructurePreference,
    StructuredBlock,
    StructuredFacePlan,
    StructuredInterface,
    StructuredLayoutPlan,
    StructuredLayoutReport,
    StructuredMeshingOptions,
    apply_structured_layout,
    commit_structured_layout,
    plan_structured_layout,
    regularity_metrics,
)
from .intersections import (
    apply_intersection_mutation,
    generate_mesh_with_intersections,
    generate_mesh_with_legacy_imprint,
    plan_intersection_mutation,
)
from .beam_connections import connect_beam_mesh, connect_shell_boundaries
from .serialize import load_mesh, mesh_from_dict, mesh_to_dict, save_mesh
from .automation import (
    MeshApplyResult,
    MeshAutomationSession,
    MeshCommand,
    MeshCommandBatch,
    MeshPlan,
)

__version__ = "0.3.1"

__all__ = [
    "ASPECT_RATIO_LIMIT",
    "Arc",
    "ArcFrame",
    "Coupling",
    "CertificationMode",
    "ChartProjection",
    "CurveShape",
    "DEFAULT_S3_QUALITY_POLICY",
    "DEFAULT_S3_REPAIR_POLICY",
    "DEFAULT_BACKEND",
    "DegenerateArcError",
    "ELEMENT_ORDERS",
    "Edge",
    "EntityKind",
    "EntityRef",
    "Face",
    "FaceChart",
    "GeometryError",
    "GeometryModel",
    "HybridMeshResult",
    "MappabilityReport",
    "Mesh",
    "MeshApplyResult",
    "MeshAutomationSession",
    "MeshBackend",
    "MeshCommand",
    "MeshCommandBatch",
    "MeshError",
    "MeshPlan",
    "MeshQuality",
    "MeshQualityPolicy",
    "MetricField",
    "MeshingStrategy",
    "OrientedEdge",
    "EdgeFlipResult",
    "PANEL_EDGE_IDS",
    "PANEL_FACE_ID",
    "PanelMeshConfig",
    "Refinement",
    "S3AdmissionReport",
    "S3ElementQuality",
    "S3QualityError",
    "S3QualityPolicy",
    "S3_QUALITY_CONTRACT_ID",
    "S3_REPAIR_CONTRACT_ID",
    "S3RepairAttempt",
    "S3RepairError",
    "S3RepairPolicy",
    "S3RepairResult",
    "STIFFENER_EDGE_ID_BASE",
    "Seeding",
    "SeedingConflict",
    "SeedEquation",
    "SmoothingResult",
    "SizeField",
    "StiffenedPanel",
    "Spline",
    "StiffenerCrossSection",
    "Straight",
    "StructurePreference",
    "StructuredBlock",
    "StructuredFacePlan",
    "StructuredInterface",
    "StructuredLayoutPlan",
    "StructuredLayoutReport",
    "StructuredMeshingOptions",
    "StructuredShellGrid",
    "StructuralPreparationOptions",
    "StructuralPreparationReport",
    "Vertex",
    "WARP_LIMIT",
    "arc_frame",
    "available_backends",
    "assert_s3_admissible",
    "apply_intersection_mutation",
    "apply_structured_layout",
    "beam_mesh",
    "build_structured_shell_grid",
    "chain_breaks",
    "chain_point",
    "check_mappable",
    "connect_beam_mesh",
    "connect_shell_boundaries",
    "commit_structured_layout",
    "coons_grid",
    "constrained_smoothing",
    "edge_demand",
    "edge_distribution",
    "evaluate_s3_admission",
    "generate_mesh",
    "generate_hybrid_mesh",
    "generate_hybrid_mesh_result",
    "load_mesh",
    "locate_shell_element_at_xy",
    "local_edge_flip",
    "mesh_from_dict",
    "mesh_to_dict",
    "metric_edge_lengths",
    "metric_tensors",
    "nodal_normals",
    "panel_edge_nodes",
    "punch_circular_hole",
    "plan_intersection_mutation",
    "plan_structured_layout",
    "prepare_structural_closure",
    "refine_around",
    "refine_at",
    "repair_s3_admission",
    "resolve_backend",
    "regularity_metrics",
    "sample_chain",
    "save_mesh",
    "shape_functions_4node",
    "shape_functions_8node",
    "simple_panel_mesh",
    "solve_seeding",
    "generate_mesh_with_intersections",
    "generate_mesh_with_legacy_imprint",
    "split_face_at",
    "split_face_between",
    "stiffened_panel_mesh",
    "strip_face",
    "surface_point",
    "triangle_to_quads",
    "verify_mesh_quality",
]
