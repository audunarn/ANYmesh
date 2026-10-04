"""Versioned current-only association receipt for a narrow authored component.

The caller supplies an already staged, linear Plane/Straight shell mesh of the
*complete* qualified Sheet-joint component. This API proves its current
associations and literal cell partition. It does not generate a mesh, remap
project source references, admit solver use, or authorize publication.
"""

from dataclasses import dataclass

from ._authored_component_cells import validate_authored_component_cells
from ._authored_component_stage import _mesh_digest
from .errors import MeshError
from .mesh import Mesh


PREPARED_CURRENT_ASSOCIATIONS_SCHEMA = "anymesher.prepared-current-associations-v1"
PREPARED_CURRENT_ASSOCIATIONS_CREATED_UV_SCHEMA = "anymesher.prepared-current-associations-v2"

_KNOWN_MESH_FIELDS = (
    "geometry_model_id", "geometry_revision", "nodes", "quads", "tris", "beams",
    "node_of_vertex", "nodes_of_edge", "offset_nodes_of_edge", "couplings",
    "grid_of_face", "block_grids_of_face", "elements_of_face", "elements_of_edge",
    "elements_of_sheet", "elements_of_member", "nodes_of_member", "activity",
    "thickness_of_face", "seeding", "order", "automatic_intersections",
    "declared_plate_junction_edges", "automatic_beam_connections",
    "automatic_shell_connections", "structural_preparation", "hybrid_diagnostics",
)


@dataclass(frozen=True)
class PreparedCurrentAssociationReceipt:
    schema: str
    association_namespace: str
    geometry_model_id: object
    geometry_revision: int
    mesh_digest: str
    owner_component: object
    authored_face_ids: tuple[int, ...]
    current_face_ids: tuple[int, ...]
    sheet_ids: tuple[int, ...]
    occurrence_correspondence: tuple
    cell_current_faces: tuple[tuple[int, int], ...]
    elements_of_face: tuple[tuple[int, tuple[int, ...]], ...]
    elements_of_sheet: tuple[tuple[int, tuple[int, ...]], ...]
    elements_of_face_use: tuple[tuple[int, tuple[int, ...]], ...]
    nodes_of_edge: tuple[tuple[int, tuple[int, ...]], ...]
    node_of_vertex: tuple[tuple[int, int], ...]
    joint_chains: tuple[tuple[int, tuple[int, ...]], ...]
    # Original loop order is owner-backed; each current edge's station order
    # is reversed only when that original edge use is reversed.
    source_boundary_chains: tuple[tuple[int, int, bool, tuple[int, ...], tuple[int, ...]], ...]
    # Current edge may have multiple original exterior ancestors. Absence here
    # is NOT a generated-only or no-source-reference assertion.
    exterior_source_edge_ancestry: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    # Empty original VERTEX ancestry does not exclude source-edge references.
    current_vertex_preimages: tuple[tuple[int, tuple[int, ...]], ...]
    offset_nodes_of_edge: tuple = ()
    elements_of_edge: tuple = ()
    elements_of_member: tuple = ()
    nodes_of_member: tuple = ()
    grid_of_face: tuple = ()
    block_grids_of_face: tuple = ()
    thickness_of_face: tuple = ()
    seeding: None = None
    activity: tuple = ()
    diagnostics: tuple = ()
    source_reference_transfer_qualified: bool = False
    solver_admitted: bool = False
    publication_qualified: bool = False
    # Version 2 only: exact original-chart UV for genuinely created nodes.
    created_material_uv_by_root: tuple = ()


def query_prepared_current_component_associations(
    geometry, component, mesh: Mesh, registry, cell_current_faces, *,
    created_material_uv_by_root=None,
    cancellation_check=None,
) -> PreparedCurrentAssociationReceipt:
    """Reprove every supported current association and return a detached record.

    Unsupported/nonempty Mesh fields refuse in the underlying linear proof;
    adding a new Mesh field also fails closed until this schema is reviewed.
    """
    try:
        from anygeometry import (
            query_prepared_vertex_preimages,
            validate_prepared_vertex_preimages_binding,
        )
    except ImportError as error:
        raise MeshError("prepared current vertex ancestry capability is unavailable") from error
    if type(mesh) is not Mesh or tuple(Mesh.__dataclass_fields__) != _KNOWN_MESH_FIELDS:
        raise MeshError("prepared current association Mesh schema changed")
    before = _mesh_digest(mesh)
    proven = validate_authored_component_cells(
        geometry, component, mesh, registry, cell_current_faces,
        created_material_uv_by_root=created_material_uv_by_root,
        cancellation_check=cancellation_check,
    )
    created_uv = ()
    if created_material_uv_by_root is not None:
        created_uv = tuple(
            (int(root), tuple(sorted((int(node), tuple(pair))
                                    for node, pair in values.items())))
            for root, values in sorted(created_material_uv_by_root.items())
        )
        if not any(values for _root, values in created_uv):
            created_uv = ()
    vertices = query_prepared_vertex_preimages(
        geometry, cancellation_check=cancellation_check,
    )
    validate_prepared_vertex_preimages_binding(
        geometry, vertices, cancellation_check=cancellation_check,
    )
    current_to_authored = dict(vertices.current_to_authored)
    vertex_preimages = tuple(
        (vertex, tuple(current_to_authored[vertex]))
        for vertex, _node in proven.vertex_nodes
    )
    ancestry = {}
    edge_nodes = dict(proven.edge_chains)
    source_chains = []
    for correspondence in component.boundary_correspondences:
        root = int(correspondence.authored_definition.face_id)
        for loop in correspondence.exterior_loops:
            for original_edge, forward, current_edges in loop:
                nodes = []
                for edge in current_edges:
                    ancestry.setdefault(int(edge), set()).add((root, int(original_edge)))
                    segment = edge_nodes[int(edge)]
                    directed = segment if forward else segment[::-1]
                    if nodes and nodes[-1] != directed[0]:
                        raise MeshError("prepared source edge chain lost owner order")
                    nodes.extend(directed if not nodes else directed[1:])
                if len(nodes) < 2:
                    raise MeshError("prepared source edge lacks an ordered chain")
                source_chains.append((root, int(original_edge), bool(forward),
                                      tuple(int(edge) for edge in current_edges),
                                      tuple(nodes)))
    edge_ancestry = tuple(
        (edge, tuple(sorted(ancestry.get(edge, ()))))
        for edge, _chain in proven.edge_chains
    )
    after = _mesh_digest(mesh)
    if after != before:
        raise MeshError("prepared current association mesh changed during proof")
    return PreparedCurrentAssociationReceipt(
        (PREPARED_CURRENT_ASSOCIATIONS_CREATED_UV_SCHEMA if created_uv
         else PREPARED_CURRENT_ASSOCIATIONS_SCHEMA), "prepared_current",
        geometry.model_id, geometry.revision, before, component.owner_receipt,
        tuple(component.authored_face_ids), tuple(component.current_face_ids),
        tuple(component.sheet_ids), tuple(component.occurrence_correspondence),
        proven.cell_current_faces, proven.current_face_cells, proven.sheet_cells,
        proven.face_use_cells, proven.edge_chains, proven.vertex_nodes,
        proven.joint_chains, tuple(source_chains), edge_ancestry, vertex_preimages,
        created_material_uv_by_root=created_uv,
    )


def validate_prepared_current_component_associations(
    geometry, receipt: PreparedCurrentAssociationReceipt, component, mesh,
    registry, cell_current_faces, *, cancellation_check=None,
) -> None:
    """Rederive complete contents; a digest alone never validates owner truth."""
    if type(receipt) is not PreparedCurrentAssociationReceipt:
        raise MeshError("prepared current associations need a versioned receipt")
    if receipt.schema not in (PREPARED_CURRENT_ASSOCIATIONS_SCHEMA,
                              PREPARED_CURRENT_ASSOCIATIONS_CREATED_UV_SCHEMA):
        raise MeshError("prepared current associations have an unsupported schema")
    if (receipt.schema == PREPARED_CURRENT_ASSOCIATIONS_SCHEMA) != (
        receipt.created_material_uv_by_root == ()
    ):
        raise MeshError("prepared current association schema and created UV disagree")
    created = (None if not receipt.created_material_uv_by_root
               else {root: dict(values)
                     for root, values in receipt.created_material_uv_by_root})
    expected = query_prepared_current_component_associations(
        geometry, component, mesh, registry, cell_current_faces,
        created_material_uv_by_root=created,
        cancellation_check=cancellation_check,
    )
    if receipt != expected:
        raise MeshError("prepared current association contents or owner binding changed")
