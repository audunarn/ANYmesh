"""Exact association remapping after detached geometry preparation.

Working geometry may contain partition faces and split edges that never exist in
the editable source document.  A published mesh must therefore expose source
identity, while retaining every real block grid and every exact descendant
element.  The mappings accepted here come from geometry replacement lineage;
coordinates are used only to order nodes along an already-declared source edge,
never to choose an owner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar

import numpy as np

from anygeometry.model import GeometryModel

from .errors import MeshError
from .mesh import Mesh
from .seeding import Seeding

__all__ = ["remap_prepared_mesh_associations"]

_Value = TypeVar("_Value")


def _declared_descendants(
    mapping: Mapping[int, Sequence[int]],
    source_ids: Sequence[int],
    working_ids: Mapping[int, _Value],
    kind: str,
) -> dict[int, tuple[int, ...]]:
    source_set = set(int(item) for item in source_ids)
    unknown_sources = sorted(set(int(item) for item in mapping).difference(source_set))
    if unknown_sources:
        raise MeshError(
            f"prepared {kind} mapping names unknown source {kind} "
            f"{unknown_sources[0]}"
        )
    result: dict[int, tuple[int, ...]] = {}
    claimed: dict[int, int] = {}
    for source_id in sorted(source_set):
        descendants = tuple(
            dict.fromkeys(int(item) for item in mapping.get(source_id, (source_id,)))
        )
        if not descendants:
            raise MeshError(
                f"prepared source {kind} {source_id} has no exact descendants"
            )
        missing = [item for item in descendants if item not in working_ids]
        if missing:
            raise MeshError(
                f"prepared source {kind} {source_id} maps to missing working "
                f"{kind} {missing[0]}"
            )
        for descendant in descendants:
            previous = claimed.setdefault(descendant, source_id)
            if previous != source_id:
                raise MeshError(
                    f"working {kind} {descendant} is claimed by source {kind}s "
                    f"{previous} and {source_id}"
                )
        result[source_id] = descendants
    return result


def _ordered_edge_associations(
    mesh: Mesh,
    source: GeometryModel,
    source_edge_id: int,
    working_edge_ids: Sequence[int],
) -> tuple[list[int], list[int], list[int]]:
    entries: list[tuple[float, int, int | None]] = []
    for working_edge_id in working_edge_ids:
        nodes = tuple(mesh.nodes_of_edge.get(working_edge_id, ()))
        offsets = tuple(mesh.offset_nodes_of_edge.get(working_edge_id, ()))
        if offsets and len(offsets) != len(nodes):
            raise MeshError(
                f"working edge {working_edge_id} has {len(nodes)} plate nodes but "
                f"{len(offsets)} offset nodes"
            )
        for index, node_id in enumerate(nodes):
            if node_id not in mesh.nodes:
                raise MeshError(
                    f"working edge {working_edge_id} references missing node {node_id}"
                )
            _point, parameter, distance = source.closest_edge_point(
                source_edge_id, mesh.nodes[node_id]
            )
            length = source.edge_length(source_edge_id)
            tolerance = max(
                source.tolerance.effective_length(length),
                128.0 * np.finfo(float).eps * max(length, 1.0),
            )
            if distance > tolerance:
                raise MeshError(
                    f"working edge {working_edge_id} node {node_id} is not on exact "
                    f"source edge {source_edge_id} (residual {distance:.6g} m)"
                )
            entries.append(
                (
                    float(parameter),
                    int(node_id),
                    None if not offsets else int(offsets[index]),
                )
            )

    entries.sort(key=lambda item: (item[0], item[1]))
    ordered: list[tuple[float, int, int | None]] = []
    parameter_tolerance = source.tolerance.effective_parameter(
        source.edge_length(source_edge_id)
    )
    for entry in entries:
        if ordered and abs(entry[0] - ordered[-1][0]) <= parameter_tolerance:
            if entry[1] != ordered[-1][1]:
                raise MeshError(
                    f"source edge {source_edge_id} has duplicate coincident mesh "
                    f"nodes {ordered[-1][1]} and {entry[1]}; preparation is not conformal"
                )
            if entry[2] != ordered[-1][2]:
                raise MeshError(
                    f"source edge {source_edge_id} has conflicting offset-node identity"
                )
            continue
        ordered.append(entry)

    nodes = [item[1] for item in ordered]
    supplied_offsets = [item[2] for item in ordered if item[2] is not None]
    if supplied_offsets and len(supplied_offsets) != len(nodes):
        raise MeshError(
            f"source edge {source_edge_id} is only partially offset after preparation"
        )
    offsets = [int(item) for item in supplied_offsets]
    elements = sorted(
        {
            int(element)
            for edge_id in working_edge_ids
            for element in mesh.elements_of_edge.get(edge_id, ())
        }
    )
    return nodes, offsets, elements


def remap_prepared_mesh_associations(
    mesh: Mesh,
    source: GeometryModel,
    working: GeometryModel,
    *,
    source_to_working_faces: Mapping[int, Sequence[int]],
    source_to_working_edges: Mapping[int, Sequence[int]],
) -> Mesh:
    """Publish one working-closure mesh against exact source identities.

    The mesh is changed only after every association has validated, so a failed
    remap cannot leave a partly source-bound artifact.  ``source`` and
    ``working`` are never mutated.
    """

    faces = _declared_descendants(
        source_to_working_faces,
        tuple(source.faces),
        working.faces,
        "face",
    )
    edges = _declared_descendants(
        source_to_working_edges,
        tuple(source.edges),
        working.edges,
        "edge",
    )

    remapped_elements_of_face: dict[int, list[int]] = {}
    remapped_grid_of_face: dict[int, np.ndarray] = {}
    remapped_block_grids: dict[int, tuple[np.ndarray, ...]] = {}
    remapped_thickness: dict[int, float] = {}
    for source_face_id, working_faces in faces.items():
        elements = sorted(
            {
                int(element)
                for face_id in working_faces
                for element in mesh.elements_of_face.get(face_id, ())
            }
        )
        if elements:
            remapped_elements_of_face[source_face_id] = elements
        grids = tuple(
            np.asarray(mesh.grid_of_face[face_id], dtype=int).copy()
            for face_id in working_faces
            if face_id in mesh.grid_of_face
        )
        if len(grids) == 1:
            remapped_grid_of_face[source_face_id] = grids[0]
        elif grids:
            remapped_block_grids[source_face_id] = grids
        thicknesses = {
            float(mesh.thickness_of_face[face_id])
            for face_id in working_faces
            if face_id in mesh.thickness_of_face
        }
        if len(thicknesses) > 1:
            raise MeshError(
                f"working descendants of source face {source_face_id} have "
                "conflicting thickness associations"
            )
        if thicknesses:
            remapped_thickness[source_face_id] = thicknesses.pop()

    remapped_nodes_of_edge: dict[int, list[int]] = {}
    remapped_offsets: dict[int, list[int]] = {}
    remapped_elements_of_edge: dict[int, list[int]] = {}
    for source_edge_id, working_edges in edges.items():
        nodes, offsets, elements = _ordered_edge_associations(
            mesh, source, source_edge_id, working_edges
        )
        if nodes:
            remapped_nodes_of_edge[source_edge_id] = nodes
        if offsets:
            remapped_offsets[source_edge_id] = offsets
        if elements:
            remapped_elements_of_edge[source_edge_id] = elements

    node_of_vertex = {
        int(vertex_id): int(mesh.node_of_vertex[vertex_id])
        for vertex_id in source.vertices
        if vertex_id in mesh.node_of_vertex
    }
    elements_of_sheet: dict[int, list[int]] = {}
    for sheet_id in source.sheets:
        elements = set(map(int, mesh.elements_of_sheet.get(sheet_id, ())))
        working_sheet = working.sheets.get(sheet_id)
        if working_sheet is not None:
            elements.update(
                int(element)
                for face_use_id in working_sheet.face_use_ids
                for element in mesh.elements_of_face.get(
                    working.face_uses[face_use_id].face_id, ()
                )
            )
        if elements:
            elements_of_sheet[int(sheet_id)] = sorted(elements)

    elements_of_member: dict[int, list[int]] = {}
    nodes_of_member: dict[int, list[int]] = {}
    for member_id in source.members:
        elements = set(map(int, mesh.elements_of_member.get(member_id, ())))
        nodes = list(map(int, mesh.nodes_of_member.get(member_id, ())))
        working_member = working.members.get(member_id)
        if working_member is not None:
            member_edges = tuple(
                working.member_edge_uses[edge_use_id].edge_id
                for edge_use_id in working_member.edge_use_ids
            )
            elements.update(
                int(element)
                for edge_id in member_edges
                for element in mesh.elements_of_edge.get(edge_id, ())
            )
            for edge_id in member_edges:
                nodes.extend(map(int, mesh.nodes_of_edge.get(edge_id, ())))
                nodes.extend(map(int, mesh.offset_nodes_of_edge.get(edge_id, ())))
        if elements:
            elements_of_member[int(member_id)] = sorted(elements)
        if nodes:
            nodes_of_member[int(member_id)] = list(dict.fromkeys(nodes))

    steps = 2 if mesh.is_quadratic else 1
    divisions = {
        edge_id: (len(nodes) - 1) // steps
        for edge_id, nodes in remapped_nodes_of_edge.items()
        if len(nodes) >= steps + 1 and (len(nodes) - 1) % steps == 0
    }
    if any(value < 1 for value in divisions.values()):
        raise MeshError("prepared mesh produced an invalid zero-division source edge")
    remapped_seeding = (
        None
        if not divisions
        else Seeding(
            divisions=divisions,
            sweeps=0 if mesh.seeding is None else int(mesh.seeding.sweeps),
            classes={edge_id: edge_id for edge_id in divisions},
            size_field=None,
        )
    )

    # Commit the validated association set as one publication step.
    mesh.geometry_model_id = source.model_id
    mesh.geometry_revision = source.revision
    mesh.node_of_vertex = node_of_vertex
    mesh.nodes_of_edge = remapped_nodes_of_edge
    mesh.offset_nodes_of_edge = remapped_offsets
    mesh.grid_of_face = remapped_grid_of_face
    mesh.block_grids_of_face = remapped_block_grids
    mesh.elements_of_face = remapped_elements_of_face
    mesh.elements_of_edge = remapped_elements_of_edge
    mesh.elements_of_sheet = elements_of_sheet
    mesh.elements_of_member = elements_of_member
    mesh.nodes_of_member = nodes_of_member
    mesh.thickness_of_face = remapped_thickness
    mesh.seeding = remapped_seeding
    return mesh
