"""Portable, detached authored two-Sheet fixture for cross-package development.

Invoke with ANYmesh ``src`` and the exact ANYgeometry e509862 wheel on
``PYTHONPATH``. This builds live owner/registry/receipt objects; it does not
publish a mesh or assert source-reference transfer or solver acceptance.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from typing import Callable

import anygeometry as owner
import numpy as np

from anymesher._authored_component_binding import bind_authored_sheet_joint_component
from anymesher._authored_route_boundary import (
    bind_authored_root_triangles_to_children,
    plan_authored_component_boundaries,
    triangulate_authored_root_boundary,
)
from anymesher._authored_staged_pair import (
    prepare_authored_component_boundary_stations,
    stage_authored_root_pair,
)
from anymesher.boundary import GlobalEdgeBoundaryRegistry
from anymesher.mesh import Mesh
from anymesher.meshing_view import GeometryMeshingView
from anymesher.native_v2 import ComponentSeedRegistry
from anymesher.s3_quality import evaluate_s3_admission


FIXTURE_SCHEMA = "anymesher.detached-authored-pair-handoff-v1"
GEOMETRY_COMMIT = "e509862bfb7f957d4200d090b05207d24f55ca82"
GEOMETRY_WHEEL_SHA256 = (
    "2761f6d62611ea62b69f80d6ec47659c4cfa1142d3ac07ed460e54fc191165ed"
)
STAGED_MESHER_BASE = "969c106b7ed94c60890b5a64b0dba9ee6b21a7c4"


@dataclass(frozen=True)
class AuthoredPairFixture:
    variant: str
    geometry: object
    component: object
    source_mesh: Mesh
    source_registry: GlobalEdgeBoundaryRegistry
    source_seeds: ComponentSeedRegistry
    packets: tuple
    triangulations: tuple
    child_bindings: tuple
    staged: object
    s3_admission: object

    @property
    def manifest(self) -> dict:
        return {
            "schema": FIXTURE_SCHEMA,
            "variant": self.variant,
            "geometry_commit": GEOMETRY_COMMIT,
            "geometry_wheel_sha256": GEOMETRY_WHEEL_SHA256,
            "geometry_import": owner.__file__,
            "mesher_base": STAGED_MESHER_BASE,
            "source_model_id": str(self.geometry.model_id),
            "source_revision": self.geometry.revision,
            "original_roots": list(self.component.authored_face_ids),
            "current_faces": list(self.component.current_face_ids),
            "sheets": list(self.component.sheet_ids),
            "nodes": len(self.staged.mesh.nodes),
            "triangles": len(self.staged.mesh.tris),
            "current_receipt_schema": self.staged.current_receipt.schema,
            "s3_shape_admitted": self.s3_admission.admitted,
            "s3_failing_cells": sum(
                not element.admitted for element in self.s3_admission.elements
            ),
            "source_reference_transfer_qualified": False,
            "solver_admitted": False,
            "publication_qualified": False,
        }


def _default_source_factory():
    geometry = owner.GeometryModel()
    root = geometry.add_plate(geometry.add_points(
        ((0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0))
    ))
    geometry.set_face_surface(
        root, owner.Plane((0, 0, 0), (1, 0, 0), (0, 1, 0))
    )
    cutter = geometry.add_plate(geometry.add_points(
        ((3, -1, -1), (3, 5, -1), (3, 5, 1), (3, -1, 1))
    ))
    geometry.add_sheet((root,), name="source root")
    geometry.add_sheet((cutter,), name="source cutter")
    return geometry, (root, cutter)


def _source_component(source_factory):
    """Call a process-bound factory before any owner intersection preparation.

    The factory owns an isolated, live *working* GeometryModel and returns
    its two mapped authored face IDs. Preparation mutates that same object.
    The caller must supply its witnessed live extraction, not rebuild a
    snapshot; this helper never reconstructs source state.
    """
    supplied = source_factory()
    if (type(supplied) is not tuple or len(supplied) != 2
            or type(supplied[0]) is not owner.GeometryModel
            or type(supplied[1]) is not tuple or len(supplied[1]) != 2
            or any(type(root) is not int for root in supplied[1])):
        raise ValueError("source_factory must return live GeometryModel and two root IDs")
    geometry, roots = supplied
    if len(set(roots)) != 2 or set(geometry.faces) != set(roots):
        raise ValueError("source_factory needs exactly two unprepared authored faces")
    sheet_by_root = {}
    for sheet_id, sheet in geometry.sheets.items():
        for use_id in sheet.face_use_ids:
            face = geometry.face_uses[use_id].face_id
            if face in roots:
                if face in sheet_by_root:
                    raise ValueError("source_factory repeats declared Sheet ownership")
                sheet_by_root[face] = sheet_id
    if set(sheet_by_root) != set(roots) or len(set(sheet_by_root.values())) != 2:
        raise ValueError("source_factory needs separate declared Sheets for both roots")
    owner.apply_intersections(
        geometry,
        owner.plan_intersections(geometry, roots, policy="connect"),
        policy="connect",
    )
    correspondences = tuple(
        owner.query_prepared_authored_boundary_correspondence(geometry, face)
        for face in roots
    )
    joint = correspondences[0].interior_incidence[0][0]
    component = bind_authored_sheet_joint_component(
        geometry, joint, correspondences, roots
    )
    if {(root, sheet) for sheet, root, _use, _current in
            component.occurrence_correspondence} != set(sheet_by_root.items()):
        raise ValueError("prepared component changed declared Sheet ownership")
    return geometry, component, correspondences


def _source_state(geometry, component, correspondences):
    mesh = Mesh(
        geometry_model_id=geometry.model_id,
        geometry_revision=geometry.revision,
        nodes={vertex: np.asarray(geometry.vertex_position(vertex), dtype=float)
               for vertex in geometry.vertices},
    )
    registry = GlobalEdgeBoundaryRegistry(GeometryMeshingView(geometry))
    edges = {
        edge for correspondence in correspondences
        for loop in correspondence.exterior_loops
        for _source, _forward, current in loop for edge in current
    } | {
        edge for correspondence in correspondences
        for edge, _uses in correspondence.interior_incidence
    }
    for edge in sorted(edges):
        current = geometry.edges[edge]
        registry.register(edge, 0.0, node_id=current.start)
        registry.register(edge, 1.0, node_id=current.end)
        mesh.nodes_of_edge[edge] = [current.start, current.end]
    mesh.node_of_vertex = {
        row["id"]: row["id"]
        for row in component.owner_receipt.current_records["vertices"]
    }
    cell_faces = {}
    next_cell = 100
    for face in component.current_face_ids:
        vertices = tuple(
            geometry.oriented_start_vertex(use) for use in geometry.faces[face].loop
        )
        if len(vertices) != 4:
            raise ValueError("portable source fixture changed its four-corner children")
        mesh.elements_of_face[face] = []
        for triangle in ((vertices[0], vertices[1], vertices[2]),
                         (vertices[0], vertices[2], vertices[3])):
            mesh.tris[next_cell] = triangle
            mesh.elements_of_face[face].append(next_cell)
            cell_faces[next_cell] = face
            next_cell += 1
    uses = {
        row["id"]: row for row in component.owner_receipt.current_records["face_uses"]
    }
    sheets = defaultdict(set)
    for sheet, _root, _original_use, current_uses in component.occurrence_correspondence:
        for use in current_uses:
            sheets[sheet].update(mesh.elements_of_face[uses[use]["face_id"]])
    mesh.elements_of_sheet = {sheet: sorted(cells) for sheet, cells in sheets.items()}
    mesh.declared_plate_junction_edges = tuple(sorted({
        (min(a, b), max(a, b))
        for edge in component.owner_receipt.joint_edge_ids
        for a, b in zip(mesh.nodes_of_edge[edge], mesh.nodes_of_edge[edge][1:])
    }))
    return mesh, registry, ComponentSeedRegistry(max(mesh.nodes) + 1)


def build_fixture(
    variant: str = "size2", *,
    source_factory: Callable[[], tuple[object, tuple[int, int]]] | None = None,
) -> AuthoredPairFixture:
    """Build ``legacy14``, corrected ``size2`` or created-UV ``interior``.

    ``source_factory`` runs once before ``plan_intersections``. ANYfem may use
    a closure over its witnessed Project extraction to return its isolated
    working GeometryModel and mapped root IDs, with separate declared Sheets.
    The returned fixture retains that exact live geometry and its receipt.
    """
    if variant not in ("legacy14", "size2", "interior"):
        raise ValueError("fixture variant must be legacy14, size2 or interior")
    geometry, component, correspondences = _source_component(
        _default_source_factory if source_factory is None else source_factory
    )
    mesh, registry, seeds = _source_state(geometry, component, correspondences)
    if variant == "size2":
        prepared = prepare_authored_component_boundary_stations(
            geometry, component, mesh, registry, seeds, target_size=2.0,
        )
        mesh, registry, seeds = prepared.mesh, prepared.registry, prepared.seeds
    packets = plan_authored_component_boundaries(geometry, component, mesh, registry)
    if variant == "interior":
        roots = tuple(triangulate_authored_root_boundary(
            packet,
            interior_metric=packet.chart.to_metric(np.asarray(((0.231, 0.317),))),
        ) for packet in packets)
    else:
        roots = tuple(triangulate_authored_root_boundary(packet) for packet in packets)
    children = tuple(
        bind_authored_root_triangles_to_children(packet, result, correspondence)
        for packet, result, correspondence in zip(
            packets, roots, component.boundary_correspondences
        )
    )
    staged = stage_authored_root_pair(
        geometry, component, mesh, registry, seeds, packets, roots, children,
    )
    normals = {
        cell: geometry.face_normal(face, 0.5, 0.5)
        for cell, face in staged.cell_current_faces
    }
    admission = evaluate_s3_admission(staged.mesh, element_owner_normals=normals)
    return AuthoredPairFixture(
        variant, geometry, component, mesh, registry, seeds,
        packets, roots, children, staged, admission,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("legacy14", "size2", "interior"))
    args = parser.parse_args()
    print(json.dumps(build_fixture(args.variant).manifest, sort_keys=True, indent=2))
