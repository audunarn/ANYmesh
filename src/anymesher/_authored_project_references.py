"""Private consumer of ANYfem's preparation-bound reference visibility.

This does not qualify child/cell membership or activate authored-root meshing.
Root-only association is refused when the project has a child-local reference.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ._authored_scope_binding import BoundAuthoredRootInputs
from ._authored_associations import AuthoredRootAssociations
from .boundary import GlobalEdgeBoundaryRegistry
from .errors import MeshError


@dataclass(frozen=True)
class BoundAuthoredProjectReferences:
    authored_face: int
    source_namespace: str
    source_face: int
    descendants: tuple[int, ...]
    required_boundary_edge_ids: tuple[int, ...]
    required_vertex_ids: tuple[int, ...]
    manifest: object

    @property
    def publication_qualified(self) -> bool:
        return False


@dataclass(frozen=True)
class AuthoredChildMembershipEvidence:
    authored_face: int
    selected_children: tuple[int, ...]
    cell_children: tuple[tuple[int, int], ...]
    required_boundary_edge_ids: tuple[int, ...]

    @property
    def publication_qualified(self) -> bool:
        return False


def bind_authored_project_references(
    project, prepared_geometry, bound: BoundAuthoredRootInputs, manifest,
    *, closure=None,
) -> BoundAuthoredProjectReferences:
    """Require complete fresh input scope before planning root associations."""
    try:
        from anyfem.prepared_reference_scope import (
            assert_authored_root_face_scope,
            validate_prepared_project_reference_scope,
        )
        from anygeometry import validate_prepared_model_scope_binding
    except ImportError as error:
        raise MeshError("prepared project reference capability is unavailable") from error
    if not isinstance(bound, BoundAuthoredRootInputs):
        raise MeshError("authored project references need a bound owner root")
    validate_prepared_project_reference_scope(
        project, prepared_geometry, manifest, closure=closure,
    )
    validate_prepared_model_scope_binding(prepared_geometry, bound.scope)
    if (bound.scope.face_preimages != manifest.owner_scope.face_preimages
            or bound.scope.authored_document != manifest.authored_document
            or bound.scope.current_document != manifest.current_document):
        raise MeshError("owner and project references bind different preparations")
    descendants = dict(manifest.owner_scope.face_preimages.face_descendants).get(
        bound.authored_face
    )
    if descendants is None or tuple(descendants) != bound.descendants:
        raise MeshError("project references do not cover the complete authored root")
    assert_authored_root_face_scope(
        manifest, bound.authored_face, bound.descendants,
    )
    source_rows = tuple(
        row for row in manifest.authored_face_sources if row[0] == bound.authored_face
    )
    if len(source_rows) != 1:
        raise MeshError("authored root lacks one explicit project source identity")
    _root, namespace, source_face = source_rows[0]
    validate_prepared_project_reference_scope(
        project, prepared_geometry, manifest, closure=closure,
    )
    return BoundAuthoredProjectReferences(
        bound.authored_face, str(namespace), int(source_face), bound.descendants,
        tuple(manifest.required_boundary_edge_ids),
        tuple(manifest.required_vertex_ids), manifest,
    )


def validate_required_project_constraints(
    project, prepared_geometry, binding: BoundAuthoredProjectReferences,
    mesh, boundary_registry: GlobalEdgeBoundaryRegistry, *, closure=None,
) -> None:
    """Check exact retained edge chains and vertices, not inferred proximity."""
    try:
        from anyfem.prepared_reference_scope import validate_prepared_project_reference_scope
    except ImportError as error:
        raise MeshError("prepared project reference capability is unavailable") from error
    if not isinstance(binding, BoundAuthoredProjectReferences):
        raise MeshError("required project constraints need a bound manifest")
    validate_prepared_project_reference_scope(
        project, prepared_geometry, binding.manifest, closure=closure,
    )




    if (binding.required_boundary_edge_ids != binding.manifest.required_boundary_edge_ids
            or binding.required_vertex_ids != binding.manifest.required_vertex_ids
            or tuple(dict(binding.manifest.owner_scope.face_preimages.face_descendants)
                     .get(binding.authored_face, ())) != binding.descendants
            or (binding.authored_face, binding.source_namespace, binding.source_face)
            not in binding.manifest.authored_face_sources):
        raise MeshError("required project constraint binding was altered")
    if not isinstance(boundary_registry, GlobalEdgeBoundaryRegistry):
        raise MeshError("required project constraints need a boundary registry")
    boundary_registry.view.assert_current(prepared_geometry)
    shell_segments: set[tuple[int, int]] = set()
    active_nodes: set[int] = set()
    for family, corners in ((mesh.tris, 3), (mesh.quads, 4)):
        for connection in family.values():
            if len(connection) not in (corners, 2 * corners):
                raise MeshError("required project constraints found invalid shell connectivity")
            active_nodes.update(int(node) for node in connection)
            for index in range(corners):
                first, second = int(connection[index]), int(connection[(index + 1) % corners])
                path = (first, second) if len(connection) == corners else (
                    first, int(connection[corners + index]), second,
                )
                for a, b in zip(path, path[1:]):
                    shell_segments.add((min(a, b), max(a, b)))
    for edge_id in binding.required_boundary_edge_ids:
        stations = boundary_registry.entries(edge_id)
        if (len(stations) < 2 or stations[0].key.parameter != 0.0
                or stations[-1].key.parameter != 1.0):
            raise MeshError(f"required project edge {edge_id} lacks complete stations")
        node_ids = tuple(entry.node_id for entry in stations)
        if any(node is None or node not in mesh.nodes for node in node_ids):
            raise MeshError(f"required project edge {edge_id} has an unpublished node")
        tolerance = boundary_registry.view.effective_length(
            boundary_registry.view.edge_length(edge_id)
        )
        for station in stations:
            xyz = np.asarray(mesh.nodes[station.node_id], dtype=float)
            if (xyz.shape != (3,) or not np.all(np.isfinite(xyz))
                    or np.linalg.norm(xyz - station.point) > tolerance):
                raise MeshError(f"required project edge {edge_id} has altered node coordinates")
        if any((min(a, b), max(a, b)) not in shell_segments
               for a, b in zip(node_ids, node_ids[1:])):
            raise MeshError(f"required project edge {edge_id} is not a conforming shell chain")
    for vertex_id in binding.required_vertex_ids:
        node_id = mesh.node_of_vertex.get(vertex_id)
        if node_id is None or node_id not in active_nodes or node_id not in mesh.nodes:
            raise MeshError(f"required project vertex {vertex_id} lacks an active node")
        xyz = np.asarray(mesh.nodes[node_id], dtype=float)
        if (xyz.shape != (3,) or not np.all(np.isfinite(xyz))
                or np.linalg.norm(xyz - prepared_geometry.vertex_position(vertex_id))
                > boundary_registry.view.effective_length()):
            raise MeshError(f"required project vertex {vertex_id} has altered coordinates")
    validate_prepared_project_reference_scope(
        project, prepared_geometry, binding.manifest, closure=closure,
    )


def validate_authored_project_stage(
    project, prepared_geometry, bound: BoundAuthoredRootInputs, manifest,
    associations: AuthoredRootAssociations, mesh, boundary_registry,
    *, closure=None,
) -> None:
    """Preflight source scope immediately before a separate full mesh gate.

    This returns no admission token. A production caller must separately prove
    coverage, exact child membership where needed, mapping, quality and work
    limits before using an atomic publication holder.
    """
    try:
        from anyfem.prepared_reference_scope import validate_prepared_project_reference_scope
        from anygeometry import validate_prepared_model_scope_binding
    except ImportError as error:
        raise MeshError("prepared project reference capability is unavailable") from error
    reference = bind_authored_project_references(
        project, prepared_geometry, bound, manifest, closure=closure,
    )
    if (reference.source_namespace != "project_authored"
            or reference.source_face != bound.authored_face):
        raise MeshError("detached authored root needs a qualified project output remap")
    if not isinstance(associations, AuthoredRootAssociations):
        raise MeshError("authored project stage needs source association evidence")
    original_uses = tuple(sorted(
        (int(use["id"]), int(use["sheet_id"]), str(use["orientation"]))
        for use in manifest.authored_document["structural"]["face_uses"]
        if int(use["face_id"]) == bound.authored_face
    ))
    ids = tuple(sorted(set(mesh.tris) | set(mesh.quads)))
    expected_sheets = {
        sheet_id: list(ids) for sheet_id in associations.sheet_ids
    }
    if (associations.source_face != bound.authored_face
            or associations.element_ids != ids or not ids
            or associations.face_uses != original_uses
            or mesh.beams or mesh.couplings
            or mesh.elements_of_face != {bound.authored_face: list(ids)}
            or mesh.elements_of_sheet != expected_sheets):
        raise MeshError("authored project stage lacks exact root-local source associations")
    validate_required_project_constraints(
        project, prepared_geometry, reference, mesh, boundary_registry,
        closure=closure,
    )
    validate_prepared_model_scope_binding(prepared_geometry, bound.scope)
    validate_prepared_project_reference_scope(
        project, prepared_geometry, manifest, closure=closure,
    )


def validate_authored_child_project_cells(
    project, prepared_geometry, bound: BoundAuthoredRootInputs, manifest,
    correspondence, mesh, boundary_registry, element_to_child,
    node_authored_uv, *, closure=None, cancellation_check=None,
) -> AuthoredChildMembershipEvidence:
    """Prove each declared linear cell fits its literal selected child.

    The caller supplies original UV from its chart; XYZ is checked against the
    owner support. This proves neither child-area completeness nor a global
    partition, and returns no mesh-publication authority.
    """
    try:
        from anyfem.prepared_reference_scope import validate_prepared_project_reference_scope
        from anygeometry import (
            evaluate_prepared_authored_face,
            validate_prepared_authored_boundary_correspondence_binding,
            validate_prepared_authored_face_child_triangles,
            validate_prepared_model_scope_binding,
        )
    except ImportError as error:
        raise MeshError("authored child coverage capability is unavailable") from error
    if not isinstance(bound, BoundAuthoredRootInputs):
        raise MeshError("authored child cells need a bound owner root")
    validate_prepared_project_reference_scope(
        project, prepared_geometry, manifest, closure=closure,
    )
    validate_prepared_model_scope_binding(prepared_geometry, bound.scope)
    validate_prepared_authored_boundary_correspondence_binding(
        prepared_geometry, correspondence, cancellation_check=cancellation_check,
    )
    if (bound.scope.face_preimages != manifest.owner_scope.face_preimages
            or bound.scope.authored_document != manifest.authored_document
            or bound.scope.current_document != manifest.current_document
            or int(correspondence.authored_definition.face_id) != bound.authored_face
            or tuple(correspondence.descendants) != bound.descendants):
        raise MeshError("child cells and project references bind different preparations")
    selected = set()
    for reference in manifest.references:
        if reference.authority == "derived_cache":
            continue
        for scope in reference.face_scopes:
            if scope.authored_face_id == bound.authored_face and not scope.whole_authored_root:
                selected.update(scope.current_face_ids)
    if not selected or not selected.issubset(set(bound.descendants)):
        raise MeshError("authored child cells need an explicit selected-child reference")
    if mesh.order != "linear" or mesh.beams or mesh.couplings:
        raise MeshError("authored child proof currently requires linear shell cells only")
    if set(mesh.tris) & set(mesh.quads):
        raise MeshError("authored child proof found reused shell element IDs")
    shells = {**mesh.tris, **mesh.quads}
    if (not shells or set(shells) != set(element_to_child)
            or any(isinstance(cell, bool) or not isinstance(cell, Integral)
                   for cell in element_to_child)
            or any(isinstance(child, bool) or not isinstance(child, Integral)
                   or int(child) not in bound.descendants
                   for child in element_to_child.values())):
        raise MeshError("authored child cells need one declared descendant per shell cell")
    if not selected.issubset({int(child) for child in element_to_child.values()}):
        raise MeshError("authored child reference has no declared shell cells")
    if mesh.elements_of_face != {bound.authored_face: sorted(shells)}:
        raise MeshError("authored child cells must retain their original-root owner")
    original_uses = (
        use for use in manifest.authored_document["structural"]["face_uses"]
        if int(use["face_id"]) == bound.authored_face
    )
    expected_sheets = {
        int(use["sheet_id"]): sorted(shells) for use in original_uses
    }
    if mesh.elements_of_sheet != expected_sheets:
        raise MeshError("authored child cells changed original Sheet ownership")
    corners = set()
    for cell_id, connection in shells.items():
        expected = 3 if cell_id in mesh.tris else 4
        if len(connection) != expected or len(set(connection)) != expected:
            raise MeshError("authored child proof needs valid linear corner connectivity")
        corners.update(int(node) for node in connection)
    if set(node_authored_uv) != corners or not corners.issubset(mesh.nodes):
        raise MeshError("authored child proof needs original UV for every active corner")
    ordered = tuple(sorted(corners))
    try:
        uv = np.asarray([node_authored_uv[node] for node in ordered], dtype=float)
    except (TypeError, ValueError) as error:
        raise MeshError("authored child UV rows are invalid") from error
    if uv.shape != (len(ordered), 2) or not np.all(np.isfinite(uv)):
        raise MeshError("authored child UV rows must be finite pairs")
    actual = np.asarray([mesh.nodes[node] for node in ordered], dtype=float)
    if actual.shape != (len(ordered), 3) or not np.all(np.isfinite(actual)):
        raise MeshError("authored child mesh coordinates are invalid")
    support = evaluate_prepared_authored_face(
        prepared_geometry, correspondence, uv,
        cancellation_check=cancellation_check,
    )
    extent = max(float(np.ptp(actual, axis=0).max()), 1.0)
    if np.max(np.linalg.norm(actual - support, axis=1)) > 1e-10 * extent:
        raise MeshError("authored child UV does not support the preserved node XYZ")
    by_node = dict(zip(ordered, uv))
    triangles_by_child: dict[int, list[np.ndarray]] = {}
    for cell_id, connection in sorted(shells.items()):
        points = np.asarray([by_node[int(node)] for node in connection], dtype=float)
        if len(connection) == 3:
            triangles = (points,)
        else:
            crosses = [float(np.linalg.det(np.stack((points[(i + 1) % 4] - points[i],
                                                     points[(i + 2) % 4] - points[(i + 1) % 4]))))
                       for i in range(4)]
            if (any(value == 0.0 for value in crosses)
                    or min(crosses) < 0 < max(crosses)):
                raise MeshError("authored child quad has no convex original-UV cover")
            triangles = (points[[0, 1, 2]], points[[0, 2, 3]])
        by_child = triangles_by_child.setdefault(int(element_to_child[cell_id]), [])
        by_child.extend(triangles)
    for child, triangles in sorted(triangles_by_child.items()):
        validate_prepared_authored_face_child_triangles(
            prepared_geometry, correspondence, child, np.asarray(triangles),
            cancellation_check=cancellation_check,
        )
    source_rows = tuple(row for row in manifest.authored_face_sources
                        if row[0] == bound.authored_face)
    if len(source_rows) != 1:
        raise MeshError("child cells lack one explicit project source identity")
    if (source_rows[0][1] != "project_authored"
            or int(source_rows[0][2]) != bound.authored_face):
        raise MeshError("detached authored child cells need a qualified project output remap")
    reference = BoundAuthoredProjectReferences(
        bound.authored_face, str(source_rows[0][1]), int(source_rows[0][2]),
        bound.descendants, tuple(manifest.required_boundary_edge_ids),
        tuple(manifest.required_vertex_ids), manifest,
    )
    validate_required_project_constraints(
        project, prepared_geometry, reference, mesh, boundary_registry,
        closure=closure,
    )
    validate_prepared_model_scope_binding(prepared_geometry, bound.scope)
    validate_prepared_project_reference_scope(
        project, prepared_geometry, manifest, closure=closure,
    )
    return AuthoredChildMembershipEvidence(
        bound.authored_face, tuple(sorted(selected)),
        tuple(sorted((int(cell), int(child)) for cell, child in element_to_child.items())),
        tuple(manifest.required_boundary_edge_ids),
    )
