"""Detached whole-component authored T3 assembly; never publishes a mesh."""

from dataclasses import dataclass

import numpy as np

from ._authored_component_stage import (
    AuthoredComponentStage, _mesh_digest, _registry_receipt,
)
from ._authored_route_boundary import (
    AuthoredRootBoundaryPacket, AuthoredRootChildBinding,
    AuthoredRootTriangulation, _assert_packet_current,
    refine_authored_root_with_ledger,
)
from ._authored_work_ledger import AuthoredWorkLedger
from .boundary import GlobalEdgeBoundaryRegistry
from .core import MeshCore
from .errors import MeshError
from .mesh import Mesh
from .native_v2 import ComponentSeedRegistry, NativeMeshingOptions
from .prepared_current_associations import (
    PreparedCurrentAssociationReceipt,
    query_prepared_current_component_associations,
    validate_prepared_current_component_associations,
)
from .quality_v2 import assert_valid_mesh


@dataclass(frozen=True)
class DetachedAuthoredPair:
    mesh: Mesh
    core: MeshCore
    current_receipt: PreparedCurrentAssociationReceipt
    created_material_uv_by_root: tuple
    cell_current_faces: tuple[tuple[int, int], ...]

    @property
    def publication_qualified(self) -> bool:
        return False

    @property
    def solver_admitted(self) -> bool:
        return False


@dataclass(frozen=True)
class AuthoredPairRefinement:
    triangulations: tuple[AuthoredRootTriangulation, AuthoredRootTriangulation]
    ledger: AuthoredWorkLedger
    reports: tuple[dict, dict]

    @property
    def publication_qualified(self) -> bool:
        return False

    @property
    def layout_eligible(self) -> bool:
        return all(report["selected_route"] not in (
            "frontal_delaunay_geometry_limited", "frontal_delaunay_budget_limited",
        ) for report in self.reports)


def refine_authored_pair_with_one_ledger(
    packets, seeds, ledger: AuthoredWorkLedger,
    original_options: NativeMeshingOptions, *, target_size: float,
    cancellation_check=None,
) -> AuthoredPairRefinement:
    """Debit both roots in owner order against one immutable original ledger."""
    packets, seeds = tuple(packets), tuple(seeds)
    if len(packets) != 2 or len(seeds) != 2:
        raise MeshError("authored refinement needs a complete two-root component")
    current = ledger
    made, reports = [], []
    for packet, seed in zip(packets, seeds):
        result, current, report = refine_authored_root_with_ledger(
            packet, seed, current, original_options, target_size=target_size,
            cancellation_check=cancellation_check,
        )
        made.append(result)
        reports.append(report)
    return AuthoredPairRefinement(tuple(made), current, tuple(reports))


def _core(mesh):
    node_ids = np.asarray(sorted(mesh.nodes), dtype=np.int64)
    triangle_ids = np.asarray(sorted(mesh.tris), dtype=np.int64)
    coordinates = np.asarray([mesh.nodes[int(node)] for node in node_ids], dtype=float)
    triangles = np.asarray([mesh.tris[int(cell)] for cell in triangle_ids],
                           dtype=np.int64).reshape((-1, 3))
    return MeshCore.from_id_connectivity(
        coordinates, node_ids=node_ids, triangles=triangles,
        triangle_ids=triangle_ids,
    )


def _validate_core(mesh, core):
    rows = {int(node): row for row, node in enumerate(core.node_ids)}
    try:
        junctions = tuple((rows[int(a)], rows[int(b)])
                          for a, b in mesh.declared_plate_junction_edges)
    except KeyError as error:
        raise MeshError("staged joint refers to a missing node") from error
    assert_valid_mesh(core, declared_plate_junction_edges=junctions)


def stage_authored_root_pair(
    geometry, component, source_mesh: Mesh, source_registry: GlobalEdgeBoundaryRegistry,
    source_seeds: ComponentSeedRegistry, packets, triangulations, child_bindings, *,
    cancellation_check=None,
) -> DetachedAuthoredPair:
    """Assemble two complete roots with one global station and created-ID space.

    Source objects stay untouched. Exact owner partition and a fresh public
    current-only receipt are required; no source-reference or quality admission
    claim follows from the returned detached artifact.
    """
    if (type(source_mesh) is not Mesh
            or not isinstance(source_registry, GlobalEdgeBoundaryRegistry)
            or not isinstance(source_seeds, ComponentSeedRegistry)):
        raise MeshError("authored pair needs neutral source mesh and component registries")
    packets = tuple(packets)
    triangulations = tuple(triangulations)
    child_bindings = tuple(child_bindings)
    roots = tuple(int(value) for value in component.authored_face_ids)
    if (len(roots) != 2 or len(packets) != 2
            or len(triangulations) != 2 or len(child_bindings) != 2
            or any(not isinstance(packet, AuthoredRootBoundaryPacket)
                   for packet in packets)
            or tuple(packet.authored_face_id for packet in packets) != roots):
        raise MeshError("authored pair needs both original roots in owner order")
    for packet, result, child in zip(packets, triangulations, child_bindings):
        _assert_packet_current(packet)
        if isinstance(result, AuthoredRootTriangulation):
            report = result.triangulation.native_diagnostics.get("native_v2", {})
            if report and (not isinstance(report, dict)
                           or report.get("selected_route") in (
                               "frontal_delaunay_geometry_limited",
                               "frontal_delaunay_budget_limited")):
                raise MeshError("authored pair has an unaccepted limited root output")
        if (packet._geometry is not geometry
                or packet._mesh is not source_mesh
                or packet._registry is not source_registry
                or not isinstance(result, AuthoredRootTriangulation)
                or result._packet is not packet
                or not isinstance(child, AuthoredRootChildBinding)
                or child.authored_face_id != packet.authored_face_id
                or len(child.triangle_current_faces) != len(result.triangulation.triangles)
                or set(child.triangle_current_faces)
                   - set(packet._correspondence.descendants)):
            raise MeshError("authored pair has a mismatched root output")
        protected_rows = {int(row) for _node, row in result.triangulation.protected_node_rows}
        constraint_rows = {int(row) for edge in result.triangulation.segments for row in edge}
        if not constraint_rows <= protected_rows:
            raise MeshError("authored pair needs authoritative created edge stations")
    source_digest = _mesh_digest(source_mesh)
    source_stations = _registry_receipt(source_registry)
    source_seed_state = source_seeds.committed_snapshot()
    station_ids = {int(entry.node_id) for entry in source_registry.entries()
                   if entry.node_id is not None}
    protected_ids = {int(node) for result in triangulations
                     for node, _row in result.triangulation.protected_node_rows}
    if not station_ids or station_ids != protected_ids or not station_ids <= set(source_mesh.nodes):
        raise MeshError("authored pair omitted or changed global protected stations")
    stage = AuthoredComponentStage(source_mesh, source_registry, source_seeds)
    try:
        mesh = stage.mesh
        mesh.nodes = {node: np.array(source_mesh.nodes[node], dtype=float, copy=True)
                      for node in sorted(station_ids)}
        mesh.tris = {}
        mesh.quads = {}
        mesh.elements_of_face = {int(face): [] for face in component.current_face_ids}
        mesh.elements_of_sheet = {int(sheet): [] for sheet in component.sheet_ids}
        created_uv = {root: {} for root in roots}
        cell_faces = {}
        next_cell = max((*source_mesh.tris, *source_mesh.quads), default=0) + 1
        for packet, result, child in zip(packets, triangulations, child_bindings):
            if cancellation_check is not None:
                cancellation_check("authored pair before root staging")
            row_ids = dict((int(row), int(node))
                           for node, row in result.triangulation.protected_node_rows)
            new_rows = tuple(row for row in range(len(result.triangulation.points))
                             if row not in row_ids)
            new_ids = stage.reserve_interior_nodes(len(new_rows))
            xyz = packet.chart.evaluate(result.triangulation.points[list(new_rows)])
            if xyz.shape != (len(new_rows), 3) or not np.isfinite(xyz).all():
                raise MeshError("authored pair created invalid owner coordinates")
            for row, node, point in zip(new_rows, new_ids, xyz):
                row_ids[row] = node
                mesh.nodes[node] = np.array(point, copy=True)
                created_uv[packet.authored_face_id][node] = result.original_uv_by_row[row]
            for triangle, face in zip(result.triangulation.triangles,
                                      child.triangle_current_faces):
                mesh.tris[next_cell] = tuple(row_ids[int(row)] for row in triangle)
                mesh.elements_of_face[int(face)].append(next_cell)
                cell_faces[next_cell] = int(face)
                next_cell += 1
        for face in mesh.elements_of_face:
            mesh.elements_of_face[face].sort()
        uses = {int(row["id"]): row
                for row in component.owner_receipt.current_records["face_uses"]}
        for sheet, _root, _original_use, current_uses in component.occurrence_correspondence:
            for use in current_uses:
                record = uses[int(use)]
                mesh.elements_of_sheet[int(sheet)].extend(
                    mesh.elements_of_face[int(record["face_id"])]
                )
        mesh.elements_of_sheet = {
            sheet: sorted(set(cells)) for sheet, cells in mesh.elements_of_sheet.items()
        }
        normalized_created = tuple(
            (root, tuple(sorted(values.items()))) for root, values in sorted(created_uv.items())
        )
        proved = {}

        def validate(candidate_mesh, candidate_registry, _candidate_seeds):
            if (_mesh_digest(source_mesh) != source_digest
                    or _registry_receipt(source_registry) != source_stations
                    or source_seeds.committed_snapshot() != source_seed_state):
                raise MeshError("authored pair source changed during staging")
            if cancellation_check is not None:
                cancellation_check("authored pair before validation")
            core = _core(candidate_mesh)
            _validate_core(candidate_mesh, core)
            receipt = query_prepared_current_component_associations(
                geometry, component, candidate_mesh, candidate_registry, cell_faces,
                created_material_uv_by_root=created_uv,
                cancellation_check=cancellation_check,
            )
            validate_prepared_current_component_associations(
                geometry, receipt, component, candidate_mesh, candidate_registry,
                cell_faces, cancellation_check=cancellation_check,
            )
            if receipt.source_reference_transfer_qualified or receipt.publication_qualified:
                raise MeshError("authored pair current receipt widened its qualification")
            if (_mesh_digest(source_mesh) != source_digest
                    or _registry_receipt(source_registry) != source_stations
                    or source_seeds.committed_snapshot() != source_seed_state):
                raise MeshError("authored pair source changed during validation")
            proved["core"], proved["receipt"] = core, receipt
            return True

        detached_mesh, _detached_registry, _detached_seeds = stage.finish(validate)
        return DetachedAuthoredPair(
            detached_mesh, proved["core"], proved["receipt"], normalized_created,
            tuple(sorted(cell_faces.items())),
        )
    except BaseException:
        stage.abort()
        raise
