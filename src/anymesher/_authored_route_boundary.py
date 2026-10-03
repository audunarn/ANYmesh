"""Source-only boundary/constraint packet for a guarded authored-root route.

The meshing engine consumes float metric coordinates; exact current-material
Fraction UV stays attached to every unchanged global station for later owner
partition proof. Building this packet does not invoke an engine or publish.
"""

from dataclasses import dataclass, field
from fractions import Fraction
import numpy as np

from ._authored_component_binding import BoundAuthoredSheetJointComponent
from ._authored_metric_chart import AuthoredMetricChart
from ._authored_planar_stations import plan_authored_planar_stations
from .boundary import GlobalEdgeBoundaryRegistry
from .errors import MeshError
from .triangulation import PlanarTriangulation, triangulate_polygon


@dataclass(frozen=True)
class AuthoredRootBoundaryPacket:
    authored_face_id: int
    outer_node_ids: tuple[int, ...]
    outer_metric: tuple[tuple[float, float], ...]
    constraint_node_pairs: tuple[tuple[int, int], ...]
    constraint_metric: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    material_uv_by_node: tuple
    chart: AuthoredMetricChart
    _geometry: object = field(repr=False, compare=False)
    _mesh: object = field(repr=False, compare=False)
    _registry: GlobalEdgeBoundaryRegistry = field(repr=False, compare=False)
    _edge_ids: tuple[int, ...] = ()
    _registry_snapshot: tuple = ()
    _node_xyz_snapshot: tuple = ()

    @property
    def publication_qualified(self) -> bool:
        return False


@dataclass(frozen=True)
class AuthoredRootTriangulation:
    """Detached T3 seed with registry IDs and original-chart UV provenance."""

    triangulation: PlanarTriangulation
    original_uv_by_row: tuple[tuple[Fraction, Fraction], ...]

    @property
    def publication_qualified(self) -> bool:
        return False


def _registry_snapshot(registry, edge_ids):
    return tuple((edge_id, tuple((entry.key.parameter, entry.node_id,
                                  tuple(float(value) for value in entry.point),
                                  entry.owners)
                                 for entry in registry.entries(edge_id)))
                 for edge_id in edge_ids)


def _node_xyz_snapshot(mesh, node_ids):
    return tuple((node, np.asarray(mesh.nodes[node], dtype=np.float64).tobytes())
                 for node in sorted(set(node_ids)))


def _assert_packet_current(packet):
    packet._registry.view.assert_current(packet._geometry)
    if _registry_snapshot(packet._registry, packet._edge_ids) != packet._registry_snapshot:
        raise MeshError("authored boundary registry changed after planning")
    try:
        current_xyz = _node_xyz_snapshot(packet._mesh, dict(packet._node_xyz_snapshot))
    except KeyError as error:
        raise MeshError("authored protected registry node is missing") from error
    if current_xyz != packet._node_xyz_snapshot:
        raise MeshError("authored protected registry node coordinates changed")
    packet.chart.to_authored_uv(np.empty((0, 2), dtype=float))


def triangulate_authored_root_boundary(
    packet: AuthoredRootBoundaryPacket, *, interior_metric=None, backend="python",
    cancellation_check=None,
) -> AuthoredRootTriangulation:
    """Carry protected global IDs through a detached constrained T3 seed.

    Only the owner material receipt supplies exact UV for protected rows. New
    rows retain the binary64 inverse-chart coordinates as rational provenance;
    they still need owner partition, work-ledger and quality checks before use.
    """
    if not isinstance(packet, AuthoredRootBoundaryPacket):
        raise MeshError("authored triangulation needs an owner boundary packet")
    _assert_packet_current(packet)
    interior = (np.empty((0, 2), dtype=float) if interior_metric is None
                else np.asarray(interior_metric, dtype=float))
    if interior.ndim != 2 or interior.shape[1] != 2 or not np.isfinite(interior).all():
        raise MeshError("authored interior metric seeds must be finite (n, 2) rows")
    source_uv = dict(packet.material_uv_by_node)
    constraint_ids = tuple(node for pair in packet.constraint_node_pairs for node in pair)
    raw_ids = (*packet.outer_node_ids, *constraint_ids)
    protected = {row: node for row, node in enumerate(packet.outer_node_ids)}
    protected.update((len(packet.outer_node_ids) + len(interior) + row, node)
                     for row, node in enumerate(constraint_ids))
    seed = triangulate_polygon(
        packet.outer_metric, constraints=packet.constraint_metric,
        interior_points=interior,
        backend=backend, protected_node_ids=protected,
        cancellation_check=cancellation_check,
    )
    if {node for node, _row in seed.protected_node_rows} != set(raw_ids):
        raise MeshError("authored triangulation lost a protected registry ID")
    inverted = packet.chart.to_authored_uv(seed.points)
    if inverted.shape != (len(seed.points), 2):
        raise MeshError("authored triangulation lost original UV row provenance")
    uv = tuple(tuple(Fraction.from_float(float(value)) for value in row)
               for row in inverted)
    mutable = list(uv)
    for node, row in seed.protected_node_rows:
        if node not in source_uv:
            raise MeshError("authored triangulation lost exact material station UV")
        mutable[row] = source_uv[node]
    _assert_packet_current(packet)
    return AuthoredRootTriangulation(seed, tuple(mutable))


def plan_authored_component_boundaries(
    geometry, component: BoundAuthoredSheetJointComponent, mesh,
    registry: GlobalEdgeBoundaryRegistry, *, cancellation_check=None,
) -> tuple[AuthoredRootBoundaryPacket, ...]:
    """Prepare every root together, preserving owner loop and station order."""
    if not isinstance(component, BoundAuthoredSheetJointComponent):
        raise MeshError("authored route boundaries need a whole component")
    if not isinstance(registry, GlobalEdgeBoundaryRegistry):
        raise MeshError("authored route boundaries need a global registry")
    registry.view.assert_current(geometry)
    packets = []
    for correspondence in component.boundary_correspondences:
        if len(correspondence.exterior_loops) != 1:
            raise MeshError("authored route currently needs one straight outer loop")
        station_plan = plan_authored_planar_stations(
            geometry, correspondence, mesh, registry,
            cancellation_check=cancellation_check,
        )
        material = dict(station_plan.node_material_uv)
        outer = []
        for _source, forward, current_edges in correspondence.exterior_loops[0]:
            for edge_id in current_edges:
                segment = tuple(entry.node_id for entry in registry.entries(edge_id))
                directed = segment if forward else segment[::-1]
                if outer and outer[-1] != directed[0]:
                    raise MeshError("authored route exterior stations lost owner order")
                outer.extend(directed if not outer else directed[1:])
        if len(outer) < 4 or outer[0] != outer[-1]:
            raise MeshError("authored route exterior stations are not closed")
        outer.pop()
        if len(outer) != len(set(outer)):
            raise MeshError("authored route exterior repeats a station")
        pairs = []
        for edge_id, _uses in correspondence.interior_incidence:
            chain = tuple(entry.node_id for entry in registry.entries(edge_id))
            pairs.extend((a, b) for a, b in zip(chain, chain[1:]))
        if not pairs:
            raise MeshError("authored route needs protected interior constraints")
        if any(node not in material for node in (*outer, *(n for pair in pairs for n in pair))):
            raise MeshError("authored route station lacks exact current-material UV")
        chart = AuthoredMetricChart(
            geometry, correspondence, cancellation_check=cancellation_check,
        )

        def metric(nodes):
            uv = np.asarray([[float(value) for value in material[node]]
                             for node in nodes], dtype=float)
            points = chart.to_metric(uv)
            if points.shape != (len(nodes), 2) or not np.isfinite(points).all():
                raise MeshError("authored route produced invalid metric stations")
            return tuple(tuple(float(value) for value in row) for row in points)

        outer_metric = metric(outer)
        constraint_metric = tuple(metric((a, b)) for a, b in pairs)
        edge_ids = tuple(dict.fromkeys(
            edge for _source, _forward, current in correspondence.exterior_loops[0]
            for edge in current
        )) + tuple(edge for edge, _uses in correspondence.interior_incidence)
        node_ids = (*outer, *(node for pair in pairs for node in pair))
        packets.append(AuthoredRootBoundaryPacket(
            int(correspondence.authored_definition.face_id), tuple(outer),
            outer_metric, tuple(pairs), constraint_metric,
            station_plan.node_material_uv, chart, geometry, mesh, registry,
            edge_ids, _registry_snapshot(registry, edge_ids),
            _node_xyz_snapshot(mesh, node_ids),
        ))
    registry.view.assert_current(geometry)
    return tuple(packets)
