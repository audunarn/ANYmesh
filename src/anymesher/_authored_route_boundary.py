"""Source-only boundary/constraint packet for a guarded authored-root route.

The meshing engine consumes float metric coordinates; exact current-material
Fraction UV stays attached to every unchanged global station for later owner
partition proof. Building this packet does not invoke an engine or publish.
"""

from dataclasses import dataclass
import numpy as np

from ._authored_component_binding import BoundAuthoredSheetJointComponent
from ._authored_metric_chart import AuthoredMetricChart
from ._authored_planar_stations import plan_authored_planar_stations
from .boundary import GlobalEdgeBoundaryRegistry
from .errors import MeshError


@dataclass(frozen=True)
class AuthoredRootBoundaryPacket:
    authored_face_id: int
    outer_node_ids: tuple[int, ...]
    outer_metric: tuple[tuple[float, float], ...]
    constraint_node_pairs: tuple[tuple[int, int], ...]
    constraint_metric: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    material_uv_by_node: tuple
    chart: AuthoredMetricChart

    @property
    def publication_qualified(self) -> bool:
        return False


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
        packets.append(AuthoredRootBoundaryPacket(
            int(correspondence.authored_definition.face_id), tuple(outer),
            outer_metric, tuple(pairs), constraint_metric,
            station_plan.node_material_uv, chart,
        ))
    registry.view.assert_current(geometry)
    return tuple(packets)
