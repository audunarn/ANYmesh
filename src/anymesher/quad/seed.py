"""Target-size-driven constrained T3 seed for the planar quad advancing front."""
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence
import numpy as np
from anygeometry.model import GeometryModel
from ..errors import MeshError
from ..triangulation import PlanarTriangulation, orient2d, triangulate_polygon
from .boundary import BoundaryStationKey, BoundaryStationRegistry
from .domain import PlanarQuadDomain, Vec2
from .state import QuadMeshState

@dataclass(frozen=True)
class PlanarQuadSeed:
    domain: PlanarQuadDomain
    registry: BoundaryStationRegistry
    target_size: float
    boundary_keys: tuple[BoundaryStationKey, ...]
    triangulation: PlanarTriangulation
    state: QuadMeshState
    station_to_node: Mapping[BoundaryStationKey, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "station_to_node", MappingProxyType(dict(self.station_to_node)))


def _on_segment(p: Vec2, a: Vec2, b: Vec2, tol: float) -> bool:
    if p[0] < min(a[0], b[0]) - tol or p[0] > max(a[0], b[0]) + tol:
        return False
    if p[1] < min(a[1], b[1]) - tol or p[1] > max(a[1], b[1]) + tol:
        return False
    return abs((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])) <= tol


def _strict_inside(p: Vec2, polygon: Sequence[Vec2], tol: float) -> bool:
    inside = False
    x, y = p
    for i, a in enumerate(polygon):
        b = polygon[(i+1) % len(polygon)]
        if _on_segment(p, a, b, tol):
            return False
        if (a[1] > y) != (b[1] > y):
            crossing = a[0] + (y-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if crossing > x:
                inside = not inside
    return inside


def _interior_lattice(polygon: Sequence[Vec2], target_size: float) -> np.ndarray:
    h = float(target_size)
    if not np.isfinite(h) or h <= 0.0:
        raise MeshError("target_size must be positive and finite")
    min_x = min(p[0] for p in polygon); max_x = max(p[0] for p in polygon)
    min_y = min(p[1] for p in polygon); max_y = max(p[1] for p in polygon)
    tol = 1.0e-10 * max(max_x-min_x, max_y-min_y, 1.0)
    xs = np.arange(min_x+h, max_x-tol, h, dtype=float)
    ys = np.arange(min_y+h, max_y-tol, h, dtype=float)
    points = [(float(x), float(y)) for y in ys for x in xs if _strict_inside((float(x),float(y)), polygon, tol)]
    return np.asarray(points, dtype=float).reshape((-1,2))


def build_planar_quad_seed(
    geometry: GeometryModel,
    face_id: int,
    target_size: float,
    *,
    domain: PlanarQuadDomain | None = None,
    registry: BoundaryStationRegistry | None = None,
) -> PlanarQuadSeed:
    start_revision = int(geometry.revision)
    start_model_id = str(geometry.model_id)
    domain = domain or PlanarQuadDomain.from_geometry(geometry, face_id)
    if domain.face_id != int(face_id):
        raise MeshError("domain face does not match requested face")
    domain.assert_current(geometry)
    registry = registry or BoundaryStationRegistry.for_domain(geometry, domain, target_size)
    registry.assert_current(geometry)
    if abs(registry.target_size-float(target_size)) > 1.0e-15*max(1.0,abs(float(target_size))):
        raise MeshError("boundary registry target_size differs from seed target_size")
    stations = registry.face_stations(domain)
    outer = tuple(domain.project(station.position) for station in stations)
    interior = _interior_lattice(outer, float(target_size))
    triangulation = triangulate_polygon(np.asarray(outer, dtype=float), interior_points=interior, backend="python")
    if len(triangulation.points) < len(stations):
        raise MeshError("triangulation lost canonical boundary stations")
    for row, point in enumerate(outer):
        if not np.array_equal(triangulation.points[row], np.asarray(point, dtype=float)):
            raise MeshError("triangulation changed canonical boundary station rows")
    nodes = {row: tuple(map(float, point)) for row, point in enumerate(triangulation.points)}
    cells = {row: tuple(map(int, tri)) for row, tri in enumerate(triangulation.triangles)}
    for body in cells.values():
        if orient2d(nodes[body[0]], nodes[body[1]], nodes[body[2]]) <= 0.0:
            raise MeshError("PQ2 seed contains a non-positive T3")
    boundary_edges = tuple(tuple(map(int, edge)) for edge in triangulation.boundary_segments)
    boundary_nodes = tuple(range(len(stations)))
    state = QuadMeshState(
        nodes, cells, {cid: "T3" for cid in cells},
        initial_front=boundary_edges,
        protected_nodes=boundary_nodes,
        protected_edges=boundary_edges,
    )
    station_to_node = {station.key: row for row, station in enumerate(stations)}
    if len(station_to_node) != len(stations):
        raise MeshError("canonical boundary station identities are not unique on the face")
    domain.assert_current(geometry); registry.assert_current(geometry)
    if int(geometry.revision) != start_revision or str(geometry.model_id) != start_model_id:
        raise MeshError("PQ2 seed mutated source geometry")
    return PlanarQuadSeed(domain, registry, float(target_size), tuple(station_to_node), triangulation, state, station_to_node)
