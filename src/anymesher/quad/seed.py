"""Target-size-driven constrained T3 seed for the planar quad advancing front."""
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence
import numpy as np
from anygeometry.model import GeometryModel
from ..errors import MeshError
from ..refinement import SizeField
from ..triangulation import PlanarTriangulation, orient2d, triangulate_polygon
from .boundary import BoundaryStationKey, BoundaryStationRegistry
from .domain import PlanarQuadDomain, Vec2
from .state import QuadMeshState

def _shoelace(loop_points: np.ndarray) -> float:
    loop = np.asarray(loop_points, dtype=float)
    if loop.ndim != 2 or loop.shape[1] < 2 or len(loop) < 3:
        raise MeshError("shoelace loop must contain at least three points")
    x = loop[:, 0]
    y = loop[:, 1]
    value = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if not np.isfinite(value):
        raise MeshError("shoelace area is not finite")
    return value


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

    @property
    def discrete_area(self) -> float:
        points = np.asarray(self.triangulation.points, dtype=float)
        area = _shoelace(points[np.asarray(self.triangulation.outer_loop, dtype=np.int64)])
        for ring in self.triangulation.hole_loops:
            area -= _shoelace(points[np.asarray(ring, dtype=np.int64)])
        if not np.isfinite(area) or area <= 0.0:
            raise MeshError("discrete meshed-domain area must be positive and finite")
        return area


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


def _interior_lattice(
    domain: PlanarQuadDomain,
    outer: Sequence[Vec2],
    holes: Sequence[Sequence[Vec2]],
    target_size: float,
    size_field: SizeField | None = None,
) -> np.ndarray:
    h = float(target_size)
    if not np.isfinite(h) or h <= 0.0:
        raise MeshError("target_size must be positive and finite")
    if size_field is None or size_field.is_uniform:
        min_x = min(p[0] for p in outer); max_x = max(p[0] for p in outer)
        min_y = min(p[1] for p in outer); max_y = max(p[1] for p in outer)
        tol = 1.0e-10 * max(max_x-min_x, max_y-min_y, 1.0)
        xs = np.arange(min_x+h, max_x-tol, h, dtype=float)
        ys = np.arange(min_y+h, max_y-tol, h, dtype=float)
        points: list[tuple[float, float]] = []
        for y in ys:
            for x in xs:
                point = (float(x), float(y))
                if not _strict_inside(point, outer, tol):
                    continue
                if any(_strict_inside(point, hole, tol) for hole in holes):
                    continue
                # Cartesian lattice coordinates are unique by construction; avoid an
                # unnecessary O(N^2) duplicate scan over previously accepted points.
                points.append(point)
        return np.asarray(points, dtype=float).reshape((-1, 2))
    min_x = min(p[0] for p in outer); max_x = max(p[0] for p in outer)
    min_y = min(p[1] for p in outer); max_y = max(p[1] for p in outer)
    tol = 1.0e-10 * max(max_x-min_x, max_y-min_y, 1.0)
    coarse = _uniform_lattice_points(outer, list(holes), h, min_x, max_x, min_y, max_y, tol)
    h_min = min(float(target_size), *(float(zone.size) for zone in size_field.zones))
    nx = int(np.ceil((max_x - min_x) / h_min))
    ny = int(np.ceil((max_y - min_y) / h_min))
    if nx * ny > 250000:
        raise MeshError("graded refinement lattice exceeds fine candidate budget")
    accepted: list[np.ndarray] = [np.asarray(p, dtype=float) for p in coarse]
    seen = {(float(p[0]), float(p[1])) for p in coarse}
    for j in range(ny):
        y = min_y + (j + 0.5) * h_min
        if y > max_y - tol:
            continue
        for i in range(nx):
            x = min_x + (i + 0.5) * h_min
            if x > max_x - tol:
                continue
            point = (float(x), float(y))
            if not _strict_inside(point, outer, tol):
                continue
            if any(_strict_inside(point, hole, tol) for hole in holes):
                continue
            key = point
            if key in seen:
                continue
            h_local = float(size_field.size_at(np.asarray([domain.lift(point)], dtype=float))[0])
            if h_local >= 0.999999 * target_size:
                continue
            if any(float(np.linalg.norm(np.asarray(p, dtype=float) - np.asarray(point, dtype=float))) < 0.75 * h_local for p in accepted):
                continue
            accepted.append(np.asarray(point, dtype=float))
            seen.add(key)
    return np.asarray(accepted, dtype=float).reshape((-1, 2))


def _uniform_lattice_points(
    outer: Sequence[Vec2],
    holes: list[Sequence[Vec2]],
    h: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    tol: float,
) -> list[tuple[float, float]]:
    xs = np.arange(min_x+h, max_x-tol, h, dtype=float)
    ys = np.arange(min_y+h, max_y-tol, h, dtype=float)
    points: list[tuple[float, float]] = []
    for y in ys:
        for x in xs:
            point = (float(x), float(y))
            if not _strict_inside(point, outer, tol):
                continue
            if any(_strict_inside(point, hole, tol) for hole in holes):
                continue
            points.append(point)
    return points


def build_planar_quad_seed(
    geometry: GeometryModel,
    face_id: int,
    target_size: float,
    *,
    domain: PlanarQuadDomain | None = None,
    registry: BoundaryStationRegistry | None = None,
    size_field: SizeField | None = None,
) -> PlanarQuadSeed:
    start_revision = int(geometry.revision)
    start_model_id = str(geometry.model_id)
    domain = domain or PlanarQuadDomain.from_geometry(geometry, face_id)
    if domain.face_id != int(face_id):
        raise MeshError("domain face does not match requested face")
    domain.assert_current(geometry)
    if size_field is not None and abs(float(size_field.target_size) - float(target_size)) > 1.0e-12 * max(1.0, abs(float(target_size))):
        raise MeshError("provided size field target_size differs from seed target_size")
    registry = registry or BoundaryStationRegistry.for_domain(geometry, domain, target_size, size_field=size_field)
    registry.assert_current(geometry)
    if abs(registry.target_size-float(target_size)) > 1.0e-15*max(1.0,abs(float(target_size))):
        raise MeshError("boundary registry target_size differs from seed target_size")
    stations = registry.outer_stations(domain)
    outer = tuple(domain.project(station.position) for station in stations)
    hole_polys: list[np.ndarray] = []
    for index in range(len(domain.hole_edge_uses)):
        hole_stations = registry.hole_stations(domain, index)
        hole_points = [domain.project(station.position) for station in hole_stations]
        if len(hole_points) < 3:
            raise MeshError("hole boundary has fewer than three stations")
        hole_polys.append(np.asarray(hole_points, dtype=float))
    interior = _interior_lattice(domain, outer, tuple(hole_polys), float(target_size), size_field)
    triangulation = triangulate_polygon(
        np.asarray(outer, dtype=float),
        holes=hole_polys,
        interior_points=interior,
        backend="python",
    )
    if len(triangulation.points) < len(outer) + sum(len(p) for p in hole_polys):
        raise MeshError("triangulation lost canonical boundary stations")
    for row, point in enumerate(outer):
        if not np.array_equal(triangulation.points[row], np.asarray(point, dtype=float)):
            raise MeshError("triangulation changed canonical outer station rows")
    cursor = len(outer)
    for hole_points in hole_polys:
        for offset, point in enumerate(hole_points):
            if not np.array_equal(triangulation.points[cursor + offset], np.asarray(point, dtype=float)):
                raise MeshError("triangulation changed canonical hole station rows")
        cursor += len(hole_points)
    nodes = {row: tuple(map(float, point)) for row, point in enumerate(triangulation.points)}
    cells = {row: tuple(map(int, tri)) for row, tri in enumerate(triangulation.triangles)}
    for body in cells.values():
        if orient2d(nodes[body[0]], nodes[body[1]], nodes[body[2]]) <= 0.0:
            raise MeshError("PQ2 seed contains a non-positive T3")
    boundary_edges = tuple(tuple(map(int, edge)) for edge in triangulation.boundary_segments)
    boundary_nodes = tuple(range(len(outer) + sum(len(p) for p in hole_polys)))
    state = QuadMeshState(
        nodes, cells, {cid: "T3" for cid in cells},
        initial_front=boundary_edges,
        protected_nodes=boundary_nodes,
        protected_edges=boundary_edges,
    )
    station_to_node: dict[BoundaryStationKey, int] = {}
    for row, station in enumerate(stations):
        station_to_node[station.key] = row
    cursor = len(stations)
    for index in range(len(domain.hole_edge_uses)):
        hole_stations = registry.hole_stations(domain, index)
        for station in hole_stations:
            if station.key in station_to_node:
                raise MeshError("canonical boundary station identities are not unique on the face")
            station_to_node[station.key] = cursor
            cursor += 1
    domain.assert_current(geometry); registry.assert_current(geometry)
    if int(geometry.revision) != start_revision or str(geometry.model_id) != start_model_id:
        raise MeshError("PQ2 seed mutated source geometry")
    return PlanarQuadSeed(domain, registry, float(target_size), tuple(station_to_node), triangulation, state, station_to_node)
