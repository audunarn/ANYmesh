"""Immutable planar-face chart used by the production quad seed."""
from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
from typing import Sequence
from anygeometry.model import GeometryModel
from ..errors import MeshError

Vec3 = tuple[float, float, float]
Vec2 = tuple[float, float]

def _sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0]-b[0]), float(a[1]-b[1]), float(a[2]-b[2]))
def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]*b[0] + a[1]*b[1] + a[2]*b[2])
def _cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[1]*b[2]-a[2]*b[1]), float(a[2]*b[0]-a[0]*b[2]), float(a[0]*b[1]-a[1]*b[0]))
def _unit(a: Sequence[float]) -> Vec3:
    n = sqrt(_dot(a,a))
    if n <= 0.0:
        raise MeshError("planar quad face has a degenerate direction")
    return (float(a[0]/n), float(a[1]/n), float(a[2]/n))

def _cross2d(ax: float, ay: float, bx: float, by: float) -> float:
    return float(ax*by - ay*bx)

def _on_segment2d(p: Sequence[float], a: Sequence[float], b: Sequence[float], tol: float) -> bool:
    x, y = float(p[0]), float(p[1])
    if x < min(a[0], b[0]) - tol or x > max(a[0], b[0]) + tol:
        return False
    if y < min(a[1], b[1]) - tol or y > max(a[1], b[1]) + tol:
        return False
    return abs(_cross2d(b[0]-a[0], b[1]-a[1], x-a[0], y-a[1])) <= tol

def _segments_intersect(a: Vec2, b: Vec2, c: Vec2, d: Vec2, tol: float) -> bool:
    def orient(p: Sequence[float], q: Sequence[float], r: Sequence[float]) -> int:
        value = _cross2d(q[0]-p[0], q[1]-p[1], r[0]-p[0], r[1]-p[1])
        if abs(value) <= tol:
            return 0
        return 1 if value > 0.0 else -1
    o1 = orient(a, b, c); o2 = orient(a, b, d)
    o3 = orient(c, d, a); o4 = orient(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment2d(c, a, b, tol):
        return True
    if o2 == 0 and _on_segment2d(d, a, b, tol):
        return True
    if o3 == 0 and _on_segment2d(a, c, d, tol):
        return True
    if o4 == 0 and _on_segment2d(b, c, d, tol):
        return True
    return False

def _strictly_inside_chart(point: Sequence[float], polygon: Sequence[Vec2], tol: float) -> bool:
    x, y = float(point[0]), float(point[1])
    for i in range(len(polygon)):
        if _on_segment2d(point, polygon[i], polygon[(i+1) % len(polygon)], tol):
            return False
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        a = polygon[i]; b = polygon[j]
        if (a[1] > y) != (b[1] > y):
            crossing = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < crossing:
                inside = not inside
        j = i
    return inside

def _signed_chart_area(chart: Sequence[Vec2]) -> float:
    n = len(chart)
    return 0.5 * sum(
        chart[i][0]*chart[(i+1) % n][1] - chart[(i+1) % n][0]*chart[i][1]
        for i in range(n)
    )

def _loop_vertex_ids(geometry: GeometryModel, oriented_edges: Sequence) -> tuple[int, ...]:
    vertices: list[int] = []
    expected: int | None = None
    for item in oriented_edges:
        edge = geometry.edges[int(item.edge)]
        start, end = (edge.start, edge.end) if bool(item.forward) else (edge.end, edge.start)
        if expected is not None and int(start) != expected:
            raise MeshError("planar boundary loop is not a connected oriented loop")
        vertices.append(int(start))
        expected = int(end)
    if expected is None or expected != vertices[0]:
        raise MeshError("planar boundary loop is not closed")
    return tuple(vertices)

def _loop_is_valid(geometry: GeometryModel, loop: Sequence, label: str) -> tuple[int, ...]:
    if len(loop) < 3:
        raise MeshError(f"{label} boundary needs at least three edges")
    vertices = _loop_vertex_ids(geometry, loop)
    if len(set(vertices)) != len(vertices):
        raise MeshError(f"{label} boundary repeats source vertices")
    return vertices

@dataclass(frozen=True)
class PlanarQuadDomain:
    model_id: str
    revision: int
    face_id: int
    vertex_ids: tuple[int, ...]
    edge_uses: tuple[tuple[int, bool], ...]
    origin: Vec3
    x_hat: Vec3
    y_hat: Vec3
    normal: Vec3
    outer_chart: tuple[Vec2, ...]
    hole_vertex_loops: tuple[tuple[int, ...], ...] = ()
    hole_edge_uses: tuple[tuple[tuple[int, bool], ...], ...] = ()
    hole_charts: tuple[tuple[Vec2, ...], ...] = ()
    outer_area: float = 0.0
    hole_areas: tuple[float, ...] = ()
    area: float = 0.0

    @classmethod
    def from_geometry(cls, geometry: GeometryModel, face_id: int) -> "PlanarQuadDomain":
        if face_id not in geometry.faces:
            raise MeshError(f"unknown planar face {face_id}")
        face = geometry.faces[face_id]
        loop = tuple(face.loop)
        if len(loop) < 3:
            raise MeshError("planar quad face needs at least three boundary edges")
        vertices = _loop_is_valid(geometry, loop, "outer")
        points = [tuple(map(float, geometry.vertex_position(v))) for v in vertices]
        origin = points[0]
        x_hat = _unit(_sub(points[1], origin))
        normal = None
        for i in range(1, len(points)-1):
            c = _cross(_sub(points[i], origin), _sub(points[i+1], origin))
            if _dot(c,c) > 1.0e-24:
                normal = _unit(c); break
        if normal is None:
            raise MeshError("planar quad face is degenerate")
        y_hat = _unit(_cross(normal, x_hat))
        chart = tuple((_dot(_sub(p, origin), x_hat), _dot(_sub(p, origin), y_hat)) for p in points)
        extent = max(max(abs(x) for x,y in chart), max(abs(y) for x,y in chart), 1.0)
        tol = 1.0e-10 * extent
        for p in points:
            if abs(_dot(_sub(p, origin), normal)) > tol:
                raise MeshError("face is not planar within PQ2 tolerance")
        signed = _signed_chart_area(chart)
        if abs(signed) <= tol*tol:
            raise MeshError("planar quad face has zero chart area")
        if signed < 0.0:
            normal = tuple(-v for v in normal)  # type: ignore[assignment]
            y_hat = tuple(-v for v in y_hat)  # type: ignore[assignment]
            chart = tuple((_dot(_sub(p, origin), x_hat), _dot(_sub(p, origin), y_hat)) for p in points)
            signed = -signed
        outer_vertices = tuple(vertices)
        hole_vertex_loops: list[tuple[int, ...]] = []
        hole_edge_uses: list[tuple[tuple[int, bool], ...]] = []
        hole_charts: list[tuple[Vec2, ...]] = []
        hole_areas: list[float] = []
        source_holes = tuple(getattr(face, "holes", ()) or ())
        for index, hole in enumerate(source_holes):
            hole = tuple(hole)
            hole_tag = f"hole {index}"
            hole_vertices = _loop_is_valid(geometry, hole, hole_tag)
            hole_points = [tuple(map(float, geometry.vertex_position(v))) for v in hole_vertices]
            for p in hole_points:
                if abs(_dot(_sub(p, origin), normal)) > tol:
                    raise MeshError(f"{hole_tag} is not planar with the face")
            hole_chart = tuple(
                (_dot(_sub(p, origin), x_hat), _dot(_sub(p, origin), y_hat)) for p in hole_points
            )
            hole_area = abs(_signed_chart_area(hole_chart))
            if hole_area <= tol * tol:
                raise MeshError(f"{hole_tag} has zero chart area")
            for p in hole_chart:
                if not _strictly_inside_chart(p, chart, tol):
                    raise MeshError(f"{hole_tag} is not strictly inside the outer boundary")
            for i in range(len(hole_chart)):
                a = hole_chart[i]; b = hole_chart[(i+1) % len(hole_chart)]
                for j in range(len(chart)):
                    if _segments_intersect(a, b, chart[j], chart[(j+1) % len(chart)], tol):
                        raise MeshError(f"{hole_tag} crosses the outer boundary")
            for k in range(len(hole_charts)):
                other = hole_charts[k]
                for p in hole_chart:
                    if _strictly_inside_chart(p, other, tol):
                        raise MeshError(f"{hole_tag} overlaps hole {k}")
                for q in other:
                    if _strictly_inside_chart(q, hole_chart, tol):
                        raise MeshError(f"hole {k} overlaps {hole_tag}")
                for i in range(len(hole_chart)):
                    for j in range(len(other)):
                        if _segments_intersect(
                            hole_chart[i], hole_chart[(i+1) % len(hole_chart)],
                            other[j], other[(j+1) % len(other)],
                            tol,
                        ):
                            raise MeshError(f"{hole_tag} intersects hole {k}")
            hole_vertex_loops.append(hole_vertices)
            hole_edge_uses.append(
                tuple((int(item.edge), bool(item.forward)) for item in hole)
            )
            hole_charts.append(hole_chart)
            hole_areas.append(float(hole_area))
        outer_area = float(signed)
        area = outer_area - float(sum(hole_areas))
        if area <= tol * tol:
            raise MeshError("planar quad face area is not positive after hole subtraction")
        return cls(
            str(geometry.model_id), int(geometry.revision), int(face_id),
            outer_vertices,
            tuple((int(item.edge), bool(item.forward)) for item in loop),
            origin, x_hat, y_hat, normal, chart,
            tuple(hole_vertex_loops), tuple(hole_edge_uses), tuple(hole_charts),
            outer_area, tuple(hole_areas), float(area),
        )

    def assert_current(self, geometry: GeometryModel) -> None:
        if str(geometry.model_id) != self.model_id or int(geometry.revision) != self.revision:
            raise MeshError("source geometry changed after planar-domain capture")
    def project(self, point: Sequence[float]) -> Vec2:
        d = _sub(point, self.origin); return (_dot(d,self.x_hat), _dot(d,self.y_hat))
    def lift(self, point: Sequence[float]) -> Vec3:
        x,y=float(point[0]),float(point[1]); return (self.origin[0]+x*self.x_hat[0]+y*self.y_hat[0], self.origin[1]+x*self.x_hat[1]+y*self.y_hat[1], self.origin[2]+x*self.x_hat[2]+y*self.y_hat[2])
