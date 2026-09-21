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
    area: float

    @classmethod
    def from_geometry(cls, geometry: GeometryModel, face_id: int) -> "PlanarQuadDomain":
        if face_id not in geometry.faces:
            raise MeshError(f"unknown planar face {face_id}")
        face = geometry.faces[face_id]
        if getattr(face, "holes", ()):
            raise MeshError("PQ2 planar seed does not support holes")
        uses = tuple((int(item.edge), bool(item.forward)) for item in face.loop)
        if len(uses) < 3:
            raise MeshError("planar quad face needs at least three boundary edges")
        vertices: list[int] = []
        expected = None
        for edge_id, forward in uses:
            edge = geometry.edges[edge_id]
            start, end = (edge.start, edge.end) if forward else (edge.end, edge.start)
            if expected is not None and start != expected:
                raise MeshError("face boundary is not a connected oriented loop")
            vertices.append(int(start)); expected = int(end)
        if expected != vertices[0]:
            raise MeshError("face boundary is not closed")
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
        signed = 0.5 * sum(chart[i][0]*chart[(i+1)%len(chart)][1] - chart[(i+1)%len(chart)][0]*chart[i][1] for i in range(len(chart)))
        if abs(signed) <= tol*tol:
            raise MeshError("planar quad face has zero chart area")
        if signed < 0.0:
            normal = tuple(-v for v in normal)  # type: ignore[assignment]
            y_hat = tuple(-v for v in y_hat)  # type: ignore[assignment]
            chart = tuple((_dot(_sub(p, origin), x_hat), _dot(_sub(p, origin), y_hat)) for p in points)
            signed = -signed
        return cls(str(geometry.model_id), int(geometry.revision), int(face_id), tuple(vertices), uses, origin, x_hat, y_hat, normal, chart, float(signed))

    def assert_current(self, geometry: GeometryModel) -> None:
        if str(geometry.model_id) != self.model_id or int(geometry.revision) != self.revision:
            raise MeshError("source geometry changed after planar-domain capture")
    def project(self, point: Sequence[float]) -> Vec2:
        d = _sub(point, self.origin); return (_dot(d,self.x_hat), _dot(d,self.y_hat))
    def lift(self, point: Sequence[float]) -> Vec3:
        x,y=float(point[0]),float(point[1]); return (self.origin[0]+x*self.x_hat[0]+y*self.y_hat[0], self.origin[1]+x*self.x_hat[1]+y*self.y_hat[1], self.origin[2]+x*self.x_hat[2]+y*self.y_hat[2])
