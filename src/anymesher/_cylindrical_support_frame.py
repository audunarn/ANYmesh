"""Support-frame face queries for an owner Cylinder with a distinct parameterization.

MESH-B option A2. On the owner-certified trimmed-cylinder route the quad-first
mesher works entirely in the support frame: chart coordinates are the support's
normalized ``(angle, height)``, points and derivatives come from the support,
and the face's own parameterization (for example a topology Coons map kept for
display) is never consulted. The owner's trims are support-frame curves and the
face map lies on that support, so the material region is the same.

Each method below reproduces, call for call, what ANYgeometry's GeometryModel
returns for the same face *without* its parameterization (the "twin"), using
the same owner helpers in the same order. A mesh of the parameterized face is
therefore bit-identical to the twin's mesh. The equivalence is tested in
tests/test_cylinder_support_frame_parameterization.py; a change to the owner's
built-in Cylinder face path must be mirrored here.
"""
from __future__ import annotations

import numpy as np
from anygeometry import GeometryError, GeometryModel
from anygeometry.curves import Straight
from anygeometry.surfaces import CoonsSurface, Cylinder, _evaluate_surface_many, _surface_derivatives_many

from .errors import MeshError

# GeometryModel.face_trim_loops_uv default, used by project_to_face(_many).
_TRIM_CURVE_SAMPLES = 17


def _signed_area(loop) -> float:
    points = np.asarray(loop, dtype=float)
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


class CylinderSupportFrame:
    """Face queries of one parameterized Cylinder face, in its support frame."""

    __slots__ = ("owner", "face_id", "face", "surface", "_trim_loops")

    def __init__(self, owner, face_id: int) -> None:
        if type(owner) is not GeometryModel:
            # The reproduced owner helpers are those of GeometryModel itself.
            raise MeshError("support-frame cylinder chart requires an owner GeometryModel")
        face = owner.faces[int(face_id)]
        surface = face.surface
        if (not isinstance(surface, Cylinder) or face.parameterization is None
                or face.parameterization is surface):
            raise MeshError("support-frame chart requires a Cylinder with a distinct parameterization")
        self.owner = owner
        self.face_id = int(face_id)
        self.face = face
        self.surface = surface
        self._trim_loops = None
        self._admit_orientation()

    def _admit_orientation(self) -> None:
        """Refuse unless the face's own orientation provably matches the support's.

        The support-frame mesh is oriented by the support normal (du x dv of
        the support), exactly like the twin's. The face's own normal comes from
        its parameterization, so a reversed parameterization would publish
        elements against the face normal. Admission is decided only where the
        orientation follows from the loop: a topology Coons map sends the unit
        square's counter-clockwise boundary onto the outer loop in loop order
        (ANYgeometry ``_topology_face_point_and_derivatives``), so its degree is
        the winding of that loop in the support (u, v) plane. Any other
        parameterization is undecided and refused. The loop must wind
        positively both as its vertex polygon and as its sampled trim polygon,
        and the parameterization normal at its centre must agree with the
        support normal there; otherwise the face is refused (fail closed).
        Folding (a sign change inside a positively wound map) is not decided
        here.
        """
        face, owner = self.face, self.owner
        parameterization = face.parameterization
        if (type(parameterization) is not CoonsSurface or parameterization.has_boundaries
                or len(face.corners) != 4):
            raise MeshError("support-frame cylinder chart admits only a topology Coons face "
                            "parameterization (its orientation relative to the support is undecided)")
        vertices = np.asarray([
            self.face_local_uv(self.face_id, owner.vertex_position(owner.oriented_start_vertex(item)))
            for item in face.loop], dtype=float)
        loops = (vertices, self.face_trim_loops_uv(self.face_id)[0])
        if any(len(loop) < 3 or not _signed_area(loop) > 0.0 for loop in loops):
            raise MeshError("support-frame cylinder chart refuses a face parameterization whose "
                            "orientation is reversed relative to its support")
        centre = np.asarray(owner.face_point(self.face_id, .5, .5), dtype=float)
        support_normal = self.face_normal_many(
            self.face_id, np.asarray([self.face_local_uv(self.face_id, centre)], dtype=float))[0]
        own_normal = np.asarray(owner.face_normal(self.face_id, .5, .5), dtype=float)
        if not float(own_normal @ support_normal) > 0.0:
            raise MeshError("support-frame cylinder chart refuses a face parameterization whose "
                            "orientation is reversed relative to its support")

    def _check(self, face_id) -> None:
        if int(face_id) != self.face_id:
            raise MeshError("support-frame chart is bound to another face")
        face = self.owner.faces.get(self.face_id)
        if face is not self.face or face.surface is not self.surface:
            raise MeshError("support-frame chart face changed after binding")

    # -- inverse --------------------------------------------------------------
    def face_local_uv(self, face_id, point):
        self._check(face_id)
        local = self.surface.local_uv(point)
        return (float(np.clip(local[0], 0.0, 1.0)), float(np.clip(local[1], 0.0, 1.0)))

    def face_local_uv_many(self, face_id, points):
        self._check(face_id)
        values = np.asarray(points, dtype=float)
        if values.size == 0:
            if values.ndim != 2 or values.shape[1] != 3:
                raise GeometryError("face local UV points must have shape (n, 3)")
            return np.empty((0, 2), dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise GeometryError("face local UV points must be finite (n, 3) coordinates")
        return np.clip(GeometryModel._builtin_local_uv_many(self.surface, values), 0.0, 1.0)

    # -- evaluation -----------------------------------------------------------
    def face_point(self, face_id, u, v):
        self._check(face_id)
        return np.asarray(self.surface.evaluate(float(u), float(v)), dtype=float)

    def evaluate_face_many(self, face_id, uv):
        self._check(face_id)
        values = self.owner._uv_array(uv)
        return _evaluate_surface_many(self.surface, values).reshape((-1, 3))

    def face_derivatives_many(self, face_id, uv):
        self._check(face_id)
        values = self.owner._uv_array(uv)
        return _surface_derivatives_many(self.surface, values)

    def face_normal_many(self, face_id, uv):
        du, dv = self.face_derivatives_many(face_id, uv)
        normals = np.cross(du, dv)
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths <= 0.0):
            raise GeometryError(f"face {self.face_id} has a degenerate normal")
        return normals / lengths[:, None]

    def face_normal(self, face_id, u, v):
        # GeometryModel.face_normal: one-sided finite differences of face_point.
        step = 1.0e-6
        u0, v0 = float(u), float(v)
        point = self.face_point(face_id, u0, v0)
        du = self.face_point(face_id, min(1.0, u0 + step), v0) - point
        dv = self.face_point(face_id, u0, min(1.0, v0 + step)) - point
        if float(np.linalg.norm(du)) <= 0.0:
            du = point - self.face_point(face_id, max(0.0, u0 - step), v0)
        if float(np.linalg.norm(dv)) <= 0.0:
            dv = point - self.face_point(face_id, u0, max(0.0, v0 - step))
        normal = np.cross(du, dv)
        length = float(np.linalg.norm(normal))
        if length <= 0.0:
            raise GeometryError(f"face {self.face_id} has a degenerate normal")
        return normal / length

    # -- trim-aware projection ------------------------------------------------
    def face_trim_loops_uv(self, face_id):
        self._check(face_id)
        if self._trim_loops is None:
            owner = self.owner

            def polygon(loop):
                points = []
                for item in loop:
                    edge = owner.edges[item.edge]
                    samples = owner.sample_edge(item.edge, np.linspace(
                        0.0, 1.0, 2 if isinstance(edge.curve, Straight) else _TRIM_CURVE_SAMPLES))
                    if not item.forward:
                        samples = samples[::-1]
                    points.extend(samples[:-1])
                if not points:
                    return np.asarray([], dtype=float)
                return np.asarray([self.face_local_uv(self.face_id, point) for point in points],
                                  dtype=float).reshape((-1, 2))

            self._trim_loops = tuple(polygon(loop) for loop in (self.face.loop,) + tuple(self.face.holes))
        return tuple(loop.copy() for loop in self._trim_loops)

    def _project_builtin(self, targets, trim_loops):
        owner = self.owner
        uv, inside_domain = owner._builtin_projection_uv_many(self.surface, targets)
        projected = _evaluate_surface_many(self.surface, uv).reshape((-1, 3))
        inside_trim = owner._points_in_face_trim(uv, trim_loops)
        for index in np.flatnonzero(~(inside_domain & inside_trim)):
            # Frame-free: the closest point of the face's boundary edges.
            boundary = owner._closest_face_boundary_point(self.face_id, targets[index])
            projected[index] = boundary
            uv[index] = np.asarray(self.face_local_uv(self.face_id, boundary), dtype=float)
        distances = np.linalg.norm(projected - targets, axis=1)
        return projected, uv, distances

    def project_to_face(self, face_id, point):
        target = np.asarray(point, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise GeometryError("projection point must be a finite 3-vector")
        trim_loops = self.face_trim_loops_uv(face_id)
        projected, uv, distances = self._project_builtin(target.reshape(1, 3), trim_loops)
        return projected[0], (float(uv[0, 0]), float(uv[0, 1])), float(distances[0])

    def project_to_face_many(self, face_id, xyz, initial_uv=None):
        self._check(face_id)
        points = np.asarray(xyz, dtype=float)
        if points.ndim == 1 and points.size == 0:
            points = points.reshape(0, 3)
        if points.ndim == 1 and points.shape == (3,):
            points = points.reshape(1, 3)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise GeometryError("projection points must be a finite (n, 3) array")
        if initial_uv is not None and len(self.owner._uv_array(initial_uv)) != len(points):
            raise GeometryError("initial_uv must match the number of projection points")
        if len(points) == 0:
            return np.empty((0, 3)), np.empty((0, 2)), np.empty(0)
        return self._project_builtin(points, self.face_trim_loops_uv(face_id))
