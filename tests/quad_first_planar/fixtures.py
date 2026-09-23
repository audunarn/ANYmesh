from __future__ import annotations

import math

from anygeometry.model import GeometryModel


def _transform(points: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    angle = math.radians(30.0)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return tuple(
        (cos_a * x - sin_a * y + 7.0, sin_a * x + cos_a * y - 3.0, z + 2.0)
        for x, y, z in points
    )


def p01_geometry() -> tuple[GeometryModel, int]:
    """Return the frozen P01 one-face 10 m x 6 m rectangle."""
    geometry = GeometryModel()
    vertices = geometry.add_points(
        (
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 6.0, 0.0),
            (0.0, 6.0, 0.0),
        )
    )
    face = geometry.add_plate(vertices)
    return geometry, face


def p02_geometry() -> tuple[GeometryModel, int]:
    """Return the P02 rigidly rotated (+30 deg Z) and translated rectangle."""
    geometry = GeometryModel()
    vertices = geometry.add_points(
        tuple(_transform(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 6.0, 0.0), (0.0, 6.0, 0.0))))
    )
    face = geometry.add_plate(vertices)
    return geometry, face
