"""Identity-built source models for the opt-in irregular-shell layout."""
from __future__ import annotations

import sys
from math import ceil
from pathlib import Path

from anygeometry import GeometryModel, punch_hole
from anymesher.refinement import Refinement
from benchmarks.sg1.fixtures import build as build_sg1

CASES = ("planar_opening", "planar_concave", "cylinder_step", "cone_unequal", "ruled", "coons")
SIZES = {
    "planar_opening": (.8, .4), "planar_concave": (.8, .4),
    "cylinder_step": (.6, .3), "cone_unequal": (.6, .3),
    "ruled": (.6, .3), "coons": (.6, .3),
}


def build(case: str, h: float):
    if case == "planar_opening":
        geometry = GeometryModel()
        vertices = geometry.add_points(((0., 0., 0.), (3., .4, 0.),
                                        (3.5, 2.6, 0.), (.5, 2.2, 0.)))
        face = geometry.add_face(geometry.add_polyline(vertices, close=True), surface=None)
        face, _ = punch_hole(geometry, face, (1.8, 1.3, 0.), .25)
        faces = (face,)
        center = (1.8, 1.3, 0.)
        overrides = {}
    elif case == "planar_concave":
        geometry = GeometryModel()
        vertices = geometry.add_points(((0., 0., 0.), (4., 0., 0.),
                                        (4., .8, 0.), (1.2, .8, 0.),
                                        (1.2, 2.6, 0.), (0., 2.6, 0.)))
        face = geometry.add_face(geometry.add_polyline(vertices, close=True), surface=None)
        faces = (face,)
        center = (1.2, .8, 0.)
        overrides = {}
    elif case == "cylinder_step":
        tests = Path(__file__).resolve().parents[2] / "tests"
        if str(tests) not in sys.path:
            sys.path.insert(0, str(tests))
        from test_cylindrical_atlas_binding import _sector_model

        geometry, uses = _sector_model(True)
        faces = tuple(int(geometry.face_uses[item.id].face_id) for item in uses)
        center = (1., 0., 1.)
        overrides = {}
    elif case in ("cone_unequal", "ruled", "coons"):
        fixture = build_sg1({"cone_unequal": "CO", "ruled": "R", "coons": "C"}[case])
        geometry, faces = fixture.geometry, fixture.faces
        center = tuple(float(v) for v in fixture.center)
        overrides = {}
        if case == "cone_unequal":
            # One explicit owner-edge division breaks opposite-chain equality.
            edge = int(fixture.exterior[0])
            overrides[edge] = ceil(geometry.edge_length(edge) / h) + 1
    else:
        raise ValueError(f"unknown IS1 case {case!r}")
    return geometry, tuple(faces), tuple(center), overrides


def refinements(case: str, h: float, center):
    return (Refinement(size=h/3., radius=.35, center=center,
                       growth=1.5, name=f"IS1/{case}"),)
