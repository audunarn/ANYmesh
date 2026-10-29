"""S1: geometry-level admission predicate for the NURBS tensor route, with stable reason codes."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from anygeometry import GeometryModel, Plane
from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface

from test_qf_nurbs_s0_characterisation import nurbs_patch
from anymesher._parametric_tensor import TensorAdmission, nurbs_tensor_admission


class Proxy:
    """Real geometry with selected face attributes overridden (faces are immutable owner objects)."""
    def __init__(self, model, face_id, **changes):
        self._model, self._face_id = model, face_id
        face = model.faces[face_id]
        data = {name: getattr(face, name) for name in ('surface', 'parameterization', 'loop', 'holes')}
        data.update(changes)
        self.faces = {face_id: SimpleNamespace(**data)}

    def __getattr__(self, name):
        return getattr(self._model, name)


def test_untrimmed_four_edge_patch_is_admitted():
    model, face = nurbs_patch()
    result = nurbs_tensor_admission(model, face)
    assert isinstance(result, TensorAdmission) and result.eligible and result.code == 'eligible'


def test_warped_patch_is_admitted_geometrically():
    model, face = nurbs_patch(warp=.5)
    assert nurbs_tensor_admission(model, face).eligible


def test_triangle_is_refused_with_code():
    model, face = nurbs_patch(triangle=True)
    result = nurbs_tensor_admission(model, face)
    assert not result.eligible and result.code == 'not_four_edges'


def test_hole_is_refused_with_code():
    model, face = nurbs_patch()
    result = nurbs_tensor_admission(Proxy(model, face, holes=(model.faces[face].loop,)), face)
    assert not result.eligible and result.code == 'holes'


def test_plane_face_is_not_a_nurbs_face():
    model = GeometryModel()
    pts = [model.add_point(*p) for p in ((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.))]
    face = model.add_face([model.add_line(pts[i], pts[(i + 1) % 4]) for i in range(4)])
    result = nurbs_tensor_admission(model, face)
    assert not result.eligible and result.code == 'not_nurbs_surface'


def test_missing_face_is_refused_not_raised():
    model, _ = nurbs_patch()
    assert nurbs_tensor_admission(model, 10_000).code == 'no_such_face'


def test_no_boundary_uv_is_refused():
    model, face = nurbs_patch()
    class NoUV(Proxy):
        def face_boundary_uv(self, *a, **k):
            return None
    result = nurbs_tensor_admission(NoUV(model, face), face)
    assert not result.eligible and result.code == 'no_boundary_uv'


def test_boundary_off_the_uv_box_is_refused():
    """A diagonal boundary edge is not an iso-parametric side of a rectangle."""
    model, face = nurbs_patch()
    class Diagonal(Proxy):
        def face_boundary_uv(self, face_id, edge, t):
            t = np.asarray(t, dtype=float)
            if edge == self._model.faces[face_id].loop[0].edge:
                return np.stack((t, t), axis=-1)
            return self._model.face_boundary_uv(face_id, edge, t)
    result = nurbs_tensor_admission(Diagonal(model, face), face)
    assert not result.eligible and result.code == 'not_iso_parametric_rectangle'


def test_collapsed_side_is_a_degenerate_pole():
    """The owner refuses to author a collapsed side, so present a face whose support collapses the v=1 side."""
    model, face = nurbs_patch()
    pole = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((0., 0., 0.), (0., 0., 1.)), ((1., 0., 0.), (0., 0., 1.))))
    result = nurbs_tensor_admission(Proxy(model, face, surface=pole), face)
    assert not result.eligible and result.code == 'degenerate_edge'
