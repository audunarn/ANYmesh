"""Analytic trim-certificate tests; no sampled curved-area oracle."""

from types import SimpleNamespace

import numpy as np
import pytest

from anygeometry.curves import Arc, Straight
from anygeometry.entities import OrientedEdge
from anygeometry.surfaces import Plane
from anymesher._trim_certificates import certified_complementary_arc_domains


def _model(*, turns=1, radius=.5):
    count = 12 * turns
    vertices = {1: np.array((-1., -1., 0.)), 2: np.array((1., -1., 0.)),
                3: np.array((1., 1., 0.)), 4: np.array((-1., 1., 0.))}
    endpoints = {i + 1: (i + 1, (i + 1) % 4 + 1) for i in range(4)}
    edges = {i: SimpleNamespace(curve=Straight()) for i in endpoints}
    for i in range(count):
        angle = i * np.pi / 6.
        middle = (i + .5) * np.pi / 6.
        vertices[10 + i] = np.array((radius * np.cos(angle), radius * np.sin(angle), 0.))
        vertices[100 + i] = np.array((radius * np.cos(middle), radius * np.sin(middle), 0.))
        endpoints[10 + i] = (10 + i, 10 + (i + 1) % count)
        edges[10 + i] = SimpleNamespace(curve=Arc(via_vertex=100 + i))
    plane = Plane(origin=np.zeros(3), u_vector=np.array((1., 0., 0.)), v_vector=np.array((0., 1., 0.)))
    outer = tuple(OrientedEdge(i, True) for i in range(1, 5))
    ring = tuple(OrientedEdge(10 + i, True) for i in range(count))
    hole = tuple(OrientedEdge(item.edge, False) for item in reversed(ring))
    model = SimpleNamespace(
        faces={26: SimpleNamespace(loop=outer, holes=(hole,), surface=plane),
               27: SimpleNamespace(loop=ring, holes=(), surface=plane)},
        edges=edges,
        vertex_position=lambda vertex: vertices[vertex],
        oriented_start_vertex=lambda item: endpoints[item.edge][0 if item.forward else 1],
        oriented_end_vertex=lambda item: endpoints[item.edge][1 if item.forward else 0],
        tolerance=SimpleNamespace(effective_length=lambda scale: 1.e-9 * scale),
        _entity_bounds=lambda key: (
            *np.min(np.asarray(tuple(vertices.values())), axis=0),
            *np.max(np.asarray(tuple(vertices.values())), axis=0),
        ),
    )
    return model, vertices, endpoints


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("rotate", [False, True])
def test_owned_complement_is_certified_without_mutation(reverse, rotate):
    model, vertices, _ = _model()
    if reverse:
        for face in model.faces.values():
            face.loop = tuple(OrientedEdge(item.edge, not item.forward) for item in reversed(face.loop))
            face.holes = tuple(tuple(OrientedEdge(item.edge, not item.forward) for item in reversed(loop))
                               for loop in face.holes)
    if rotate:
        rotation = np.array(((0., 0., 1.), (1., 0., 0.), (0., 1., 0.)))
        offset = np.array((2., -3., 4.))
        for key in vertices:
            vertices[key] = rotation @ vertices[key] + offset
        plane = Plane(origin=offset, u_vector=rotation[:, 0], v_vector=rotation[:, 1])
        for face in model.faces.values():
            face.surface = plane
    before = {key: value.tobytes() for key, value in vertices.items()}
    assert certified_complementary_arc_domains(model, 26, 27)
    assert certified_complementary_arc_domains(model, 27, 26)
    assert before == {key: value.tobytes() for key, value in vertices.items()}


def test_two_turns_are_not_a_simple_trim():
    model, _, _ = _model(turns=2)
    assert not certified_complementary_arc_domains(model, 26, 27)


@pytest.mark.parametrize("radius", [1., 1.1])
def test_trim_contact_or_overlap_has_no_exemption(radius):
    model, _, _ = _model(radius=radius)
    assert not certified_complementary_arc_domains(model, 26, 27)


@pytest.mark.parametrize("defect", ["open", "bowtie", "nonplanar", "long_arc", "not_complementary"])
def test_unproven_domains_retain_owner_path(defect):
    model, vertices, endpoints = _model()
    if defect == "open":
        endpoints[10] = (10, 12)
    elif defect == "bowtie":
        vertices[2], vertices[3] = vertices[3], vertices[2]
    elif defect == "nonplanar":
        vertices[100][2] = .01
    elif defect == "long_arc":
        vertices[100] = -vertices[100]
    else:
        model.faces[26].holes = ()
    assert not certified_complementary_arc_domains(model, 26, 27)


def test_certificate_cancellation_propagates():
    model, _, _ = _model()
    failure = RuntimeError("cancel certificate")

    def cancel(phase):
        assert phase == "structural preparation overlap narrow phase"
        raise failure

    with pytest.raises(RuntimeError) as caught:
        certified_complementary_arc_domains(model, 26, 27, cancellation_check=cancel)
    assert caught.value is failure


def _translate(model, vertices, offset):
    for key in vertices:
        vertices[key] = vertices[key] + offset
    for face in model.faces.values():
        surface = face.surface
        face.surface = Plane(
            origin=surface.origin + offset,
            u_vector=surface.u_vector,
            v_vector=surface.v_vector,
        )


@pytest.mark.parametrize("factor", [1.e-12, 1., 1.e12])
def test_chart_rescaling_keeps_physical_tolerance_and_certificate(factor):
    model, _, _ = _model()
    extents = []

    def tolerance(extent):
        extents.append(extent)
        return 1.e-9 * extent

    model.tolerance = SimpleNamespace(effective_length=tolerance)
    for face in model.faces.values():
        surface = face.surface
        face.surface = Plane(
            origin=surface.origin,
            u_vector=surface.u_vector * factor,
            v_vector=surface.v_vector / factor,
        )
    assert certified_complementary_arc_domains(model, 26, 27)
    assert extents == [np.sqrt(8.)]


@pytest.mark.parametrize("distance,expected", [(10., True), (1.e9, False)])
def test_translation_does_not_inflate_owner_tolerance(distance, expected):
    model, vertices, _ = _model()
    _translate(model, vertices, np.array((distance, -distance, distance)))
    extents = []

    def tolerance(extent):
        extents.append(extent)
        return 1.e-9 * extent

    model.tolerance = SimpleNamespace(effective_length=tolerance)
    assert certified_complementary_arc_domains(model, 26, 27) is expected
    assert extents == [np.sqrt(8.)]


@pytest.mark.parametrize("distance", [0., 1.e9])
@pytest.mark.parametrize("factor", [1., 1.e12])
def test_out_of_plane_defect_cannot_borrow_coordinate_or_chart_scale(distance, factor):
    model, vertices, _ = _model()
    vertices[100][2] = 1.e-6
    _translate(model, vertices, np.array((distance, -distance, distance)))
    for face in model.faces.values():
        surface = face.surface
        face.surface = Plane(
            origin=surface.origin,
            u_vector=surface.u_vector * factor,
            v_vector=surface.v_vector * factor,
        )
    assert not certified_complementary_arc_domains(model, 26, 27)


@pytest.mark.parametrize("bounds", [None, (0., 0., 0., -1., 1., 1.), (0., 0., 0., np.inf, 1., 1.)])
def test_missing_or_invalid_owner_bounds_fail_closed(bounds):
    model, _, _ = _model()
    model._entity_bounds = lambda key: bounds
    assert not certified_complementary_arc_domains(model, 26, 27)
