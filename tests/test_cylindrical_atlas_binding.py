"""Owner-atlas consumption tests, not cylindrical meshing activation tests."""

from dataclasses import replace
from contextlib import ExitStack
from fractions import Fraction
import math

import numpy as np
import pytest

from anygeometry import (
    Cylinder,
    CylinderAtlasError,
    CylinderAtlasErrorCode,
    CylinderAtlasPolicy,
    CylinderAtlasStatus,
    CylinderOccurrenceRequest,
    GeometryModel,
    OrientedEdge,
    query_cylinder_atlas,
)
from anymesher._cylindrical_atlas import prepare_cylindrical_atlas
from anymesher.errors import MeshError


def _sector_model(hole=False):
    """Public owner authoring with a symbolic registry, never coordinate welding."""
    model = GeometryModel()
    vertices, edges = {}, {}
    faces = []

    def vertex(angle, z):
        key = (Fraction(angle) % 8, Fraction(z))
        if key not in vertices:
            theta = float(key[0]) * math.pi / 4
            vertices[key] = model.add_point(math.cos(theta), math.sin(theta), float(z))
        return vertices[key]

    def edge(a, b):
        ka, kb = (a[0] % 8, a[1]), (b[0] % 8, b[1])
        circular = a[1] == b[1]
        key = (circular, *sorted((ka, kb)))
        start, end = vertex(*a), vertex(*b)
        if key not in edges:
            edges[key] = (
                model.add_arc(start, vertex((a[0] + b[0]) / 2, a[1]), end)
                if circular else model.add_line(start, end)
            )
        identifier = edges[key]
        return OrientedEdge(identifier, model.edges[identifier].start == start)

    rectangle = ((0, 0), (1, 0), (1, 2), (0, 2))
    first = ((0, 0), (1, 0), (1, 2), (0, 2), (0, 1.25), (.25, 1.25), (.25, .75), (0, .75))
    last = ((0, 0), (1, 0), (1, .75), (.75, .75), (.75, 1.25), (1, 1.25), (1, 2), (0, 2))
    with model.transaction():
        for index in range(8):
            local = first if hole and index == 0 else last if hole and index == 7 else rectangle
            points = tuple((Fraction(index) + Fraction(u), Fraction(z)) for u, z in local)
            loop = tuple(edge(a, b) for a, b in zip(points, points[1:] + points[:1]))
            surface = Cylinder(
                origin=(0., 0., 0.), axis=(0., 0., 1.), radial_direction=(1., 0., 0.),
                radius=1., height=2., start_angle=index * math.pi / 4, sweep_angle=math.pi / 4,
            )
            faces.append(model.add_face_from_loop(loop, surface=surface))
        part = model.add_part(name="mesher atlas consumer")
        sheet = model.add_sheet(faces, part_id=part)
    selected = tuple(model.handle("face_use", key) for key in model.sheets[sheet].face_use_ids)
    return model, selected


@pytest.fixture(scope="module", params=((False, 0), (True, 0), (True, 3)))
def prepared(request):
    hole, reference_index = request.param
    model, selected = _sector_model(hole)
    before = repr(model.__dict__)
    binding = prepare_cylindrical_atlas(
        model, selected, reference_face_use=selected[reference_index]
    )
    assert repr(model.__dict__) == before
    return model, selected, binding, hole


def test_owner_seam_keys_and_native_parameter_are_preserved(prepared):
    model, selected, binding, hole = prepared
    before = repr(model.__dict__)
    seam = next(item for item in binding.atlas.interfaces if item.is_reference_seam)
    parameters = (0., .125, .5, .875, 1.)
    requests = tuple(
        CylinderOccurrenceRequest(occurrence, t)
        for t in parameters for occurrence in seam.occurrences
    )
    result = binding.evaluate(requests)
    for t, a, b in zip(parameters, result.samples[::2], result.samples[1::2]):
        assert a.parameter == b.parameter == t
        assert a.edge == b.edge == seam.edge
        assert a.equivalence_key == b.equivalence_key
        assert a.point == b.point
    assert binding.face_uses == selected
    assert len(binding.charts) == 8
    assert sorted(c.winding for c in binding.atlas.boundary_cycles) == (
        [-1, 0, 1] if hole else [-1, 1]
    )
    if hole:
        for interface in binding.atlas.interfaces:
            owner_edge = model.edges[interface.edge.id]
            assert {
                model.vertices[owner_edge.start].position[2],
                model.vertices[owner_edge.end].position[2],
            } != {.75, 1.25}
    assert repr(model.__dict__) == before


def test_sector_charts_use_physical_lengths_and_keep_face_ownership(prepared):
    model, _, binding, _ = prepared
    before = repr(model.__dict__)
    for sector, (face_use, chart) in zip(binding.atlas.sectors, binding.charts):
        assert face_use == sector.face_use
        assert chart.face_chart.face_id == sector.face.id
        assert chart.circumferential_length == pytest.approx(math.pi / 4)
        assert chart.axial_length == 2.0
        assert chart.model_id == binding.atlas.model_id
        assert chart.revision == binding.atlas.revision
    chart = binding.chart_for(binding.charts[0][0])
    uv = np.array(((0., 0.), (.25, .5), (1., 1.)))
    np.testing.assert_allclose(chart.to_parameters(chart.to_chart(uv)), uv)
    assert repr(model.__dict__) == before


def test_caller_selection_tampering_remains_owner_error(prepared):
    _, selected, binding, _ = prepared
    with pytest.raises(CylinderAtlasError):
        replace(binding, face_uses=selected[::-1]).validate()
    wrong_reference = next(item for item in selected if item != binding.reference_face_use)
    with pytest.raises(CylinderAtlasError):
        replace(binding, reference_face_use=wrong_reference).evaluate(())


def test_owner_cancellation_exception_is_not_relabelled(prepared):
    _, _, binding, _ = prepared
    error = RuntimeError("cancel atlas consumer")

    def cancel(phase):
        raise error

    with pytest.raises(RuntimeError) as caught:
        binding.evaluate((), cancellation_check=cancel)
    assert caught.value is error


def test_stale_model_cannot_consume_owner_rows_or_physical_chart():
    model, selected = _sector_model()
    binding = prepare_cylindrical_atlas(model, selected, reference_face_use=selected[0])
    with model.transaction():
        model.add_point(10., 10., 10.)
    with pytest.raises(CylinderAtlasError):
        binding.evaluate(())
    with pytest.raises(MeshError):
        binding.charts[0][1].to_chart(((0., 0.),))


def test_unqualified_owner_budget_does_not_publish_a_binding():
    model, selected = _sector_model()
    before = repr(model.__dict__)
    with pytest.raises(CylinderAtlasError):
        prepare_cylindrical_atlas(
            model, selected, reference_face_use=selected[0],
            policy=CylinderAtlasPolicy(max_interval_operations=1),
        )
    assert repr(model.__dict__) == before


@pytest.mark.parametrize("open_transaction", (False, True))
def test_final_callback_owner_changes_cannot_publish_binding(open_transaction):
    model, selected = _sector_model()
    phases = []
    with ExitStack() as transactions:
        def callback(phase):
            phases.append(phase)
            if phase == "cylindrical atlas preparation complete":
                if open_transaction:
                    transactions.enter_context(model.transaction())
                else:
                    with model.transaction():
                        model.add_point(10., 10., 10.)

        with pytest.raises(CylinderAtlasError) as caught:
            prepare_cylindrical_atlas(
                model, selected, reference_face_use=selected[0],
                cancellation_check=callback,
            )
        assert caught.value.code is (
            CylinderAtlasErrorCode.BUSY_MODEL if open_transaction
            else CylinderAtlasErrorCode.STALE_REVISION
        )
        assert phases[-1] == "cylindrical atlas preparation complete"
        assert phases.count("cylindrical atlas preparation complete") == 1


def test_normal_budget_incomplete_cylinder_is_refused_before_charts(monkeypatch):
    model, selected = _sector_model()
    selected = selected[:4]  # Genuine half-cylinder selection, not a full-period atlas.
    before = repr(model.__dict__)
    result = query_cylinder_atlas(
        model, selected, reference_face_use=selected[0], expected_revision=model.revision,
    )
    assert result.status is not CylinderAtlasStatus.QUALIFIED
    assert not result.certificate.complete

    def forbidden_chart(*args, **kwargs):
        pytest.fail("unqualified atlas must be refused before chart construction")

    monkeypatch.setattr(
        "anymesher._cylindrical_atlas.CylindricalMetricChart.from_geometry", forbidden_chart
    )
    with pytest.raises(CylinderAtlasError) as caught:
        prepare_cylindrical_atlas(model, selected, reference_face_use=selected[0])
    assert caught.value.code is CylinderAtlasErrorCode.UNQUALIFIED_RESULT
    assert caught.value.diagnostics == result.diagnostics
    assert repr(model.__dict__) == before
