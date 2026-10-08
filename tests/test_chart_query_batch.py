"""Focused tests for batched trimmed-chart qualification (M6)."""

from __future__ import annotations

import numpy as np
import pytest
from anygeometry import GeometryModel, query_trimmed_surface_charts
from anygeometry.errors import GeometryError
from anygeometry.surfaces import CoonsSurface

from anymesher._runtime_counters import (
    MESHER_CHART_QUERY_BATCHES,
    MESHER_CHART_QUERY_FACES,
    operation_counts_scope,
)
from anymesher.preparation import _face_chart_errors, prepare_structural_closure


def _plates(count: int) -> tuple[GeometryModel, tuple[int, ...]]:
    geometry = GeometryModel()
    faces = []
    for index in range(count):
        x = 3.0 * index
        vertices = geometry.add_points(
            ((x, 0.0, 0.0), (x + 2.0, 0.0, 0.0), (x + 2.0, 1.0, 0.0), (x, 1.0, 0.0))
        )
        faces.append(geometry.add_plate(vertices))
    return geometry, tuple(faces)


def _coons_face(geometry: GeometryModel, offset: float) -> int:
    p00 = (offset, 0.0, 0.0)
    p10 = (offset + 2.0, 0.0, 0.0)
    p01 = (offset, 1.0, 0.5)
    p11 = (offset + 2.0, 1.0, 0.5)
    surface = CoonsSurface(
        bottom=np.asarray((p00, p10)),
        right=np.asarray((p10, p11)),
        top=np.asarray((p01, p11)),
        left=np.asarray((p00, p01)),
    )
    uv = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    points = [geometry.add_point(*surface.evaluate(u, v)) for u, v in uv]
    # A spline top boundary makes the Coons face genuinely curved, which
    # chart qualification rejects exactly as the legacy branch expects.
    mid = geometry.add_point(*surface.evaluate(0.5, 1.0))
    edges = [
        geometry.add_line(points[0], points[1]),
        geometry.add_line(points[1], points[2]),
        geometry.add_spline(points[2], (mid,), points[3]),
        geometry.add_line(points[3], points[0]),
    ]
    return geometry.add_face(edges, corners=(0, 1, 2, 3), surface=surface)


def test_one_batched_by_face_query_serves_all_selected_faces(monkeypatch) -> None:
    import anygeometry

    geometry, faces = _plates(3)
    real = anygeometry.query_trimmed_surface_charts_by_face
    calls: list[tuple[int, ...]] = []

    def counting(model, operands=None, **kwargs):
        calls.append(tuple(int(handle.id) for handle in operands))
        return real(model, operands, **kwargs)

    monkeypatch.setattr(
        anygeometry, "query_trimmed_surface_charts_by_face", counting
    )
    counts: dict[str, int] = {}
    with operation_counts_scope(counts):
        errors = _face_chart_errors(geometry, faces, None, None)
    assert calls == [faces]
    assert errors == {face: None for face in faces}
    assert counts == {MESHER_CHART_QUERY_BATCHES: 1, MESHER_CHART_QUERY_FACES: 3}


def test_per_face_errors_are_preserved_against_single_queries() -> None:
    geometry, plates = _plates(1)
    curved = _coons_face(geometry, 5.0)
    faces = (*plates, curved)
    errors = _face_chart_errors(geometry, faces, None, None)
    for face in faces:
        try:
            query_trimmed_surface_charts(
                geometry, (geometry.handle("face", face),)
            )
            expected = None
        except GeometryError as error:
            expected = str(error)
        assert errors[face] == expected


def test_older_geometry_without_the_batch_api_keeps_per_face_queries(
    monkeypatch,
) -> None:
    import anygeometry

    geometry, faces = _plates(2)
    monkeypatch.delattr(
        anygeometry, "query_trimmed_surface_charts_by_face", raising=False
    )
    counts: dict[str, int] = {}
    with operation_counts_scope(counts):
        errors = _face_chart_errors(geometry, faces, None, None)
    assert errors == {face: None for face in faces}
    assert counts == {}


class _Row:
    def __init__(self, face_id, error) -> None:
        self.face = type("Handle", (), {"id": face_id})()
        self.error = error


class _Rows:
    def __init__(self, rows) -> None:
        self.results = tuple(rows)


def test_batch_rows_map_to_per_face_errors(monkeypatch) -> None:
    import anygeometry

    geometry, faces = _plates(2)
    monkeypatch.setattr(
        anygeometry,
        "query_trimmed_surface_charts_by_face",
        lambda model, operands=None, **kwargs: _Rows(
            (
                _Row(faces[0], "face 0 has curved Coons boundaries"),
                _Row(faces[1], None),
            )
        ),
    )
    counts: dict[str, int] = {}
    with operation_counts_scope(counts):
        errors = _face_chart_errors(geometry, faces, None, None)
    assert errors == {
        faces[0]: "face 0 has curved Coons boundaries",
        faces[1]: None,
    }
    assert counts == {MESHER_CHART_QUERY_BATCHES: 1, MESHER_CHART_QUERY_FACES: 2}


def test_batch_query_guards_abort_the_whole_preparation(monkeypatch) -> None:
    import anygeometry

    geometry, faces = _plates(2)

    def stale(model, operands=None, **kwargs):
        raise GeometryError("trimmed surface chart query revision is stale")

    monkeypatch.setattr(anygeometry, "query_trimmed_surface_charts_by_face", stale)
    with pytest.raises(GeometryError, match="revision is stale"):
        _face_chart_errors(geometry, faces, None, None)


def test_legacy_curved_classification_matches_the_per_face_route(monkeypatch) -> None:
    import anygeometry

    def coons_model() -> GeometryModel:
        geometry = GeometryModel()
        first = _coons_face(geometry, 0.0)
        second = _coons_face(geometry, 4.0)
        return geometry, (first, second)

    geometry, faces = coons_model()
    counts: dict[str, int] = {}
    with operation_counts_scope(counts):
        working, report = prepare_structural_closure(
            geometry, face_ids=faces, beam_edges=()
        )
    assert report is not None
    assert (
        "legacy curved topology discretization; no automatic interior joints"
        in report.diagnostics
    )
    assert counts == {
        MESHER_CHART_QUERY_BATCHES: 1,
        MESHER_CHART_QUERY_FACES: 2,
    }

    # The same model through the older per-face route classifies identically.
    geometry, faces = coons_model()
    monkeypatch.delattr(
        anygeometry, "query_trimmed_surface_charts_by_face", raising=False
    )
    counts = {}
    with operation_counts_scope(counts):
        working_legacy, report_legacy = prepare_structural_closure(
            geometry, face_ids=faces, beam_edges=()
        )
    assert report_legacy is not None
    assert report_legacy.diagnostics == report.diagnostics
    assert counts == {}


def test_controls_pass_through_to_the_batch_query(monkeypatch) -> None:
    import anygeometry

    geometry, faces = _plates(2)
    real = anygeometry.query_trimmed_surface_charts_by_face
    seen: dict = {}

    def recording(model, operands=None, **kwargs):
        seen.update(kwargs)
        return real(model, operands, **kwargs)

    monkeypatch.setattr(
        anygeometry, "query_trimmed_surface_charts_by_face", recording
    )

    def cancellation_check(phase):
        return None

    errors = _face_chart_errors(
        geometry, faces, cancellation_check, geometry.revision
    )
    assert errors == {face: None for face in faces}
    assert seen["expected_revision"] == geometry.revision
    assert seen["cancellation_check"] is cancellation_check


def test_controls_pass_through_to_the_per_face_fallback(monkeypatch) -> None:
    import anygeometry

    geometry, faces = _plates(2)
    monkeypatch.delattr(
        anygeometry, "query_trimmed_surface_charts_by_face", raising=False
    )
    real = anygeometry.query_trimmed_surface_charts
    seen: list[dict] = []

    def recording(model, operands=None, **kwargs):
        seen.append(kwargs)
        return real(model, operands, **kwargs)

    monkeypatch.setattr(anygeometry, "query_trimmed_surface_charts", recording)

    def cancellation_check(phase):
        return None

    errors = _face_chart_errors(
        geometry, faces, cancellation_check, geometry.revision
    )
    assert errors == {face: None for face in faces}
    assert seen and all(
        item["expected_revision"] == geometry.revision
        and item["cancellation_check"] is cancellation_check
        for item in seen
    )


class _Cancelled(Exception):
    pass


def test_prepare_structural_closure_callback_cancellation_aborts() -> None:
    geometry, faces = _plates(2)

    def cancellation_check(phase):
        if "trimmed surface chart" in phase:
            raise _Cancelled("cancelled during " + phase)
        return None

    with pytest.raises(_Cancelled):
        prepare_structural_closure(
            geometry, face_ids=faces, cancellation_check=cancellation_check
        )


def test_prepare_structural_closure_callback_mutation_aborts(monkeypatch) -> None:
    import anygeometry

    geometry, faces = _plates(2)
    real = anygeometry.query_trimmed_surface_charts_by_face
    holder: dict = {}

    def recording(model, operands=None, **kwargs):
        holder["model"] = model
        return real(model, operands, **kwargs)

    monkeypatch.setattr(
        anygeometry, "query_trimmed_surface_charts_by_face", recording
    )

    def cancellation_check(phase):
        model = holder.get("model")
        if model is not None and "trimmed surface chart" in phase:
            model.add_point(99.0, 99.0, 99.0)
        return None

    with pytest.raises(
        GeometryError, match="geometry changed during trimmed chart query"
    ):
        prepare_structural_closure(
            geometry, face_ids=faces, cancellation_check=cancellation_check
        )


def test_isolated_single_face_route_issues_no_chart_query(monkeypatch) -> None:
    import anygeometry

    geometry, faces = _plates(1)
    calls: list[str] = []
    real_batch = getattr(
        anygeometry, "query_trimmed_surface_charts_by_face", None
    )
    real_single = anygeometry.query_trimmed_surface_charts

    if real_batch is not None:
        def batch(model, operands=None, **kwargs):
            calls.append("batch")
            return real_batch(model, operands, **kwargs)

        monkeypatch.setattr(
            anygeometry, "query_trimmed_surface_charts_by_face", batch
        )

    def single(model, operands=None, **kwargs):
        calls.append("single")
        return real_single(model, operands, **kwargs)

    monkeypatch.setattr(anygeometry, "query_trimmed_surface_charts", single)
    counts: dict[str, int] = {}
    with operation_counts_scope(counts):
        working, report = prepare_structural_closure(geometry, face_ids=faces)
    assert working is not None
    assert calls == []
    assert counts == {}
