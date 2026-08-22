"""Atomic detached application and exact feature adoption regressions."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from anygeometry import (
    Cylinder,
    EntityRef,
    GeometryModel,
    OrientedEdge,
    Plane,
    to_dict,
)
from anygeometry.features import FeatureHistory
from anygeometry.operations import trim_face

from anymesher import MeshError
from anymesher.structured import (
    StructuredLayoutReport,
    apply_structured_layout,
    commit_structured_layout,
    plan_structured_layout,
)

HAS_FROZEN_ADOPTION = hasattr(FeatureHistory, "adopt_frozen")


PLANE = Plane(
    np.asarray((0.0, 0.0, 0.0)),
    np.asarray((1.0, 0.0, 0.0)),
    np.asarray((0.0, 1.0, 0.0)),
)


def _document_bytes(geometry: GeometryModel) -> bytes:
    return json.dumps(
        to_dict(geometry, include_features=True),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _pentagon(*, sheet: bool = False) -> tuple[GeometryModel, int]:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        (
            (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.5, 1.0, 0.0),
            (1.0, 2.0, 0.0), (-0.5, 1.0, 0.0),
        )
    )
    face = geometry.add_face_from_loop(
        geometry.order_loop(geometry.add_polyline(vertices, close=True)),
        surface=PLANE,
    )
    if sheet:
        geometry.add_sheet((face,))
    return geometry, face


def _planar_ogrid() -> tuple[GeometryModel, int]:
    geometry = GeometryModel()
    outer = geometry.add_points(
        ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 4.0, 0.0), (0.0, 4.0, 0.0))
    )
    face = geometry.add_face_from_loop(
        geometry.order_loop(geometry.add_polyline(outer, close=True)), surface=PLANE
    )
    hole = geometry.add_points(
        ((1.0, 1.0, 0.0), (3.0, 1.0, 0.0), (3.0, 3.0, 0.0), (1.0, 3.0, 0.0))
    )
    hole_edges = geometry.add_polyline(hole, close=True)
    trim_face(
        geometry,
        face,
        (tuple(OrientedEdge(edge, True) for edge in hole_edges),),
    )
    return geometry, face


def _neutral_quad() -> tuple[GeometryModel, int]:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    face = geometry.add_face_from_loop(
        geometry.order_loop(geometry.add_polyline(vertices, close=True)),
        surface=PLANE,
    )
    return geometry, face


def _unmapped_cylinder() -> tuple[GeometryModel, int]:
    geometry = GeometryModel()
    surface = Cylinder(
        np.zeros(3),
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((1.0, 0.0, 0.0)),
        2.0,
        3.0,
        0.0,
        0.5 * np.pi,
    )
    lower = geometry.add_points(tuple(surface.evaluate(u, 0.0) for u in (0.0, 0.5, 1.0)))
    upper = geometry.add_points(tuple(surface.evaluate(u, 1.0) for u in (0.0, 0.5, 1.0)))
    bottom = geometry.add_arc(*lower)
    far_side = geometry.add_line(lower[-1], upper[-1])
    top = geometry.add_arc(*upper)
    near_side = geometry.add_line(lower[0], upper[0])
    face = geometry.add_face_from_loop(
        (
            OrientedEdge(bottom, True), OrientedEdge(far_side, True),
            OrientedEdge(top, False), OrientedEdge(near_side, False),
        ),
        surface=surface,
    )
    return geometry, face


def test_direct_apply_is_exact_and_source_document_is_byte_identical() -> None:
    geometry, face = _pentagon(sheet=True)
    before = _document_bytes(geometry)
    plan = plan_structured_layout(geometry, target_size=0.5)

    working, report = apply_structured_layout(geometry, plan)

    assert _document_bytes(geometry) == before
    assert working.model_id != geometry.model_id
    assert report.plan.model_id == str(geometry.model_id)
    assert report.source_to_working_faces[face] == tuple(
        item.working_face_id for item in report.blocks
    )
    assert len(report.blocks) == 5
    assert all(item.working_face_id in working.faces for item in report.blocks)
    assert len(report.seed_equations) == 10
    for equation in report.seed_equations:
        assert sum(report.seed_solution[item] for item in equation.left_edges) == sum(
            report.seed_solution[item] for item in equation.right_edges
        )
    sheet = next(iter(working.sheets.values()))
    owned_faces = {
        working.face_uses[use_id].face_id for use_id in sheet.face_use_ids
    }
    assert owned_faces == set(report.source_to_working_faces[face])
    assert working.validate_topology() == ()
    assert report.working_model_id == str(working.model_id)
    assert report.working_revision == working.revision
    assert report.to_dict()["working_binding"].startswith("sha256:")
    with pytest.raises(MeshError, match="working binding is inconsistent"):
        replace(report, working_binding="sha256:" + "0" * 64)


@pytest.mark.parametrize(
    ("factory", "action"),
    (
        (_pentagon, "radial_partition"),
        (_neutral_quad, "promote_quad"),
        (_planar_ogrid, "ogrid_partition"),
    ),
)
def test_every_partition_preserves_face_metadata_and_parameterization(
    factory,
    action: str,
) -> None:
    geometry, face = factory()
    parameterization = Plane(
        np.asarray((0.25, 0.5, 0.0)),
        np.asarray((2.0, 0.0, 0.0)),
        np.asarray((0.0, 3.0, 0.0)),
    )
    geometry.set_face_metadata(face, {"section": "A", "nested": {"id": 7}})
    geometry.set_face_parameterization(face, parameterization)
    plan = plan_structured_layout(geometry, target_size=0.5)
    assert plan.faces[0].action == action

    working, report = apply_structured_layout(geometry, plan)

    for child in report.source_to_working_faces[face]:
        made = working.faces[child]
        assert made.metadata.to_dict() == {"section": "A", "nested": {"id": 7}}
        assert made.parameterization is not None
        assert np.allclose(
            made.parameterization.evaluate(0.2, 0.7),
            parameterization.evaluate(0.2, 0.7),
        )


def test_planar_ogrid_has_eight_exact_descendant_blocks() -> None:
    geometry, face = _planar_ogrid()
    before = _document_bytes(geometry)
    plan = plan_structured_layout(geometry, target_size=0.5)

    assert plan.faces[0].action == "ogrid_partition"
    working, report = apply_structured_layout(geometry, plan)

    assert _document_bytes(geometry) == before
    assert len(report.source_to_working_faces[face]) == 8
    assert len(report.blocks) == 8
    assert len(report.seed_equations) == 16
    assert all(len(working.faces[int(item.working_face_id)].corners) == 4 for item in report.blocks)


def test_adjacent_radial_and_existing_mapped_face_share_split_side_chain() -> None:
    geometry = GeometryModel()
    a, b, c, d, e, u, v = geometry.add_points(
        (
            (0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
            (2.5, 1.0, 0.0), (1.0, 2.0, 0.0), (-0.5, 1.0, 0.0),
            (2.0, -1.0, 0.0), (0.0, -1.0, 0.0),
        )
    )
    shared = geometry.add_line(a, b)
    radial = geometry.add_face_from_loop(
        (
            OrientedEdge(shared, True),
            OrientedEdge(geometry.add_line(b, c), True),
            OrientedEdge(geometry.add_line(c, d), True),
            OrientedEdge(geometry.add_line(d, e), True),
            OrientedEdge(geometry.add_line(e, a), True),
        ),
        surface=PLANE,
    )
    mapped = geometry.add_face_from_loop(
        (
            OrientedEdge(shared, False),
            OrientedEdge(geometry.add_line(a, v), True),
            OrientedEdge(geometry.add_line(v, u), True),
            OrientedEdge(geometry.add_line(u, b), True),
        ),
        (0, 1, 2, 3),
        surface=PLANE,
    )
    before = _document_bytes(geometry)
    plan = plan_structured_layout(geometry, target_size=0.5)
    assert [item.action for item in plan.faces] == ["radial_partition", "mapped"]

    working, report = apply_structured_layout(geometry, plan)

    assert _document_bytes(geometry) == before
    assert report.source_to_working_faces[mapped] == (mapped,)
    assert len(report.source_to_working_faces[radial]) == 5
    assert len(report.source_to_working_edges[shared]) == 2
    assert len(working.faces[mapped].sides()[0]) == 2
    mapped_block = next(item for item in report.blocks if item.source_face_id == mapped)
    mapped_equations = [item for item in report.seed_equations if item.block_id == mapped_block.block_id]
    assert len(mapped_equations) == 2


def test_unmapped_cylinder_remains_native_and_never_uses_ogrid() -> None:
    geometry, face = _unmapped_cylinder()
    plan = plan_structured_layout(geometry, target_size=0.5)

    assert plan.faces[0].source_face_id == face
    assert plan.faces[0].action == "native"
    assert plan.faces[0].surface_family == "cylinder"
    assert "Plane" in plan.faces[0].reason
    assert plan.blocks == ()


def test_report_recursively_freezes_caller_owned_maps() -> None:
    geometry, _face = _pentagon()
    plan = plan_structured_layout(geometry, target_size=0.5)
    _working, report = apply_structured_layout(geometry, plan)
    report = replace(
        report,
        quality={"limits": {"angles": [30.0, 150.0]}},
    )

    with pytest.raises(TypeError):
        report.source_to_working_faces[1] = (999,)  # type: ignore[index]
    with pytest.raises(TypeError):
        report.quality["limits"]["angles"] = ()  # type: ignore[index]
    assert report.to_dict()["quality"]["limits"]["angles"] == [30.0, 150.0]


def test_apply_cancellation_rolls_back_detached_work_and_preserves_source() -> None:
    geometry, _face = _pentagon()
    before = _document_bytes(geometry)
    plan = plan_structured_layout(geometry, target_size=0.5)

    def cancel(phase: str) -> None:
        if phase == "structured block materialization":
            raise RuntimeError("cancel after shared-edge partition")

    with pytest.raises(RuntimeError, match="cancel"):
        apply_structured_layout(
            geometry, plan, cancellation_check=cancel
        )
    assert _document_bytes(geometry) == before


@pytest.mark.skipif(
    not HAS_FROZEN_ADOPTION,
    reason="requires the upcoming ANYgeometry FeatureHistory.adopt_frozen release",
)
def test_frozen_commit_records_exact_upstream_feature_dependencies() -> None:
    geometry, face = _pentagon()
    base = geometry.features.adopt_frozen(
        geometry,
        kind="test.base",
        outputs={"shell": EntityRef("face", face)},
    )
    before = _document_bytes(geometry)
    plan = plan_structured_layout(geometry, target_size=0.5)

    working, report, record = commit_structured_layout(geometry, plan)

    assert _document_bytes(geometry) == before
    assert report.status == "committed"
    assert record.dependencies == (base.feature_id,)
    assert record.inputs["upstream"][0].feature_id == base.feature_id
    assert record.inputs["upstream"][0].output_key == "shell"
    assert record.materialization_checksum is not None
    assert working.features.validate_materialization(record, working) is None


@pytest.mark.skipif(
    not HAS_FROZEN_ADOPTION,
    reason="requires the upcoming ANYgeometry FeatureHistory.adopt_frozen release",
)
def test_failed_frozen_adoption_is_non_mutating(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry, _face = _pentagon()
    before = _document_bytes(geometry)
    plan = plan_structured_layout(geometry, target_size=0.5)

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected frozen adoption failure")

    monkeypatch.setattr(FeatureHistory, "adopt_frozen", fail)
    with pytest.raises(MeshError, match="failed atomically"):
        commit_structured_layout(geometry, plan)
    assert _document_bytes(geometry) == before


def test_direct_report_constructor_does_not_retain_nested_mutable_aliases() -> None:
    geometry, _face = _pentagon()
    plan = plan_structured_layout(geometry, target_size=0.5)
    nested = {"outer": {"values": [1, 2]}}
    report = StructuredLayoutReport(plan, {}, {}, quality=nested)
    nested["outer"]["values"].append(3)

    assert report.to_dict()["quality"] == {"outer": {"values": [1, 2]}}
