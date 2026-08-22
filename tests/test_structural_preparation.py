from __future__ import annotations

from types import MappingProxyType

from anygeometry import GeometryModel
from anygeometry.serialization import to_dict
import pytest

from anymesher import MeshError
from anymesher.preparation import prepare_structural_closure


def _crossing_faces() -> tuple[GeometryModel, int, int]:
    geometry = GeometryModel()
    horizontal = geometry.add_plate(
        geometry.add_points(
            ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0))
        )
    )
    vertical = geometry.add_plate(
        geometry.add_points(
            ((-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1))
        )
    )
    return geometry, horizontal, vertical


def test_crossing_faces_prepare_on_clone_with_exact_lineage() -> None:
    geometry, first, second = _crossing_faces()
    before = to_dict(geometry)

    working, report = prepare_structural_closure(
        geometry,
        face_ids=(first, second),
    )

    assert report is not None
    assert to_dict(geometry) == before
    assert working is not geometry
    assert len(working.faces) == 4
    assert report.face_connections == 1
    assert report.applications == 1
    assert len(report.temporary_sheet_ids) == 2
    assert len(report.source_to_working_faces[first]) == 2
    assert len(report.source_to_working_faces[second]) == 2
    assert report.preparation_hash.startswith("sha256:")
    assert isinstance(report.source_to_working_faces, MappingProxyType)
    shared = {
        item.edge
        for face_id in report.source_to_working_faces[first]
        for item in working.faces[face_id].loop
    }.intersection(
        item.edge
        for face_id in report.source_to_working_faces[second]
        for item in working.faces[face_id].loop
    )
    assert len(shared) == 1


def test_crossing_members_create_declared_junction_without_source_mutation() -> None:
    geometry = GeometryModel()
    first_points = geometry.add_points(((-1, 0, 0), (1, 0, 0)))
    second_points = geometry.add_points(((0, -1, 0), (0, 1, 0)))
    first = geometry.add_line(*first_points)
    second = geometry.add_line(*second_points)
    before = to_dict(geometry)

    working, report = prepare_structural_closure(
        geometry,
        face_ids=(),
        beam_edges=(first, second),
    )

    assert report is not None
    assert to_dict(geometry) == before
    assert report.member_connections == 1
    assert len(report.temporary_member_ids) == 2
    assert len(working.junctions) == 1
    assert len(report.source_to_working_edges[first]) == 2
    assert len(report.source_to_working_edges[second]) == 2


def test_one_member_connects_at_multiple_distinct_intersections() -> None:
    geometry = GeometryModel()
    horizontal = geometry.add_line(
        *geometry.add_points(((-2, 0, 0), (2, 0, 0)))
    )
    first_crossing = geometry.add_line(
        *geometry.add_points(((-1, -1, 0), (-1, 1, 0)))
    )
    second_crossing = geometry.add_line(
        *geometry.add_points(((1, -1, 0), (1, 1, 0)))
    )

    working, report = prepare_structural_closure(
        geometry,
        face_ids=(),
        beam_edges=(horizontal, first_crossing, second_crossing),
    )

    assert report is not None
    assert report.member_connections == 2
    assert len(working.junctions) == 2
    assert len(report.source_to_working_edges[horizontal]) == 3


def test_member_sheet_connection_is_declared_on_working_clone() -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(
            ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0))
        )
    )
    endpoints = geometry.add_points(((0, 0, -1), (0, 0, 1)))
    edge = geometry.add_line(*endpoints)

    working, report = prepare_structural_closure(
        geometry,
        face_ids=(face,),
        beam_edges=(edge,),
    )

    assert report is not None
    assert report.member_sheet_connections == 1
    assert len(working.attachments) == 1
    assert len(working.junctions) == 1
    assert geometry.attachments == {}
    assert geometry.junctions == {}


def test_coplanar_overlap_blocks_without_mutating_source() -> None:
    geometry = GeometryModel()
    first = geometry.add_plate(
        geometry.add_points(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)))
    )
    second = geometry.add_plate(
        geometry.add_points(((1, 0, 0), (3, 0, 0), (3, 2, 0), (1, 2, 0)))
    )
    before = to_dict(geometry)

    with pytest.raises(MeshError, match="Fragment Overlaps"):
        prepare_structural_closure(geometry, face_ids=(first, second))

    assert to_dict(geometry) == before


def test_candidate_resource_limit_fails_closed_and_source_stays_unchanged() -> None:
    geometry, first, second = _crossing_faces()
    third = geometry.add_plate(
        geometry.add_points(
            ((0, -1, -1), (0, 1, -1), (0, 1, 1), (0, -1, 1))
        )
    )
    before = to_dict(geometry)

    with pytest.raises(MeshError, match="maximum_candidate_pairs=1"):
        prepare_structural_closure(
            geometry,
            face_ids=(first, second, third),
            options={"maximum_candidate_pairs": 1},
        )

    assert to_dict(geometry) == before


def test_overlap_narrow_phase_is_cancellable_before_geometry_mutation() -> None:
    geometry, first, second = _crossing_faces()
    before = to_dict(geometry)

    def cancel(phase: str) -> None:
        if phase == "structural preparation overlap narrow phase":
            raise RuntimeError("cancel overlap audit")

    with pytest.raises(RuntimeError, match="cancel overlap audit"):
        prepare_structural_closure(
            geometry,
            face_ids=(first, second),
            cancellation_check=cancel,
        )

    assert to_dict(geometry) == before


def test_preparation_can_be_explicitly_disabled() -> None:
    geometry, _first, _second = _crossing_faces()
    before = to_dict(geometry)

    working, report = prepare_structural_closure(geometry, options=False)

    assert working is not geometry
    assert report is not None
    assert report.applications == 0
    assert report.temporary_sheet_ids == ()
    assert to_dict(geometry) == before
    assert tuple(working.vertices) == tuple(geometry.vertices)
    assert tuple(working.edges) == tuple(geometry.edges)
    assert tuple(working.faces) == tuple(geometry.faces)
    assert working.validate_topology() == ()
