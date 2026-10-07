"""Focused tests for the vectorized linear-mesh validity certification (M5).

The original scanning validator stays the authority: every fallback case must
reproduce its exact report, and a certification must equal the report the
scanner would produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from anymesher.core import MeshCore
from anymesher.errors import MeshError
from anymesher.quality_v2 import (
    MeshValidityError,
    _certified_linear_validity,
    _scan_mesh_validity,
    assert_valid_mesh,
    evaluate_quality,
    validate_mesh,
)


SQUARE_NODES = np.asarray(
    (
        (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0),
        (2.0, 0.0), (2.0, 1.0),
    )
)
TWO_QUADS = np.asarray(((0, 1, 2, 3), (1, 4, 5, 2)), dtype=np.int64)


def _mesh(
    triangles=None,
    quads=None,
    triangle_active=None,
    quad_active=None,
    nodes=SQUARE_NODES,
) -> MeshCore:
    return MeshCore(
        nodes,
        triangles,
        quads,
        triangle_active=triangle_active,
        quad_active=quad_active,
    )


def _scan(
    mesh: MeshCore,
    tolerance=None,
    declared_plate_junction_edges=(),
):
    declared = tuple(
        tuple(int(value) for value in raw_edge)
        for raw_edge in declared_plate_junction_edges
    )
    return _scan_mesh_validity(mesh, tolerance, False, declared)


def test_valid_linear_mesh_certification_equals_scanner_report() -> None:
    mesh = _mesh(quads=TWO_QUADS)
    certified = _certified_linear_validity(mesh, None, ())
    assert certified is not None
    assert validate_mesh(mesh) == _scan(mesh) == certified


def test_valid_triangle_mesh_certifies() -> None:
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    mesh = _mesh(triangles=triangles)
    assert validate_mesh(mesh).valid
    assert validate_mesh(mesh) == _scan(mesh)


def test_valid_mixed_mesh_certifies() -> None:
    triangles = np.asarray(((0, 1, 2),), dtype=np.int64)
    quads = np.asarray(((0, 4, 5, 2),), dtype=np.int64)
    mesh = _mesh(triangles=triangles, quads=quads)
    assert validate_mesh(mesh).valid
    assert validate_mesh(mesh) == _scan(mesh)


def test_repeated_corner_falls_back_with_original_report() -> None:
    quads = np.asarray(((0, 1, 2, 2), (1, 4, 5, 2)), dtype=np.int64)
    mesh = _mesh(quads=quads)
    report = validate_mesh(mesh)
    expected = _scan(mesh)
    assert not report.valid
    assert report == expected
    assert any("repeats a corner node" in error for error in report.errors)


def test_duplicate_elements_fall_back_with_original_report() -> None:
    quads = np.asarray(((0, 1, 2, 3), (3, 2, 1, 0)), dtype=np.int64)
    mesh = _mesh(quads=quads)
    report = validate_mesh(mesh)
    assert report == _scan(mesh)
    assert any("duplicate the same corners" in error for error in report.errors)


def test_nonmanifold_edge_falls_back_with_original_report() -> None:
    triangles = np.asarray(((0, 1, 2), (0, 1, 3), (0, 1, 4)), dtype=np.int64)
    nodes = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (2.0, 2.0), (2.0, 3.0))
    )
    mesh = _mesh(triangles=triangles, nodes=nodes)
    report = validate_mesh(mesh)
    assert report == _scan(mesh)
    assert any("non-manifold edge" in error for error in report.errors)


def test_declared_junction_uses_scanner_authority() -> None:
    triangles = np.asarray(((0, 1, 2), (0, 1, 3), (0, 1, 4)), dtype=np.int64)
    nodes = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (2.0, 2.0), (2.0, 3.0))
    )
    mesh = _mesh(triangles=triangles, nodes=nodes)
    declared = ((0, 1),)
    report = validate_mesh(mesh, declared_plate_junction_edges=declared)
    assert report.valid
    assert report == _scan(mesh, declared_plate_junction_edges=declared)
    # The certification itself must decline declared junctions.
    assert _certified_linear_validity(mesh, None, ((0, 1),)) is None


def test_declared_junction_generator_is_consumed_once() -> None:
    triangles = np.asarray(((0, 1, 2), (0, 1, 3), (0, 1, 4)), dtype=np.int64)
    nodes = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (2.0, 2.0), (2.0, 3.0))
    )
    mesh = _mesh(triangles=triangles, nodes=nodes)
    edges = iter(((0, 1),))
    assert validate_mesh(mesh, declared_plate_junction_edges=edges).valid
    with pytest.raises(MeshError, match="distinct node-row pairs"):
        validate_mesh(mesh, declared_plate_junction_edges=iter(((1, 1),)))


def test_malformed_tolerance_outranks_malformed_declared_edge() -> None:
    # The original validator evaluates tolerance before converting any
    # declared edge, so a tolerance that cannot convert must surface its
    # error first, even when a declared edge is also malformed.
    mesh = _mesh(quads=TWO_QUADS)
    with pytest.raises(ValueError, match="could not convert string to float"):
        validate_mesh(
            mesh,
            tolerance="not-a-number",
            declared_plate_junction_edges=(("a", "b"),),
        )


def test_malformed_tolerance_alone_keeps_original_error() -> None:
    mesh = _mesh(quads=TWO_QUADS)
    with pytest.raises(ValueError, match="could not convert string to float"):
        validate_mesh(mesh, tolerance="not-a-number")


def test_malformed_declared_edges_keep_scanner_errors() -> None:
    mesh = _mesh(quads=TWO_QUADS)
    with pytest.raises(MeshError, match="distinct node-row pairs"):
        validate_mesh(mesh, declared_plate_junction_edges=((0, 1, 2),))
    with pytest.raises(MeshError, match="distinct node-row pairs"):
        validate_mesh(mesh, declared_plate_junction_edges=((1, 1),))
    with pytest.raises(MeshError, match="outside the mesh"):
        validate_mesh(mesh, declared_plate_junction_edges=((0, 99),))


def test_declared_junction_nested_generators_are_consumed_once() -> None:
    mesh = _mesh(quads=TWO_QUADS)
    edges = (iter((0, 1)), iter((1, 2)))
    assert validate_mesh(mesh, declared_plate_junction_edges=edges).valid
    assert validate_mesh(
        mesh, declared_plate_junction_edges=iter(((0, 1),))
    ).valid
    # A malformed edge behind a single-use generator still raises the
    # scanner's own error.
    with pytest.raises(MeshError, match="distinct node-row pairs"):
        validate_mesh(mesh, declared_plate_junction_edges=(iter((1, 1)),))


def test_inactive_bad_elements_are_ignored() -> None:
    # An inactive triangle with a repeated corner must not block a valid
    # active quad mesh, exactly as in the scanner.
    bad = np.asarray(((0, 1, 1),), dtype=np.int64)
    mesh = _mesh(
        triangles=bad,
        quads=TWO_QUADS,
        triangle_active=np.asarray((False,)),
    )
    assert validate_mesh(mesh).valid
    assert validate_mesh(mesh) == _scan(mesh)


def test_inactive_quadratic_rows_do_not_block_certification() -> None:
    quadratic = np.asarray(
        ((0, 1, 2, 3, 4, 5), (0, 2, 5, 3, 1, 0)), dtype=np.int64
    )
    mesh = _mesh(
        triangles=quadratic,
        quads=TWO_QUADS,
        triangle_active=np.asarray((False, False)),
    )
    assert validate_mesh(mesh).valid
    assert validate_mesh(mesh) == _scan(mesh)


def test_active_quadratic_mesh_uses_scanner_authority() -> None:
    # T6 with a midside equal to one of its edge corners: only the scanner
    # detects this.
    triangles = np.asarray(((0, 1, 2, 3, 4, 0),), dtype=np.int64)
    mesh = _mesh(triangles=triangles)
    report = validate_mesh(mesh)
    assert report == _scan(mesh)
    assert any("midside equal to an edge corner" in error for error in report.errors)


def test_zero_area_element_falls_back_with_original_report() -> None:
    triangles = np.asarray(((0, 1, 2),), dtype=np.int64)
    nodes = np.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))
    mesh = _mesh(triangles=triangles, nodes=nodes)
    report = validate_mesh(mesh)
    assert report == _scan(mesh)
    assert any("zero or non-finite area" in error for error in report.errors)


def test_inverted_quad_falls_back_with_original_report() -> None:
    quads = np.asarray(((0, 1, 3, 2),), dtype=np.int64)
    mesh = _mesh(quads=quads)
    report = validate_mesh(mesh)
    assert report == _scan(mesh)
    assert any(
        "inverted, concave, or self-intersecting" in error for error in report.errors
    )


def test_raise_on_error_preserves_scanner_behavior() -> None:
    quads = np.asarray(((0, 1, 3, 2),), dtype=np.int64)
    mesh = _mesh(quads=quads)
    with pytest.raises(MeshValidityError) as raised:
        validate_mesh(mesh, raise_on_error=True)
    assert str(raised.value) == "; ".join(_scan(mesh).errors)


def test_tolerance_is_shared_by_certification_and_scanner() -> None:
    # A generous tolerance certifies a sliver the default tolerance rejects;
    # both paths must agree at either tolerance.
    triangles = np.asarray(((0, 1, 2),), dtype=np.int64)
    nodes = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0e-15)))
    mesh = _mesh(triangles=triangles, nodes=nodes)
    strict = validate_mesh(mesh)
    assert strict == _scan(mesh)
    assert not strict.valid
    loose = validate_mesh(mesh, tolerance=1.0e-16)
    assert loose.valid
    assert loose == _scan(mesh, tolerance=1.0e-16)


def test_caller_integration_quality_pipeline_uses_validation() -> None:
    mesh = _mesh(quads=TWO_QUADS)
    quality = evaluate_quality(mesh)
    assert quality.validity.valid
    assert_valid_mesh(mesh)

    bad = _mesh(quads=np.asarray(((0, 1, 3, 2),), dtype=np.int64))
    with pytest.raises(MeshValidityError):
        evaluate_quality(bad)
