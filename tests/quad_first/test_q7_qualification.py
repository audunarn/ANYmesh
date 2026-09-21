"""Q7 qualification tranche A+B+C+F.

D (source/wheel parity) and E (formal timing) are intentionally deferred to the
next bounded tranche.  These tests are deterministic and small.
"""
from __future__ import annotations

import inspect
import os
import sys

_REPOSITORY_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_SRC = os.path.normcase(os.path.realpath(os.path.join(_REPOSITORY_ROOT, "src")))


def _is_src(entry: object) -> bool:
    if not isinstance(entry, str) or not entry:
        return False
    try:
        return os.path.normcase(os.path.realpath(entry)) == _SRC
    except (OSError, ValueError):
        return False


if os.path.isdir(os.path.join(_REPOSITORY_ROOT, "src")):
    sys.path[:] = [entry for entry in sys.path if not _is_src(entry)]
    sys.path.insert(0, os.path.join(_REPOSITORY_ROOT, "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from anygeometry import GeometryModel, OrientedEdge, Plane  # noqa: E402
from anygeometry.operations import trim_face  # noqa: E402
from anymesher.hybrid import generate_hybrid_mesh_result  # noqa: E402
from anymesher.quad import area2, front_step  # noqa: E402
from anymesher.quad.options import QuadMeshingOptions  # noqa: E402
from anymesher.quad.public_integration import (  # noqa: E402
    QuadPublicUnsupported,
    advertise_quad_capabilities,
    coerce_public_quad_options,
)
from anymesher.quad.state import QuadMeshState  # noqa: E402


def _plane_face(size: float = 2.0) -> tuple[GeometryModel, int]:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (size, 0.0, 0.0),
         (size, size, 0.0), (0.0, size, 0.0))
    )
    return geometry, geometry.add_plate(vertices)


def _off_centre_hole_face() -> tuple[GeometryModel, int]:
    geometry = GeometryModel()
    plane = Plane(np.asarray((0.0, 0.0, 0.0)),
                  np.asarray((4.0, 0.0, 0.0)),
                  np.asarray((0.0, 4.0, 0.0)))
    outer = geometry.add_points(
        ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0),
         (4.0, 4.0, 0.0), (0.0, 4.0, 0.0))
    )
    face = geometry.add_face_from_loop(
        geometry.order_loop(geometry.add_polyline(outer, close=True)),
        surface=plane,
    )
    hole = geometry.add_points(
        ((0.75, 1.25, 0.0), (1.75, 1.25, 0.0),
         (1.75, 2.25, 0.0), (0.75, 2.25, 0.0))
    )
    hole_edges = geometry.add_polyline(hole, close=True)
    trim_face(
        geometry,
        face,
        (tuple(OrientedEdge(edge, True) for edge in hole_edges),),
    )
    return geometry, face


def _concave_five_corner_face() -> tuple[GeometryModel, int]:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0),
         (1.0, 0.35, 0.0), (0.0, 1.0, 0.0)),
        dtype=float,
    )
    geometry = GeometryModel()
    vertices = geometry.add_points(points)
    span = np.ptp(points, axis=0)
    face = geometry.add_face(
        geometry.add_polyline(vertices, close=True),        surface=Plane(
            np.min(points, axis=0),
            np.asarray((span[0], 0.0, 0.0)),
            np.asarray((0.0, span[1], 0.0)),
        ),
    )
    return geometry, face


def test_c_public_quad_rejects_off_centre_hole_without_fallback() -> None:
    geometry, face = _off_centre_hole_face()
    with pytest.raises(QuadPublicUnsupported):
        generate_hybrid_mesh_result(
            geometry,
            target_size=0.5,
            face_ids=(face,),
            quad_options=QuadMeshingOptions(),
        )


def test_c_public_quad_rejects_concave_non_four_corner_without_fallback() -> None:
    geometry, face = _concave_five_corner_face()
    assert len(geometry.faces[face].loop) != 4
    with pytest.raises(QuadPublicUnsupported):
        generate_hybrid_mesh_result(
            geometry,
            target_size=0.5,
            face_ids=(face,),
            quad_options=QuadMeshingOptions(),
        )


def _regular_strip_state(count: int = 8) -> tuple[QuadMeshState, list[tuple[int, int]]]:
    nodes: dict[int, tuple[float, float]] = {}
    cells: dict[int, tuple[int, int, int]] = {}
    bottom_edges: list[tuple[int, int]] = []
    for index in range(count + 1):
        nodes[2 * index] = (float(index), 0.0)
        nodes[2 * index + 1] = (float(index), 1.0)
    for index in range(count):
        bl = 2 * index
        tl = bl + 1
        br = 2 * (index + 1)
        tr = br + 1
        cells[2 * index] = (bl, br, tr)
        cells[2 * index + 1] = (bl, tr, tl)
        bottom_edges.append((bl, br))

    front = list(bottom_edges)
    front.extend((2 * index + 1, 2 * (index + 1) + 1) for index in range(count))
    front.extend(((0, 1), (2 * count, 2 * count + 1)))
    return QuadMeshState(nodes=nodes, cells=cells, initial_front=front), bottom_edges


def _q4_fractions(state: QuadMeshState) -> tuple[float, float]:
    cell_ids = sorted(state.cells)
    q4_ids = [cid for cid in cell_ids if state.cell_kind(cid) == "Q4"]
    total_double_area = sum(abs(area2(state, state.cell(cid))) for cid in cell_ids)
    q4_double_area = sum(abs(area2(state, state.cell(cid))) for cid in q4_ids)
    return len(q4_ids) / len(cell_ids), q4_double_area / total_double_area


def test_a_driver_regular_planar_quality_gate() -> None:
    state, bottom_edges = _regular_strip_state()
    for edge in bottom_edges:
        front_step(state, edge)

    count_fraction, area_fraction = _q4_fractions(state)
    assert count_fraction >= 0.85
    assert area_fraction >= 0.75
    assert count_fraction == pytest.approx(1.0)
    assert area_fraction == pytest.approx(1.0)

def _mixed_strip_faces(count: int = 8) -> tuple[GeometryModel, tuple[int, ...], int]:
    geometry = GeometryModel()
    bottom = geometry.add_points([(float(i), 0.0, 0.0) for i in range(count + 1)])
    top = geometry.add_points([(float(i), 1.0, 0.0) for i in range(count + 1)])
    vertical = [geometry.add_line(bottom[i], top[i]) for i in range(count + 1)]
    lower = [geometry.add_line(bottom[i], bottom[i + 1]) for i in range(count)]
    upper = [geometry.add_line(top[i], top[i + 1]) for i in range(count)]

    quad_faces: list[int] = []
    for i in range(count):
        loop = (
            OrientedEdge(lower[i], True),
            OrientedEdge(vertical[i + 1], True),
            OrientedEdge(upper[i], False),
            OrientedEdge(vertical[i], False),
        )
        face = geometry.add_face_from_loop(loop, (0, 1, 2, 3))
        geometry.add_sheet((face,))
        quad_faces.append(face)

    apex = geometry.add_point(float(count + 1), 0.5, 0.0)
    tri_loop = (
        OrientedEdge(vertical[count], False),
        OrientedEdge(geometry.add_line(bottom[count], apex), True),
        OrientedEdge(geometry.add_line(apex, top[count]), True),
    )
    tri_face = geometry.add_face_from_loop(        tri_loop,
        corners=None,
        surface=Plane(
            np.asarray((float(count), 0.0, 0.0)),
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
        ),
    )
    geometry.add_sheet((tri_face,))
    return geometry, tuple(quad_faces), tri_face


def test_b_meaningful_mixed_q4_s3_route_and_ownership() -> None:
    geometry, quad_faces, tri_face = _mixed_strip_faces()
    selected = quad_faces + (tri_face,)
    result = generate_hybrid_mesh_result(
        geometry,
        target_size=1.0,
        strategy="native",
        native_backend="python",
        recombine=False,
        structural_preparation=False,
        face_ids=selected,
        quad_options=QuadMeshingOptions(),
        quad_face_ids=quad_faces,
        qualified_s3=True,
    )
    mesh = result.mesh

    assert all(result.strategy_by_face[face] == "quad_first" for face in quad_faces)
    assert result.strategy_by_face[tri_face] == "native"
    assert mesh.quads and mesh.tris
    assert 0.0 < len(mesh.tris) / (len(mesh.quads) + len(mesh.tris)) <= 0.25
    for face in quad_faces:
        owned = set(mesh.elements_of_face[face])
        assert owned
        assert owned <= set(mesh.quads)
    tri_owned = set(mesh.elements_of_face[tri_face])
    assert tri_owned
    assert tri_owned <= set(mesh.tris)

    record = mesh.structural_preparation["qualified_s3"]
    assert record["status"] == "ADMITTED"
    assert record["legacy_fallback"] == "FORBIDDEN"
    assert record["contract_id"] == "ANYMESHER_QUALIFIED_S3_PRODUCTION_PREPARATION_V1"
    assert sorted(record["element_ids"]) == sorted(mesh.tris)
    assert advertise_quad_capabilities().mixed_q4_s3 is True


def test_f_quad_first_remains_explicit_opt_in_not_default() -> None:
    assert coerce_public_quad_options(None) is None
    parameter = inspect.signature(generate_hybrid_mesh_result).parameters["quad_options"]
    assert parameter.default is None

    geometry, face = _plane_face()
    phases: list[str] = []
    result = generate_hybrid_mesh_result(
        geometry,
        target_size=1.0,
        face_ids=(face,),
        quad_options=None,
        cancellation_check=phases.append,
    )
    assert not any(phase.startswith("quad-first") for phase in phases)
    assert result.mesh.hybrid_diagnostics.get("route") != "quad-first"
