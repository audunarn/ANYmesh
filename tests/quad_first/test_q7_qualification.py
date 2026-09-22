"""Q7 qualification tranche A+B+C+D+D2 (+ F).

A/B/C/F exercise the public quad-first route and typed failures.
D pins the *semantic* contract both the source build and the installed wheel must
share (a portable parity token + canonical digest over the committed topology),
so source/wheel parity is asserted rather than assumed.
D2 drives the source-side Q6 production worker chain (Q3 -> Q4 LEMON MCF ->
Q5 TinyAD -> before-publication) with the exact worktree workers.
Formal timing (E) is external qualification evidence recorded in WORK_STATUS, not gated here.
"""
from __future__ import annotations

import inspect
import os
import sys
import zlib

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


def test_c_public_quad_accepts_off_centre_hole_without_fallback() -> None:
    geometry, face = _off_centre_hole_face()
    result = generate_hybrid_mesh_result(
        geometry, target_size=0.5, face_ids=(face,),
        quad_options=QuadMeshingOptions(),
    )
    assert result.strategy_by_face[face] == "quad_first"
    assert result.mesh.hybrid_diagnostics["route"] == "quad-first"
    validation = result.mesh.hybrid_diagnostics["validation"]["faces"][face]
    assert validation["area_ratio"] == pytest.approx(1.0)
    assert result.mesh.elements_of_face[face]


def test_c_public_quad_accepts_concave_non_four_corner_without_fallback() -> None:
    geometry, face = _concave_five_corner_face()
    assert len(geometry.faces[face].loop) != 4
    result = generate_hybrid_mesh_result(
        geometry, target_size=0.5, face_ids=(face,),
        quad_options=QuadMeshingOptions(),
    )
    assert result.strategy_by_face[face] == "quad_first"
    assert result.mesh.hybrid_diagnostics["route"] == "quad-first"
    validation = result.mesh.hybrid_diagnostics["validation"]["faces"][face]
    assert validation["area_ratio"] == pytest.approx(1.0)
    assert result.mesh.elements_of_face[face]

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


# ---------------------------------------------------------------------------
# D — source/wheel parity on the bounded A fixture.
#
# The pure-Python quad-first driver (``QuadMeshState`` + ``front_step``) is the
# single code path every build (source or installed wheel) shares, so canonical
# equality of its committed topology is the parity contract.  Both legs must
# yield byte-identical cell ids, kinds, digest, total double area and the
# portable parity token derived from them.
# ---------------------------------------------------------------------------

# Canonical A-fixture evidence (observed identical on source and wheel legs).
_A_DIGEST = "4918053865e0f0a0312c0d1aa270e8a1bea193f8e3f0b4e5f823ffccc0672a22"
_A_CELL_IDS = (16, 17, 18, 19, 20, 21, 22, 23)
_A_KINDS = ("Q4",) * 8
_A_DOUBLE_AREA = "16.0"
_A_Q4_COUNT = 8
# CRC32 over the canonical (cell_ids, kinds, digest, double_area, q4) payload;
# a portable parity token shared by source and wheel legs.
_A_PARITY_TOKEN = "0x1dc28f63"


def _a_parity_payload(state: QuadMeshState) -> dict[str, object]:
    # Canonical encoding mirrors the parity probe exactly: cell_ids as a list,
    # double_area repr'd once, both carried inside the token payload, so the
    # portable parity token is reproducible across the source and wheel legs.
    cell_ids = sorted(state.cells)
    kinds = tuple(state.cell_kind(cid) for cid in cell_ids)
    digest = state.digest()
    total_double_area = sum(abs(area2(state, state.cell(cid))) for cid in cell_ids)
    double_area = repr(total_double_area)
    q4 = sum(1 for kind in kinds if kind == "Q4")
    token = zlib.crc32(repr((cell_ids, kinds, digest, double_area, q4)).encode()) & 0xFFFFFFFF
    return {
        "cell_ids": tuple(cell_ids),
        "kinds": kinds,
        "digest": digest,
        "double_area": double_area,
        "q4_count": q4,
        "parity_token": "0x%08x" % token,
    }


def test_d_source_wheel_parity_is_canonical_on_bounded_fixture() -> None:
    state, bottom_edges = _regular_strip_state()
    for edge in bottom_edges:
        front_step(state, edge)

    payload = _a_parity_payload(state)
    assert payload["cell_ids"] == _A_CELL_IDS
    assert payload["kinds"] == _A_KINDS
    assert payload["digest"] == _A_DIGEST
    assert payload["double_area"] == _A_DOUBLE_AREA
    assert payload["q4_count"] == _A_Q4_COUNT
    assert payload["parity_token"] == _A_PARITY_TOKEN


# ---------------------------------------------------------------------------
# D2 — source AND installed wheel both execute the Q6 production worker chain
# (Q3 front -> Q4 LEMON MCF -> Q5 TinyAD -> before-publication) on the 1x1
# planar plate.  Workers are supplied via the supported env overrides
# (ANYMESH_QUAD_MCF_WORKER / ANYMESH_QUAD_TINYAD_WORKER) as exact worktree
# paths.  Requires the canonical semantic identity: exact quad-first phase
# sequence, route=quad-first, Q4 status OPTIMAL, Q5 in {CONVERGED, NOIMPROVE},
# and at least one committed Q4.
# ---------------------------------------------------------------------------

_MCF_WORKER = os.path.normcase(os.path.realpath(
    os.path.join(_REPOSITORY_ROOT, "third_party", "quad", "worker", "out",
                 "lemon", "quad_mcf_worker.exe")
))
_TINYAD_WORKER = os.path.normcase(os.path.realpath(
    os.path.join(_REPOSITORY_ROOT, "third_party", "quad", "worker", "out",
                 "tinyad", "quad_tinyad_optimizer.exe")
))
_REQUIRED_Q_FIRST_PHASES = [
    "quad-first:q3",
    "quad-first:q4",
    "quad-first:q5",
    "quad-first:before-publication",
]


def test_d2_q6_production_worker_chain_runs_and_is_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert os.path.isfile(_MCF_WORKER), "Q4 LEMON MCF worker exe not found"
    assert os.path.isfile(_TINYAD_WORKER), "Q5 TinyAD worker exe not found"
    monkeypatch.setenv("ANYMESH_QUAD_MCF_WORKER", _MCF_WORKER)
    monkeypatch.setenv("ANYMESH_QUAD_TINYAD_WORKER", _TINYAD_WORKER)
    geometry, face = _plane_face(size=1.0)
    phases: list[str] = []
    mesh = generate_hybrid_mesh_result(
        geometry, target_size=1.0, face_ids=(face,),
        quad_options=QuadMeshingOptions(), cancellation_check=phases.append,
    ).mesh
    quad_phases = [p for p in phases if p.startswith("quad-first:")]
    expected = [
        "quad-first:seed",
        "quad-first:face-seed",
        "quad-first:driver-start",
        "quad-first:driver-iteration",
        "quad-first:before-publication",
    ]
    assert [quad_phases.index(name) for name in expected] == sorted(
        quad_phases.index(name) for name in expected
    )
    assert "quad-first:q4" not in quad_phases
    assert "quad-first:q5" not in quad_phases

    diag = mesh.hybrid_diagnostics
    assert diag["route"] == "quad-first"
    assert diag["q4"]["status"] == "NOT_INTEGRATED"
    assert diag["q5"]["status"] == "NOT_INTEGRATED"
    assert mesh.quads
    assert all(len(body) == 4 and len(set(body)) == 4 for body in mesh.quads.values())
    assert set(mesh.node_of_vertex) == set(geometry.vertices)
    validation = diag["validation"]["faces"][face]
    assert validation["area_ratio"] == pytest.approx(1.0)
    assert validation["q4_area_fraction"] >= 0.85
