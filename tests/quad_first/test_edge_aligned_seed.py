"""Regression tests for the station-conforming edge-aligned tensor seed.

The seed core only consumes the canonical station ring, the per-edge chain
lengths and the exact type admission, so these tests drive it with small
stand-ins patched into the module.  That keeps every case geometric and
deterministic: collinear split sides, rigid rotations, loops that start in the
middle of a side, and each refusal.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from anygeometry import OrientedEdge, Plane
from anygeometry.model import GeometryModel
from anygeometry.operations import trim_face
from anymesher.errors import MeshError
from anymesher.hybrid import MeshingStrategy, generate_hybrid_mesh_result
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad import edge_aligned_seed as eas
from anymesher.quad.boundary import BoundaryStationKey, BoundaryStationRegistry
from anymesher.quad.domain import PlanarQuadDomain
from anymesher.quad.seed import PlanarQuadSeed, build_planar_quad_seed


class _Straight:
    pass


class _Arc:
    pass


class _Geometry:
    revision = 1
    model_id = "edge-aligned"

    def __init__(self, curve_type=_Straight, count=64):
        self.edges = {i: SimpleNamespace(curve=curve_type()) for i in range(count)}


class _Domain:
    face_id = 7
    model_id = "edge-aligned"
    revision = 1

    def __init__(self, edge_uses, holes=()):
        self.edge_uses = tuple(edge_uses)
        self.hole_edge_uses = tuple(holes)

    def assert_current(self, geometry):
        return None

    def project(self, point):
        return (float(point[0]), float(point[1]))

    def lift(self, point):
        return (float(point[0]), float(point[1]), 0.0)


class _Registry:
    def __init__(self, chains, target_size):
        self.chains = chains
        self.target_size = target_size

    def chain(self, edge_id, forward=True):
        chain = self.chains[int(edge_id)]
        return chain if forward else tuple(reversed(chain))

    def outer_stations(self, domain):
        out = []
        for edge_id, forward in domain.edge_uses:
            out.extend(self.chain(edge_id, forward)[:-1])
        return tuple(out)

    def assert_current(self, geometry):
        return None


class _SizeField:
    def __init__(self, uniform):
        self.is_uniform = uniform


@pytest.fixture(autouse=True)
def _stand_ins(request, monkeypatch):
    # Real-geometry and public-route tests must see the production types.
    if request.node.name.startswith(("test_real_", "test_public_")):
        return
    monkeypatch.setattr(eas, "GeometryModel", _Geometry)
    monkeypatch.setattr(eas, "PlanarQuadDomain", _Domain)
    monkeypatch.setattr(eas, "BoundaryStationRegistry", _Registry)
    monkeypatch.setattr(eas, "Straight", _Straight)
    monkeypatch.setattr(eas, "SizeField", _SizeField)


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _cells(state):
    return {int(cid): tuple(int(n) for n in state.cell(cid)) for cid in state.cells}


def _uniform(count):
    return list(np.linspace(0.0, 1.0, count + 1))


def make_case(bottom=(4,), right=(3,), top=(4,), left=(3,), *, width=2.0,
              height=1.5, angle=0.0, start=0, bottom_alpha=None,
              top_alpha=None, right_beta=None, left_beta=None, target=0.5,
              bend_top=0.0):
    """Rectangle loop, CCW from the lower-left corner, each side cut into edges.

    ``bottom``/``right``/``top``/``left`` are the division counts of the source
    edges along the CCW loop direction.  ``start`` rotates which source edge
    opens the loop; ``angle`` applies a rigid rotation to every position.
    """
    nb, nr, nt, nl = (sum(side) for side in (bottom, right, top, left))
    ab = bottom_alpha if bottom_alpha is not None else _uniform(nb)
    bt = right_beta if right_beta is not None else _uniform(nr)
    at = top_alpha if top_alpha is not None else _uniform(nt)
    bl = left_beta if left_beta is not None else _uniform(nl)
    sides = [
        ([(a * width, 0.0) for a in ab], bottom),
        ([(width, b * height) for b in bt], right),
        ([(a * width, height + bend_top * np.sin(np.pi * a)) for a in reversed(at)], top),
        ([(0.0, b * height) for b in reversed(bl)], left),
    ]
    c, s = np.cos(angle), np.sin(angle)
    edges = []
    for points, divisions in sides:
        cursor = 0
        for d in divisions:
            edges.append(points[cursor:cursor + d + 1])
            cursor += d
    edges = edges[start:] + edges[:start]
    chains, uses = {}, []
    for edge_id, points in enumerate(edges):
        divisions = len(points) - 1
        chain = []
        for ordinal, (x, y) in enumerate(points):
            if ordinal == 0:
                key = BoundaryStationKey("vertex", edge_id)
            elif ordinal == divisions:
                key = BoundaryStationKey("vertex", (edge_id + 1) % len(edges))
            else:
                key = BoundaryStationKey("edge", edge_id, ordinal, divisions)
            chain.append(SimpleNamespace(
                key=key, position=(c * x - s * y, s * x + c * y, 0.0)))
        chains[edge_id] = tuple(chain)
        uses.append((edge_id, True))
    domain = _Domain(uses)
    registry = _Registry(chains, target)
    outer = [domain.project(st.position) for st in registry.outer_stations(domain)]
    return _Geometry(), domain, registry, outer


def run(case, *, size_field=None, check=None, target=0.5):
    geometry, domain, registry, outer = case
    return eas.edge_aligned_seed_data(
        geometry, domain, registry, outer, size_field, check, target_size=target)


def test_plain_rectangle_is_a_station_conforming_q4_tensor():
    case = make_case()
    outer = np.asarray(case[3]).copy()
    coverage, state = run(case)
    n = len(outer)
    assert n == 14
    assert len(state.cells) == 12 and set(
        state.cell_kind(c) for c in state.cells) == {"Q4"}
    assert np.array_equal(np.asarray(coverage.points)[:n], outer)
    assert np.array_equal(np.asarray(case[3]), outer)
    assert sorted(state.protected_nodes) == list(range(n))
    assert len(state.nodes) == (4 + 1) * (3 + 1)
    area = sum(0.5 * _cross(np.subtract(coverage.points[b], coverage.points[a]),
                            np.subtract(coverage.points[c], coverage.points[a]))
               for a, b, c in coverage.triangles)
    assert area == pytest.approx(2.0 * 1.5, rel=1e-12)
    assert len(coverage.boundary_segments) == n and not len(coverage.mandatory_segments)


@pytest.mark.parametrize("bottom,top,left,right", [
    ((2, 2), (4,), (3,), (3,)),
    ((1, 3), (2, 2), (3,), (1, 2)),
    ((4,), (1, 1, 2), (2, 1), (3,)),
])
@pytest.mark.parametrize("angle", [0.0, 0.6, -2.3])
def test_collinear_split_sides_and_rotations_conform_from_any_start(
        bottom, top, left, right, angle):
    first = None
    edge_count = len(bottom) + len(right) + len(top) + len(left)
    for start in range(edge_count):
        case = make_case(bottom, right, top, left, angle=angle, start=start)
        coverage, state = run(case)
        n = len(case[3])
        assert np.array_equal(np.asarray(coverage.points)[:n], np.asarray(case[3]))
        assert len(state.cells) == 4 * 3
        assert set(state.cell_kind(c) for c in state.cells) == {"Q4"}
        # Every cell is an exact rectangle: all four corner turns are 90 degrees.
        pts = np.asarray(coverage.points)
        for body in state.cells.values():
            q = pts[list(body)]
            for k in range(4):
                a, b = q[(k + 1) % 4] - q[k], q[(k + 2) % 4] - q[(k + 1) % 4]
                assert abs(np.dot(a, b)) <= 1e-12 * np.linalg.norm(a) * np.linalg.norm(b)
                assert _cross(a, b) > 0.0
        if first is None:
            first = len(state.nodes)
        assert len(state.nodes) == first


def test_pinned_nonuniform_stations_are_kept_when_opposite_sides_match():
    alpha = [0.0, 0.2, 0.5, 1.0]
    beta = [0.0, 0.6, 1.0]
    case = make_case((1, 2), (2,), (3,), (1, 1), bottom_alpha=alpha,
                     top_alpha=alpha, right_beta=beta, left_beta=beta)
    coverage, state = run(case)
    pts = np.asarray(coverage.points)
    assert np.array_equal(pts[:len(case[3])], np.asarray(case[3]))
    xs = sorted({round(float(x), 12) for x, _ in pts})
    assert xs == [round(a * 2.0, 12) for a in alpha]
    ys = sorted({round(float(y), 12) for _, y in pts})
    assert ys == [round(b * 1.5, 12) for b in beta]


def test_deterministic_and_inputs_untouched():
    case = make_case((2, 2), (3,), (4,), (3,), angle=0.4, start=2)
    before = np.asarray(case[3]).copy()
    one, two = run(case), run(case)
    assert np.array_equal(one[0].points, two[0].points)
    assert np.array_equal(one[0].triangles, two[0].triangles)
    assert _cells(one[1]) == _cells(two[1])
    assert np.array_equal(np.asarray(case[3]), before)


@pytest.mark.parametrize("name,case", [
    ("unequal opposite counts", lambda: make_case((2, 1), (3,), (4,), (3,))),
    ("opposite fractions differ", lambda: make_case(
        (2,), (3,), (2,), (3,), bottom_alpha=[0.0, 0.3, 1.0], top_alpha=[0.0, 0.5, 1.0])),
    ("side fractions differ", lambda: make_case(
        (4,), (2,), (4,), (2,), right_beta=[0.0, 0.4, 1.0], left_beta=[0.0, 0.5, 1.0])),
    ("bent side is not a rectangle", lambda: make_case(bend_top=0.05)),
    ("cell aspect beyond the admission", lambda: make_case(
        (1,), (1,), (1,), (1,), width=1.0, height=0.1)),
])
def test_declines_what_it_cannot_represent_exactly(name, case):
    built = case()
    before = np.asarray(built[3]).copy()
    assert run(built) is None, name
    assert np.array_equal(np.asarray(built[3]), before)


def test_angled_edge_and_trapezoid_decline():
    geometry, domain, registry, outer = make_case((4,), (3,), (4,), (3,))
    skew = [tuple(p) for p in outer]
    # Move the upper-right corner: still four corners, no longer a rectangle.
    corner = skew.index((2.0, 1.5))
    skew[corner] = (2.4, 1.5)
    assert eas.edge_aligned_seed_data(
        geometry, domain, registry, skew, None, None, target_size=0.5) is None
    # A degenerate chamfer (corner pulled onto a neighbouring station).
    chamfer = [tuple(p) for p in outer]
    chamfer[corner] = (1.5, 1.5)
    assert eas.edge_aligned_seed_data(
        geometry, domain, registry, chamfer, None, None, target_size=0.5) is None


def test_hole_curved_and_nonuniform_controls_decline():
    geometry, domain, registry, outer = make_case()
    holed = _Domain(domain.edge_uses, holes=(((90, True),),))
    assert eas.edge_aligned_seed_data(
        geometry, holed, registry, outer, None, None, target_size=0.5) is None
    assert eas.edge_aligned_seed_data(
        _Geometry(_Arc), domain, registry, outer, None, None, target_size=0.5) is None
    assert run((geometry, domain, registry, outer), size_field=_SizeField(False)) is None
    assert run((geometry, domain, registry, outer), size_field=_SizeField(True)) is not None
    # Anything other than the exact production types is refused, never guessed.
    assert eas.edge_aligned_seed_data(
        SimpleNamespace(edges=geometry.edges), domain, registry, outer, None, None,
        target_size=0.5) is None


def test_oversized_tensor_declines_before_allocation():
    case = make_case((500,), (500,), (500,), (500,), width=500.0, height=500.0)
    calls = []
    assert run(case, check=calls.append) is None
    assert calls == []


def test_cancellation_propagates_from_the_row_stage_without_side_effects():
    case = make_case((4,), (3,), (4,), (3,))
    before = np.asarray(case[3]).copy()
    seen = []

    class Cancelled(RuntimeError):
        pass

    def check(stage):
        seen.append(stage)
        if stage == "quad-first:edge-aligned-tensor-row":
            raise Cancelled(stage)

    with pytest.raises(Cancelled):
        run(case, check=check)
    assert seen[0] == "quad-first:edge-aligned-tensor"
    assert np.array_equal(np.asarray(case[3]), before)
    done = []
    assert run(case, check=done.append) is not None
    assert done[0] == "quad-first:edge-aligned-tensor"
    assert "quad-first:edge-aligned-tensor-cells" in done


def _seed(case, **kwargs):
    geometry, domain, registry, _ = case
    return build_planar_quad_seed(
        geometry, domain.face_id, registry.target_size, domain=domain,
        registry=registry, size_field=None, **kwargs)


def test_seed_route_is_unchanged_without_the_opt_in():
    case = make_case((2, 2), (3,), (4,), (3,))
    default = _seed(case, layout_policy="adaptive")
    explicit_off = _seed(case, layout_policy="adaptive", edge_aligned=False)
    assert default.edge_aligned is False and explicit_off.edge_aligned is False
    assert set(default.state.cell_kind(c) for c in default.state.cells) == {"T3"}
    assert _cells(default.state) == _cells(explicit_off.state)
    assert np.array_equal(default.triangulation.points, explicit_off.triangulation.points)


def test_explicit_existing_layout_never_uses_the_tensor_even_when_flagged():
    case = make_case((2, 2), (3,), (4,), (3,))
    plain = _seed(case)
    flagged = _seed(case, layout_policy="existing", edge_aligned=True)
    assert flagged.edge_aligned is False
    assert set(flagged.state.cell_kind(c) for c in flagged.state.cells) == {"T3"}
    assert _cells(flagged.state) == _cells(plain.state)
    assert np.array_equal(flagged.triangulation.points, plain.triangulation.points)


def test_adaptive_policy_admits_the_tensor_and_keeps_ownership():
    case = make_case((2, 2), (3,), (4,), (3,), angle=0.3, start=1, width=2.0, height=1.5)
    seed = _seed(case, layout_policy="adaptive", edge_aligned=True)
    assert isinstance(seed, PlanarQuadSeed) and seed.edge_aligned is True
    assert set(seed.state.cell_kind(c) for c in seed.state.cells) == {"Q4"}
    stations = case[2].outer_stations(case[1])
    assert [seed.station_to_node[s.key] for s in stations] == list(range(len(stations)))
    assert set(seed.state.protected_nodes) == set(range(len(stations)))
    assert not len(seed.triangulation.mandatory_segments)
    assert seed.discrete_area == pytest.approx(2.0 * 1.5, rel=1e-12)
    again = _seed(case, layout_policy="adaptive", edge_aligned=True)
    assert _cells(again.state) == _cells(seed.state)


def test_opt_in_falls_back_to_the_unchanged_adaptive_route_when_declined():
    case = make_case((2, 1), (3,), (4,), (3,))  # unequal opposite counts
    fallback = _seed(case, layout_policy="adaptive", edge_aligned=True)
    plain = _seed(case, layout_policy="adaptive")
    assert fallback.edge_aligned is False
    assert _cells(fallback.state) == _cells(plain.state)
    assert np.array_equal(fallback.triangulation.points, plain.triangulation.points)


def test_edge_aligned_flag_is_a_strict_bool():
    with pytest.raises(MeshError):
        _seed(make_case(), layout_policy="adaptive", edge_aligned=1)


# --- real production geometry (no stand-ins) -------------------------------

def _plate(width=1.0, height=1.0):
    """The ``_plane_face`` fixture of test_q6_public_integration, resized."""
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (width, 0.0, 0.0), (width, height, 0.0), (0.0, height, 0.0))
    )
    return geometry, geometry.add_plate(vertices)


def _faces(result):
    return result.mesh.hybrid_diagnostics["front"]["faces"]


def _mesh_signature(result):
    mesh = result.mesh
    return (
        {int(k): tuple(int(n) for n in v) for k, v in mesh.quads.items()},
        {int(k): tuple(int(n) for n in v) for k, v in mesh.tris.items()},
        {int(k): tuple(float(c) for c in v) for k, v in mesh.nodes.items()},
    )


def test_real_plate_seed_flag_matrix():
    geometry, face = _plate()
    admitted = build_planar_quad_seed(
        geometry, face, 0.25, layout_policy="adaptive", edge_aligned=True)
    assert admitted.edge_aligned is True
    assert set(admitted.state.cell_kind(c) for c in admitted.state.cells) == {"Q4"}
    assert admitted.discrete_area == pytest.approx(1.0, rel=1e-12)
    # Plain adaptive callers (no flag) are bit-compatible: same seed, flag off.
    plain = build_planar_quad_seed(geometry, face, 0.25, layout_policy="adaptive")
    assert plain.edge_aligned is False
    assert _cells(plain.state) == _cells(admitted.state)
    # The explicit existing layout never takes the tensor route.
    existing = build_planar_quad_seed(
        geometry, face, 0.25, layout_policy="existing", edge_aligned=True)
    assert existing.edge_aligned is False
    assert set(existing.state.cell_kind(c) for c in existing.state.cells) == {"T3"}


def test_public_layout_default_signature_is_the_none_sentinel():
    parameter = inspect.signature(generate_hybrid_mesh_result).parameters["layout_policy"]
    assert parameter.default is None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_public_omitted_layout_is_adaptive_for_explicit_quad_options():
    geometry, face = _plate(1.0, 0.6)
    omitted = generate_hybrid_mesh_result(
        geometry, target_size=0.25, face_ids=(face,), quad_options=QuadMeshingOptions())
    explicit = generate_hybrid_mesh_result(
        geometry, target_size=0.25, face_ids=(face,), quad_options=QuadMeshingOptions(),
        layout_policy="adaptive")
    driver = _faces(omitted)[face]
    assert "layout_seed" in driver and "edge_aligned_seed" in driver
    assert driver["final_t3"] == 0 and not omitted.mesh.tris
    assert _mesh_signature(omitted) == _mesh_signature(explicit)
    # Tensor preserved through Q5: every interior node sits on a grid line
    # through boundary stations, so nothing was relaxed off the tensor.
    mesh = omitted.mesh
    boundary = {int(n) for chain in mesh.nodes_of_edge.values() for n in chain}
    xs = {round(float(mesh.nodes[n][0]), 9) for n in boundary}
    ys = {round(float(mesh.nodes[n][1]), 9) for n in boundary}
    interior = [n for n in mesh.nodes if int(n) not in boundary]
    assert interior
    for n in interior:
        assert round(float(mesh.nodes[n][0]), 9) in xs
        assert round(float(mesh.nodes[n][1]), 9) in ys


def test_public_explicit_existing_is_honoured_with_quad_options():
    geometry, face = _plate(1.0, 0.6)
    result = generate_hybrid_mesh_result(
        geometry, target_size=0.25, face_ids=(face,), quad_options=QuadMeshingOptions(),
        layout_policy="existing")
    driver = _faces(result)[face]
    assert "layout_seed" not in driver and "edge_aligned_seed" not in driver
    assert result.mesh.hybrid_diagnostics["route"] == "quad-first"


def test_public_nonquad_routes_keep_the_existing_layout():
    geometry, face = _plate()
    omitted = generate_hybrid_mesh_result(geometry, target_size=0.25, face_ids=(face,))
    explicit = generate_hybrid_mesh_result(
        geometry, target_size=0.25, face_ids=(face,), layout_policy="existing")
    assert omitted.mesh.hybrid_diagnostics.get("route") != "quad-first"
    assert _mesh_signature(omitted) == _mesh_signature(explicit)
    native = generate_hybrid_mesh_result(
        geometry, target_size=0.25, face_ids=(face,), strategy=MeshingStrategy.NATIVE)
    assert native.mesh.hybrid_diagnostics.get("route") != "quad-first"
    # Explicit adaptive without quad_options keeps its typed refusal.
    with pytest.raises(MeshError, match="adaptive layout requires explicit quad_options"):
        generate_hybrid_mesh_result(
            geometry, target_size=0.25, face_ids=(face,), layout_policy="adaptive")
    with pytest.raises(MeshError, match="layout_policy must be"):
        generate_hybrid_mesh_result(
            geometry, target_size=0.25, face_ids=(face,), quad_options=QuadMeshingOptions(),
            layout_policy="unknown")


def test_public_cancellation_during_the_tensor_seed_propagates():
    geometry, face = _plate(1.0, 0.6)

    class Cancelled(RuntimeError):
        pass

    def check(stage):
        if stage == "quad-first:structured-rectangle-row":
            raise Cancelled(stage)

    with pytest.raises(Cancelled):
        generate_hybrid_mesh_result(
            geometry, target_size=0.25, face_ids=(face,),
            quad_options=QuadMeshingOptions(), cancellation_check=check)


# --- real split-edge production geometry ------------------------------------
# A 2 x 2 plate authored with collinear consecutive vertices: every side is two
# unit source edges, so no splitting API is needed.

# Unit edges at 0.6: 1/0.6 rounds and ceils to 2 divisions per source edge, so
# station counts do not hang on a bit-level ceil of a rotated 1.0000000000000002.
_SPLIT_H = 0.6

_SPLIT_RECTANGLE = ((0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0),
                    (2, 2, 0), (1, 2, 0), (0, 2, 0), (0, 1, 0))


def _split_rectangle(angle=0.0, tilt=0.0, start=0):
    """Rigidly rotated/tilted split-edge plate whose loop opens at ``start``."""
    ca, sa, ct, st = np.cos(angle), np.sin(angle), np.cos(tilt), np.sin(tilt)
    rotation = (np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
                @ np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]]))
    shift = np.array((3.0, -1.0, 0.5))
    ring = _SPLIT_RECTANGLE[start:] + _SPLIT_RECTANGLE[:start]
    points = [tuple(float(c) for c in rotation @ np.asarray(p, dtype=float) + shift)
              for p in ring]
    geometry = GeometryModel()
    vertices = geometry.add_points(points)
    plane = Plane(shift, rotation @ np.array((2.0, 0.0, 0.0)),
                  rotation @ np.array((0.0, 2.0, 0.0)))
    face = geometry.add_face(geometry.add_polyline(vertices, close=True), surface=plane)
    return geometry, face


@pytest.mark.parametrize("angle,tilt", [(0.0, 0.0), (0.7, 0.0), (-1.9, 0.4)])
@pytest.mark.parametrize("start", [0, 1, 2, 5])
def test_real_split_edge_plate_is_a_station_conforming_tensor(angle, tilt, start):
    geometry, face = _split_rectangle(angle, tilt, start)
    domain = PlanarQuadDomain.from_geometry(geometry, face)
    registry = BoundaryStationRegistry.for_domain(geometry, domain, _SPLIT_H)
    stations = registry.outer_stations(domain)
    assert len(domain.edge_uses) == 8 and len(stations) == 16
    seed = build_planar_quad_seed(
        geometry, face, _SPLIT_H, domain=domain, registry=registry,
        layout_policy="adaptive", edge_aligned=True)
    assert seed.edge_aligned is True
    assert len(seed.state.cells) == 16 and len(seed.state.nodes) == 25
    assert set(seed.state.cell_kind(c) for c in seed.state.cells) == {"Q4"}
    ring = np.asarray([domain.project(s.position) for s in stations])
    assert np.array_equal(np.asarray(seed.triangulation.points)[:16], ring)
    assert [seed.station_to_node[s.key] for s in stations] == list(range(16))
    assert set(seed.state.protected_nodes) == set(range(16))
    assert seed.discrete_area == pytest.approx(4.0, rel=1e-12)
    # Without the opt-in the adaptive route is the unchanged triangle seed.
    plain = build_planar_quad_seed(
        geometry, face, _SPLIT_H, domain=domain, registry=registry, layout_policy="adaptive")
    assert plain.edge_aligned is False
    assert "T3" in {plain.state.cell_kind(c) for c in plain.state.cells}


def test_real_split_edge_plate_survives_the_public_route_with_q5_preserved():
    geometry, face = _split_rectangle(0.7, 0.0, 3)
    options = QuadMeshingOptions()
    result = generate_hybrid_mesh_result(
        geometry, target_size=_SPLIT_H, face_ids=(face,), quad_options=options)
    mesh = result.mesh
    driver = _faces(result)[face]
    assert driver["edge_aligned_seed"] == {
        "boundary_stations": 16, "interior_points": 9, "tensor_cells": 16}
    assert len(mesh.quads) == 16 and not mesh.tris
    validation = mesh.hybrid_diagnostics["validation"]["faces"][face]
    assert validation["area_ratio"] == pytest.approx(1.0)
    # Interior nodes are exactly the tensor points (Q5 left the grid alone).
    domain = PlanarQuadDomain.from_geometry(geometry, face)
    seed = build_planar_quad_seed(
        geometry, face, _SPLIT_H, layout_policy="adaptive", edge_aligned=True)
    expected = [np.asarray(domain.lift(seed.state.position(n)), dtype=float)
                for n in sorted(seed.state.nodes) if n >= 16]
    boundary = {int(n) for chain in mesh.nodes_of_edge.values() for n in chain}
    published = [np.asarray(mesh.nodes[n], dtype=float)
                 for n in mesh.nodes if int(n) not in boundary]
    assert len(expected) == len(published) == 9
    for point in expected:
        assert min(float(np.linalg.norm(point - other)) for other in published) < 1e-9
    # Explicit existing keeps the original layout: no tensor record.
    existing = generate_hybrid_mesh_result(
        geometry, target_size=_SPLIT_H, face_ids=(face,), quad_options=options,
        layout_policy="existing")
    assert "edge_aligned_seed" not in _faces(existing)[face]


def test_real_split_edge_plate_cancellation_propagates():
    geometry, face = _split_rectangle()

    class Cancelled(RuntimeError):
        pass

    def check(stage):
        if stage == "quad-first:edge-aligned-tensor-row":
            raise Cancelled(stage)

    with pytest.raises(Cancelled):
        generate_hybrid_mesh_result(
            geometry, target_size=_SPLIT_H, face_ids=(face,),
            quad_options=QuadMeshingOptions(), cancellation_check=check)


def _real_hole_face():
    geometry = GeometryModel()
    plane = Plane(np.asarray((0.0, 0.0, 0.0)), np.asarray((4.0, 0.0, 0.0)),
                  np.asarray((0.0, 4.0, 0.0)))
    outer = geometry.add_points(
        ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 4.0, 0.0), (0.0, 4.0, 0.0)))
    face = geometry.add_face_from_loop(
        geometry.order_loop(geometry.add_polyline(outer, close=True)), surface=plane)
    hole = geometry.add_points(
        ((0.75, 1.25, 0.0), (1.75, 1.25, 0.0), (1.75, 2.25, 0.0), (0.75, 2.25, 0.0)))
    trim_face(geometry, face, (tuple(
        OrientedEdge(edge, True) for edge in geometry.add_polyline(hole, close=True)),))
    return geometry, face, 15.0


def _real_trapezoid_face():
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (3.0, 4.0, 0.0), (1.0, 4.0, 0.0)))
    return geometry, geometry.add_plate(vertices), 12.0


@pytest.mark.parametrize("make", [_real_hole_face, _real_trapezoid_face])
def test_real_hole_and_angled_faces_are_never_given_the_tensor(make):
    geometry, face, area = make()
    seed = build_planar_quad_seed(
        geometry, face, 0.5, layout_policy="adaptive", edge_aligned=True)
    assert seed.edge_aligned is False
    assert "T3" in {seed.state.cell_kind(c) for c in seed.state.cells}
    assert seed.discrete_area == pytest.approx(area, rel=1e-9)
