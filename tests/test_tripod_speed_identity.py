"""Speed slice (tripod 2026-11-25): each shortcut is decision-identical to the route it replaces.

1. ``_finite_hull_seed``: bounding-box prefilter + tuple coordinates == the plain scan (driven through an
   ndarray subclass, which keeps the original code path), callbacks and errors included, with far fewer
   orientation predicates.
4. ``prepare_qualified_s3_mesh``: owner normals evaluated before the S3 repair are reused afterwards on the same
   inputs; the result equals the route with the memo disabled.
5. ``repair_s3_admission``: bisection candidates read the caller's incidence map; the result equals the route
   that rebuilds it.
"""
from __future__ import annotations

from math import radians, tan

import numpy as np
import pytest

from anygeometry import GeometryModel

from anymesher import Mesh, MeshError, generate_hybrid_mesh_result, repair_s3_admission
from anymesher import s3_production, s3_repair, triangulation
from anymesher.s3_production import S3OwnerAuthorityError


class _PlainArray(np.ndarray):
    """Same values, but not exactly ``np.ndarray``: ``_finite_hull_seed`` keeps its original scan."""


def _plain(points: np.ndarray) -> np.ndarray:
    return np.array(points, dtype=points.dtype, copy=True).view(_PlainArray)


def _cases():
    rng = np.random.default_rng(20261125)
    made = []
    for count in (6, 14, 30):
        # Quad-seed-like inputs: a lattice (cocircular quadruples), collinear boundary stations,
        # interior jitter and points exactly on lattice edges.
        xs, ys = np.meshgrid(np.linspace(0.0, 2.0, count), np.linspace(0.0, 1.0, max(3, count // 2)))
        lattice = np.c_[xs.ravel(), ys.ravel()]
        on_edges = np.c_[rng.integers(0, count - 1, count) * (2.0 / (count - 1)) + 1.0 / (count - 1),
                         np.zeros(count)]
        jitter = rng.uniform(0.0, 1.0, (count, 2)) * [2.0, 1.0]
        made.append(np.unique(np.r_[lattice, on_edges, jitter], axis=0))
    angle = 0.3
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xs, ys = np.meshgrid(np.linspace(0.0, 1.0, 12), np.linspace(0.0, 1.0, 9))
    made.append(np.unique(np.c_[xs.ravel(), ys.ravel()] @ rotation.T + [1.0e3, -2.5e2], axis=0))
    made.append(np.unique(rng.uniform(-1.0, 1.0, (120, 2)), axis=0))
    side = np.linspace(0.0, 1.0, 9)
    made.append(np.unique(np.r_[np.c_[side, 0 * side], np.c_[1 + 0 * side, side], np.c_[side, 1 + 0 * side],
                                np.c_[0 * side, side], [[0.5, 0.5]]], axis=0))
    # Duplicate stations (zero-area cells whose turns are all zero outside their box; Codex r1).
    made.append(np.array([(0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (1.0, 1.0), (1.0, 1.0), (2.0, 2.0)]))
    made.append(np.r_[made[0], made[0][::3]])
    # Extreme magnitudes, subnormal offsets and signed zeros.
    made.append(np.unique(np.r_[made[1] * 1.0e300, [[1.0e300, 5.0e299]]], axis=0))
    tiny = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (5.0e-324, 0.5), (0.5, 5.0e-324),
                     (0.5, 0.5), (-0.0, 0.25), (0.25, -0.0)])
    made.append(tiny)
    return made


CASES = _cases()


def _outcome(points, mutate=None):
    stages = []

    def callback(stage):
        stages.append(stage)
        if mutate is not None:
            mutate(points, stage, sum(1 for seen in stages if seen == stage))

    try:
        result = ("ok", triangulation._finite_hull_seed(points, callback))
    except Exception as error:  # compared, never swallowed: the type and message are the outcome
        result = ("error", type(error).__name__, str(error))
    return result, stages


@pytest.mark.parametrize("index", range(len(CASES)))
def test_finite_hull_seed_prefilter_matches_the_plain_scan(index):
    points = CASES[index]
    assert type(points) is np.ndarray and points.dtype == np.float64
    assert _outcome(points.copy()) == _outcome(_plain(points))


@pytest.mark.parametrize("stage,when", [("insertion", 1), ("insertion", 3), ("edge", 1), ("edge", 7),
                                        ("legality", 1)])
def test_finite_hull_seed_follows_a_callback_that_moves_the_input(stage, when):
    # Codex r1: a callback that changes the input must see the live-array behaviour of the original.
    def move(points, seen, count):
        if seen.endswith(stage) and count == when:
            points[-1] = (8.0, 8.0)
            points[len(points) // 2] = points[len(points) // 2] * 0.5

    for points in (np.array([(0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (1.0, 1.0)]), CASES[1]):
        assert _outcome(points.copy(), move) == _outcome(_plain(points), move)


def test_finite_hull_seed_non_float64_input_keeps_the_plain_scan():
    as_float32 = CASES[0].astype(np.float32)
    assert _outcome(as_float32.copy()) == _outcome(as_float32.view(_PlainArray))


def test_finite_hull_seed_prefilter_skips_cells_that_cannot_contain_the_point(monkeypatch):
    # Fails on the old code, whose location scan tests every cell for every inserted point.
    points = CASES[2]
    calls = [0]
    original = triangulation.orient2d

    def counted(*args):
        calls[0] += 1
        return original(*args)

    monkeypatch.setattr(triangulation, "orient2d", counted)
    fast = triangulation._finite_hull_seed(points)
    fast_calls, calls[0] = calls[0], 0
    plain = triangulation._finite_hull_seed(_plain(points))
    assert fast == plain
    assert fast_calls < 0.75 * calls[0]


def _square_mesh(geometry_type=GeometryModel, owner=True):
    geometry = geometry_type()
    face_id = geometry.add_plate(
        geometry.add_points(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))))
    if owner:
        geometry.add_sheet((face_id,))
    result = generate_hybrid_mesh_result(geometry, target_size=0.25, strategy="native",
                                         native_backend="python", recombine=False,
                                         structural_preparation=False)
    return geometry, result.mesh


def _counting(monkeypatch):
    calls = [0]
    original = s3_production._physical_owner_normal

    def counted(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(s3_production, "_physical_owner_normal", counted)
    return calls


def _qualified(geometry):
    made = generate_hybrid_mesh_result(geometry, target_size=0.25, strategy="native", native_backend="python",
                                       recombine=False, structural_preparation=False, qualified_s3=True).mesh
    return (tuple((k, tuple(map(float, made.nodes[k]))) for k in sorted(made.nodes)),
            tuple((k, tuple(v)) for k, v in sorted(made.shells.items())),
            repr(made.structural_preparation["qualified_s3"])), len(made.shells)


def test_qualified_s3_with_the_owner_memo_equals_the_route_without_it(monkeypatch):
    geometry, _ = _square_mesh()
    calls = _counting(monkeypatch)
    with_memo, shells = _qualified(geometry)
    with_memo_calls, calls[0] = calls[0], 0
    monkeypatch.setattr(s3_production, "_owner_normal_key", lambda *args: None)
    without, _ = _qualified(geometry)
    assert with_memo == without
    assert calls[0] >= 2 * shells
    # Fails on the old code: unchanged shells are not evaluated a second time.
    assert with_memo_calls < calls[0]


def test_owner_normal_memo_reuses_only_identical_inputs(monkeypatch):
    geometry, mesh = _square_mesh()
    fresh, fresh_sources = s3_production._shell_owner_authority(geometry, mesh)
    memo = {"geometry": geometry, "normals": {}}
    calls = _counting(monkeypatch)
    first, first_sources = s3_production._shell_owner_authority(geometry, mesh, _normal_memo=memo)
    second, second_sources = s3_production._shell_owner_authority(geometry, mesh, _normal_memo=memo)
    assert calls[0] == len(mesh.shells)
    assert first == second == fresh and first_sources == second_sources == fresh_sources
    node = sorted(mesh.nodes)[0]
    mesh.nodes[node] = mesh.nodes[node] + np.array([1.0e-3, 0.0, 0.0])
    users = [element for element in mesh.shells if node in mesh.corners_of(element)]
    calls[0] = 0
    moved, _ = s3_production._shell_owner_authority(geometry, mesh, _normal_memo=memo)
    assert calls[0] == len(users)
    monkeypatch.undo()
    assert moved == s3_production._shell_owner_authority(geometry, mesh)[0]


class _Subclassed(GeometryModel):
    """A subclass may have impure geometry protocols (Codex r1): never memoised."""


def test_owner_normal_memo_is_not_used_for_a_subclass_or_another_geometry(monkeypatch):
    geometry, mesh = _square_mesh()
    other, _ = _square_mesh()
    subclassed, sub_mesh = _square_mesh(_Subclassed)
    calls = _counting(monkeypatch)
    memo = {"geometry": geometry, "normals": {}}
    s3_production._shell_owner_authority(geometry, mesh, _normal_memo=memo)
    calls[0] = 0
    s3_production._shell_owner_authority(other, mesh, _normal_memo=memo)
    assert calls[0] == len(mesh.shells)
    memo = {"geometry": subclassed, "normals": {}}
    calls[0] = 0
    s3_production._shell_owner_authority(subclassed, sub_mesh, _normal_memo=memo)
    s3_production._shell_owner_authority(subclassed, sub_mesh, _normal_memo=memo)
    assert calls[0] == 2 * len(sub_mesh.shells) and not memo["normals"]


def test_owner_normal_memo_keeps_the_original_refusal_without_a_face_use():
    geometry, mesh = _square_mesh(owner=False)
    with pytest.raises(S3OwnerAuthorityError) as plain:
        s3_production._shell_owner_authority(geometry, mesh)
    with pytest.raises(S3OwnerAuthorityError) as memoised:
        s3_production._shell_owner_authority(geometry, mesh, _normal_memo={"geometry": geometry, "normals": {}})
    assert str(memoised.value) == str(plain.value)


OWNER = np.asarray((0.0, 0.0, 1.0))


def _sliver_patch(angle_deg: float) -> Mesh:
    return Mesh(
        nodes={
            1: np.asarray((0.0, 0.0, 0.0)),
            2: np.asarray((1.0, 0.0, 0.0)),
            3: np.asarray((1.0, tan(radians(angle_deg)), 0.0)),
            4: np.asarray((0.0, 1.0, 0.0)),
            5: np.asarray((2.0, 0.5, 0.0)),
        },
        tris={10: (1, 2, 3), 20: (1, 3, 4), 30: (2, 5, 3)},
        elements_of_face={7: [10, 20, 30]},
        elements_of_sheet={8: [10, 20, 30]},
    )


def _mesh_signature(mesh):
    return (tuple((k, tuple(float(v) for v in mesh.nodes[k])) for k in sorted(mesh.nodes)),
            tuple((k, tuple(mesh.tris[k])) for k in sorted(mesh.tris)),
            tuple((k, tuple(v)) for k, v in sorted(mesh.elements_of_face.items())))


def _candidate_signature(made):
    owners = None if made[1] is None else tuple((k, tuple(v)) for k, v in sorted(made[1].items()))
    return None if made[0] is None else _mesh_signature(made[0]), owners, made[2], made[3], made[4]


def test_candidate_bisection_with_the_callers_incidence_matches_a_rebuild():
    mesh = _sliver_patch(10.0)
    owners = {element: OWNER for element in mesh.tris}
    selected = frozenset(mesh.tris)
    incidence = s3_repair._shell_incidence(mesh)
    for edge in sorted(incidence):
        attached = tuple(sorted(owner for kind, owner in incidence[edge] if kind == "tri"))
        rebuilt = s3_repair._candidate_bisection(mesh, edge, attached, owners, selected,
                                                 next_node_id=6, next_element_id=31)
        reused = s3_repair._candidate_bisection(mesh, edge, attached, owners, selected,
                                                next_node_id=6, next_element_id=31, incidence=incidence)
        assert _candidate_signature(reused) == _candidate_signature(rebuilt)


def _repair_outcome():
    mesh = _sliver_patch(10.0)
    try:
        result = repair_s3_admission(mesh, element_owner_normals={element: tuple(OWNER) for element in mesh.tris})
    except Exception as error:  # compared, never swallowed: the type and message are part of the outcome
        return ("error", type(error).__name__, str(error))
    return ("ok", _mesh_signature(result.mesh), result.admission.admitted, result.edge_flips,
            tuple((a.action, a.status, a.detail) for a in result.attempts))


def test_repair_with_incidence_reuse_equals_the_rebuilding_route(monkeypatch):
    inside = [False]
    rebuilt_inside = [0]
    called = [0]
    original_incidence = s3_repair._shell_incidence
    original_bisection = s3_repair._candidate_bisection

    def incidence(mesh):
        if inside[0]:
            rebuilt_inside[0] += 1
        return original_incidence(mesh)

    def bisection(*args, **kwargs):
        called[0] += 1
        inside[0] = True
        try:
            return original_bisection(*args, **kwargs)
        finally:
            inside[0] = False

    monkeypatch.setattr(s3_repair, "_shell_incidence", incidence)
    monkeypatch.setattr(s3_repair, "_candidate_bisection", bisection)
    reused = _repair_outcome()
    # Fails on the old code, where every bisection candidate rebuilt the whole-mesh map.
    assert called[0] > 0 and rebuilt_inside[0] == 0
    monkeypatch.setattr(s3_repair, "_plain_shells", lambda mesh: False)
    rebuilt_inside[0] = 0
    rebuilding = _repair_outcome()
    assert rebuilt_inside[0] > 0
    assert reused == rebuilding


class _LoggingTuple(tuple):
    """Connectivity whose reads are observable (Codex r1): the map is rebuilt, never reused."""


def test_incidence_is_not_reused_for_side_effecting_connectivity():
    mesh = _sliver_patch(10.0)
    assert s3_repair._plain_shells(mesh)
    mesh.tris[10] = _LoggingTuple(mesh.tris[10])
    assert not s3_repair._plain_shells(mesh)


def test_finite_hull_seed_follows_a_callback_that_reshapes_the_input():
    # Codex r2: a byte-identical in-place reshape must also drop to the live array.
    def reshape(points, seen, count):
        if seen.endswith("insertion") and count == 1:
            points.shape = (2, 4)

    points = np.array([(0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (1.0, 1.0)])
    assert _outcome(points.copy(), reshape) == _outcome(_plain(points), reshape)


def test_finite_hull_seed_calls_no_extra_predicates_before_the_first_insertion(monkeypatch):
    # Codex r2: the zero-area flags come from a vectorised float filter, not from extra orient2d calls.
    counts = {}
    original = triangulation.orient2d
    calls = [0]

    def counted(*args):
        calls[0] += 1
        return original(*args)

    monkeypatch.setattr(triangulation, "orient2d", counted)
    for label, points in (("fast", CASES[2].copy()), ("plain", _plain(CASES[2]))):
        calls[0] = 0
        seen = []

        def callback(stage):
            if stage.endswith("insertion") and not seen:
                seen.append(calls[0])

        triangulation._finite_hull_seed(points, callback)
        counts[label] = seen[0]
    assert counts["fast"] == counts["plain"]


def test_owner_normal_memo_needs_default_numpy_error_handling(monkeypatch):
    geometry, mesh = _square_mesh()
    calls = _counting(monkeypatch)
    memo = {"geometry": geometry, "normals": {}}
    old = np.seterrcall(lambda *args: None)
    try:
        with np.errstate(under="call"):
            s3_production._shell_owner_authority(geometry, mesh, _normal_memo=memo)
            s3_production._shell_owner_authority(geometry, mesh, _normal_memo=memo)
    finally:
        np.seterrcall(old)
    assert calls[0] == 2 * len(mesh.shells) and not memo["normals"]


def test_owner_normal_memo_needs_library_values_by_identity():
    from anygeometry.surfaces import Plane

    class OwnPlane(Plane):
        pass

    origin, u, v = np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    assert s3_production._library_value(Plane(origin, u, v))
    assert not s3_production._library_value(OwnPlane(origin, u, v))
    assert not s3_production._library_value(1.0e155) and not s3_production._library_value(1.0e-30)
    assert s3_production._library_value(0.0) and s3_production._library_value((1.0, -2.0, 3))

def test_plain_shells_never_calls_user_code_and_needs_plain_nodes():
    class Hostile(type):
        def __eq__(cls, other):
            raise RuntimeError("compared")

        def __hash__(cls):
            raise RuntimeError("hashed")

    class Odd(tuple, metaclass=Hostile):
        pass

    mesh = _sliver_patch(10.0)
    mesh.tris[10] = Odd(mesh.tris[10])
    assert not s3_repair._plain_shells(mesh)
    mesh = _sliver_patch(10.0)

    class Nodes(dict):
        pass

    mesh.nodes = Nodes(mesh.nodes)
    assert not s3_repair._plain_shells(mesh)


def test_owner_normal_memo_declines_magnitudes_that_could_warn(monkeypatch):
    # Codex r3: a reused evaluation must not skip a NumPy warning; huge coordinates are evaluated each time.
    geometry, mesh = _square_mesh()
    for node in mesh.nodes:
        mesh.nodes[node] = mesh.nodes[node] + np.array([0.0, 0.0, 1.0e155])
    calls = _counting(monkeypatch)
    memo = {"geometry": geometry, "normals": {}}
    with np.errstate(over="ignore", invalid="ignore"):
        first = s3_production._shell_owner_authority(geometry, mesh, _normal_memo=memo)
        second = s3_production._shell_owner_authority(geometry, mesh, _normal_memo=memo)
    assert first == second
    assert calls[0] == 2 * len(mesh.shells) and not memo["normals"]


def test_plain_shells_needs_plain_beams_couplings_and_float64_nodes():
    class Beams(dict):
        pass

    mesh = _sliver_patch(10.0)
    mesh.beams = Beams()
    assert not s3_repair._plain_shells(mesh)
    mesh = _sliver_patch(10.0)
    mesh.nodes[1] = mesh.nodes[1].astype(np.float32)
    assert not s3_repair._plain_shells(mesh)


def test_plain_shells_needs_exact_integer_keys_in_beams_and_couplings():
    # Codex r4: an int subclass key can run user code (e.g. in max()) after the map was captured.
    class Key(int):
        pass

    mesh = _sliver_patch(10.0)
    mesh.beams = {Key(99): (1, 2)}
    assert not s3_repair._plain_shells(mesh)
    mesh = _sliver_patch(10.0)
    mesh.beams = {99: (1, Key(2))}
    assert not s3_repair._plain_shells(mesh)


def test_owner_normal_memo_never_hashes_the_model_id(monkeypatch):
    # Codex r4: the key hashed a UUID subclass model id; the call-local memo no longer reads it.
    import uuid

    class Id(uuid.UUID):
        def __hash__(self):
            raise RuntimeError("hashed")

    geometry, mesh = _square_mesh()
    object.__setattr__(geometry, "_model_id", Id(str(geometry.model_id)))
    assert type(geometry.model_id) is Id
    expected = s3_production._shell_owner_authority(geometry, mesh)
    memo = {"geometry": geometry, "normals": {}}
    assert s3_production._shell_owner_authority(geometry, mesh, _normal_memo=memo) == expected
    assert s3_production._shell_owner_authority(geometry, mesh, _normal_memo=memo) == expected


def test_owner_normal_memo_declines_a_nurbs_surface_without_raising():
    # Codex r4: the guard read vars() of the slotted NurbsSurface and raised TypeError.
    from anygeometry.nurbs import NurbsSurface

    surface = NurbsSurface(1, 1, (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0),
                           (((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0))))
    assert s3_production._library_value(surface) is False
    geometry, mesh = _square_mesh()
    face_id = next(iter(geometry.faces))
    geometry.set_face_parameterization(face_id, surface)
    uses = tuple(int(u) for u in s3_production._face_use_ids(geometry).get(face_id, ()))
    assert s3_production._owner_face_admitted(geometry, face_id, uses) is False


def test_owner_normal_memo_checks_the_effective_parameterization(monkeypatch):
    # Codex r4: a parameterization installed with set_face_parameterization is what face_point evaluates.
    from anygeometry.surfaces import Plane

    class OwnPlane(Plane):
        pass

    geometry, mesh = _square_mesh()
    face_id = next(iter(geometry.faces))
    uses = tuple(int(u) for u in s3_production._face_use_ids(geometry).get(face_id, ()))
    assert s3_production._owner_face_admitted(geometry, face_id, uses)
    origin, u, v = np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    geometry.set_face_parameterization(face_id, OwnPlane(origin, u, v))
    assert not s3_production._owner_face_admitted(geometry, face_id, uses)
    calls = _counting(monkeypatch)
    memo = {"geometry": geometry, "normals": {}}
    s3_production._shell_owner_authority(geometry, mesh, _normal_memo=memo)
    s3_production._shell_owner_authority(geometry, mesh, _normal_memo=memo)
    assert calls[0] == 2 * len(mesh.shells) and not memo["normals"]
    geometry.set_face_parameterization(face_id, Plane(origin, u, v))
    assert s3_production._owner_face_admitted(geometry, face_id, uses)
