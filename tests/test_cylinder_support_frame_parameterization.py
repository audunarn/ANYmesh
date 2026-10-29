"""MESH-B A2: an owner Cylinder face with a distinct (Coons) parameterization.

The quad-first public route meshes such a face in the support frame on the
owner-certified trimmed-cylinder binding. The parameterization is not used
inside meshing, so the mesh must be bit-identical to the same face without
it (its "twin"). Every other route keeps refusing the parameterized face.

The model mirrors one AIP Gen2 tripod skin: a 6-degree patch of a full-turn
support Cylinder with mitred EllipticArc ends and straight generators, and a
topology ``CoonsSurface()`` parameterization (ruled, not affine, in the
support frame).
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest

from anygeometry import EllipticArc, GeometryError, GeometryModel
from anygeometry.cylinder_patch import CylinderPatchError
from anygeometry.structural import ParameterRange
from anygeometry.surfaces import CoonsSurface, Cylinder, Plane

from anymesher._cylindrical_chart import CylindricalMetricChart
from anymesher._cylindrical_public import prepare_bindings
from anymesher.errors import MeshError
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.native_v2 import NativeMeshingOptions
from anymesher.quad.options import QuadMeshingOptions
from anymesher.serialize import mesh_to_dict


_DISTINCT = "distinct face parameterization"


def _mitred_patch(parameterized: bool, *, R=6.5, a=.3, sweep=math.radians(6.), z0=0., z1=6.,
                  k0=math.tan(math.pi / 8), k1=-.3):
    """z1=6 keeps the "top" mitre above the "bottom" one (an upright patch).

    With z1=3 (the 646da87 default) the mitres cross over: the top end lies
    below the bottom end, so the loop winds clockwise on the support and the
    Coons normal points inward, against the support. Fix round 1 refuses that
    reversed case (test_reversed_mitred_patch_is_refused).
    """
    model = GeometryModel()

    def point(theta, z, k):
        return (R * math.cos(theta), R * math.sin(theta), z + k * R * math.cos(theta))

    v = model.add_points((point(a, z0, k0), point(a + sweep, z0, k0),
                          point(a + sweep, z1, k1), point(a, z1, k1)))
    bottom = model.add_curve(v[0], v[1], EllipticArc((0., 0., z0), (R, 0., k0 * R), (0., R, 0.), a, sweep))
    right = model.add_line(v[1], v[2])
    top = model.add_curve(v[3], v[2], EllipticArc((0., 0., z1), (R, 0., k1 * R), (0., R, 0.), a, sweep))
    left = model.add_line(v[0], v[3])
    # Full-turn support whose height exceeds the patch, as in the tripod repair.
    support = Cylinder((0., 0., z0 - 7.), (0., 0., 1.), (1., 0., 0.), R, (z1 - z0) + 14., a - .25)
    face = model.add_face((bottom, right, top, left), surface=support)
    # The twin also records one parameterization change, so revisions match.
    model.set_face_parameterization(face, CoonsSurface() if parameterized else None)
    model.add_sheet((face,))
    return model, face


def _web(parameterized: bool):
    """A planar web whose Plane basis (SVD) differs from its Coons uv."""
    model = GeometryModel()
    corners = np.array(((0., 0., 0.), (2.3, .4, .9), (2.1, 1.9, 1.6), (-.2, 1.7, .5)))
    origin = corners[0]
    normal = np.cross(corners[1] - origin, corners[3] - origin)
    corners[2] -= ((corners[2] - origin) @ normal) / (normal @ normal) * normal
    v = model.add_points(tuple(map(tuple, corners)))
    edges = [model.add_line(v[i], v[(i + 1) % 4]) for i in range(4)]
    _, _, basis = np.linalg.svd(corners - origin, full_matrices=False)
    face = model.add_face(edges, surface=Plane(origin, basis[0], basis[1]))
    # The twin also records one parameterization change, so revisions match.
    model.set_face_parameterization(face, CoonsSurface() if parameterized else None)
    model.add_sheet((face,))
    return model, face


def _generate(model, face, order="linear", target=.25):
    return generate_hybrid_mesh_result(
        model, face_ids=(face,), target_size=target, strategy="native",
        native_backend="python", recombine=True, order=order,
        quad_options=QuadMeshingOptions(),
    ).mesh


def _bits(value):
    """JSON-safe structure with every float replaced by its exact hex form."""
    if isinstance(value, float):
        return float.hex(value)
    if isinstance(value, np.floating):
        return float.hex(float(value))
    if isinstance(value, np.ndarray):
        return _bits(value.tolist())
    if isinstance(value, dict):
        return {str(key): _bits(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bits(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    return value


def _canonical(mesh):
    """Every float as hex; the (random) owner model id replaced by a token."""
    model_id = str(mesh.geometry_model_id)
    assert model_id and model_id != "None"
    document = json.dumps(_bits(mesh_to_dict(mesh)), sort_keys=True, default=repr)
    diagnostics = json.dumps(_bits(mesh.hybrid_diagnostics), sort_keys=True, default=repr)
    return document.replace(model_id, "<model>"), diagnostics.replace(model_id, "<model>")


def test_the_mitred_coons_map_is_not_the_support_frame():
    """Teeth: the face map and the support disagree inside the patch."""
    twin, face = _mitred_patch(False)
    model, same = _mitred_patch(True)
    assert same == face
    uv = np.array(((.5, .5), (.25, .75)))
    assert np.max(np.abs(model.evaluate_face_many(face, uv) - twin.evaluate_face_many(face, uv))) > 1.
    corner = model.vertex_position(model.edges[model.faces[face].loop[0].edge].start)
    assert np.max(np.abs(np.subtract(model.face_local_uv(face, corner), twin.face_local_uv(face, corner)))) > .1


@pytest.mark.parametrize("order", ("linear", "quadratic"))
def test_parameterized_mitred_patch_meshes_bit_identical_to_its_twin(order):
    twin, face = _mitred_patch(False)
    model, same = _mitred_patch(True)
    expected = _generate(twin, face, order)
    actual = _generate(model, same, order)
    assert actual.quads or actual.tris
    assert actual.hybrid_diagnostics["route"] == "quad-first-cylindrical"
    expected_document, expected_diagnostics = _canonical(expected)
    actual_document, actual_diagnostics = _canonical(actual)
    assert actual_document == expected_document
    assert actual_diagnostics == expected_diagnostics
    for node, point in expected.nodes.items():
        assert np.asarray(actual.nodes[node], dtype=np.float64).tobytes() == np.asarray(point, dtype=np.float64).tobytes()


def test_parameterized_planar_web_meshes_bit_identical_to_its_twin():
    twin, face = _web(False)
    model, same = _web(True)
    assert np.max(np.abs(np.subtract(model.face_local_uv(face, (1., 1., .8)),
                                     twin.face_local_uv(face, (1., 1., .8))))) > 1e-3
    expected = _generate(twin, face)
    actual = _generate(model, same)
    assert actual.quads
    assert _canonical(actual) == _canonical(expected)
    points = np.asarray(list(actual.nodes.values()), dtype=float)
    plane = model.faces[same].surface
    normal = np.cross(plane.u_vector, plane.v_vector)
    assert np.max(np.abs((points - plane.origin) @ normal)) <= 1e-12 * max(1., float(np.max(np.abs(points))))


def test_direct_chart_route_still_refuses_a_distinct_parameterization():
    model, face = _mitred_patch(True)
    with pytest.raises(MeshError, match=_DISTINCT):
        CylindricalMetricChart.from_geometry(model, face)


def test_native_frontal_route_still_refuses_a_distinct_parameterization():
    model, face = _mitred_patch(True)
    with pytest.raises(MeshError, match=_DISTINCT):
        prepare_bindings(model, (face,), NativeMeshingOptions(point_placement="frontal_delaunay"))


def test_support_frame_chart_is_refused_by_native_binding_consumers():
    from anymesher._trimmed_cylinder_binding import prepare_trimmed_cylinders
    model, face = _mitred_patch(True)
    use = model.handle("face_use", next(i for i, u in model.face_uses.items() if u.face_id == face))
    binding = prepare_trimmed_cylinders(model, (use,), support_frame_parameterizations=True)
    ((_, chart),) = binding.charts
    assert chart.support_view is not None
    with pytest.raises(MeshError, match="quad-first"):
        binding.chart_for(use)


def test_legacy_patch_route_still_refuses_a_distinct_parameterization():
    from test_cylindrical_atlas_binding import _sector_model
    from anymesher._cylindrical_patch import prepare_cylindrical_patch
    model, selected = _sector_model(False)
    face = int(model.face_uses[selected[0].id].face_id)
    model.set_face_parameterization(face, CoonsSurface())
    # The legacy owner patch contract refuses first (support_or_parameterization_out_of_scope).
    with pytest.raises((MeshError, CylinderPatchError), match=f"{_DISTINCT}|support_or_parameterization_out_of_scope"):
        prepare_cylindrical_patch(model, (selected[0],))


# --- MESH-B fix round 1: point attachments, orientation, seam, hole ----------

def _arc(model, start, end, z, theta, sweep, R):
    return model.add_curve(start, end, EllipticArc((0., 0., z), (R, 0., 0.), (0., R, 0.), theta, sweep))


def _rect_patch(parameterized: bool, *, rotate=0, reverse=False, attach=None, R=6.5, a=.3,
                sweep=math.radians(20.), z0=0., z1=3., support_start=None, support_sweep=None,
                hole=None):
    """Straight-ended cylinder patch; the support matches the patch unless told otherwise.

    ``rotate`` starts the loop at another edge, so the topology Coons map is
    turned by 90 degrees against the support (u along the axis). ``reverse``
    traverses the loop the other way round (a reversed face parameterization).
    ``attach`` = ((u, v), xyz) declares one member_through_face point
    attachment, with (u, v) in this face's own frame.
    """
    model = GeometryModel()

    def point(theta, z):
        return (R * math.cos(theta), R * math.sin(theta), z)

    v = model.add_points((point(a, z0), point(a + sweep, z0), point(a + sweep, z1), point(a, z1)))
    edges = [_arc(model, v[0], v[1], z0, a, sweep, R), model.add_line(v[1], v[2]),
             _arc(model, v[3], v[2], z1, a, sweep, R), model.add_line(v[0], v[3])]
    edges = edges[rotate:] + edges[:rotate]
    if reverse:
        edges = edges[::-1]
    support = Cylinder((0., 0., z0), (0., 0., 1.), (1., 0., 0.), R, z1 - z0,
                       a if support_start is None else support_start,
                       sweep if support_sweep is None else support_sweep)
    face = model.add_face(tuple(edges), surface=support)
    if hole is not None:
        from dataclasses import replace
        from anygeometry.entities import OrientedEdge
        (b0, b1), (h0, h1) = hole
        w = model.add_points((point(b0, h0), point(b1, h0), point(b1, h1), point(b0, h1)))
        ring = (_arc(model, w[0], w[1], h0, b0, b1 - b0, R), model.add_line(w[1], w[2]),
                _arc(model, w[3], w[2], h1, b0, b1 - b0, R), model.add_line(w[0], w[3]))
        loop = (OrientedEdge(ring[3], True), OrientedEdge(ring[2], True),
                OrientedEdge(ring[1], False), OrientedEdge(ring[0], False))
        with model.transaction():
            model._put_entity("face", replace(model.faces[face], holes=(loop,)))
            assert not model.validate_topology()
    model.set_face_parameterization(face, CoonsSurface() if parameterized else None)
    model.add_sheet((face,))
    if attach is not None:
        (u, w_), xyz = attach
        xyz = np.asarray(xyz, dtype=float)
        radial = np.array((xyz[0], xyz[1], 0.)) / math.hypot(xyz[0], xyz[1])
        ends = model.add_points((tuple(xyz - .5 * radial), tuple(xyz + .5 * radial)))
        member = model.add_member((model.add_line(ends[0], ends[1]),))
        model.add_attachment(member, "member_through_face", "face", face, ParameterRange.point(.5),
                             [ParameterRange.point(u), ParameterRange.point(w_)])
    return model, face


def _shape_jacobian(model, face, order="linear"):
    return generate_hybrid_mesh_result(
        model, face_ids=(face,), target_size=.25, strategy="native", native_backend="python",
        recombine=True, order=order, quad_options=QuadMeshingOptions(quality_model="shape_jacobian"),
    ).mesh


_PREPARATION_HASH = re.compile(r'"preparation_hash": "sha256:[0-9a-f]+"')


def _canonical_prepared(mesh):
    """_canonical without the structural preparation hash (salted by the random model id)."""
    document, diagnostics = _canonical(mesh)
    return _PREPARATION_HASH.sub('"preparation_hash": "<salted>"', document), diagnostics


def _assert_twins(actual, expected):
    assert actual.hybrid_diagnostics["route"] == "quad-first-cylindrical"
    assert _canonical_prepared(actual) == _canonical_prepared(expected)
    for node, point in expected.nodes.items():
        assert np.asarray(actual.nodes[node], dtype=np.float64).tobytes() == \
            np.asarray(point, dtype=np.float64).tobytes()


def _element_normal_signs(model, face, mesh):
    """Sign of (element normal . the face's own normal) per element."""
    signs = set()
    for element in mesh.elements_of_face[face]:
        corners = [np.asarray(mesh.nodes[node], dtype=float) for node in mesh.corners_of(element)]
        normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        u, v = model.face_local_uv(face, np.mean(corners, axis=0))
        signs.add(float(np.sign(normal @ model.face_normal(face, u, v))))
    return signs


@pytest.mark.parametrize("rotate", (1, 0))
def test_point_attachment_seed_matches_the_twin_with_the_same_physical_anchor(rotate):
    """Finding P2: the point-seed lattice and axis lengths come from the support frame.

    The attachment keeps its parameters in the face's own frame (owner
    contract). The twin declares the same physical point in its own frame,
    which is the support's. With ``rotate=1`` the two frames are turned by
    90 degrees against each other.
    """
    target = (.37, .41)
    probe, face = _rect_patch(True, rotate=rotate)
    anchor = probe.face_point(face, *target)
    support_uv = tuple(float(value) for value in probe.faces[face].surface.local_uv(anchor))
    if rotate:
        # Teeth: the declared parameters differ strongly between the frames.
        assert max(abs(p - q) for p, q in zip(target, support_uv)) > .2
    twin, same = _rect_patch(False, rotate=rotate, attach=(support_uv, anchor))
    model, face = _rect_patch(True, rotate=rotate, attach=(target, anchor))
    assert same == face
    expected = _shape_jacobian(twin, same)
    actual = _shape_jacobian(model, face)
    _assert_twins(actual, expected)
    # The published anchor node sits on the declared physical point.
    distances = [float(np.linalg.norm(np.asarray(point) - anchor)) for point in actual.nodes.values()]
    assert min(distances) <= 1e-10


def test_reversed_face_parameterization_is_refused_and_its_twin_still_meshes():
    """Finding orientation: a reversed loop reverses the topology Coons normal."""
    twin, face = _rect_patch(False, reverse=True)
    model, same = _rect_patch(True, reverse=True)
    # Teeth: the face's own normal opposes the support normal.
    centre = model.face_point(same, .5, .5)
    assert float(model.face_normal(same, .5, .5) @ twin.face_normal(face, *twin.face_local_uv(face, centre))) < 0.
    expected = _generate(twin, face)
    assert _element_normal_signs(twin, face, expected) == {1.}
    with pytest.raises(MeshError, match="orientation is reversed relative to its support"):
        _generate(model, same)
    with pytest.raises(MeshError, match="orientation is reversed relative to its support"):
        CylindricalMetricChart.from_geometry(model, same, support_frame_parameterization=True)


def test_admitted_parameterized_meshes_follow_the_face_normal():
    for rotate in (0, 1):
        model, face = _rect_patch(True, rotate=rotate)
        mesh = _generate(model, face)
        assert _element_normal_signs(model, face, mesh) == {1.}


def test_non_topology_parameterization_is_refused_as_undecided():
    model, face = _rect_patch(False)
    support = model.faces[face].surface
    # A distinct (equal) surface object: orientation is not decided by the loop.
    model.set_face_parameterization(face, Cylinder(support.origin, support.axis, support.radial_direction,
                                                   support.radius, support.height, support.start_angle,
                                                   support.sweep_angle))
    with pytest.raises(MeshError, match="admits only a topology Coons"):
        CylindricalMetricChart.from_geometry(model, face, support_frame_parameterization=True)


@pytest.mark.parametrize("support_start", (-1., -.15), ids=("interior", "seam-at-u0"))
def test_parameterized_patch_on_a_full_turn_support_matches_its_twin(support_start):
    kw = dict(a=-.15, sweep=.3, support_start=support_start, support_sweep=2 * math.pi)
    twin, face = _rect_patch(False, **kw)
    model, same = _rect_patch(True, **kw)
    _assert_twins(_generate(model, same), _generate(twin, face))


@pytest.mark.parametrize("support_start", (0., .15, .15 - 2 * math.pi),
                         ids=("straddles-seam", "seam-at-u1", "seam-at-u1-wrapped"))
def test_seam_crossing_patch_is_refused_alike_with_and_without_parameterization(support_start):
    kw = dict(a=-.15, sweep=.3, support_start=support_start, support_sweep=2 * math.pi)
    messages = []
    for parameterized in (False, True):
        with pytest.raises(Exception) as caught:
            _generate(*_rect_patch(parameterized, **kw))
        messages.append((type(caught.value).__name__, str(caught.value)))
    # Measured: the owner refuses the face itself ("loop 0 self-intersects").
    assert messages[0] == messages[1]


def test_reversed_mitred_patch_is_refused():
    """The 646da87 mitred model (z1=3): crossed mitres, so a reversed parameterization."""
    twin, face = _mitred_patch(False, z1=3.)
    model, same = _mitred_patch(True, z1=3.)
    centre = model.face_point(same, .5, .5)
    assert float(model.face_normal(same, .5, .5) @ twin.face_normal(face, *twin.face_local_uv(face, centre))) < 0.
    assert _element_normal_signs(twin, face, _generate(twin, face)) == {1.}
    with pytest.raises(MeshError, match="orientation is reversed relative to its support"):
        _generate(model, same)


@pytest.mark.parametrize("order", ("linear", "quadratic"))
def test_parameterized_patch_with_a_hole_matches_its_twin(order):
    kw = dict(sweep=.4, hole=((.42, .56), (1.1, 1.9)))
    twin, face = _rect_patch(False, **kw)
    model, same = _rect_patch(True, **kw)
    assert len(model.faces[same].holes) == 1
    _assert_twins(_generate(model, same, order), _generate(twin, face, order))


# --- MESH-B fix round 2: folded or off-support maps inside a positive loop ---

_FAMILY = "admits only a ruled cylinder strip"
_CERTIFICATE = "cannot certify that the face parameterization keeps the support orientation"


def _l_shaped(parameterized: bool):
    """Review R2 repro: a concave L-shaped trim with four declared Coons corners.

    Its loop winds positively and the Coons normal at (0.5, 0.5) agrees with
    the support, but the Coons map folds (two elements opposed the face
    normal on 8e14b93) and leaves the support by up to 6.3 mm.
    """
    model = GeometryModel()
    R, a, sweep, height = 6.5, .3, .35, 3.
    uv = [(0, 0), (1, 0), (1, .4), (.4, .4), (.4, 1), (0, 1)]
    xyz = [(R * math.cos(a + sweep * u), R * math.sin(a + sweep * u), height * v) for u, v in uv]
    vertices = model.add_points(xyz)
    edges = []
    for i, (u, v) in enumerate(uv):
        w, _ = uv[(i + 1) % len(uv)]
        j = (i + 1) % len(uv)
        edges.append(model.add_line(vertices[i], vertices[j]) if u == w else model.add_curve(
            vertices[i], vertices[j],
            EllipticArc((0, 0, height * v), (R, 0, 0), (0, R, 0), a + sweep * u, sweep * (w - u))))
    face = model.add_face(edges, corners=[0, 1, 2, 5],
                          surface=Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0), R, height, a, sweep))
    model.set_face_parameterization(face, CoonsSurface() if parameterized else None)
    model.add_sheet((face,))
    return model, face


def _strip(parameterized: bool = True, *, R=6.5, a=.3, sweep=.6, z0=0., k0=0., z1=3., k1=0.,
           top_offset=(0., 0., 0.), split_left=False):
    """A ruled strip: mitred EllipticArc rails z = z_i + k_i R cos(theta), Straight generators.

    ``top_offset`` moves the top rail's centre (off the support); its vertices
    stay at the rail's own ends. ``split_left`` splits the left generator into
    two collinear Straight edges (a five-edge loop with four corners).
    """
    model = GeometryModel()
    bottom_arc = EllipticArc((0., 0., z0), (R, 0., k0 * R), (0., R, 0.), a, sweep)
    top_arc = EllipticArc(tuple(np.add((0., 0., z1), top_offset)), (R, 0., k1 * R), (0., R, 0.), a, sweep)
    b, t = bottom_arc.evaluate(np.array((0., 1.))), top_arc.evaluate(np.array((0., 1.)))
    v = model.add_points((tuple(b[0]), tuple(b[1]), tuple(t[1]), tuple(t[0])))
    edges = [model.add_curve(v[0], v[1], bottom_arc), model.add_line(v[1], v[2]),
             model.add_curve(v[3], v[2], top_arc)]
    corners = None
    if split_left:
        (middle,) = model.add_points((tuple(.5 * (t[0] + b[0])),))
        edges += [model.add_line(middle, v[3]), model.add_line(v[0], middle)]
        corners = [0, 1, 2, 3]
    else:
        edges.append(model.add_line(v[0], v[3]))
    support = Cylinder((0., 0., -10.), (0., 0., 1.), (1., 0., 0.), R, 30., a - .25)
    face = model.add_face(tuple(edges), corners=corners, surface=support)
    model.set_face_parameterization(face, CoonsSurface() if parameterized else None)
    model.add_sheet((face,))
    return model, face


def _support_chart(model, face):
    return CylindricalMetricChart.from_geometry(model, face, support_frame_parameterization=True)


def _radial_dot(model, face, u, v):
    """face normal . the radial direction at the face's own point (support normal, R > 0)."""
    point = model.face_point(face, u, v)
    return float(model.face_normal(face, u, v)[:2] @ point[:2])


def test_l_shaped_trim_with_four_declared_corners_is_refused():
    """Finding R2 (P2): positive winding plus a centre normal admitted a folded Coons map."""
    model, face = _l_shaped(True)
    grid = [(u, v) for u in np.linspace(.01, .99, 30) for v in np.linspace(.01, .99, 30)]
    # Teeth: the Coons map leaves the support by millimetres and its normal turns inward somewhere.
    assert max(abs(float(np.hypot(*model.face_point(face, u, v)[:2])) - 6.5) for u, v in grid) > 5e-3
    assert min(_radial_dot(model, face, u, v) for u, v in grid) < 0.
    assert _radial_dot(model, face, .5, .5) > 0.
    with pytest.raises(MeshError, match=_FAMILY):
        _support_chart(model, face)
    with pytest.raises(MeshError, match=_FAMILY):
        _generate(model, face)
    twin, same = _l_shaped(False)
    assert _element_normal_signs(twin, same, _generate(twin, same)) == {1.}


class _RailSwap:
    """The owner queries _certify_ruled_strip reads, with the top rail's curve replaced.

    The owner refuses a crossed or off-support loop when the face is made, so
    the certificate's own refusal (defence in depth) is exercised on the valid
    face's topology with a substituted rail; its vertices move to the new
    rail's ends, so the corner residual stays at rounding level.
    """

    def __init__(self, model, face, curve):
        from dataclasses import replace
        loop = model.faces[face].loop
        top = model.edges[loop[2].edge]
        self.edges = dict(model.edges)
        self.edges[top.id] = replace(top, curve=curve)
        ends = curve.evaluate(np.array((0., 1.)))
        self._moved = {top.start: ends[0], top.end: ends[1]}
        self._model = model
        self.tolerance = model.tolerance

    def vertex_position(self, vertex):
        return self._moved.get(vertex, self._model.vertex_position(vertex))


def _certify_with_top_rail(curve, **kw):
    from anymesher._cylindrical_support_frame import _certify_ruled_strip
    model, face = _strip(**kw)
    record = model.faces[face]
    return _certify_ruled_strip(_RailSwap(model, face, curve), record, record.surface)


_FOLD_K = 10. / 6.5  # top rail z = z1 + 10 cos(theta) on theta in [0.3, 0.9]


@pytest.mark.parametrize("dip", (-2e-3, -.5), ids=("2mm", "500mm"))
def test_folded_strip_is_refused_by_the_certificate(dip):
    """The top rail crosses below the bottom rail near u=1: h(u) changes sign.

    The owner already refuses that loop (it self-intersects); the certificate
    refuses the same rail on its own, while its vertex polygon still winds
    positively and h at the centre is positive.
    """
    z1 = -10. * math.cos(.9) + dip
    with pytest.raises(GeometryError, match="self-intersects"):
        _strip(z1=z1, k1=_FOLD_K)
    folded = EllipticArc((0., 0., z1), (6.5, 0., _FOLD_K * 6.5), (0., 6.5, 0.), .3, .6)
    ends = folded.evaluate(np.array((0., .5, 1.)))[:, 2]
    assert ends[0] > 2. and ends[1] > 1. and ends[2] < 0.
    with pytest.raises(MeshError, match=_CERTIFICATE):
        _certify_with_top_rail(folded, z1=1., k1=_FOLD_K)


@pytest.mark.parametrize("gap", (2e-3, 1.), ids=("nearly-folded-2mm", "plain"))
def test_unfolded_strip_is_certified(gap):
    model, face = _strip(z1=-10. * math.cos(.9) + gap, k1=_FOLD_K)
    certificate = _support_chart(model, face).support_view.orientation_certificate
    assert .9 * gap < certificate["h_min"] <= gap
    assert certificate["support_deviation"] <= 1e-11
    assert certificate["normal_lower"] >= .99 * certificate["normal_ideal"] > 0.


@pytest.mark.parametrize("offset", (2e-8, 1e-6, 1e-3))
def test_off_support_rail_is_refused_by_the_certificate(offset):
    """A rail whose centre is off the axis: the map leaves its support by about ``offset``."""
    if offset >= 1e-6:
        with pytest.raises(GeometryError, match="inconsistent with its explicit surface"):
            _strip(top_offset=(offset, 0., 0.))
    shifted = EllipticArc((offset, 0., 3.), (6.5, 0., 0.), (0., 6.5, 0.), .3, .6)
    with pytest.raises(MeshError, match=_CERTIFICATE):
        _certify_with_top_rail(shifted)


def test_on_support_rail_within_the_surface_residual_is_admitted():
    model, face = _strip(top_offset=(1e-12, 0., 0.))
    certificate = _support_chart(model, face).support_view.orientation_certificate
    assert certificate["support_deviation"] <= model.tolerance.surface_residual


def test_split_generator_side_is_outside_the_certified_family():
    """Collinear split sides are not in the proven family (narrower than 8e14b93): refused."""
    model, face = _strip(split_left=True)
    assert len(model.faces[face].loop) == 5
    with pytest.raises(MeshError, match=_FAMILY):
        _support_chart(model, face)


def test_concave_strip_in_the_family_matches_its_twin_and_follows_the_face_normal():
    """The top rail dips in the middle (concave in the support frame) but never folds."""
    kw = dict(a=-.3, sweep=.6, z1=3., k1=-.4)
    twin, face = _strip(False, **kw)
    model, same = _strip(True, **kw)
    top = [float(model.face_point(same, u, 1.)[2]) for u in (0., .5, 1.)]
    assert top[1] < min(top[0], top[2]) - .1
    assert _support_chart(model, same).support_view.orientation_certificate["h_min"] > .3
    actual = _generate(model, same)
    _assert_twins(actual, _generate(twin, face))
    assert _element_normal_signs(model, same, actual) == {1.}


@pytest.mark.parametrize("rotate", (0, 1))
def test_existing_admitted_models_carry_a_tight_certificate(rotate):
    for model, face in (_rect_patch(True, rotate=rotate), _mitred_patch(True),
                        _rect_patch(True, rotate=rotate, sweep=.4, hole=((.42, .56), (1.1, 1.9)))):
        certificate = _support_chart(model, face).support_view.orientation_certificate
        assert certificate["support_deviation"] <= 1e-11
        assert certificate["normal_lower"] >= .99 * certificate["normal_ideal"] > 0.
