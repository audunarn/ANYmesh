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

import numpy as np
import pytest

from anygeometry import EllipticArc, GeometryModel
from anygeometry.cylinder_patch import CylinderPatchError
from anygeometry.surfaces import CoonsSurface, Cylinder, Plane

from anymesher._cylindrical_chart import CylindricalMetricChart
from anymesher._cylindrical_public import prepare_bindings
from anymesher.errors import MeshError
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.native_v2 import NativeMeshingOptions
from anymesher.quad.options import QuadMeshingOptions
from anymesher.serialize import mesh_to_dict


_DISTINCT = "distinct face parameterization"


def _mitred_patch(parameterized: bool, *, R=6.5, a=.3, sweep=math.radians(6.), z0=0., z1=3.,
                  k0=math.tan(math.pi / 8), k1=-.3):
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
