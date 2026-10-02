"""Small owner-candidate integration checks for exact extrusions and branch edges.

These tests run when the installed ANYgeometry exposes its schema-6 surface.
They do not change the legacy package's supported geometry floor.
"""

from __future__ import annotations

import pytest

from anygeometry import (
    GeometryModel, apply_intersections, from_dict, plan_intersections, to_dict,
)

try:
    from anygeometry import ConnectionIntent, BezierQuadricCurve, ExtrudedSurface
except ImportError:
    pytest.skip("schema-6 ANYgeometry candidate is unavailable", allow_module_level=True)

from anygeometry.generators import cylinder

from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.preparation import prepare_structural_closure
from anymesher.quad.options import QuadMeshingOptions
from anymesher.serialize import mesh_from_dict, mesh_to_dict


def _mesh(model: GeometryModel, face_ids: tuple[int, ...], order: str):
    return generate_hybrid_mesh_result(
        model,
        face_ids=face_ids,
        beam_edges=(),
        member_ids=(),
        target_size=0.75,
        strategy="native",
        native_backend="python",
        recombine=True,
        order=order,
        quad_options=QuadMeshingOptions(),
    ).mesh


def _extruded_faces(model: GeometryModel) -> tuple[int, ...]:
    return tuple(sorted(
        face_id for face_id, face in model.faces.items()
        if isinstance(face.surface, ExtrudedSurface)
    ))


def test_split_extrusion_keeps_shared_ids_and_quadratic_owner_mapping():
    model = GeometryModel()
    plate = model.add_plate(model.add_points((
        (-1., -2., 1.), (4., -2., 1.), (4., 3., 1.), (-1., 3., 1.),
    )))
    controls = model.add_points((
        (0., 0., 0.), (1., .8, 0.), (2., -.4, 0.), (3., 0., 0.),
    ))
    wall = model.extrude([
        model.add_spline(controls[0], tuple(controls[1:-1]), controls[-1])
    ], (0., 0., 2.))[0]
    model.add_sheet((plate,))
    wall_sheet = model.add_sheet((wall,))
    source_document = to_dict(model)
    prepared, report = prepare_structural_closure(
        model, face_ids=(plate, wall), beam_edges=(),
    )
    assert report is not None
    assert to_dict(model) == source_document
    model = prepared
    faces = _extruded_faces(model)
    assert len(faces) == 2
    document = to_dict(model)
    assert document["version"] == 6
    assert to_dict(from_dict(document)) == document

    linear = _mesh(model, faces, "linear")
    quadratic = _mesh(model, faces, "quadratic")
    assert to_dict(model) == document
    assert linear.quads and linear.tris
    assert quadratic.quads and quadratic.tris
    assert {eid: body[:4] for eid, body in quadratic.quads.items()} == linear.quads
    assert {eid: body[:3] for eid, body in quadratic.tris.items()} == linear.tris
    certificate = quadratic.hybrid_diagnostics["high_order_geometry"]
    assert certificate["status"] == "CERTIFIED_POSITIVE"
    assert {report["geometry_family"] for report in certificate["reports"]} == {"extruded"}
    assert set(quadratic.elements_of_sheet[wall_sheet]) == {
        element_id for face_id in faces
        for element_id in quadratic.elements_of_face[face_id]
    }
    shared = {
        use.edge for use in model.faces[faces[0]].loop
    } & {use.edge for use in model.faces[faces[1]].loop}
    assert shared
    for edge_id in shared:
        chain = quadratic.nodes_of_edge[edge_id]
        assert len(chain) >= 3
        assert chain[::2] == linear.nodes_of_edge[edge_id]
        for face_id in faces:
            owned = set(quadratic.elements_of_face[face_id])
            incident = {
                node for eid in owned
                for node in quadratic.quads.get(eid, quadratic.tris.get(eid, ()))
            }
            assert set(chain) <= incident
    assert mesh_to_dict(mesh_from_dict(mesh_to_dict(quadratic))) == mesh_to_dict(quadratic)


def test_bezier_quadric_joint_edge_is_sampled_from_its_owner():
    model = GeometryModel()
    controls = model.add_points((
        (0., 0., 0.), (1., 2., 0.), (2., -1., 0.), (3., 1., 0.),
    ))
    edge = model.add_spline(controls[0], tuple(controls[1:-1]), controls[-1])
    wall = model.extrude([edge], (.25, 0., 1.5))[0]
    before = set(model.faces)
    model.insert_model(cylinder(
        .7, 8., origin=(-2., .4, .6), axis=(1., 0., 0.),
        radial_direction=(0., 1., 0.), circumferential_segments=8,
    ))
    faces = (wall, *sorted(set(model.faces) - before))
    plan = plan_intersections(
        model, [model.handle("face", fid) for fid in faces],
        policy=ConnectionIntent.CONNECT,
    )
    apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)
    branch_edges = {
        edge_id for edge_id, item in model.edges.items()
        if isinstance(item.curve, BezierQuadricCurve)
    }
    assert branch_edges
    candidates = [
        face_id for face_id in _extruded_faces(model)
        if len(model.faces[face_id].loop) == 3
        and any(use.edge in branch_edges for use in model.faces[face_id].loop)
    ]
    assert candidates
    document = to_dict(model)
    assert document["version"] == 6
    mesh = _mesh(model, (candidates[0],), "quadratic")
    assert to_dict(model) == document
    certificate = mesh.hybrid_diagnostics["high_order_geometry"]
    assert certificate["status"] == "CERTIFIED_POSITIVE"
    assert certificate["reports"][0]["geometry_family"] == "extruded"
    assert mesh.quads and mesh.tris
    for use in model.faces[candidates[0]].loop:
        if use.edge in branch_edges:
            assert len(mesh.nodes_of_edge[use.edge]) >= 3
