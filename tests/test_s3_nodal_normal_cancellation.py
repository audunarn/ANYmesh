"""Qualified-S3 nodal normals where oriented owner normals cancel at a node.

Reproduces the floating-tripod hub: two coplanar plates whose sheets are
oriented oppositely (mirror-image webs) meet at a node that no other plate
uses.  The owner-aligned element normals cancel exactly, so the signed
average has no direction although the plane normal is well defined.
"""
from __future__ import annotations

import numpy as np
import pytest

from anygeometry import GeometryModel
from anygeometry.structural import Orientation

from anymesher import Mesh, S3OwnerAuthorityError, prepare_qualified_s3_mesh


def _two_plates(points, quad, tri, orientations):
    geometry = GeometryModel()
    ids = geometry.add_points(points)
    quad_ids = tuple(ids[i] for i in quad)
    tri_ids = tuple(ids[i] for i in tri)
    quad_face = geometry.add_plate(quad_ids)
    tri_face = geometry.add_plate(tri_ids)
    geometry.add_sheet((quad_face,), orientations=(orientations[0],))
    geometry.add_sheet((tri_face,), orientations=(orientations[1],))
    mesh = Mesh(
        geometry_model_id=geometry.model_id,
        geometry_revision=geometry.revision,
        nodes={i: np.asarray(geometry.vertex_position(i), dtype=float) for i in ids},
        quads={10: quad_ids},
        tris={20: tri_ids},
        elements_of_face={quad_face: [10], tri_face: [20]},
    )
    return geometry, mesh, ids


_COPLANAR = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
             (2.0, 1.0, 0.0), (2.0, 2.0, 0.0))


def test_opposite_coplanar_owners_meeting_at_one_node_get_the_plane_director() -> None:
    geometry, mesh, ids = _two_plates(
        _COPLANAR, (0, 1, 2, 3), (2, 4, 5), (Orientation.FORWARD, Orientation.REVERSED))

    _prepared, record = prepare_qualified_s3_mesh(mesh, geometry)

    normals = {int(k): np.asarray(v) for k, v in record["nodal_normals"].items()}
    assert set(normals) == set(ids)
    # The shared node takes the plane normal, signed by the lowest element
    # (quad 10, owner +z); each plate's own nodes keep their owner side.
    np.testing.assert_array_equal(normals[ids[2]], (0.0, 0.0, 1.0))
    for node in (ids[0], ids[1], ids[3]):
        np.testing.assert_allclose(normals[node], (0.0, 0.0, 1.0))
    for node in (ids[4], ids[5]):
        np.testing.assert_allclose(normals[node], (0.0, 0.0, -1.0))


def test_tilted_opposite_coplanar_owners_get_the_plane_director() -> None:
    # A rotated plane, so the cancellation is no longer exact in floating point
    # and the signed sum would otherwise leave a rounding-noise direction.
    c, s = np.cos(0.3), np.sin(0.3)
    rot = np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))) @ np.array(
        ((np.cos(0.7), 0.0, np.sin(0.7)), (0.0, 1.0, 0.0), (-np.sin(0.7), 0.0, np.cos(0.7))))
    points = tuple(tuple(float(x) for x in rot @ np.asarray(p) * 3.1) for p in _COPLANAR)
    geometry, mesh, ids = _two_plates(
        points, (0, 1, 2, 3), (2, 4, 5), (Orientation.FORWARD, Orientation.REVERSED))

    _prepared, record = prepare_qualified_s3_mesh(mesh, geometry)

    plane = rot @ np.array((0.0, 0.0, 1.0))
    shared = np.asarray(record["nodal_normals"][str(ids[2])])
    assert abs(float(np.linalg.norm(shared)) - 1.0) <= 1.0e-12
    assert float(np.dot(shared, plane)) >= 1.0 - 1.0e-9


def _y_junction():
    # Three plates meeting on one line at 120 degrees, owner normals
    # symmetric about it: they cancel without sharing a plane (the tripod hub
    # axis, where the three arm-plane webs meet).
    geometry = GeometryModel()
    pts = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    for k in range(3):
        a = 2.0 * np.pi * k / 3.0
        d = (float(np.cos(a)), float(np.sin(a)), 0.0)
        pts += [(d[0], d[1], 1.0), (d[0], d[1], 0.0)]
    ids = geometry.add_points(pts)
    faces, quads = [], {}
    for k in range(3):
        loop = (ids[0], ids[2 + 2 * k + 1], ids[2 + 2 * k], ids[1])
        face = geometry.add_plate(loop)
        geometry.add_sheet((face,))
        faces.append(face)
        quads[10 + k] = loop
    mesh = Mesh(
        geometry_model_id=geometry.model_id,
        geometry_revision=geometry.revision,
        nodes={i: np.asarray(geometry.vertex_position(i), dtype=float) for i in ids},
        quads=quads,
        tris={},
        elements_of_face={f: [10 + k] for k, f in enumerate(faces)},
    )
    owners = {10 + k: tuple(np.cross(np.subtract(pts[3 + 2 * k], pts[0]),
                                     np.subtract(pts[1], pts[0])))
              for k in range(3)}
    return mesh, ids, owners


def test_non_parallel_cancelling_owners_take_the_lowest_element_normal() -> None:
    # Owner decision D1 (2026-11-21): no director exists at such a junction;
    # the record takes the owner-side normal of the lowest-numbered element.
    from anymesher.mapped import oriented_shell_normal_contributions
    from anymesher.s3_production import _qualified_nodal_normals

    mesh, ids, owners = _y_junction()
    normals = _qualified_nodal_normals(mesh, owners)

    lowest = {e: n for e, _nodes, n in oriented_shell_normal_contributions(
        mesh, element_owner_normals=owners, include_triangles=True)}[10]
    for node in (ids[0], ids[1]):
        np.testing.assert_array_equal(normals[node], lowest)
    assert set(normals) == set(ids)


def test_a_node_without_any_normal_term_stays_a_typed_refusal() -> None:
    mesh, ids, owners = _y_junction()
    # Collapse element 12 so it has no normal; its two outer nodes then have
    # no term at all.
    mesh.nodes[ids[6]] = mesh.nodes[ids[0]].copy()
    mesh.nodes[ids[7]] = mesh.nodes[ids[1]].copy()
    from anymesher.s3_production import _qualified_nodal_normals

    with pytest.raises(S3OwnerAuthorityError, match="ambiguous"):
        _qualified_nodal_normals(mesh, owners)
