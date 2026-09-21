from __future__ import annotations

import os
import sys

_REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

from anymesher.hybrid import generate_hybrid_mesh_result  # noqa: E402
from anymesher.quad.options import QuadMeshingOptions  # noqa: E402
from .fixtures import p01_geometry  # noqa: E402


def _corner_vertex_ids(geometry, face_id: int) -> tuple[int, ...]:
    face = geometry.faces[face_id]
    out = []
    for corner_index in face.corners:
        use = face.loop[int(corner_index)]
        edge = geometry.edges[int(use.edge)]
        out.append(int(edge.start if use.forward else edge.end))
    return tuple(out)


def test_pq0_current_public_route_is_one_q4_and_size_nonresponsive() -> None:
    """Freeze baseline limitation; PQ3 intentionally replaces this expectation."""
    geometry, face = p01_geometry()
    before_revision = geometry.revision
    before_face_ids = tuple(sorted(geometry.faces))
    before_edge_ids = tuple(sorted(geometry.edges))
    before_vertex_ids = tuple(sorted(geometry.vertices))
    before_loop = tuple(
        (int(use.edge), bool(use.forward)) for use in geometry.faces[face].loop
    )
    before_corners = tuple(int(i) for i in geometry.faces[face].corners)
    expected_corner_vertices = set(_corner_vertex_ids(geometry, face))

    coarse = generate_hybrid_mesh_result(
        geometry,
        target_size=1.0,
        face_ids=(face,),
        quad_options=QuadMeshingOptions(),
    )
    fine = generate_hybrid_mesh_result(
        geometry,
        target_size=0.5,
        face_ids=(face,),
        quad_options=QuadMeshingOptions(),
    )

    for result in (coarse, fine):
        mesh = result.mesh
        assert len(mesh.quads) == 1
        assert len(mesh.tris) == 0
        assert len(mesh.nodes) == 4
        assert mesh.elements_of_face[face] == list(mesh.quads)
        assert set(mesh.node_of_vertex) == expected_corner_vertices

    coarse_body = tuple(int(n) for n in next(iter(coarse.mesh.quads.values())))
    fine_body = tuple(int(n) for n in next(iter(fine.mesh.quads.values())))
    assert coarse_body == fine_body
    assert geometry.revision == before_revision
    assert tuple(sorted(geometry.faces)) == before_face_ids
    assert tuple(sorted(geometry.edges)) == before_edge_ids
    assert tuple(sorted(geometry.vertices)) == before_vertex_ids
    assert (
        tuple((int(use.edge), bool(use.forward)) for use in geometry.faces[face].loop)
        == before_loop
    )
    assert tuple(int(i) for i in geometry.faces[face].corners) == before_corners
