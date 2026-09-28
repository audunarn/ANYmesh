from __future__ import annotations

import numpy as np
import pytest
import anygeometry

from test_cylindrical_atlas_binding import _sector_model
from test_curved_native_qualification import _model_face

from anymesher.quad.public_integration import QuadPublicUnsupported
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.options import QuadMeshingOptions


def _run(model, faces, size, order):
    return generate_hybrid_mesh_result(
        model, face_ids=faces, target_size=size, strategy="native",
        native_backend="python", order=order, layout_policy="adaptive",
        quad_options=QuadMeshingOptions(quality_model="shape_jacobian"),
    ).mesh


@pytest.mark.parametrize("family", ("cone", "ruled", "coons"))
def test_adaptive_owner_chart_preserves_quadratic_corners(family):
    model, face, surface = _model_face(family)
    before = (str(model.model_id), int(model.revision))
    linear = _run(model, (face,), .6, "linear")
    quadratic = _run(model, (face,), .6, "quadratic")
    assert (str(model.model_id), int(model.revision)) == before
    assert linear.quads or linear.tris
    assert {eid: tuple(body[:4]) for eid, body in quadratic.quads.items()} == linear.quads
    assert {eid: tuple(body[:3]) for eid, body in quadratic.tris.items()} == linear.tris
    for node in linear.nodes:
        np.testing.assert_array_equal(quadratic.nodes[node], linear.nodes[node])
    assert quadratic.hybrid_diagnostics["high_order_geometry"]["status"] == "CERTIFIED_POSITIVE"


def test_adaptive_cylinder_retains_reversed_shared_edge_nodes():
    model, uses = _sector_model(False)
    faces = tuple(int(model.face_uses[use.id].face_id) for use in uses[:2])
    if not hasattr(anygeometry, "query_cylinder_open_component"):
        # The frozen 0.5.0 dependency pin predates the owner API for an open
        # two-face cylindrical component. Verify that the optional route is
        # rejected explicitly while package import and supported routes remain
        # usable; never approximate the missing owner qualification in ANYmesh.
        with pytest.raises(QuadPublicUnsupported, match="owner-qualified"):
            _run(model, faces, .6, "linear")
        return
    linear = _run(model, faces, .6, "linear")
    quadratic = _run(model, faces, .6, "quadratic")
    common = {use.edge for use in model.faces[faces[0]].loop} & {
        use.edge for use in model.faces[faces[1]].loop
    }
    assert len(common) == 1
    edge = next(iter(common))
    assert tuple(quadratic.nodes_of_edge[edge][::2]) == tuple(linear.nodes_of_edge[edge])
    assert {eid: tuple(body[:4]) for eid, body in quadratic.quads.items()} == linear.quads
    assert quadratic.hybrid_diagnostics["high_order_geometry"]["status"] == "CERTIFIED_POSITIVE"
