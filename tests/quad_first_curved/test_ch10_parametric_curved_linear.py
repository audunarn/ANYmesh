"""CH10 metric-aware explicit quad-first contract for ruled/Coons faces."""
from __future__ import annotations

import numpy as np
import pytest

from test_cylindrical_frontal_integration import _persistent_state
from test_curved_native_qualification import _model_face

from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.domain import ParametricQuadDomain
from anymesher.quad.options import QuadMeshingOptions


def _generate(model, face, h, *, order="linear", cancellation_check=None):
    return generate_hybrid_mesh_result(
        model, face_ids=(face,), target_size=float(h), strategy="native",
        native_backend="python", recombine=True, order=order,
        quad_options=QuadMeshingOptions(), cancellation_check=cancellation_check,
    ).mesh


def _n_eq(mesh):
    return float(len(mesh.quads)) + 0.5 * float(len(mesh.tris))


def _max_edge(mesh):
    return max(float(np.linalg.norm(np.asarray(mesh.nodes[b]) - np.asarray(mesh.nodes[a])))
               for eid in mesh.shells for a, b in zip(mesh.corners_of(eid), mesh.corners_of(eid)[1:] + mesh.corners_of(eid)[:1]))


def _support_residual(mesh, surface):
    return max(float(np.linalg.norm(np.asarray(surface.evaluate(*surface.local_uv(tuple(p)))) - np.asarray(p)))
               for p in mesh.nodes.values())


@pytest.mark.parametrize("name", ("ruled", "coons"))
def test_ch10_parametric_domain_roundtrip_and_source_immutability(name):
    model, face, surface = _model_face(name)
    before = _persistent_state(model)
    domain = ParametricQuadDomain.from_geometry(model, face)
    assert domain.model_id == str(model.model_id)
    assert domain.revision == int(model.revision)
    assert domain.face_id == int(face)
    assert domain.area > 0.0
    for uv in ((0.0, 0.0), (0.25, 0.35), (0.75, 0.65), (1.0, 1.0)):
        world = tuple(np.asarray(surface.evaluate(*uv), dtype=float))
        np.testing.assert_allclose(domain.lift(domain.project(world)), world, atol=1.0e-9, rtol=0.0)
    assert _persistent_state(model) == before


@pytest.mark.parametrize("name", ("ruled", "coons"))
def test_ch10_public_linear_is_physically_target_size_causal(name):
    coarse_model, coarse_face, coarse_surface = _model_face(name)
    coarse = _generate(coarse_model, coarse_face, 0.6)
    fine_model, fine_face, fine_surface = _model_face(name)
    fine = _generate(fine_model, fine_face, 0.3)
    assert _n_eq(fine) > _n_eq(coarse) > 0.0
    assert _max_edge(fine) < _max_edge(coarse)
    assert _support_residual(coarse, coarse_surface) <= 1.0e-10
    assert _support_residual(fine, fine_surface) <= 1.0e-10
    for mesh, face in ((coarse, coarse_face), (fine, fine_face)):
        assert set(mesh.elements_of_face[face]) == set(mesh.quads) | set(mesh.tris)
        assert mesh.hybrid_diagnostics["route"] == "quad-first-parametric-curved"
        assert mesh.hybrid_diagnostics["geometry_family_by_face"][face] == name
        assert mesh.hybrid_diagnostics["front"]["faces"][face]["attempts"] > 0
        assert all(len(mesh.nodes_of_edge[int(use.edge)]) >= 2 for use in (coarse_model if mesh is coarse else fine_model).faces[face].loop)


@pytest.mark.parametrize("name", ("ruled", "coons"))
def test_ch10_linear_repeat_is_deterministic_and_source_unchanged(name):
    model, face, _ = _model_face(name)
    before = _persistent_state(model)
    first = _generate(model, face, 0.6)
    repeat_model, repeat_face, _ = _model_face(name)
    repeat = _generate(repeat_model, repeat_face, 0.6)
    assert first.quads == repeat.quads
    assert first.tris == repeat.tris
    assert first.nodes_of_edge == repeat.nodes_of_edge
    assert _persistent_state(model) == before


@pytest.mark.parametrize("name", ("ruled", "coons"))
def test_ch10_cancellation_is_atomic(name):
    model, face, _ = _model_face(name)
    before = _persistent_state(model)

    class Cancelled(RuntimeError):
        pass

    def cancel(phase):
        if phase == "quad-first:face-seed":
            raise Cancelled(phase)

    with pytest.raises(Cancelled):
        _generate(model, face, 0.6, cancellation_check=cancel)
    assert _persistent_state(model) == before
