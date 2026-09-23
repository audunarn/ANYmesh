"""The final user callback cannot invalidate a returned mesh binding."""

import pytest
from anygeometry import GeometryModel

from anymesher import hybrid
from anymesher.errors import MeshError
from test_intersection_meshing import _plate


def _model():
    model = GeometryModel()
    _plate(model, ((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)))
    return model


@pytest.mark.parametrize("policy", (
    hybrid.GeometryMutationPolicy.READ_ONLY,
    hybrid.GeometryMutationPolicy.WORKING_COPY,
))
def test_committed_final_callback_mutation_cannot_publish(policy):
    model = _model()
    revision = model.revision

    def mutate(stage):
        if stage == "hybrid generation complete":
            model.add_point(5., 5., 5.)

    with pytest.raises(MeshError, match="stale"):
        hybrid.generate_hybrid_mesh_result(
            model, target_size=.25, strategy="mapped",
            mutation_policy=policy, cancellation_check=mutate,
        )
    # The caller's edit is preserved; only mesh publication is refused.
    assert model.revision > revision


@pytest.mark.parametrize("policy", (
    hybrid.GeometryMutationPolicy.READ_ONLY,
    hybrid.GeometryMutationPolicy.WORKING_COPY,
))
def test_open_final_callback_transaction_cannot_publish(policy):
    model = _model()
    opened = []

    def begin(stage):
        if stage == "hybrid generation complete":
            transaction = model.transaction()
            transaction.__enter__()
            opened.append(transaction)

    try:
        with pytest.raises(MeshError, match="open geometry transaction"):
            hybrid.generate_hybrid_mesh_result(
                model, target_size=.25, strategy="mapped",
                mutation_policy=policy, cancellation_check=begin,
            )
    finally:
        for transaction in reversed(opened):
            transaction.__exit__(None, None, None)


def test_final_callback_cancellation_propagates():
    model = _model()
    revision = model.revision

    def cancel(stage):
        if stage == "hybrid generation complete":
            raise RuntimeError("cancelled at final publication")

    with pytest.raises(RuntimeError, match="cancelled at final publication"):
        hybrid.generate_hybrid_mesh_result(
            model, target_size=.25, strategy="mapped", cancellation_check=cancel,
        )
    assert model.revision == revision


def test_observational_final_callback_keeps_valid_publication():
    model = _model()
    stages = []
    result = hybrid.generate_hybrid_mesh_result(
        model, target_size=.25, strategy="mapped", cancellation_check=stages.append,
    )
    assert result.mesh.quads
    assert stages[-1] == "hybrid generation complete"
    assert result.mesh.geometry_model_id == model.model_id
    assert result.mesh.geometry_revision == model.revision
