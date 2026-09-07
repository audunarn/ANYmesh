"""Bounded consumer contracts against the accepted ANYgeometry owner API.

Local qualification requires owner commit
12683ae5d6dbb2620f5020b67c0a7673f2601766 (not version 0.4.3 alone).
These tests do not generate meshes, invoke a compiler, or publish artifacts.
"""

from types import SimpleNamespace

import anygeometry as owner
import pytest

from anymesher._owner_trim_domains import validated_complementary_trim_domains
from anymesher.errors import MeshError
from anymesher import preparation


def _reconstruct_pair():
    from anygeometry.generators import cylinder

    model = cylinder(
        radius=.5, height=2., circumferential_segments=12,
        origin=(0., 0., 0.), axis=(0., 0., 1.),
        radial_direction=(1., 0., 0.), longitudinal_spacing=.5, ring_spacing=1.,
    )
    wall = tuple(model.group("shell"))[0]
    model.features.capture_baseline(model)
    feature = model.features.append("generator.plate", parameters={
        "length": 2., "width": 2., "origin": (-1., -1., 1.),
        "u_direction": (1., 0., 0.), "v_direction": (0., 1., 0.),
        "semantic_group": "shell",
    })
    assert model.regenerate_features().success
    plate = model.features.get(feature.feature_id).outputs["face/1"]
    result = owner.query_intersection(
        model, model.handle("face", plate.id), model.handle("face", wall.id),
    )
    plan = owner.plan_imprint(model, result, policy=owner.ConnectionIntent.CONNECT)
    application = owner.apply_imprint(model, plan, policy=owner.ConnectionIntent.CONNECT)
    parents = tuple(item.id for item in application.face_intersection.first_faces)
    assert parents == (26, 27)
    assert model.validate_topology() == ()
    return model, parents


@pytest.fixture(scope="module")
def recovered_pair():
    return _reconstruct_pair()


def snapshot(model):
    return (
        model.revision,
        tuple((key, value.position.tobytes()) for key, value in model.vertices.items()),
        repr(tuple(model.edges.items())), repr(tuple(model.faces.items())),
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_public_recovered_pair_requires_live_owner_validation(recovered_pair, reverse, monkeypatch):
    model, parents = recovered_pair
    if reverse:
        parents = parents[::-1]
    before = snapshot(model)
    calls = []
    validate = owner.validate_trim_domain_binding

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return validate(*args, **kwargs)

    monkeypatch.setattr(owner, "validate_trim_domain_binding", record)
    assert validated_complementary_trim_domains(model, *parents)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is model
    assert args[2:] == tuple(model.handle("face", item) for item in parents)
    assert kwargs["expected_revision"] == before[0]
    assert snapshot(model) == before
    assert all(len(model.faces_using_edge(edge)) == 4 for edge in range(13, 25))


@pytest.mark.parametrize("site", ["query_trim_domain_relation", "validate_trim_domain_binding"])
def test_operational_failures_propagate_without_mutation(recovered_pair, monkeypatch, site):
    model, parents = recovered_pair
    before = snapshot(model)
    failure = RuntimeError("owner operation failed")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(owner, site, fail)
    with pytest.raises(RuntimeError) as caught:
        validated_complementary_trim_domains(model, *parents)
    assert caught.value is failure
    assert snapshot(model) == before


def test_cancellation_propagates(recovered_pair):
    model, parents = recovered_pair
    before = snapshot(model)
    failure = RuntimeError("cancel owner qualification")

    def cancel(phase):
        raise failure

    with pytest.raises(RuntimeError) as caught:
        validated_complementary_trim_domains(model, *parents, cancellation_check=cancel)
    assert caught.value is failure
    assert snapshot(model) == before


@pytest.mark.parametrize("field", ["revision", "model_id", "first_parent"])
def test_stale_foreign_and_wrong_parent_evidence_is_not_accepted(monkeypatch, field):
    model, parents = _reconstruct_pair()
    handles = tuple(model.handle("face", item) for item in parents)
    issuing_model = model
    issuing_handles = handles
    if field == "model_id":
        issuing_model, issuing_parents = _reconstruct_pair()
        issuing_handles = tuple(issuing_model.handle("face", item) for item in issuing_parents)
        assert issuing_model.model_id != model.model_id
    elif field == "first_parent":
        issuing_handles = handles[::-1]
    result = owner.query_trim_domain_relation(
        issuing_model, *issuing_handles, expected_revision=issuing_model.revision,
    )
    assert isinstance(result, owner.TrimDomainResult)
    # Establish that the owner constructed and live-validated genuine evidence
    # before deliberately replaying it against a different request binding.
    validate = owner.validate_trim_domain_binding
    validate(issuing_model, result, *issuing_handles, expected_revision=issuing_model.revision)
    if field == "revision":
        with model.transaction():
            model.add_points([(3., 3., 3.)])
        assert result.revision < model.revision
    before = snapshot(model)
    validation_calls = []

    def observe_real_validation(*args, **kwargs):
        validation_calls.append((args, kwargs))
        return validate(*args, **kwargs)

    monkeypatch.setattr(owner, "query_trim_domain_relation", lambda *args, **kwargs: result)
    monkeypatch.setattr(owner, "validate_trim_domain_binding", observe_real_validation)
    with pytest.raises(owner.TrimDomainError):
        validated_complementary_trim_domains(model, *parents)
    assert len(validation_calls) == 1
    args, kwargs = validation_calls[0]
    assert args[0] is model and args[1] is result
    assert args[2:] == handles
    assert kwargs["expected_revision"] == model.revision
    assert snapshot(model) == before


@pytest.mark.parametrize("missing", ["query_trim_domain_relation", "validate_trim_domain_binding"])
def test_missing_capabilities_fail_closed(recovered_pair, monkeypatch, missing):
    model, parents = recovered_pair
    before = snapshot(model)
    monkeypatch.delattr(owner, missing)
    with pytest.raises(ImportError):
        validated_complementary_trim_domains(model, *parents)
    assert snapshot(model) == before


@pytest.mark.parametrize("defect", ["relation", "complementary", "boundary"])
def test_every_positive_condition_is_required(recovered_pair, monkeypatch, defect):
    model, parents = recovered_pair
    fields = dict(
        relation=owner.TrimInteriorRelation.DISJOINT_INTERIORS,
        complementary=True, boundary=owner.TrimBoundaryContact.CURVE,
    )
    fields[defect] = False if defect == "complementary" else object()
    result = SimpleNamespace(**fields)
    calls = []
    monkeypatch.setattr(owner, "query_trim_domain_relation", lambda *args, **kwargs: result)
    monkeypatch.setattr(owner, "validate_trim_domain_binding", lambda *args, **kwargs: calls.append(args))
    assert not validated_complementary_trim_domains(model, *parents)
    assert len(calls) == 1


def test_unexempted_owner_overlap_is_never_filtered(recovered_pair, monkeypatch):
    from anymesher import _owner_trim_domains as adapter

    model, parents = recovered_pair
    before = snapshot(model)
    overlap = SimpleNamespace(first=parents[0], second=parents[1], area=.125)
    monkeypatch.setattr(preparation, "_face_pairs", lambda *args, **kwargs: (parents,))
    monkeypatch.setattr(adapter, "validated_complementary_trim_domains", lambda *args, **kwargs: False)
    monkeypatch.setattr(preparation, "find_coplanar_overlaps", lambda *args, **kwargs: [overlap])
    with pytest.raises(MeshError, match="positive-area coplanar overlap"):
        preparation.prepare_structural_closure(model, face_ids=parents, options=False)
    assert snapshot(model) == before
