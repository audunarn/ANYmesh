"""Owner capability negotiation and exact serial output, without worker pools."""
import json
from functools import wraps
from types import SimpleNamespace

import pytest
import anygeometry
from anygeometry import GeometryModel
from anymesher import component_parallel as module
from anymesher.serialize import mesh_to_dict


def plate():
    model = GeometryModel()
    model.add_plate(model.add_points(((0.,0.,0.),(2.,0.,0.),(2.,1.,0.),(0.,1.,0.))))
    return model


def refused():
    return SimpleNamespace(certified=False, components=(), merge_reasons=(),
                           refusals=(SimpleNamespace(reason="owner declined"),))


@pytest.mark.parametrize("kind", ["explicit", "legacy", "kwargs", "uninspectable", "positional_only"])
def test_only_explicit_keyword_capability_receives_opt_in(monkeypatch, kind):
    seen = []
    if kind == "explicit":
        def planner(model, *, stop_if_single_component=False, **options):
            seen.append((stop_if_single_component, options))
            return refused()
    elif kind == "legacy":
        def planner(model, *, edge_ids, separation, expected_revision, cancellation_check):
            seen.append((False, {}))
            return refused()
    elif kind == "positional_only":
        def planner(model, stop_if_single_component=False, /, **options):
            seen.append((stop_if_single_component, options))
            return refused()
    else:
        def planner(model, **options):
            seen.append((options.get("stop_if_single_component", False), options))
            return refused()
        if kind == "uninspectable":
            planner.__signature__ = "unavailable"
    monkeypatch.setattr(anygeometry, "plan_independent_components", planner)
    monkeypatch.setattr(module, "generate_hybrid_mesh_result", lambda *args, **kwargs:
        SimpleNamespace(mesh=SimpleNamespace(hybrid_diagnostics={})))
    result = module.generate_hybrid_mesh_result_parallel(plate(), target_size=.5)
    assert len(seen) == 1
    assert seen[0][0] is (kind == "explicit")
    assert "stop_if_single_component" not in seen[0][1]
    assert result.mesh.hybrid_diagnostics["parallel"]["used"] is False


def test_internal_planner_typeerror_is_not_retried_or_masked(monkeypatch):
    marker = TypeError("inside owner planner")
    calls = []
    def planner(model, *, stop_if_single_component=False, **options):
        calls.append(stop_if_single_component)
        raise marker
    monkeypatch.setattr(anygeometry, "plan_independent_components", planner)
    monkeypatch.setattr(module, "_serial", lambda *args:pytest.fail("no retry/fallback for TypeError"))
    with pytest.raises(TypeError) as caught:
        module.generate_hybrid_mesh_result_parallel(plate(), target_size=.5)
    assert caught.value is marker and calls == [True]


@pytest.mark.parametrize("forwarding", [False, True])
def test_decorated_wrapper_uses_its_actual_callable_signature(monkeypatch, forwarding):
    owner = anygeometry.plan_independent_components
    calls = []
    if forwarding:
        @wraps(owner)
        def planner(*args, **options):
            calls.append(dict(options))
            return owner(*args, **options)
    else:
        @wraps(owner)
        def planner(model, *, edge_ids, separation, expected_revision, cancellation_check):
            options = dict(edge_ids=edge_ids, separation=separation,
                expected_revision=expected_revision, cancellation_check=cancellation_check)
            calls.append(options)
            return owner(model, **options)
    monkeypatch.setattr(anygeometry, "plan_independent_components", planner)
    monkeypatch.setattr(module, "generate_hybrid_mesh_result", lambda *args, **kwargs:
        SimpleNamespace(mesh=SimpleNamespace(hybrid_diagnostics={})))
    result = module.generate_hybrid_mesh_result_parallel(plate(), target_size=.5)
    assert len(calls) == 1 and "stop_if_single_component" not in calls[0]
    assert result.mesh.hybrid_diagnostics["parallel"]["reason"] == "1 component(s)"


@pytest.mark.parametrize("strategy", ["mapped", "native"])
def test_public_serial_mesh_exact_with_new_and_legacy_geometry_capability(monkeypatch, strategy):
    model = plate()
    original = anygeometry.plan_independent_components
    arguments = dict(target_size=.5, order="quadratic", strategy=strategy, native_backend="python")
    current = module.generate_hybrid_mesh_result_parallel(model, **arguments).mesh
    def legacy(model, *, edge_ids, separation, expected_revision, cancellation_check):
        return original(model, edge_ids=edge_ids, separation=separation,
                        expected_revision=expected_revision, cancellation_check=cancellation_check)
    monkeypatch.setattr(anygeometry, "plan_independent_components", legacy)
    previous = module.generate_hybrid_mesh_result_parallel(model, **arguments).mesh
    a, b = mesh_to_dict(current), mesh_to_dict(previous)
    # Preserve every serialized physical field, global ID, high-order node,
    # station/ownership/constraint and scalar bit, including signed zero.
    for key in a:
        if key not in ("hybrid_diagnostics", "structural_preparation"):
            assert json.dumps(a[key], sort_keys=True) == json.dumps(b[key], sort_keys=True), key
    for diagnostic in ("high_order_geometry", "mesh_quality", "quad_quality_repair"):
        assert json.dumps(a["hybrid_diagnostics"].get(diagnostic), sort_keys=True) == json.dumps(
            b["hybrid_diagnostics"].get(diagnostic), sort_keys=True)
    assert current.hybrid_diagnostics["parallel"]["used"] is False
    assert "without full independence certification" in current.hybrid_diagnostics["parallel"]["reason"]
    assert previous.hybrid_diagnostics["parallel"]["reason"] == "1 component(s)"
