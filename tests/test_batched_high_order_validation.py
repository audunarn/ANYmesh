"""Exact root batching, scalar fallback and final-publication ordering."""
import struct

import numpy as np
import pytest

import anymesher.hybrid as hybrid
from anymesher.errors import MeshError
from anymesher.quad import high_order as h
from anymesher.quad.timing import collect_quad_stage_timings


def _nodes(family):
    corners = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    if family == "T6":
        corners = corners[[0, 1, 3]]
    return np.vstack((corners, (corners + np.roll(corners, -1, axis=0)) / 2))


def _bits(value):
    if isinstance(value, float):
        return struct.pack("!d", value)
    if isinstance(value, (tuple, list)):
        return tuple(map(_bits, value))
    return value


def _report_bits(report):
    return tuple((key, _bits(value)) for key, value in report.to_dict().items())


def _compare(rows, family, **kwargs):
    expected = [h.certify_mapping_validity(row, family, **kwargs) for row in rows]
    actual = list(h._certify_mapping_validity_batch(rows, family, **kwargs))
    assert list(map(_report_bits, actual)) == list(map(_report_bits, expected))
    return actual


@pytest.mark.parametrize("family", ["Q8", "T6"])
@pytest.mark.parametrize("limits", [(0, 0), (2, 7), (6, 4096)])
def test_complete_report_bits_for_positive_folded_and_thin_elements(family, limits):
    rng = np.random.default_rng(202607)
    base = _nodes(family)
    cases = [base.copy(), base.copy()]
    cases[1][:, 1] *= 2.**-44
    for _ in range(20):
        cases.append(base + rng.uniform(-.45, .45, base.shape))
    for scale in (2.**-100, 2.**100):
        cases.append(base * scale)
    cases.append(base @ np.array([[1., .3, .1], [.2, 1., -.2], [.1, .2, 1.]]))
    cases.append(base * np.array([-1., 1., -0.]))
    reports = _compare(np.array(cases), family,
                       max_depth=limits[0], max_subdivisions=limits[1])
    assert reports[0].status is h.ValidityStatus.CERTIFIED_POSITIVE
    assert reports[1].status is h.ValidityStatus.UNRESOLVED
    assert any(report.status is h.ValidityStatus.INVALID for report in reports)


@pytest.mark.parametrize("family", ["Q8", "T6"])
def test_native_chunks_are_bounded_and_readonly_inputs_work(monkeypatch, family):
    original = h._batch_root_values
    sizes = []
    def observe(xyz, *args):
        sizes.append(len(xyz))
        return original(xyz, *args)
    monkeypatch.setattr(h, "_batch_root_values", observe)
    rows = np.repeat(_nodes(family)[None], 529, axis=0)
    rows.flags.writeable = False
    _compare(rows, family)
    assert sizes == [256, 256, 17]


@pytest.mark.parametrize("kind", ["noncontiguous", "subclass", "two_dimensions", "list", "error_policy"])
def test_unsupported_routes_use_scalar_without_batch_products(monkeypatch, kind):
    rows = np.repeat(_nodes("Q8")[None], 3, axis=0)
    if kind == "noncontiguous":
        rows = np.repeat(rows[..., None], 2, axis=-1)[..., 0]
    elif kind == "subclass":
        class Custom(np.ndarray):
            pass
        rows = rows.view(Custom)
    elif kind == "two_dimensions":
        rows = rows[:, :, :2]
    elif kind == "list":
        rows = rows.tolist()
    def unexpected(*args):
        raise AssertionError("unsupported route batched")
    monkeypatch.setattr(h, "_batch_root_values", unexpected)
    with np.errstate(over="raise" if kind == "error_policy" else "warn"):
        _compare(rows, "Q8")


@pytest.mark.parametrize("bad", ["nan", "zero", "extreme"])
def test_unsafe_input_refusal_matches_scalar_order(bad):
    rows = np.repeat(_nodes("Q8")[None], 2, axis=0)
    if bad == "nan":
        rows[1, 0, 0] = np.nan
    elif bad == "zero":
        rows[1] = 0.
    else:
        rows[1] *= 2.**200
    iterator = h._certify_mapping_validity_batch(rows, "Q8")
    assert _report_bits(next(iterator)) == _report_bits(h.certify_mapping_validity(rows[0], "Q8"))
    try:
        expected = h.certify_mapping_validity(rows[1], "Q8")
    except Exception as expected_error:
        with pytest.raises(type(expected_error), match=str(expected_error)):
            next(iterator)
    else:
        assert _report_bits(next(iterator)) == _report_bits(expected)


def test_alias_mutation_between_yields_is_observed():
    rows = np.repeat(_nodes("Q8")[None], 3, axis=0)
    iterator = h._certify_mapping_validity_batch(rows, "Q8")
    next(iterator)
    alias = rows[1]
    alias[4] = [.5, 3., 0.]
    assert _report_bits(next(iterator)) == _report_bits(h.certify_mapping_validity(alias, "Q8"))
    rows[2, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        next(iterator)


def test_error_policy_change_between_chunks_keeps_scalar_route(monkeypatch):
    rows = np.repeat(_nodes("Q8")[None], 257, axis=0)
    original = h._batch_root_values
    calls = []
    def observe(*args):
        calls.append(None)
        return original(*args)
    monkeypatch.setattr(h, "_batch_root_values", observe)
    iterator = h._certify_mapping_validity_batch(rows, "Q8")
    next(iterator)
    with np.errstate(over="raise"):
        assert len(list(iterator)) == 256
    assert len(calls) == 1


def test_callback_cancellation_and_mutation_are_scalar(monkeypatch):
    rows = np.repeat(_nodes("Q8")[None], 2, axis=0)
    error = RuntimeError("callback identity")
    stages = []
    def callback(stage):
        stages.append(stage)
        if len(stages) == 1:
            rows[0, 4] = [.5, .1, 0.]
        else:
            raise error
    def unexpected(*args):
        raise AssertionError("callbacks cannot batch")
    monkeypatch.setattr(h, "_batch_root_values", unexpected)
    with pytest.raises(RuntimeError) as caught:
        next(h._certify_mapping_validity_batch(rows, "Q8", cancellation_check=callback))
    assert caught.value is error
    assert stages == ["high-order validity start", "high-order validity square patch"]
    assert rows[0, 4, 1] == .1


def _gate_data(first):
    nodes = {i: row.copy() for i, row in enumerate(first)}
    nodes.update({i + len(first): row.copy() for i, row in enumerate(_nodes("Q8"))})
    return nodes, {20: tuple(range(8, 16)), 10: tuple(range(8))}


@pytest.mark.parametrize("later", ["missing_node", "zero_skeleton", "nan"])
def test_final_gate_earlier_invalid_wins_over_later_error(later):
    invalid = _nodes("Q8")
    invalid[4] = [.5, 3., 0.]
    assert h.certify_mapping_validity(invalid, "Q8").status is h.ValidityStatus.INVALID
    nodes, elements = _gate_data(invalid)
    if later == "missing_node":
        del nodes[15]
    elif later == "zero_skeleton":
        for node in range(8, 16):
            nodes[node][:] = 0.
    else:
        nodes[8][0] = np.nan
    with pytest.raises(MeshError, match="final Q8 element 10 is INVALID"):
        hybrid._validate_promoted_element_family(nodes, elements, "Q8")


def test_final_gate_custom_conversion_is_not_prefetched():
    invalid = _nodes("Q8")
    invalid[4] = [.5, 3., 0.]
    nodes, elements = _gate_data(invalid)
    class Custom:
        def __array__(self, dtype=None, copy=None):
            raise AssertionError("later custom conversion was prefetched")
    nodes[8] = Custom()
    with pytest.raises(MeshError, match="final Q8 element 10 is INVALID"):
        hybrid._validate_promoted_element_family(nodes, elements, "Q8")


def test_custom_certifier_is_respected_by_final_gate(monkeypatch):
    nodes, elements = _gate_data(_nodes("Q8"))
    error = RuntimeError("custom validator")
    def custom(*args, **kwargs):
        raise error
    monkeypatch.setattr(hybrid, "certify_mapping_validity", custom)
    with pytest.raises(RuntimeError) as caught:
        hybrid._validate_promoted_element_family(nodes, elements, "Q8")
    assert caught.value is error


def test_module_local_certifier_replacement_does_not_replace_callers_alias(monkeypatch):
    nodes, elements = _gate_data(_nodes("Q8"))
    def unexpected(*args, **kwargs):
        raise AssertionError("hybrid's unchanged imported scalar alias was replaced")
    monkeypatch.setattr(h, "certify_mapping_validity", unexpected)
    hybrid._validate_promoted_element_family(nodes, elements, "Q8")


def test_custom_stored_dictionary_key_is_not_looked_up_before_earlier_failure():
    invalid = _nodes("Q8")
    invalid[4] = [.5, 3., 0.]
    nodes, elements = _gate_data(invalid)
    callbacks = []
    class CustomKey(int):
        __hash__ = int.__hash__
        def __eq__(self, other):
            callbacks.append(other)
            raise AssertionError("later key callback was prefetched")
    value = nodes.pop(15)
    nodes[CustomKey(15)] = value
    with pytest.raises(MeshError, match="final Q8 element 10 is INVALID"):
        hybrid._validate_promoted_element_family(nodes, elements, "Q8")
    assert callbacks == []


@pytest.mark.parametrize("surface", ["cone", "cylinder"])
def test_connected_public_promotion_matches_forced_scalar(monkeypatch, surface):
    from anymesher.serialize import mesh_to_dict
    from test_curved_native_qualification import _model_face
    from test_cylindrical_frontal_integration import _persistent_state
    from quad_first_curved.test_ch8_conical_public import _generate
    if surface == "cylinder":
        from test_cylindrical_atlas_binding import _sector_model
        from quad_first_curved.test_ch3_cylindrical_public import _face_id
        model, selected = _sector_model(False)
        face = _face_id(model, selected[0])
    else:
        model, face, _ = _model_face(surface)
    target = .5 if surface == "cylinder" else .6
    before = _persistent_state(model)
    with collect_quad_stage_timings() as timing:
        actual = _generate(model, (face,), target, order="quadratic").mesh
    assert timing["strict_validation_batched_root_products"] > 0.
    def scalar(rows, family):
        return (h.certify_mapping_validity(row, family) for row in rows)
    monkeypatch.setattr(hybrid, "_certify_mapping_validity_batch", scalar)
    expected = _generate(model, (face,), target, order="quadratic").mesh
    assert mesh_to_dict(actual) == mesh_to_dict(expected)
    assert _persistent_state(model) == before
