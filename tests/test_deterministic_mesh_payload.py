"""Pure fixture-normalization controls; no geometry or mesher execution."""
from copy import deepcopy
import pytest
from _deterministic_mesh_payload import GEOMETRY_TIMING_KEYS, deterministic_mesh_payload


def payload():
    return {"nodes": [[1., 2., 3.]], "hybrid_diagnostics": {"runtime": {
        "operation_counts": {**dict.fromkeys(GEOMETRY_TIMING_KEYS, 17),
                             "geometry.intersection_pairs": 4},
        "phase_seconds": {"planning": 1., "application": 2.}}}}


def counts(value):
    return value["hybrid_diagnostics"]["runtime"]["operation_counts"]


@pytest.mark.parametrize("key", GEOMETRY_TIMING_KEYS)
def test_only_named_elapsed_sample_values_compare_equal(key):
    first = payload(); second = deepcopy(first)
    counts(second)[key] = 2**100
    assert deterministic_mesh_payload(first) == deterministic_mesh_payload(second)


@pytest.mark.parametrize("bad", [True, -1, 1., None, "1"])
@pytest.mark.parametrize("key", GEOMETRY_TIMING_KEYS)
def test_invalid_named_elapsed_samples_refuse(key, bad):
    document = payload(); counts(document)[key] = bad
    with pytest.raises(ValueError, match="nonnegative exact integer"):
        deterministic_mesh_payload(document)


@pytest.mark.parametrize("key", GEOMETRY_TIMING_KEYS)
def test_elapsed_key_membership_remains_compared(key):
    first = payload(); second = deepcopy(first); del counts(second)[key]
    assert deterministic_mesh_payload(first) != deterministic_mesh_payload(second)


@pytest.mark.parametrize("fault", ["operation_count", "coordinate", "unknown_timing", "same_name_other_path"])
def test_non_elapsed_payload_changes_remain_visible(fault):
    first = payload(); second = deepcopy(first)
    if fault == "operation_count": counts(second)["geometry.intersection_pairs"] += 1
    elif fault == "coordinate": second["nodes"][0][0] += .5
    elif fault == "unknown_timing":
        counts(first)["geometry.future_time_ns"] = 1
        counts(second)["geometry.future_time_ns"] = 2
    else:
        first[GEOMETRY_TIMING_KEYS[0]] = 1
        second[GEOMETRY_TIMING_KEYS[0]] = 2
    assert deterministic_mesh_payload(first) != deterministic_mesh_payload(second)


def test_phase_membership_and_input_are_preserved():
    first = payload(); before = deepcopy(first); second = deepcopy(first)
    second["hybrid_diagnostics"]["runtime"]["phase_seconds"]["planning"] = 99.
    assert deterministic_mesh_payload(first) == deterministic_mesh_payload(second)
    assert first == before
    second["hybrid_diagnostics"]["runtime"]["phase_seconds"]["extra"] = 0.
    assert deterministic_mesh_payload(first) != deterministic_mesh_payload(second)
