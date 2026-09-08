"""Exact reserved midside identities at the private split-commit boundary."""
from fractions import Fraction

import numpy as np
import pytest

from anymesher.errors import MeshError
from anymesher.native_v2 import ComponentSeedRegistry


def publish_reserved(registry, edge, station, node, publish=lambda value: value):
    return registry._resolve_and_publish(
        edge, station.numerator, station.denominator, publish,
        existing_node_id=node,
    )


def test_reserved_midside_does_not_call_external_allocator():
    def forbidden():
        pytest.fail("reserved midside must not allocate another global node")
    registry = ComponentSeedRegistry(100, node_id_allocator=forbidden)
    station = Fraction.from_float(0.3)
    assert publish_reserved(registry, 7, station, 23) == 23
    assert registry.resolve(7, station.numerator, station.denominator) == 23
    assert registry.assigned_node_ids == (23,)
    assert registry._next == 100


def test_large_exact_denominators_and_reduced_keys():
    registry = ComponentSeedRegistry(100)
    station = Fraction.from_float(float.fromhex("0x1.0000000000001p-80"))
    assert station.denominator > 2**64
    assert publish_reserved(registry, 4, station, 17) == 17
    assert registry.resolve(4, station.numerator * 3, station.denominator * 3) == 17
    assert registry.assigned_node_ids == (17,)


def test_adjacent_binary64_stations_do_not_alias():
    registry = ComponentSeedRegistry(100)
    first = Fraction.from_float(0.3)
    second = Fraction.from_float(float(np.nextafter(0.3, 1.)))
    assert first != second
    assert publish_reserved(registry, 4, first, 17) == 17
    assert publish_reserved(registry, 4, second, 18) == 18
    assert registry.assigned_node_ids == (17, 18)


def test_conflicting_station_identity_preserves_registry_and_skips_publish():
    registry = ComponentSeedRegistry(100)
    station = Fraction(1, 3)
    publish_reserved(registry, 4, station, 17)
    before = dict(registry._values), registry._next
    with pytest.raises(MeshError, match="conflicts"):
        publish_reserved(registry, 4, station, 18, lambda _: pytest.fail("must not publish"))
    assert (registry._values, registry._next) == before


def test_one_reserved_id_cannot_alias_distinct_station_keys():
    registry = ComponentSeedRegistry(100)
    publish_reserved(registry, 4, Fraction(1, 3), 17)
    before = dict(registry._values), registry._next
    with pytest.raises(MeshError, match="invalid node identity"):
        publish_reserved(registry, 4, Fraction(2, 3), 17)
    assert (registry._values, registry._next) == before


@pytest.mark.parametrize("already_bound", (False, True))
def test_publisher_failure_rolls_back_only_provisional_registration(already_bound):
    registry = ComponentSeedRegistry(100)
    publish_reserved(registry, 4, Fraction(1, 4), 17)
    station = Fraction(3, 4)
    if already_bound:
        publish_reserved(registry, 4, station, 201)
    before = dict(registry._values), registry._next
    error = RuntimeError("detached topology publication failed")
    def fail(node):
        assert node == 201
        raise error
    with pytest.raises(RuntimeError) as caught:
        publish_reserved(registry, 4, station, 201, fail)
    assert caught.value is error
    assert (registry._values, registry._next) == before
    assert not registry._resolving


@pytest.mark.parametrize("node", (0, -1, True, np.bool_(True), 1.5, "17"))
def test_invalid_reserved_identity_cannot_publish(node):
    registry = ComponentSeedRegistry(100)
    with pytest.raises(MeshError, match="positive integer"):
        publish_reserved(registry, 4, Fraction(1, 2), node,
                         lambda _: pytest.fail("must not publish"))
    assert registry.assigned_node_ids == () and registry._next == 100


def test_reentrant_resolution_cannot_steal_reserved_identity():
    registry = ComponentSeedRegistry(100)
    def recurse(_):
        return publish_reserved(registry, 4, Fraction(3, 4), 19)
    with pytest.raises(MeshError, match="already active"):
        publish_reserved(registry, 4, Fraction(1, 4), 17, recurse)
    assert registry.assigned_node_ids == () and registry._next == 100


def test_default_allocation_remains_monotonic_after_reserved_reuse():
    registry = ComponentSeedRegistry(100)
    assert publish_reserved(registry, 4, Fraction(1, 4), 17) == 17
    assert registry.resolve(4, 1, 2) == 100
    assert registry.resolve(4, 2, 4) == 100
    assert registry.resolve(4, 3, 4) == 101
