"""Literal station-driver contracts; no geometry model or meshing execution."""
import ast
from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import pytest


class Refusal(RuntimeError):
    pass


@pytest.fixture
def api(monkeypatch):
    path = Path(__file__).parents[1] / 'src/anymesher/_material_region_binding.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'_pair_sampling_order', '_pair_sampling_parameters',
             '_station_balance_splits', '_station_balance_batch',
             '_boundary_literal_pin', '_boundary_callable_pin',
             '_prepare_curvilinear_station_union'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'Fraction': Fraction, 'MeshError': Refusal, '__name__': 'anymesher._station_literal',
                 'np': NS(ufunc=type(None)), '_BOUNDARY_ARRAY_DISPATCH_TYPE': type('Dispatch', (), {})}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    legacy = deepcopy(next(node for node in nodes if node.name == '_prepare_curvilinear_station_union'))
    legacy.name = '_legacy_station_union'
    for node in ast.walk(legacy):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == 'round_number'):
            node.body = ast.parse("raise MeshError('boundary station balancing qualification depth exhausted')").body
    ast.fix_missing_locations(legacy)
    exec(compile(ast.Module(body=[legacy], type_ignores=[]), str(path), 'exec'), namespace)
    hybrid = ModuleType('anymesher.hybrid')
    hybrid._station_parameters = lambda geometry, edge, count, field: geometry.parameters[edge]
    operation = ModuleType('anygeometry.prepared_material_curvilinear')

    @contextmanager
    def scope():
        yield 32

    operation.prepared_material_boundary_sampling_operation = scope
    monkeypatch.setitem(__import__('sys').modules, 'anymesher.hybrid', hybrid)
    monkeypatch.setitem(__import__('sys').modules, 'anygeometry.prepared_material_curvilinear', operation)
    return NS(**namespace)


def wave():
    parameters = tuple([i / 64 for i in range(41)] + [1.])
    coordinates = {t: (0. if i == 0 else .125 + i - 1) for i, t in enumerate(parameters)}
    return NS(edges={1: NS(start=1, end=2)}, parameters={1: parameters}), coordinates


class Binding:
    requires_curvilinear = True
    representative = 1
    region = NS(boundaries=((NS(source_edge=1),),))

    def __init__(self, geometry, coordinates):
        self.geometry, self.coordinates = geometry, coordinates
        self.calls, self.validations = [], 0

    def _uses_pair_boundary_sampling(self):
        return False

    def validate(self, callback):
        self.validations += 1

    def prepare_curvilinear_seed(self, requested, callback, *, refine=True):
        if callback is not None:
            callback()
        self.calls.append((deepcopy(requested), refine))
        result = {}
        initial = sorted(self.coordinates)
        for edge, values in requested.items():
            targets = {}
            for t in values:
                if t in self.coordinates:
                    x = self.coordinates[t]
                else:
                    a, b = next((a, b) for a, b in zip(initial, initial[1:]) if a < t < b)
                    alpha = (Fraction(t) - Fraction(a)) / (Fraction(b) - Fraction(a))
                    x = float(Fraction(self.coordinates[a]) + alpha *
                              (Fraction(self.coordinates[b]) - Fraction(self.coordinates[a])))
                targets[t] = (x, 0., 0.)
            result[edge] = targets
        return result


def run(api, geometry, bindings, callback=None):
    return api._prepare_curvilinear_station_union(geometry, bindings, {1: 1}, None,
                                                  callback, maximum_gradation=1.5)


def test_long_literal_wave_finishes_within_unchanged_32_rounds(api):
    geometry, coordinates = wave()
    before = deepcopy(coordinates)
    binding = Binding(geometry, coordinates)
    result = run(api, geometry, (binding,))
    assert not api._station_balance_splits(geometry, result, 1.5)
    assert set(coordinates) <= set(result[1])
    assert coordinates == before
    assert binding.validations == 1
    assert sum(not refine for _, refine in binding.calls) == 34
    # Every proposed station goes through a refining owner invocation before
    # the final non-refining agreement read. No interpolated proof is returned.
    final = binding.calls[-1][0]
    assert any(refine and requested == final for requested, refine in binding.calls[:-1])
    assert len(result[1]) <= len(coordinates) + 4096 + 64


def test_already_balanced_output_and_input_are_unchanged(api):
    geometry = NS(edges={1: NS(start=1, end=2)}, parameters={1: (0., .5, 1.)})
    coordinates = {0.: 0., .5: .5, 1.: 1.}
    binding = Binding(geometry, coordinates)
    expected = {1: {t: (x, 0., 0.) for t, x in coordinates.items()}}
    assert run(api, geometry, (binding,)) == expected
    assert len(binding.calls) == 2
    assert geometry.parameters == {1: (0., .5, 1.)}
    assert coordinates == {0.: 0., .5: .5, 1.: 1.}


def test_cancellation_is_terminal(api):
    geometry, coordinates = wave()
    binding = Binding(geometry, coordinates)
    def cancel():
        raise Refusal('cancelled')
    with pytest.raises(Refusal, match='cancelled'):
        run(api, geometry, (binding,), cancel)
    assert binding.calls == []


def test_owner_mutation_refusal_is_terminal(api):
    geometry, coordinates = wave()
    binding = Binding(geometry, coordinates)
    def stale(*args, **kwargs):
        raise Refusal('source changed')
    binding.prepare_curvilinear_seed = stale
    with pytest.raises(Refusal, match='source changed'):
        run(api, geometry, (binding,))
    assert binding.validations == 0


def test_incident_owner_station_disagreement_is_terminal(api):
    geometry, coordinates = wave()
    first, second = Binding(geometry, coordinates), Binding(geometry, coordinates)
    original = second.prepare_curvilinear_seed
    def disagree(*args, **kwargs):
        result = original(*args, **kwargs)
        result[1][0.] = (-1., 0., 0.)
        return result
    second.prepare_curvilinear_seed = disagree
    with pytest.raises(Refusal, match='disagree'):
        run(api, geometry, (first, second))
    assert first.validations == second.validations == 0


def test_depth32_refusal_remains_terminal(api):
    geometry, coordinates = wave()
    # A hostile proposal source remains bounded by the real driver's32 gate.
    api._prepare_curvilinear_station_union.__globals__['_station_balance_splits'] = lambda *args: ((1, .5),)
    api._prepare_curvilinear_station_union.__globals__['_station_balance_batch'] = lambda *args: ()
    binding = Binding(geometry, coordinates)
    with pytest.raises(Refusal, match='depth exhausted'):
        run(api, geometry, (binding,))
    assert sum(not refine for _, refine in binding.calls) == 33
    assert binding.validations == 0


@pytest.mark.parametrize('unit_intervals', [0, 1, 3, 31, 32])
def test_old_successful_query_sequence_and_output_remain_exact(api, unit_intervals):
    parameters = tuple(i / (unit_intervals+1) for i in range(unit_intervals+2))
    coordinates = {t: (0. if i == 0 else .5+i-1) for i, t in enumerate(parameters)}
    geometry = NS(edges={1: NS(start=1, end=2)}, parameters={1: parameters})
    old, new = Binding(geometry, coordinates), Binding(geometry, coordinates)
    expected = api._legacy_station_union(geometry, (old,), {1: 1}, None, maximum_gradation=1.5)
    actual = run(api, geometry, (new,))
    assert actual == expected
    assert new.calls == old.calls
    assert new.validations == old.validations == 1


def test_eligible_finishing_candidate_that_fails_ratio_is_not_retried(api):
    geometry, coordinates = wave()
    api._prepare_curvilinear_station_union.__globals__['_station_balance_splits'] = lambda *args: ((1, .5),)
    binding = Binding(geometry, coordinates)
    with pytest.raises(Refusal, match='depth exhausted'):
        run(api, geometry, (binding,))
    assert sum(not refine for _, refine in binding.calls) == 34
    assert sum(refine for _, refine in binding.calls) == 34
    assert binding.validations == 0


@pytest.mark.parametrize('reason', ['cancelled', 'source changed', 'box budget exhausted'])
def test_finishing_owner_errors_are_terminal(api, reason):
    geometry, coordinates = wave()
    binding = Binding(geometry, coordinates)
    original = binding.prepare_curvilinear_seed
    def fail(requested, callback, *, refine=True):
        if refine and sum(not r for _, r in binding.calls) == 33:
            raise Refusal(reason)
        return original(requested, callback, refine=refine)
    binding.prepare_curvilinear_seed = fail
    with pytest.raises(Refusal, match=reason):
        run(api, geometry, (binding,))
    assert sum(not refine for _, refine in binding.calls) == 33
    assert sum(refine for _, refine in binding.calls) == 33
    assert binding.validations == 0


def test_oversized_complete_batch_is_ineligible_without_mutation(api):
    geometry = NS(edges={1: NS(start=1, end=2)})
    stations = {1: {0.: (0., 0., 0.), .5: (1e-8, 0., 0.), 1.: (1., 0., 0.)}}
    before = deepcopy(stations)
    assert api._station_balance_batch(geometry, stations, 1.5) == ()
    assert stations == before


def test_batch_is_bounded_and_source_identity_scoped(api):
    geometry, coordinates = wave()
    stations = {1: {t: (x, 0., 0.) for t, x in coordinates.items()}}
    expected = api._station_balance_batch(geometry, stations, 1.5)
    geometry.edges[2] = NS(start=8, end=9)
    stations[2] = {0.: (0., 0., 0.), 1.: (1e-12, 0., 0.)}
    before = deepcopy(stations)
    assert api._station_balance_batch(geometry, stations, 1.5) == expected
    assert 0 < len(expected) <= 4096
    assert len(set(expected)) == len(expected)
    assert all(edge == 1 and parameter not in stations[edge] for edge, parameter in expected)
    assert stations == before


def test_finishing_owner_added_station_is_in_final_ratio_check(api):
    geometry, coordinates = wave()
    binding = Binding(geometry, coordinates)
    original = binding.prepare_curvilinear_seed
    added = 1. / 1048576
    def add_station(requested, callback, *, refine=True):
        if refine and sum(not r for _, r in binding.calls) == 33:
            requested = {edge: tuple(sorted(set(values) | {added}))
                         for edge, values in requested.items()}
        return original(requested, callback, refine=refine)
    binding.prepare_curvilinear_seed = add_station
    with pytest.raises(Refusal, match='depth exhausted'):
        run(api, geometry, (binding,))
    assert added in binding.calls[-1][0][1]
    assert sum(not refine for _, refine in binding.calls) == 34
    assert binding.validations == 0


def test_finishing_incident_disagreement_is_terminal(api):
    geometry, coordinates = wave()
    first, second = Binding(geometry, coordinates), Binding(geometry, coordinates)
    original = second.prepare_curvilinear_seed
    def disagree(requested, callback, *, refine=True):
        result = original(requested, callback, refine=refine)
        if not refine and sum(not r for _, r in second.calls) == 34:
            result[1][0.] = (-1., 0., 0.)
        return result
    second.prepare_curvilinear_seed = disagree
    with pytest.raises(Refusal, match='disagree'):
        run(api, geometry, (first, second))
    assert first.validations == second.validations == 0


@pytest.mark.parametrize('target,mode', [
    ('_station_balance_splits', 'rebind'), ('_station_balance_batch', 'rebind'),
    ('_boundary_callable_pin', 'rebind'), ('Fraction', 'rebind'),
    ('_station_balance_splits', 'code'), ('_station_balance_batch', 'code'),
    ('_boundary_callable_pin', 'code'),
])
def test_finishing_callback_runtime_drift_refuses(api, target, mode):
    geometry, coordinates = wave()
    binding = Binding(geometry, coordinates)
    original = binding.prepare_curvilinear_seed
    namespace = api._prepare_curvilinear_station_union.__globals__
    def drift(requested, callback, *, refine=True):
        result = original(requested, callback, refine=refine)
        if refine and sum(not r for _, r in binding.calls) == 33:
            if mode == 'rebind':
                namespace[target] = lambda *args, **kwargs: ()
            else:
                namespace[target].__code__ = (lambda *args, **kwargs: ()).__code__
        return result
    binding.prepare_curvilinear_seed = drift
    with pytest.raises(Refusal, match='runtime changed'):
        run(api, geometry, (binding,))
    assert binding.validations == 0


@pytest.mark.parametrize('balanced', [True, False])
def test_entry_pin_resource_error_propagates_before_any_owner_query(api, balanced):
    if balanced:
        geometry = NS(edges={1: NS(start=1, end=2)}, parameters={1: (0., 1.)})
        coordinates = {0.: 0., 1.: 1.}
    else:
        geometry, coordinates = wave()
    binding = Binding(geometry, coordinates)
    failure = MemoryError('entry pin allocation failed')
    def fail(value):
        raise failure
    api._prepare_curvilinear_station_union.__globals__['_boundary_callable_pin'] = fail
    with pytest.raises(MemoryError) as captured:
        run(api, geometry, (binding,))
    assert captured.value is failure
    assert binding.calls == []
    assert binding.validations == 0
