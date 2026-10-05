"""Geometry-only consumer guards; no meshing engine is exercised."""
from dataclasses import replace
import json

import numpy as np
import pytest
from anygeometry import GeometryModel, Plane, apply_intersections, plan_intersections, to_dict
from anygeometry.structural import ParameterRange
from anymesher._prepared_member_accounting import (
    query_prepared_member_accounting as query,
    validate_prepared_member_accounting as validate,
)
from anymesher.errors import MeshError


def build(reverse=False):
    model = GeometryModel()
    for points, plane in (
        (((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)),
         Plane((0, 0, 0), (1, 0, 0), (0, 1, 0))),
        (((-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)),
         Plane((0, 0, 0), (1, 0, 0), (0, 0, 1))),
    ):
        face = model.add_plate(model.add_points(points))
        model.set_face_surface(face, plane)
        model.add_sheet((face,))
    edge = next(e.id for e in model.edges.values()
                if all(model.vertex_position(v)[0] == -1 and model.vertex_position(v)[2] == 0
                       for v in (e.start, e.end)))
    member = model.add_member((edge,))
    if reverse:
        model.reverse_member(member)
    vertex = model.add_point(*model.sample_edge(edge, np.array([.75]))[0])
    model.add_attachment(None, 'vertex_on_edge', 'edge', edge, ParameterRange.point(0.),
        (ParameterRange.point(.75),), source_kind='vertex', source_id=vertex,
        evidence='exact', tolerance_used=1e-9)
    apply_intersections(model, plan_intersections(model, tuple(model.faces), policy='connect'),
                        policy='connect')
    joint = next(a.target_id for a in model.attachments.values() if a.kind == 'sheet_on_joint')
    return model, joint


@pytest.mark.parametrize('reverse', [False, True])
def test_complete_accounting_retains_order_station_and_boundaries(reverse):
    model, joint = build(reverse)
    before = to_dict(model)
    receipt = query(model, joint)
    assert len(receipt.member_traversal) == 2
    assert [row[2] for row in receipt.member_traversal] == [
        'reversed' if reverse else 'forward'] * 2
    assert [row[3] for row in receipt.member_traversal] == [(0., .5), (.5, 1.)]
    assert [row[4] for row in receipt.member_traversal] == (
        [((1, 2), (1, 1)), ((0, 1), (1, 2))] if reverse else
        [((0, 1), (1, 2)), ((1, 2), (1, 1))])
    assert receipt.point_station[2:] == ((3, 4), (1, 2))
    assert len(receipt.junction_membership) == 1
    assert receipt.junction_membership[0][3] == ()
    assert len([row for row in receipt.accounted_records if row[1] == 'attachments']) == 4
    assert not any((receipt.beam_discretization_qualified,
                    receipt.external_reference_transfer_qualified,
                    receipt.solver_admitted, receipt.publication_qualified))
    validate(model, receipt)
    assert to_dict(model) == before


@pytest.mark.parametrize('field', ['accounted_records', 'member_traversal',
                                  'junction_membership', 'point_station', 'relation_json'])
def test_forged_accounting_cannot_omit_or_change_records(field):
    model, joint = build()
    receipt = query(model, joint)
    forged = replace(receipt, **{field: '{}' if field == 'relation_json' else ()})
    with pytest.raises(MeshError, match='contents changed'):
        validate(model, forged)


@pytest.mark.parametrize('evidence', ['owner', 'inventory', 'ancestry'])
def test_forged_input_evidence_is_rederived(evidence):
    model, joint = build()
    receipt = query(model, joint)
    if evidence == 'owner':
        payload = receipt.owner_receipt.member_relation
        payload['current_member_uses'].pop()
        arguments = dict(owner_receipt=replace(receipt.owner_receipt,
                          member_relation_json=json.dumps(payload)))
    elif evidence == 'inventory':
        payload = receipt.constraint_receipt.inventory
        payload['current']['records']['member_edge_uses'].clear()
        arguments = dict(constraint_receipt=replace(receipt.constraint_receipt,
                          inventory_json=json.dumps(payload)))
    else:
        arguments = dict(edge_preimages=replace(receipt.edge_preimages, records=()))
    with pytest.raises(MeshError):
        query(model, joint, **arguments)


def test_wrong_model_stale_and_same_revision_edits_refuse():
    model, joint = build()
    receipt = query(model, joint)
    with pytest.raises(MeshError):
        validate(model.clone(preserve_identity=True), receipt)
    key = next(iter(model.members))
    model._members[key] = replace(model.members[key], name='same revision edit')
    with pytest.raises(MeshError):
        validate(model, receipt)
    model, joint = build()
    receipt = query(model, joint)
    model.add_point(9, 9, 9)
    with pytest.raises(MeshError):
        validate(model, receipt)


@pytest.mark.parametrize('mutation', ['model', 'evidence'])
def test_callback_mutation_cannot_publish_or_repair_forged_evidence(mutation):
    model, joint = build()
    receipt = query(model, joint)
    owner = receipt.owner_receipt
    if mutation == 'evidence':
        owner = replace(owner, member_relation_json='{}')
    calls = []
    def mutate(_phase):
        if not calls:
            calls.append(True)
            if mutation == 'model':
                key = next(iter(model.members))
                model._members[key] = replace(model.members[key], name='callback edit')
            else:
                object.__setattr__(owner, 'member_relation_json', receipt.owner_receipt.member_relation_json)
        return False
    with pytest.raises(MeshError):
        query(model, joint, owner_receipt=owner, cancellation_check=mutate)
    assert bool(calls) is (mutation == 'model')  # Invalid evidence refuses before callbacks.


def test_cancellation_is_nonpublication_and_preserves_geometry():
    model, joint = build()
    before = to_dict(model)
    with pytest.raises(MeshError, match='cancelled'):
        query(model, joint, cancellation_check=lambda _phase: True)
    assert to_dict(model) == before


def test_valid_supplied_evidence_mutation_is_detected_after_owner_callbacks():
    model, joint = build()
    receipt = query(model, joint)
    owner = receipt.owner_receipt
    calls = []
    def mutate(_phase):
        if not calls:
            calls.append(True)
            object.__setattr__(owner, 'member_relation_json', '{}')
        return False
    with pytest.raises(MeshError):
        query(model, joint, owner_receipt=owner, cancellation_check=mutate)


def test_missing_additive_owner_capability_is_typed(monkeypatch):
    import builtins
    real_import = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == 'anygeometry':
            raise ImportError('test absent additive contract')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', missing)
    with pytest.raises(MeshError, match='capability unavailable'):
        query(None, 1)


@pytest.mark.parametrize('kind', ['owner', 'inventory', 'ancestry'])
@pytest.mark.parametrize('entry', ['query', 'validate'])
def test_copy_hook_cannot_repair_original_evidence(kind, entry):
    model, joint = build()
    valid = query(model, joint)
    field, evidence, argument = {
        'owner': ('member_relation_json', valid.owner_receipt, 'owner_receipt'),
        'inventory': ('inventory_json', valid.constraint_receipt, 'constraint_receipt'),
        'ancestry': ('records', valid.edge_preimages, 'edge_preimages'),
    }[kind]
    good_value = getattr(evidence, field)
    calls = []
    class RepairString(str):
        def __deepcopy__(self, memo):
            calls.append(True)
            object.__setattr__(self.holder, field, good_value)
            return good_value
    class RepairTuple(tuple):
        def __deepcopy__(self, memo):
            calls.append(True)
            object.__setattr__(self.holder, field, good_value)
            return good_value
    value = RepairTuple() if kind == 'ancestry' else RepairString('{}')
    forged = replace(evidence, **{field: value})
    value.holder = forged
    with pytest.raises(MeshError, match='plain immutable fields'):
        if entry == 'query':
            query(model, joint, **{argument: forged})
        else:
            validate(model, replace(valid, **{argument: forged}))
    assert not calls
    assert getattr(forged, field) is value


@pytest.mark.parametrize('entry', ['query', 'validate'])
def test_alias_returning_copy_hooks_are_never_executed(entry, monkeypatch):
    model, joint = build()
    valid = query(model, joint)
    calls = []
    def alias(self, memo):
        calls.append(type(self).__name__)
        return self
    for cls in {type(valid), type(valid.owner_receipt), type(valid.constraint_receipt),
                type(valid.edge_preimages)}:
        monkeypatch.setattr(cls, '__deepcopy__', alias, raising=False)
    if entry == 'query':
        result = query(model, joint, owner_receipt=valid.owner_receipt,
                       constraint_receipt=valid.constraint_receipt,
                       edge_preimages=valid.edge_preimages)
        assert result == valid
    else:
        validate(model, valid)
    assert not calls


@pytest.mark.parametrize('kind', ['duplicate_use', 'missing_target', 'zero_interval'])
def test_owner_rederivation_prevents_raw_derive_failures(kind):
    model, joint = build()
    receipt = query(model, joint)
    payload = receipt.owner_receipt.member_relation
    if kind == 'duplicate_use':
        payload['current_member_uses'].append(payload['current_member_uses'][0])
    elif kind == 'missing_target':
        payload['current_attachment']['target_id'] = -1
    else:
        payload['source_member_uses'][0]['parent_range'] = [0., 0.]
    forged = replace(receipt.owner_receipt, member_relation_json=json.dumps(payload))
    with pytest.raises(MeshError, match='owner refusal'):
        query(model, joint, owner_receipt=forged)


@pytest.mark.parametrize('change', ['opaque', 'scope', 'ancestry', 'isolated'])
def test_synthetic_accounting_cross_checks_fail_closed(change):
    # Synthetic branch checks do not claim that forged owner evidence validates.
    from anymesher._prepared_member_accounting import _derive
    model, joint = build()
    receipt = query(model, joint)
    inventory = receipt.constraint_receipt.inventory
    constraints, ancestry = receipt.constraint_receipt, receipt.edge_preimages
    if change == 'opaque':
        inventory['original']['opaque_unqualified']['groups'] = {'unknown': [1]}
    elif change == 'isolated':
        inventory['original']['isolated_vertex_ids'].append(-1)
    elif change == 'scope':
        constraints = replace(constraints, selected_root_ids=())
    else:
        ancestry = replace(ancestry, records=())
    if change in ('opaque', 'isolated'):
        constraints = replace(constraints, inventory_json=json.dumps(inventory))
    with pytest.raises(MeshError):
        _derive(receipt.owner_receipt, constraints, ancestry)


@pytest.mark.parametrize('field', ['schema', 'relation_json'])
def test_accounting_scalar_hooks_cannot_run_before_snapshot(field):
    model, joint = build()
    valid = query(model, joint)
    calls = []
    class RepairString(str):
        def __eq__(self, other):
            calls.append('equality')
            object.__setattr__(self.holder, field, getattr(valid, field))
            return True
        def __deepcopy__(self, memo):
            calls.append('copy')
            object.__setattr__(self.holder, field, getattr(valid, field))
            return getattr(valid, field)
    value = RepairString('{}')
    forged = replace(valid, **{field: value})
    value.holder = forged
    with pytest.raises(MeshError, match='plain immutable fields'):
        validate(model, forged)
    assert not calls


@pytest.mark.parametrize('kind', ['owner', 'inventory', 'ancestry', 'accounting'])
def test_source_serialization_cannot_repair_pinned_original_evidence(kind):
    model, joint = build()
    valid = query(model, joint)
    field, evidence, argument = {
        'owner': ('member_relation_json', valid.owner_receipt, 'owner_receipt'),
        'inventory': ('inventory_json', valid.constraint_receipt, 'constraint_receipt'),
        'ancestry': ('records', valid.edge_preimages, 'edge_preimages'),
        'accounting': ('relation_json', valid, None),
    }[kind]
    good = getattr(evidence, field)
    forged = replace(evidence, **{field: () if kind == 'ancestry' else '{}'})
    calls = []
    class RepairDuringGeometryCopy(dict):
        def __deepcopy__(self, memo):
            calls.append(True)
            object.__setattr__(forged, field, good)
            model._serialization_extensions = {}
            return {}
    model._serialization_extensions = RepairDuringGeometryCopy()
    with pytest.raises(MeshError, match='changed'):
        if kind == 'accounting':
            validate(model, forged)
        else:
            query(model, joint, **{argument: forged})
    assert calls


@pytest.mark.parametrize('target', ['accounting', 'ancestry'])
@pytest.mark.parametrize('window', ['closing_owner', 'final_serialization'])
def test_generated_result_is_pinned_across_closing_mutation(target, window, monkeypatch):
    import anygeometry
    import anymesher._prepared_member_accounting as accounting
    model, joint = build()
    real_derive, real_to_dict = accounting._derive, anygeometry.to_dict
    results, calls = [], []
    def mutate():
        if not calls:
            calls.append(True)
            if target == 'accounting':
                object.__setattr__(results[0], 'accounted_records', ())
            else:
                object.__setattr__(results[0].edge_preimages, 'source_checksum', 'late mutation')
    class MutateDuringGeometryCopy(dict):
        def __deepcopy__(self, memo):
            mutate()
            model._serialization_extensions = {}
            return {}
    def derive(*args):
        result = real_derive(*args)
        results.append(result)
        if window == 'closing_owner':
            model._serialization_extensions = MutateDuringGeometryCopy()
        return result
    def closing_to_dict(owner):
        if results and window == 'final_serialization':
            mutate()
        return real_to_dict(owner)
    monkeypatch.setattr(accounting, '_derive', derive)
    monkeypatch.setattr(anygeometry, 'to_dict', closing_to_dict)
    with pytest.raises(MeshError):
        query(model, joint)  # No optional receipts; supplied tuple contains only None.
    assert calls


def test_unknown_metaclass_hooks_never_execute_during_capture():
    from anymesher._prepared_member_accounting import _snapshot
    calls = []
    class Trap(type):
        def __eq__(cls, other):
            calls.append('eq')
            return False
        def __hash__(cls):
            calls.append('hash')
            return 1
        def __getattribute__(cls, name):
            calls.append(name)
            return super().__getattribute__(name)
    class Unknown(metaclass=Trap):
        pass
    unknown = object.__new__(Unknown)
    for evidence in (unknown, (unknown,)):
        with pytest.raises(MeshError, match='plain immutable fields'):
            _snapshot(evidence)
    assert not calls


@pytest.mark.parametrize('malformed', ['cycle', 'depth'])
def test_malformed_evidence_graphs_raise_typed_errors(malformed):
    from anymesher._prepared_member_accounting import _snapshot
    if malformed == 'cycle':
        model, joint = build()
        evidence = query(model, joint)
        object.__setattr__(evidence, 'relation_json', evidence)
    else:
        evidence = ()
        for _ in range(70):
            evidence = (evidence,)
    with pytest.raises(MeshError, match='cyclic|nesting'):
        _snapshot(evidence)


def test_snapshot_breadth_is_not_a_model_size_limit():
    from anymesher._prepared_member_accounting import _snapshot
    evidence = tuple(range(10000))
    assert len(_snapshot(evidence)[1]) == len(evidence)


def test_late_ancestry_guard_mutation_refuses_before_derivation(monkeypatch):
    import anygeometry
    import anymesher._prepared_member_accounting as accounting
    model, joint = build()
    valid = query(model, joint)
    real_validate = anygeometry.validate_prepared_edge_subcurve_preimages_binding
    real_derive = accounting._derive
    mutated, derived = [], []
    def callback(phase):
        if phase == 'after final callback-bearing ancestry guard':
            payload = valid.owner_receipt.member_relation
            payload['source_member_uses'].clear()
            object.__setattr__(valid.owner_receipt, 'member_relation_json', json.dumps(payload))
            mutated.append(True)
        return False
    def late_guard(*args, **kwargs):
        real_validate(*args, **kwargs)
        if kwargs.get('cancellation_check') is not None:
            kwargs['cancellation_check']('after final callback-bearing ancestry guard')
    def derive(*args):
        derived.append(True)
        return real_derive(*args)
    monkeypatch.setattr(anygeometry, 'validate_prepared_edge_subcurve_preimages_binding', late_guard)
    monkeypatch.setattr(accounting, '_derive', derive)
    with pytest.raises(MeshError, match='before derivation'):
        query(model, joint, owner_receipt=valid.owner_receipt,
              constraint_receipt=valid.constraint_receipt,
              edge_preimages=valid.edge_preimages, cancellation_check=callback)
    assert mutated and not derived


def test_generated_owner_late_mutation_refuses_before_derivation(monkeypatch):
    import anygeometry
    import anymesher._prepared_member_accounting as accounting
    model, joint = build()
    real_query = anygeometry.query_prepared_member_sheet_joint_component
    real_validate = anygeometry.validate_prepared_edge_subcurve_preimages_binding
    real_derive = accounting._derive
    acquired, mutated, derived = [], [], []
    def capture_owner(*args, **kwargs):
        result = real_query(*args, **kwargs)
        acquired.append(result)
        return result
    def late_guard(*args, **kwargs):
        real_validate(*args, **kwargs)
        if kwargs.get('cancellation_check') is not None:
            payload = acquired[0].member_relation
            payload['source_member_uses'].clear()
            object.__setattr__(acquired[0], 'member_relation_json', json.dumps(payload))
            mutated.append(True)
    def derive(*args):
        derived.append(True)
        return real_derive(*args)
    monkeypatch.setattr(anygeometry, 'query_prepared_member_sheet_joint_component', capture_owner)
    monkeypatch.setattr(anygeometry, 'validate_prepared_edge_subcurve_preimages_binding', late_guard)
    monkeypatch.setattr(accounting, '_derive', derive)
    with pytest.raises(MeshError, match='acquired evidence changed before derivation'):
        query(model, joint, cancellation_check=lambda _phase: False)
    assert mutated and not derived


def test_partial_missing_snapshot_capability_is_typed(monkeypatch):
    import anygeometry
    model, joint = build()
    monkeypatch.delattr(anygeometry, 'AuthoredFaceDefinition')
    with pytest.raises(MeshError, match='snapshot capability unavailable'):
        query(model, joint)
