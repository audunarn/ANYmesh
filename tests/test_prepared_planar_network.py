"""Focused non-native checks for the connected planar network consumer.

Covers the complete relation inventory, shared shell/beam corner and midside
node IDs for linear and quadratic orders, B3 start-mid-end bodies, request
bound ``AuthoredMaterialStations`` proofs, wrong-model/stale/mutated receipt
refusals, cooperative cancellation, absent capability typing, the
nonuniform canonical station override contract and legacy absent-option
compatibility.  Hub meshing checks are Python-only: they skip when a native
runtime is registered.
"""
import importlib.util
import json
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from anygeometry import GeometryModel, apply_intersections, plan_intersections, to_dict

from anymesher.errors import MeshError
from anymesher.native import has_native_boundary
from anymesher._prepared_planar_network import (
    PLANAR_ACCOUNTING_SCHEMA,
    PLANAR_STATION_SCHEMA,
    PreparedPlanarNetworkAccounting,
    query_prepared_planar_network_accounting as query_accounting,
    validate_prepared_planar_network_accounting as validate_accounting,
    query_prepared_planar_network_stations as query_stations,
    validate_prepared_planar_network_stations as validate_stations,
    generate_prepared_planar_network_mesh as generate_mesh,
    validate_prepared_planar_network_mesh as validate_mesh,
)

NATIVE_REGISTERED = has_native_boundary()
python_only = pytest.mark.skipif(
    NATIVE_REGISTERED,
    reason='native runtime registered; Python-only hub mesh checks not applicable')

_spec = importlib.util.spec_from_file_location('planar_network_fixtures',
    Path(__file__).parents[1] / 'tools/capture_inputs/large_connected_fixtures.py')
builders = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = builders
_spec.loader.exec_module(builders)


def build(family='connected_hub', count=4):
    fixture = getattr(builders, family)(count)
    model = fixture.model
    apply_intersections(model, plan_intersections(model, fixture.operands,
                                                  policy='connect'),
                        policy='connect')
    joint = next(a.target_id for a in model.attachments.values()
                 if a.kind == 'sheet_on_joint')
    return model, joint


@pytest.fixture(scope='session')
def hub():
    model, joint = build()
    before = to_dict(model)
    accounting = query_accounting(model, joint)
    stations = query_stations(model, accounting, target_size=2.0)
    return SimpleNamespace(model=model, joint=joint, before=before,
                           accounting=accounting, stations=stations)


def test_hub_accounting_accounts_complete_relation_inventory(hub):
    model, accounting = hub.model, hub.accounting
    owner = accounting.owner_receipt
    assert accounting.schema == PLANAR_ACCOUNTING_SCHEMA
    assert type(accounting) is PreparedPlanarNetworkAccounting
    assert not accounting.mesh_qualified and not accounting.beam_discretization_qualified
    assert not accounting.solver_admitted and not accounting.publication_qualified
    assert accounting.constraint_receipt is not None
    current = owner.current_records
    source = owner.source_records
    attachments = {row['id']: row for row in current['attachments']}
    junctions = {row['id']: row for row in current['junctions']}
    # Every structural record of both documents is accounted exactly once.
    expected = sorted((side, kind, row['id'])
                      for side, records in (('original', source), ('current', current))
                      for kind in ('sheets', 'parts', 'members', 'member_edge_uses',
                                   'attachments', 'junctions')
                      for row in records[kind])
    assert sorted(accounting.accounted_records) == expected
    # Complete traversal: every current member edge use appears once.
    assert {row[2] for row in accounting.traversal} == \
        {row['id'] for row in current['member_edge_uses']}
    assert len({row[2] for row in accounting.traversal}) == len(accounting.traversal)
    # Complete generated member joint inventory.
    assert {row[0] for row in accounting.member_joints} == \
        {key for key, row in attachments.items()
         if row['kind'] == 'member_on_face_boundary'}
    assert {row[2] for row in accounting.member_joints} == \
        {key for key, row in junctions.items() if row['kind'] == 'overlap'}
    # Complete generated point contact inventory.
    assert {row[0] for row in accounting.point_contacts} == \
        {key for key, row in attachments.items()
         if row['kind'] == 'member_through_face'}
    assert {row[2] for row in accounting.point_contacts} == \
        {key for key, row in junctions.items()
         if row['kind'] in ('crossing', 'endpoint', 'multi_way')}
    relation = json.loads(accounting.relation_json)
    assert sorted(relation) == ['attachments', 'member_joints', 'members',
                                'point_contacts']
    validate_accounting(model, accounting)
    assert to_dict(model) == hub.before


def _shell_chain(model, accounting, mesh, member):
    rows = [row for row in accounting.traversal if row[0] == member]
    chain = []
    for index, row in enumerate(rows):
        sequence = list(mesh.nodes_of_edge[row[3]])
        piece = sequence if row[4] == 'forward' else list(reversed(sequence))
        if index == 0:
            chain.extend(piece)
        else:
            assert chain[-1] == piece[0]
            chain.extend(piece[1:])
    return chain


@python_only
def test_linear_mesh_shares_shell_corner_ids(hub):
    model, accounting, stations = hub.model, hub.accounting, hub.stations
    result = generate_mesh(model, stations, order='linear')
    assert result.order == 'linear' and result.provisional
    assert not result.mesh_qualified and not result.beam_discretization_qualified
    assert not result.load_transfer_qualified and not result.solver_admitted
    assert not result.publication_qualified
    mesh = result.shell.mesh
    assert mesh.order == 'linear' and not mesh.is_quadratic
    assert set(mesh.beams).isdisjoint(mesh.shells)
    assert all(len(body) == 2 for body in mesh.beams.values())
    members = {row[0] for row in accounting.traversal}
    assert {member for member, _chain in result.member_chains} == members
    for member, chain in result.member_chains:
        assert list(chain) == _shell_chain(model, accounting, mesh, member)
        # Consecutive B2 bodies share exactly one canonical chain node.
        beams = [body for _id, _member, body in result.beams if _member == member]
        assert beams == [tuple(chain[i:i+2]) for i in range(len(chain)-1)]
        assert mesh.elements_of_member[member] == [
            _id for _id, _member, _body in result.beams if _member == member]
        assert mesh.nodes_of_member[member] == list(chain)
    # Contact anchors reuse the owner canonical junction vertex's shell node.
    by_attachment = {row[0]: row for row in stations.contact_anchors}
    for attachment, member, node, face, sheet, station in result.contact_anchors:
        anchor = by_attachment[attachment]
        assert (member, face, sheet, station) == \
            (anchor[1], anchor[3], anchor[4], anchor[5])
        assert mesh.node_of_vertex[anchor[2]] == node
    assert to_dict(model) == hub.before


@python_only
def test_quadratic_mesh_shares_corner_and_midside_ids(hub):
    model, accounting, stations = hub.model, hub.accounting, hub.stations
    result = generate_mesh(model, stations, order='quadratic')
    assert result.order == 'quadratic'
    mesh = result.shell.mesh
    assert mesh.is_quadratic
    assert set(mesh.beams).isdisjoint(mesh.shells)
    assert all(len(body) == 8 for body in mesh.quads.values())
    assert all(len(body) == 6 for body in mesh.tris.values())
    assert all(len(body) == 3 for body in mesh.beams.values())
    canonical = {}
    for row in stations.stations:
        canonical.setdefault(row[0], []).append(Fraction(*row[1]))
    for member, chain in result.member_chains:
        rows = [row for row in accounting.traversal if row[0] == member]
        pieces = []
        midsides = set()
        for row in rows:
            sequence = list(mesh.nodes_of_edge[row[3]])
            assert len(sequence) == 2*len(canonical[row[3]])-1
            midsides.update(sequence[1::2])
            piece = sequence if row[4] == 'forward' else list(reversed(sequence))
            pieces.append(piece)
        expected = list(pieces[0])
        for piece in pieces[1:]:
            assert expected[-1] == piece[0]
            expected.extend(piece[1:])
        assert list(chain) == expected
        beams = [body for _id, _member, body in result.beams if _member == member]
        # B3 bodies are consecutive start-mid-end triples sharing corner nodes.
        assert beams == [tuple(chain[i:i+3]) for i in range(0, len(chain)-1, 2)]
        for _id, _member, body in result.beams:
            if _member != member:
                continue
            assert body[1] in midsides
            assert body[0] not in midsides and body[2] not in midsides
    assert to_dict(model) == hub.before


@python_only
def test_mesh_validation_roundtrip_and_mutation_refuse(hub):
    model, stations = hub.model, hub.stations
    result = generate_mesh(model, stations)
    validate_mesh(model, result)
    with pytest.raises(MeshError, match='mesh contents changed'):
        validate_mesh(model, replace(result, beams=result.beams[:-1]))
    with pytest.raises(MeshError, match='wrong planar network mesh receipt'):
        validate_mesh(model, object())
    # Sheet ownership is part of the mesh result, not only receipt accounting.
    original_groups = dict(result.shell.mesh.elements_of_sheet)
    result.shell.mesh.elements_of_sheet.clear()
    try:
        with pytest.raises(MeshError, match='mesh contents changed'):
            validate_mesh(model, result)
    finally:
        result.shell.mesh.elements_of_sheet.update(original_groups)
    assert to_dict(model) == hub.before


def test_strip_correspondence_contract_matches_owner_capability():
    model, joint = build('connected_strip', 10)
    before = to_dict(model)
    accounting = query_accounting(model, joint)
    # The unified shared exterior edges of the strip are refused by the
    # boundary-correspondence contract; the whole-document owner receipt is
    # self-sufficient, so accounting succeeds with a None constraint receipt.
    from dataclasses import fields
    from anygeometry import PreparedEdgeSubcurvePreimages
    has_alias_contract = any(f.name == 'alias_records' for f in fields(PreparedEdgeSubcurvePreimages))
    validate_accounting(model, accounting)
    if has_alias_contract:
        assert accounting.constraint_receipt is not None
        assert query_stations(model, accounting).stations
    else:
        assert accounting.constraint_receipt is None
        with pytest.raises(MeshError,
                           match='planar network stations require qualified '
                                 'boundary correspondences'):
            query_stations(model, accounting)
    assert to_dict(model) == before


@python_only
def test_generation_closing_source_mutation_refuses(monkeypatch):
    import anymesher.hybrid as hybrid
    model, joint = build()
    accounting = query_accounting(model, joint)
    stations = query_stations(model, accounting, target_size=2.)
    original = hybrid._quad_first_execute
    def changed(*args, **kwargs):
        shell = original(*args, **kwargs)
        model.add_point(99., 99., 99.)
        return shell
    monkeypatch.setattr(hybrid, '_quad_first_execute', changed)
    with pytest.raises(MeshError, match='source changed during mesh generation'):
        generate_mesh(model, stations)


@python_only
def test_final_closing_callback_source_mutation_refuses():
    model, joint = build()
    accounting = query_accounting(model, joint)
    stations = query_stations(model, accounting, target_size=2.)
    mutated_phases = []
    accepted_result = None
    def mutate_only_at_closing(phase):
        if phase == 'planar network closing':
            mutated_phases.append(phase)
            model.add_point(99., 99., 99.)
        return False
    with pytest.raises(MeshError, match='source or stations changed during closing publication'):
        accepted_result = generate_mesh(
            model, stations, cancellation_check=mutate_only_at_closing)
    assert mutated_phases == ['planar network closing']
    assert accepted_result is None


def test_optional_inventory_never_swallows_arbitrary_owner_refusal(hub, monkeypatch):
    import anygeometry
    def cancelled(*args, **kwargs):
        raise anygeometry.GeometryError('constraint inventory cancelled')
    monkeypatch.setattr(anygeometry, 'query_prepared_authored_constraint_scope', cancelled)
    with pytest.raises(MeshError, match='constraint inventory cancelled'):
        query_accounting(hub.model, hub.joint, owner_receipt=hub.accounting.owner_receipt)


def test_material_station_proofs_are_request_bound(monkeypatch):
    import anygeometry
    model, joint = build()
    accounting = query_accounting(model, joint)
    real = anygeometry.query_prepared_authored_material_stations
    seen = []

    def stale_proof(geometry, correspondence, edge, params, cancellation_check=None):
        proof = real(geometry, correspondence, edge, params,
                     cancellation_check=cancellation_check)
        if seen:
            return seen[0]
        seen.append(proof)
        return proof

    monkeypatch.setattr(anygeometry,
                        'query_prepared_authored_material_stations', stale_proof)
    with pytest.raises(MeshError,
                       match='owner material station proof differs from the '
                             'requested edge or correspondence'):
        query_stations(model, accounting)

    def short_sequence(geometry, correspondence, edge, params,
                       cancellation_check=None):
        proof = real(geometry, correspondence, edge, params,
                     cancellation_check=cancellation_check)
        return replace(proof, parameters=proof.parameters[:-1])

    monkeypatch.setattr(anygeometry,
                        'query_prepared_authored_material_stations', short_sequence)
    with pytest.raises(MeshError,
                       match='owner material station proof lost sequence or count'):
        query_stations(model, accounting)


def test_wrong_model_stale_and_mutated_receipts_refuse(hub):
    # Wrong model: the receipt belongs to a different document.
    other = GeometryModel()
    other.add_plate(other.add_points(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))))
    with pytest.raises(MeshError, match='owner refusal'):
        validate_accounting(other, hub.accounting)
    # Stale: the live document changed after the receipt was acquired.
    model, joint = build()
    accounting = query_accounting(model, joint)
    model.add_point(9, 9, 9)
    with pytest.raises(MeshError):
        validate_accounting(model, accounting)
    # Mutated: field-level tampering is detected by re-derivation.
    with pytest.raises(MeshError, match='accounting contents changed'):
        validate_accounting(hub.model, replace(hub.accounting,
                                                member_joints=hub.accounting.member_joints[:-1]))
    with pytest.raises(MeshError, match='station contents changed'):
        validate_stations(hub.model, replace(hub.stations,
                                              stations=hub.stations.stations[:-1]))
    with pytest.raises(MeshError, match='wrong planar network accounting receipt'):
        validate_accounting(hub.model, object())
    with pytest.raises(MeshError, match='wrong planar network station receipt'):
        validate_stations(hub.model, object())


def test_cancellation_is_cooperative_and_preserves_geometry(hub):
    model, joint = build()
    before = to_dict(model)
    with pytest.raises(MeshError, match='owner refusal'):
        query_accounting(model, joint, cancellation_check=lambda _phase: True)
    accounting = query_accounting(model, joint)
    with pytest.raises(MeshError, match='cancelled'):
        query_stations(model, accounting, cancellation_check=lambda _phase: True)
    with pytest.raises(MeshError, match='cancelled'):
        generate_mesh(hub.model, hub.stations,
                      cancellation_check=lambda _phase: True)
    assert to_dict(model) == before
    assert to_dict(hub.model) == hub.before


def test_callback_mutation_is_detected(hub):
    model, joint = build()
    accounting = query_accounting(model, joint)
    calls = []

    def corrupt_accounting(_phase):
        if not calls:
            calls.append(True)
            object.__setattr__(accounting, 'traversal', ())
        return False

    with pytest.raises(MeshError, match='changed'):
        query_stations(model, accounting, cancellation_check=corrupt_accounting)
    assert calls
    original = hub.stations.stations

    def corrupt_stations(_phase):
        if len(calls) == 1:
            calls.append(True)
            object.__setattr__(hub.stations, 'stations', ())
        return False

    try:
        with pytest.raises(MeshError, match='changed'):
            generate_mesh(hub.model, hub.stations,
                          cancellation_check=corrupt_stations)
        assert len(calls) == 2
    finally:
        object.__setattr__(hub.stations, 'stations', original)


def test_absent_capability_is_typed_mesh_error(monkeypatch, hub):
    import anygeometry
    model = GeometryModel()
    # The owner contract is exported lazily; a module without the planar
    # network contract makes the import fail, which must surface as a typed
    # MeshError, never a raw ImportError.
    fake = ModuleType('anygeometry')
    monkeypatch.setitem(sys.modules, 'anygeometry', fake)
    with pytest.raises(MeshError,
                       match='prepared planar network capability unavailable'):
        query_accounting(model, 1)
    monkeypatch.undo()
    import anygeometry as fresh
    monkeypatch.setattr(fresh, 'query_prepared_planar_member_sheet_network',
                        None, raising=False)
    with pytest.raises(MeshError,
                       match='prepared planar network capability unavailable'):
        query_accounting(model, 1)
    monkeypatch.undo()
    monkeypatch.setattr(fresh, 'query_prepared_authored_material_stations',
                        None, raising=False)
    with pytest.raises(MeshError,
                       match='prepared planar network station capability '
                             'unavailable'):
        query_stations(hub.model, hub.accounting)


def test_station_input_validation_and_budget_rules(hub):
    with pytest.raises(MeshError, match='invalid planar network station target size'):
        query_stations(hub.model, hub.accounting, target_size=0)
    with pytest.raises(MeshError, match='invalid planar network station target size'):
        query_stations(hub.model, hub.accounting, target_size=float('nan'))
    with pytest.raises(MeshError, match='invalid planar network station budget'):
        query_stations(hub.model, hub.accounting, max_stations=-1)
    with pytest.raises(MeshError, match='wrong planar network accounting receipt'):
        query_stations(hub.model, object())
    with pytest.raises(MeshError, match='planar network station budget exhausted'):
        query_stations(hub.model, hub.accounting, max_stations=4)
    with pytest.raises(MeshError,
                       match='planar network validation budget may only decrease'):
        validate_stations(hub.model, hub.stations, max_stations=10)
    assert hub.stations.schema == PLANAR_STATION_SCHEMA


def test_mesh_input_validation_refuses_before_any_meshing(hub):
    with pytest.raises(MeshError, match='wrong planar network station receipt'):
        generate_mesh(hub.model, object())
    with pytest.raises(MeshError,
                       match='planar network mesh order must be linear or quadratic'):
        generate_mesh(hub.model, hub.stations, order='cubic')
    with pytest.raises(MeshError, match='unknown planar network certification mode'):
        generate_mesh(hub.model, hub.stations, certification_mode='bogus')
    with pytest.raises(MeshError,
                       match='planar network mesh options must be QuadMeshingOptions'):
        generate_mesh(hub.model, hub.stations, options=object())
    assert to_dict(hub.model) == hub.before


def plate_domain():
    from anymesher.quad.domain import PlanarQuadDomain
    geometry = GeometryModel()
    face = geometry.add_plate(geometry.add_points(
        ((0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0))))
    domain = PlanarQuadDomain.from_geometry(geometry, face)
    edges = sorted(edge for edge, _forward in domain.edge_uses)
    return geometry, domain, edges


@pytest.mark.parametrize('parameters,overrides,match', [
    ((0., .5, .5, 1.), None, 'increasing finite parameters from 0 to 1'),
    ((.1, .5, 1.), None, 'increasing finite parameters from 0 to 1'),
    ((0., .5, .9), None, 'increasing finite parameters from 0 to 1'),
    ((0., float('nan'), 1.), None, 'increasing finite parameters from 0 to 1'),
    ((0., -.5, 1.), None, 'increasing finite parameters from 0 to 1'),
    ((0., 1.5, 1.), None, 'increasing finite parameters from 0 to 1'),
    ((0.,), None, 'increasing finite parameters from 0 to 1'),
    ((0., .5, 1.), 3, 'division override conflicts with its canonical stations'),
])
def test_canonical_station_override_invalid_inputs(parameters, overrides, match):
    from anymesher.quad.boundary import BoundaryStationRegistry
    geometry, domain, edges = plate_domain()
    edge = edges[0]
    with pytest.raises(MeshError, match=match):
        BoundaryStationRegistry.for_domains(
            geometry, (domain,), 1.0, overrides={edge: overrides} if overrides else None,
            _canonical_stations={edge: parameters})
    with pytest.raises(MeshError, match='references unknown edge 999'):
        BoundaryStationRegistry.for_domains(
            geometry, (domain,), 1.0, _canonical_stations={999: (0., .5, 1.)})


def test_canonical_override_pins_exact_nonuniform_parameters():
    from anymesher.quad.boundary import BoundaryStationRegistry
    geometry, domain, edges = plate_domain()
    edge = edges[0]
    parameters = (0.0, 0.25, 0.5, 1.0)
    registry = BoundaryStationRegistry.for_domains(
        geometry, (domain,), 1.0, _canonical_stations={edge: parameters})
    assert registry.divisions[edge] == 3
    assert tuple(station.parameter for station in registry.chain(edge)) == parameters
    legacy = BoundaryStationRegistry.for_domains(geometry, (domain,), 1.0)
    for other in edges[1:]:
        assert registry.divisions[other] == legacy.divisions[other]
        assert registry.chain(other) == legacy.chain(other)


def test_legacy_absent_option_compatibility():
    from anymesher.quad.boundary import BoundaryStationRegistry
    geometry, domain, edges = plate_domain()
    legacy = BoundaryStationRegistry.for_domains(geometry, (domain,), 1.0)
    absent = BoundaryStationRegistry.for_domains(
        geometry, (domain,), 1.0, _canonical_stations=None)
    assert absent.divisions == legacy.divisions
    assert absent.chains == legacy.chains
    for edge in edges:
        parameters = [station.parameter for station in legacy.chain(edge)]
        assert parameters[0] == 0.0 and parameters[-1] == 1.0
        assert all(second > first for first, second in zip(parameters, parameters[1:]))
    pinned = BoundaryStationRegistry.for_domains(
        geometry, (domain,), 1.0, overrides={edges[0]: 3})
    assert pinned.divisions[edges[0]] == 3
