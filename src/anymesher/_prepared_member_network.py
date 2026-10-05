"""Experimental whole-network accounting/stations; no mesh or beam authority."""
from dataclasses import dataclass, fields
from fractions import Fraction
from math import ceil, isfinite, sqrt

from ._prepared_member_accounting import _require, _snapshot, _json
from .errors import MeshError

ACCOUNTING_SCHEMA = 'anymesher.prepared-member-network-accounting-v1'
STATION_SCHEMA = 'anymesher.prepared-member-network-stations-v1'


@dataclass(frozen=True, slots=True)
class PreparedMemberNetworkAccounting:
    schema: str
    owner_receipt: object
    constraint_receipt: object
    accounted_records: tuple
    # member, original use, current use, edge, orientation, native carrier interval,
    # exact persisted global Member interval, original carrier, full certificate JSON.
    traversal: tuple
    # original attachment, point vertex, current edge, original/local native station,
    # original carrier, full relation JSON (includes retained error bounds).
    points: tuple
    relation_json: str

    @property
    def mesh_qualified(self):
        return False

    @property
    def beam_discretization_qualified(self):
        return False

    @property
    def solver_admitted(self):
        return False

    @property
    def publication_qualified(self):
        return False


@dataclass(frozen=True, slots=True)
class PreparedMemberNetworkStations:
    schema: str
    accounting: object
    target_size: float
    max_stations: int
    # edge, local station, canonical endpoint or unique attachment vertex,
    # exact current polynomial XYZ, authored root and ancestor parameter.
    stations: tuple
    # traversal row, ordered (local, global Member) station parameter pairs.
    traversal_stations: tuple
    point_intents: tuple

    @property
    def mesh_qualified(self):
        return False

    @property
    def publication_qualified(self):
        return False


def _pin(value):
    """Flatten only exact new classes; reuse the accepted closed immutable pin."""
    try:
        from anygeometry import PreparedMemberSheetJointNetwork
    except ImportError as error:
        raise MeshError('prepared member network capability unavailable') from error
    known = (PreparedMemberSheetJointNetwork, PreparedMemberNetworkAccounting,
             PreparedMemberNetworkStations)
    active = set()

    def flatten(item, depth=0):
        _require(depth <= 64, 'malformed network evidence nesting')
        cls = type(item)
        if cls is tuple or any(cls is candidate for candidate in known):
            _require(id(item) not in active, 'cyclic network evidence')
            active.add(id(item))
            try:
                if cls is tuple:
                    return tuple(flatten(child, depth+1) for child in item)
                return (cls.__name__, tuple((field.name, flatten(
                    object.__getattribute__(item, field.name), depth+1)) for field in fields(cls)))
            finally:
                active.remove(id(item))
        return item
    return _snapshot(flatten(value))


def _derive(owner, constraints):
    _require(owner.scope == constraints.scope and owner.authored_face_ids == constraints.selected_root_ids
             and owner.current_face_ids == constraints.current_face_ids and not constraints.outside_root_ids,
             'incomplete network inventory selection')
    _require(owner.bounded_relation_mapping_qualified and owner.occurrence_mapping_qualified
             and not owner.semantic_mapping_qualified and not owner.beam_discretization_qualified
             and not owner.external_reference_transfer_qualified and not owner.publication_qualified
             and constraints.typed_inventory_complete and not constraints.publication_qualified,
             'unexpected network authority')
    inventory = constraints.inventory
    accounted = []
    for side, records in (('original', owner.source_records), ('current', owner.current_records)):
        # Sheets/Parts are complete in the whole-document owner receipt; the
        # constraint inventory intentionally exposes only structural constraints.
        for kind in ('sheets','parts'):
            rows = records[kind]
            _require(len({row['id'] for row in rows}) == len(rows),'duplicate network records')
            accounted.extend((side,kind,row['id']) for row in rows)
        for kind in ('members', 'member_edge_uses', 'attachments', 'junctions'):
            rows = inventory[side]['records'][kind]
            _require(sorted(rows, key=lambda row: row['id']) ==
                     sorted(records[kind], key=lambda row: row['id']), 'unaccounted network '+kind)
            _require(len({row['id'] for row in rows}) == len(rows), 'duplicate network records')
            accounted.extend((side, kind, row['id']) for row in rows)
    relation = owner.relations
    traversal = []
    all_uses = set()
    for member in relation['members']:
        ordered = []
        for mapping in member['source_use_mappings']:
            source = mapping['source_member_use']
            certificates = mapping['station_certificates']
            uses = mapping['current_member_uses']
            _require(len(uses) == len(certificates), 'incomplete native station certificates')
            for use, certificate in zip(uses, certificates):
                _require(use['id'] == certificate['member_edge_use'] and
                         use['edge_id'] == certificate['edge'], 'wrong native certificate use')
                _require(use['id'] not in all_uses, 'duplicate current network use')
                all_uses.add(use['id']); ordered.append(use['id'])
                global_span = tuple((Fraction(float(t)).numerator, Fraction(float(t)).denominator)
                                    for t in use['parent_range'])
                traversal.append((member['source_member']['id'], source['id'], use['id'],
                    use['edge_id'], use['orientation'], tuple(tuple(t) for t in certificate['native_parent_interval']),
                    global_span, mapping['source_carrier'], _json(certificate)))
        _require(ordered == [use['id'] for use in member['current_member_uses']],
                 'network traversal order changed')
    _require(all_uses == {row['id'] for row in owner.current_records['member_edge_uses']},
             'lost current network uses')
    points = []
    for point in relation['attachments']:
        old, current = point['source_attachment'], point['current_attachment']
        t = Fraction(float(old['target_parameters'][0][0]))
        local = Fraction(float(current['target_parameters'][0][0]))
        points.append((old['id'], point['current_vertex']['id'], current['target_id'],
            (t.numerator,t.denominator), (local.numerator,local.denominator),
            point['source_carrier'], _json(point)))
    for side in ('original','current'):
        _require(set(inventory[side]['isolated_vertex_ids']) <= {row[1] for row in points},
                 'unaccounted isolated network vertices')
    return PreparedMemberNetworkAccounting(ACCOUNTING_SCHEMA, owner, constraints,
        tuple(sorted(accounted)), tuple(traversal), tuple(points), _json(relation))


def query_prepared_member_network_accounting(geometry, joint_edge_id, *, owner_receipt=None,
        constraint_receipt=None, cancellation_check=None):
    try:
        from anygeometry import (GeometryError, PreparedMemberSheetJointNetwork, to_dict,
            query_prepared_member_sheet_joint_network as query_owner,
            validate_prepared_member_sheet_joint_network_binding as validate_owner,
            query_prepared_authored_constraint_scope as query_constraints,
            validate_prepared_authored_constraint_scope_binding as validate_constraints)
    except ImportError as error:
        raise MeshError('prepared member network capability unavailable') from error
    supplied = (owner_receipt, constraint_receipt)
    pinned = _pin(supplied)
    before = _json(to_dict(geometry))
    _require(_pin(supplied) == pinned, 'network evidence changed during source serialization')
    try:
        for value, validator in ((owner_receipt,validate_owner),(constraint_receipt,validate_constraints)):
            if value is not None:
                validator(geometry,value)
        _require(_pin(supplied) == pinned,'network evidence changed before acquisition')
        owner = owner_receipt if owner_receipt is not None else query_owner(
            geometry,joint_edge_id,cancellation_check=cancellation_check)
        owner_pin = _pin(owner)
        _require(type(owner) is PreparedMemberSheetJointNetwork and type(joint_edge_id) is int
                 and owner.joint_edge_id == joint_edge_id, 'wrong network receipt or joint')
        validate_owner(geometry,owner,cancellation_check=cancellation_check)
        constraints = constraint_receipt if constraint_receipt is not None else query_constraints(
            geometry,owner.authored_face_ids,cancellation_check=cancellation_check)
        constraint_pin = _pin(constraints)
        validate_constraints(geometry,constraints,cancellation_check=cancellation_check)
        _require(_pin(supplied) == pinned and _pin(owner) == owner_pin
                 and _pin(constraints) == constraint_pin,'network evidence changed before derivation')
        result = _derive(owner,constraints)
        result_pin = _pin(result)
        validate_owner(geometry,owner)
        validate_constraints(geometry,constraints)
        _require(_json(to_dict(geometry)) == before and _pin(supplied) == pinned
                 and _pin(result) == result_pin,'network contents changed during closing proof')
        return result
    except GeometryError as error:
        raise MeshError('prepared member network owner refusal: '+str(error)) from error


def validate_prepared_member_network_accounting(geometry, result, *, cancellation_check=None):
    _require(type(result) is PreparedMemberNetworkAccounting and result.schema == ACCOUNTING_SCHEMA,
             'wrong network accounting receipt')
    pin = _pin(result)
    expected = query_prepared_member_network_accounting(geometry,result.owner_receipt.joint_edge_id,
        owner_receipt=result.owner_receipt,constraint_receipt=result.constraint_receipt,
        cancellation_check=cancellation_check)
    _require(_pin(result) == pin and _pin(expected) == pin,'network accounting contents changed')


def _station_pin(proof, owner_type):
    _require(type(proof) is owner_type, 'wrong owner network station proof type')
    return _snapshot(tuple((field.name,object.__getattribute__(proof,field.name))
                           for field in fields(owner_type)))


def query_prepared_member_network_stations(geometry, accounting, *, target_size=2.,
        max_stations=10000, cancellation_check=None):
    try:
        from anygeometry import (GeometryError, AuthoredBoundaryStations,
            query_prepared_authored_boundary_stations as query,
            validate_prepared_authored_boundary_station_coordinates as validate)
    except ImportError as error:
        raise MeshError('prepared member network station capability unavailable') from error
    _require(callable(query) and callable(validate) and isinstance(AuthoredBoundaryStations,type),
             'prepared member network station capability unavailable')
    _require(type(target_size) in (int,float) and isfinite(target_size) and target_size > 0,
             'invalid network station target size')
    _require(type(max_stations) is int and max_stations >= 0,'invalid network station budget')
    pinned = _pin(accounting)
    try:
        validate_prepared_member_network_accounting(geometry,accounting,cancellation_check=cancellation_check)
        rows, edge_parameters, proofs = [], {}, []
        records = {row.edge_id:row for row in accounting.owner_receipt.edge_preimages.records}
        for edge in sorted({row[3] for row in accounting.traversal}):
            if cancellation_check is not None and cancellation_check('member network stations'):
                raise MeshError('member network stations cancelled')
            _require(_pin(accounting) == pinned,'network accounting changed during stations')
            record = records[edge]
            controls = record.current_definition.controls
            _require(len(controls) == 2,'network station carrier is not straight')
            length = sqrt(sum((float(Fraction(*b))-float(Fraction(*a)))**2 for a,b in zip(*controls)))
            ratio = length/target_size
            _require(isfinite(ratio) and ratio <= max_stations,'network station budget exhausted')
            count = max(1,ceil(ratio))
            params = {Fraction(i,count) for i in range(count+1)}
            params.update(Fraction(*point[4]) for point in accounting.points if point[2] == edge)
            params = tuple(sorted(params))
            _require(len(rows)+len(params) <= max_stations,'network station budget exhausted')
            root = record.ancestor.definition.edge_id
            matches = [c for c in accounting.constraint_receipt.boundary_correspondences
                if any(original == root and edge in children for loop in c.exterior_loops
                       for original,_forward,children in loop)]
            _require(bool(matches),'network edge lacks owner boundary occurrence')
            first = None
            for correspondence in sorted(matches,key=lambda c:c.authored_definition.face_id):
                request_pin = _snapshot(correspondence)
                proof = query(geometry,correspondence,edge,params,cancellation_check=cancellation_check)
                proof_pin = _station_pin(proof,AuthoredBoundaryStations)
                _require(type(proof.edge_id) is int and proof.edge_id == edge
                    and _snapshot(correspondence) == request_pin
                    and _snapshot(proof.correspondence) == request_pin,
                    'owner network station proof differs from requested edge or occurrence')
                _require(all(_station_pin(prior,AuthoredBoundaryStations) == prior_pin
                    for prior,prior_pin in proofs),'network station proof changed during later callbacks')
                _require(tuple(Fraction(*t) for t in proof.parameters) == params
                    and all(len(values) == len(params) for values in (proof.authored_parameters,
                        proof.authored_points,proof.current_polynomial_points)), 'network station proof lost sequence')
                coordinates = tuple(tuple(float(Fraction(*v)) for v in xyz) for xyz in proof.current_polynomial_points)
                validate(geometry,proof,coordinates,cancellation_check=cancellation_check)
                _require(all(_station_pin(prior,AuthoredBoundaryStations) == prior_pin
                    for prior,prior_pin in proofs),'network station proof changed during later callbacks')
                _require(_station_pin(proof,AuthoredBoundaryStations) == proof_pin,
                    'network station proof changed')
                proofs.append((proof,proof_pin))
                _require(_pin(accounting) == pinned,'network accounting changed during owner station proof')
                if first is None:
                    first = proof
                    selected_root = correspondence.authored_definition.face_id
                else:
                    _require(proof.current_polynomial_points == first.current_polynomial_points,
                             'shared network station coordinates disagree')
            for t,xyz,ancestor in zip(params,first.current_polynomial_points,first.authored_parameters):
                vertices = {point[1] for point in accounting.points if point[2] == edge and Fraction(*point[4]) == t}
                vertex = (record.current_definition.start if t == 0 else record.current_definition.end
                    if t == 1 else next(iter(vertices)) if len(vertices) == 1 else None)
                rows.append((edge,(t.numerator,t.denominator),vertex,xyz,selected_root,ancestor))
            edge_parameters[edge] = params
        traversals = []
        for row in accounting.traversal:
            p,q = (Fraction(*t) for t in row[6])
            params = edge_parameters[row[3]]
            reverse = row[4] == 'reversed'
            ordered = tuple(reversed(params)) if reverse else params
            sequence = []
            for t in ordered:
                global_t = p+(q-p)*(1-t if reverse else t)
                sequence.append(((t.numerator,t.denominator),(global_t.numerator,global_t.denominator)))
            traversals.append((row,tuple(sequence)))
        result = PreparedMemberNetworkStations(STATION_SCHEMA,accounting,float(target_size),max_stations,
            tuple(rows),tuple(traversals),accounting.points)
        pin = _pin(result)
        validate_prepared_member_network_accounting(geometry,accounting)
        _require(all(_station_pin(proof,AuthoredBoundaryStations) == proof_pin
            for proof,proof_pin in proofs),'network station proof changed during later callbacks')
        _require(_pin(accounting) == pinned and _pin(result) == pin,'network station closing proof changed')
        return result
    except GeometryError as error:
        raise MeshError('prepared member network station owner refusal: '+str(error)) from error


def validate_prepared_member_network_stations(geometry, result, *, max_stations=None,
        cancellation_check=None):
    _require(type(result) is PreparedMemberNetworkStations and result.schema == STATION_SCHEMA,
             'wrong network station receipt')
    pin = _pin(result)
    if max_stations is not None:
        _require(type(max_stations) is int and 0 <= max_stations <= result.max_stations
                 and len(result.stations) <= max_stations,'network validation budget may only decrease')
    expected = query_prepared_member_network_stations(geometry,result.accounting,
        target_size=result.target_size,max_stations=result.max_stations,cancellation_check=cancellation_check)
    _require(_pin(result) == pin and _pin(expected) == pin,'network station contents changed')
