"""Experimental connected planar member/Sheet network consumer.

Private additive consumer of the public whole-document
``PreparedPlanarMemberSheetNetwork`` owner receipt.  It freshly revalidates the
owner binding, accounts every source/current structural record, native
multi-use Member station mapping, original point attachments and the complete
generated member-joint/point-contact inventory, then prepares owner-certified
canonical stations through the public ``AuthoredMaterialStations`` contract.
The mesh stage generates the shell first through the existing quad-first route
with private canonical station overrides, then builds B2/B3 beam bodies from
the shell's canonical edge chains with shared corner and midside node IDs.

Nothing here grants beam discretization, load transfer, solver, external
reference or publication authority; the produced mesh is provisional.  Ideal
chart ancestry (owner material stations) is kept separate from the native
persisted Member station mapping.  Aliases are retained through explicit
source-point/target relations, never proximity or coordinate welding.
"""
from dataclasses import dataclass, fields
from fractions import Fraction
from math import ceil, isfinite, sqrt
from uuid import UUID

from ._prepared_member_accounting import _json, _require
from .errors import MeshError

PLANAR_ACCOUNTING_SCHEMA = 'anymesher.prepared-planar-network-accounting-v1'
PLANAR_STATION_SCHEMA = 'anymesher.prepared-planar-network-stations-v1'
PLANAR_MESH_SCHEMA = 'anymesher.prepared-planar-network-mesh-v1'


@dataclass(frozen=True, slots=True)
class PreparedPlanarNetworkAccounting:
    schema: str
    owner_receipt: object
    constraint_receipt: object
    accounted_records: tuple
    # member, original use, current use, edge, orientation, native carrier
    # interval, exact persisted global Member interval, original carrier,
    # full certificate JSON (includes retained error bounds).
    traversal: tuple
    # original attachment, point vertex, current edge, original/native station,
    # current local station, original carrier, full relation JSON.
    points: tuple
    # attachment, member, junction, sheets, boundary faces, exact member range,
    # target interval, carrier orientation, source carrier, full joint JSON.
    member_joints: tuple
    # attachment, member, junction, junction kind, face, material root, sheet,
    # current carrier, exact member station, target UV, source carrier, JSON.
    point_contacts: tuple
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
class PreparedPlanarNetworkStations:
    schema: str
    accounting: object
    target_size: float
    max_stations: int
    # edge, exact local station, canonical endpoint or unique attachment
    # vertex, exact owner current XYZ, authored root face, trace kind.
    stations: tuple
    # traversal row, ordered exact (local, global Member) station pairs.
    traversal_stations: tuple
    # attachment, source point vertex, current edge, local/native station.
    point_intents: tuple
    # attachment, member, canonical junction vertex, face, sheet, exact
    # member station, exact carrier-local station.
    contact_anchors: tuple

    @property
    def mesh_qualified(self):
        return False

    @property
    def publication_qualified(self):
        return False


@dataclass(frozen=True, slots=True)
class PreparedPlanarNetworkMesh:
    schema: str
    stations: object
    shell: object
    order: str
    options_json: str
    certification_mode: str
    # member, ordered shared node chain (corners, then midsides when quadratic).
    member_chains: tuple
    # beam element ID, member, node body (B2 linear, B3 quadratic).
    beams: tuple
    # attachment, member, anchor node, contact face, sheet, exact member station.
    contact_anchors: tuple
    # attachment, member, junction, sheets, boundary faces.
    member_joints: tuple

    @property
    def provisional(self):
        return True

    @property
    def mesh_qualified(self):
        return False

    @property
    def beam_discretization_qualified(self):
        return False

    @property
    def load_transfer_qualified(self):
        return False

    @property
    def external_reference_transfer_qualified(self):
        return False

    @property
    def solver_admitted(self):
        return False

    @property
    def publication_qualified(self):
        return False


def _pin(value):
    """Pin plain immutable evidence without invoking copy/equality/user hooks.

    Only the actual owner dataclasses of the planar contract and this module's
    receipts are admissible; containers and scalar subclasses cannot smuggle
    hooks into capture.  The returned tree is exclusively builtin immutables.
    """
    try:
        from anygeometry import (
            PreparedPlanarMemberSheetNetwork, PreparedModelScope,
            PreparedFacePreimages, AuthoredFaceDefinition,
            PreparedEdgeSubcurvePreimages, EdgeSubcurvePreimage,
            PolynomialEdgeAncestor, PolynomialEdgeDefinition,
            PreparedAuthoredConstraintScope, AuthoredBoundaryCorrespondence,
            AuthoredMaterialStations)
    except ImportError as error:
        raise MeshError('prepared planar network capability unavailable') from error
    known = (PreparedPlanarMemberSheetNetwork, PreparedModelScope,
             PreparedFacePreimages, AuthoredFaceDefinition,
             PreparedEdgeSubcurvePreimages, EdgeSubcurvePreimage,
             PolynomialEdgeAncestor, PolynomialEdgeDefinition,
             PreparedAuthoredConstraintScope, AuthoredBoundaryCorrespondence,
             AuthoredMaterialStations, PreparedPlanarNetworkAccounting,
             PreparedPlanarNetworkStations)
    shapes = tuple(tuple(field.name for field in fields(cls)) for cls in known)
    active = set()

    def pin(item, depth):
        _require(depth <= 64, 'malformed planar network evidence nesting')
        cls = type(item)
        if any(cls is primitive for primitive in (type(None), bool, int, str)):
            return (cls.__name__, item)
        if cls is float:
            return ('float', item.hex())
        if cls is UUID:
            return ('UUID', pin(object.__getattribute__(item, 'int'), depth+1))
        index = next((i for i, candidate in enumerate(known) if cls is candidate), None)
        _require(cls is tuple or index is not None,
                 'planar network evidence requires plain immutable fields')
        identity = id(item)
        _require(identity not in active, 'cyclic planar network evidence')
        active.add(identity)
        try:
            if cls is tuple:
                return ('tuple', tuple(pin(child, depth+1) for child in item))
            return ('owner-dataclass', index, tuple(
                (name, pin(object.__getattribute__(item, name), depth+1))
                for name in shapes[index]))
        finally:
            active.remove(identity)

    return pin(value, 0)


def _owner_capabilities():
    try:
        from anygeometry import (
            GeometryError, PreparedPlanarMemberSheetNetwork, to_dict,
            query_prepared_planar_member_sheet_network as query_owner,
            validate_prepared_planar_member_sheet_network_binding as validate_owner,
            query_prepared_authored_constraint_scope as query_constraints,
            validate_prepared_authored_constraint_scope_binding as validate_constraints)
    except ImportError as error:
        raise MeshError('prepared planar network capability unavailable') from error
    _require(callable(query_owner) and callable(validate_owner)
             and callable(query_constraints) and callable(validate_constraints)
             and isinstance(PreparedPlanarMemberSheetNetwork, type),
             'prepared planar network capability unavailable')
    return (GeometryError, PreparedPlanarMemberSheetNetwork, to_dict,
            query_owner, validate_owner, query_constraints, validate_constraints)


def _derive(owner, constraints):
    _require(owner.bounded_relation_mapping_qualified and owner.occurrence_mapping_qualified
             and not owner.semantic_mapping_qualified and not owner.beam_discretization_qualified
             and not owner.load_transfer_qualified
             and not owner.external_reference_transfer_qualified
             and not owner.solver_qualified and not owner.publication_qualified,
             'unexpected planar network authority')
    inventory = None
    if constraints is not None:
        _require(owner.scope == constraints.scope
                 and tuple(owner.authored_face_ids) == constraints.selected_root_ids
                 and tuple(owner.current_face_ids) == constraints.current_face_ids
                 and not constraints.outside_root_ids,
                 'incomplete planar network inventory selection')
        _require(constraints.typed_inventory_complete
                 and not constraints.semantic_mapping_qualified
                 and not constraints.parameter_remapping_qualified
                 and not constraints.publication_qualified,
                 'unexpected planar network inventory authority')
        inventory = constraints.inventory
    relation = owner.relations
    _require(sorted(relation) == ['attachments', 'member_joints', 'members',
                                  'point_contacts'],
             'unexpected planar network relation payload')
    accounted = []
    for side, records in (('original', owner.source_records),
                          ('current', owner.current_records)):
        # Sheets/Parts are complete in the whole-document owner receipt; the
        # constraint inventory intentionally exposes only structural records.
        for kind in ('sheets', 'parts'):
            rows = records[kind]
            _require(len({row['id'] for row in rows}) == len(rows),
                     'duplicate planar network records')
            accounted.extend((side, kind, row['id']) for row in rows)
        for kind in ('members', 'member_edge_uses', 'attachments', 'junctions'):
            rows = records[kind]
            _require(len({row['id'] for row in rows}) == len(rows),
                     'duplicate planar network records')
            if inventory is not None:
                _require(sorted(rows, key=lambda row: row['id']) ==
                         sorted(inventory[side]['records'][kind],
                                key=lambda row: row['id']),
                         'unaccounted planar network '+kind)
            accounted.extend((side, kind, row['id']) for row in rows)
    traversal = []
    all_uses = set()
    for member in relation['members']:
        _require(member.get('member_role') == 'straight_planar_chain',
                 'unsupported planar network member role')
        _require(len(member.get('material_memberships', ())) ==
                 len(member['source_carriers']),
                 'incomplete planar material membership proof')
        ordered = []
        for mapping in member['source_use_mappings']:
            source = mapping['source_member_use']
            certificates = mapping['station_certificates']
            uses = mapping['current_member_uses']
            _require(len(uses) == len(certificates),
                     'incomplete native planar station certificates')
            p, q = (Fraction(*value) for value in mapping['source_span'])
            for use, certificate in zip(uses, certificates):
                _require(use['id'] == certificate['member_edge_use'] and
                         use['edge_id'] == certificate['edge'],
                         'wrong native planar certificate use')
                _require(use['id'] not in all_uses,
                         'duplicate current planar network use')
                all_uses.add(use['id'])
                ordered.append(use['id'])
                u, v = (Fraction(float(t)) for t in use['parent_range'])
                expected = ((1-(v-p)/(q-p), 1-(u-p)/(q-p))
                            if mapping['orientation'] == 'reversed'
                            else ((u-p)/(q-p), (v-p)/(q-p)))
                _require(tuple(Fraction(*t) for t in certificate['native_parent_interval'])
                         == expected,
                         'native planar station mapping changed')
                global_span = tuple((Fraction(float(t)).numerator,
                                     Fraction(float(t)).denominator)
                                    for t in use['parent_range'])
                traversal.append((member['source_member']['id'], source['id'],
                    use['id'], use['edge_id'], use['orientation'],
                    tuple(tuple(t) for t in certificate['native_parent_interval']),
                    global_span, mapping['source_carrier'], _json(certificate)))
        _require(ordered == [use['id'] for use in member['current_member_uses']],
                 'planar network traversal order changed')
    _require(all_uses == {row['id'] for row in owner.current_records['member_edge_uses']},
             'lost current planar network uses')
    points = []
    for point in relation['attachments']:
        old, current = point['source_attachment'], point['current_attachment']
        t = Fraction(float(old['target_parameters'][0][0]))
        local = Fraction(float(current['target_parameters'][0][0]))
        points.append((old['id'], point['current_vertex']['id'], current['target_id'],
            (t.numerator, t.denominator), (local.numerator, local.denominator),
            point['source_carrier'], _json(point)))
    if inventory is not None:
        for side in ('original', 'current'):
            _require(set(inventory[side]['isolated_vertex_ids'])
                     <= {row[1] for row in points},
                     'unaccounted isolated planar network vertices')
    current_junctions = {row['id']: row for row in owner.current_records['junctions']}
    current_attachments = {row['id']: row for row in owner.current_records['attachments']}
    member_joints = []
    for joint in relation['member_joints']:
        _require(joint['joint_role'] == 'member_on_face_boundary',
                 'unsupported planar member joint role')
        attachment, junction = joint['attachment'], joint['junction']
        _require(attachment['id'] in current_attachments
                 and junction['id'] in current_junctions
                 and current_attachments[attachment['id']]['kind'] == 'member_on_face_boundary'
                 and current_junctions[junction['id']]['kind'] == 'overlap'
                 and current_junctions[junction['id']]['attachment_ids'] == [attachment['id']],
                 'unaccounted generated planar member joint')
        member_joints.append((attachment['id'], joint['member'], junction['id'],
            tuple(joint['sheets']), tuple(joint['boundary_faces']),
            tuple(tuple(t) for t in joint['member_range']),
            tuple(tuple(t) for t in joint['target_interval']),
            joint['carrier_orientation'], joint['source_carrier'], _json(joint)))
    _require(len({row[2] for row in member_joints}) == len(member_joints)
             and {row[2] for row in member_joints} ==
             {key for key, row in current_junctions.items() if row['kind'] == 'overlap'},
             'incomplete generated planar member junction inventory')
    _require(len({row[0] for row in member_joints}) == len(member_joints)
             and {row[0] for row in member_joints} ==
             {key for key, row in current_attachments.items()
              if row['kind'] == 'member_on_face_boundary'},
             'incomplete generated planar member attachment inventory')
    point_contacts = []
    for contact in relation['point_contacts']:
        _require(contact['joint_role'] == 'member_through_face',
                 'unsupported planar point contact role')
        attachment, junction = contact['attachment'], contact['junction']
        _require(attachment['id'] in current_attachments
                 and junction['id'] in current_junctions
                 and current_attachments[attachment['id']]['kind'] == 'member_through_face'
                 and current_junctions[junction['id']]['kind'] in
                 ('crossing', 'endpoint', 'multi_way')
                 and attachment['id'] in current_junctions[junction['id']]['attachment_ids'],
                 'unaccounted generated planar point contact')
        point_contacts.append((attachment['id'], contact['member'], junction['id'],
            junction['kind'], contact['face'], contact['material_root'],
            contact['sheet'], contact['current_carrier'],
            tuple(contact['member_station']),
            tuple(tuple(t) for t in contact['target_uv']),
            contact['source_carrier'], _json(contact)))
    _require(len({row[2] for row in point_contacts}) == len(point_contacts)
             and {row[2] for row in point_contacts} ==
             {key for key, row in current_junctions.items()
              if row['kind'] in ('crossing', 'endpoint', 'multi_way')},
             'incomplete generated planar point junction inventory')
    _require(len({row[0] for row in point_contacts}) == len(point_contacts)
             and {row[0] for row in point_contacts} ==
             {key for key, row in current_attachments.items()
              if row['kind'] == 'member_through_face'},
             'incomplete generated planar point attachment inventory')
    accounted_attachments = ({row[0] for row in points}
                             | {row[0] for row in member_joints}
                             | {row[0] for row in point_contacts}
                             | {key for key, row in current_attachments.items()
                                if row['kind'] == 'sheet_on_joint'})
    _require(accounted_attachments == set(current_attachments),
             'unaccounted planar network attachments')
    accounted_junctions = ({row[2] for row in member_joints}
                           | {row[2] for row in point_contacts}
                           | {key for key, row in current_junctions.items()
                              if row['kind'] == 'sheet_joint'})
    _require(accounted_junctions == set(current_junctions),
             'unaccounted planar network junctions')
    return PreparedPlanarNetworkAccounting(PLANAR_ACCOUNTING_SCHEMA, owner,
        constraints, tuple(sorted(accounted)), tuple(traversal), tuple(points),
        tuple(member_joints), tuple(point_contacts), _json(relation))


def query_prepared_planar_network_accounting(geometry, joint_edge_id, *,
        owner_receipt=None, constraint_receipt=None, cancellation_check=None):
    """Account the whole connected planar document from a fresh owner proof.

    Supplied evidence is untrusted: it is pinned callback-free, validated
    against the live owner, and re-pinned across every closing callback.  The
    optional constraint receipt cross-checks the typed inventory; connected
    documents whose unified shared exterior edges the boundary-correspondence
    contract refuses are accounted from the self-sufficient whole-document
    owner receipt alone and recorded with ``constraint_receipt`` None.
    """
    (GeometryError, Receipt, to_dict, query_owner, validate_owner,
     query_constraints, validate_constraints) = _owner_capabilities()
    supplied = (owner_receipt, constraint_receipt)
    pinned = _pin(supplied)
    before = _json(to_dict(geometry))
    _require(_pin(supplied) == pinned,
             'planar network evidence changed during source serialization')
    try:
        for value, validator in ((owner_receipt, validate_owner),
                                 (constraint_receipt, validate_constraints)):
            if value is not None:
                validator(geometry, value)
        _require(_pin(supplied) == pinned,
                 'planar network evidence changed before acquisition')
        owner = owner_receipt if owner_receipt is not None else query_owner(
            geometry, joint_edge_id, cancellation_check=cancellation_check)
        owner_pin = _pin(owner)
        _require(type(owner) is Receipt and type(joint_edge_id) is int
                 and owner.joint_edge_id == joint_edge_id,
                 'wrong planar network receipt or joint')
        validate_owner(geometry, owner, cancellation_check=cancellation_check)
        constraints = constraint_receipt
        if constraints is None:
            # The whole-document owner receipt is self-sufficient; the
            # correspondence scope is an additional cross-check that some
            # connected documents cannot produce.
            try:
                constraints = query_constraints(geometry, owner.authored_face_ids,
                                                cancellation_check=cancellation_check)
            except GeometryError as error:
                # Only the retained, specific shared-exterior ancestry gap is
                # optional. Cancellation and unrelated owner failures remain
                # failures; they must never masquerade as missing inventory.
                if str(error) != 'authored boundary correspondence has a different original anchor':
                    raise
                constraints = None
        if constraints is not None:
            constraint_pin = _pin(constraints)
            validate_constraints(geometry, constraints,
                                 cancellation_check=cancellation_check)
        _require(_pin(supplied) == pinned and _pin(owner) == owner_pin,
                 'planar network evidence changed before derivation')
        result = _derive(owner, constraints)
        result_pin = _pin(result)
        validate_owner(geometry, owner, cancellation_check=cancellation_check)
        if constraints is not None:
            validate_constraints(geometry, constraints, cancellation_check=cancellation_check)
        _require(_json(to_dict(geometry)) == before and _pin(supplied) == pinned
                 and _pin(result) == result_pin,
                 'planar network contents changed during closing proof')
        return result
    except GeometryError as error:
        raise MeshError('prepared planar network owner refusal: '+str(error)) from error


def validate_prepared_planar_network_accounting(geometry, result, *,
        cancellation_check=None):
    _require(type(result) is PreparedPlanarNetworkAccounting
             and result.schema == PLANAR_ACCOUNTING_SCHEMA,
             'wrong planar network accounting receipt')
    pin = _pin(result)
    expected = query_prepared_planar_network_accounting(
        geometry, result.owner_receipt.joint_edge_id,
        owner_receipt=result.owner_receipt,
        constraint_receipt=result.constraint_receipt,
        cancellation_check=cancellation_check)
    _require(_pin(result) == pin and _pin(expected) == pin,
             'planar network accounting contents changed')


def query_prepared_planar_network_stations(geometry, accounting, *, target_size=2.,
        max_stations=10000, cancellation_check=None):
    """Prepare owner-certified canonical stations; assign no mesh node IDs.

    Every relevant exterior or paired-interior correspondence of a Member
    carrier is queried through the public ``AuthoredMaterialStations`` contract
    with exact rational parameters; proof type, requested edge/correspondence,
    sequence/count and all fields are pinned before consumption and across
    every closing callback.  Mandatory source-point anchors may be nonuniform,
    including endpoints; their aliases stay explicit point relations.
    """
    try:
        from anygeometry import (GeometryError, AuthoredMaterialStations,
            query_prepared_authored_material_stations as query,
            validate_prepared_authored_material_station_coordinates as validate)
    except ImportError as error:
        raise MeshError('prepared planar network station capability unavailable') from error
    _require(callable(query) and callable(validate)
             and isinstance(AuthoredMaterialStations, type),
             'prepared planar network station capability unavailable')
    _require(type(target_size) in (int, float) and isfinite(target_size)
             and target_size > 0,
             'invalid planar network station target size')
    _require(type(max_stations) is int and max_stations >= 0,
             'invalid planar network station budget')
    _require(type(accounting) is PreparedPlanarNetworkAccounting,
             'wrong planar network accounting receipt')
    _require(accounting.constraint_receipt is not None,
             'planar network stations require qualified boundary correspondences')
    pinned = _pin(accounting)
    try:
        validate_prepared_planar_network_accounting(
            geometry, accounting, cancellation_check=cancellation_check)
        _require(_pin(accounting) == pinned,
                 'planar network accounting changed before station planning')
        correspondences = accounting.constraint_receipt.boundary_correspondences
        records = {row.edge_id: row
                   for row in accounting.owner_receipt.edge_preimages.records}
        rows, edge_parameters, proofs = [], {}, []
        for edge in sorted({row[3] for row in accounting.traversal}):
            if cancellation_check is not None and cancellation_check(
                    'planar network stations'):
                raise MeshError('planar network stations cancelled')
            _require(_pin(accounting) == pinned,
                     'planar network accounting changed during stations')
            record = records.get(edge)
            _require(record is not None,
                     'planar network carrier lacks retained ancestry')
            controls = record.current_definition.controls
            _require(len(controls) == 2,
                     'planar network station carrier is not straight')
            length = sqrt(sum((float(Fraction(*b))-float(Fraction(*a)))**2
                              for a, b in zip(*controls)))
            ratio = length/target_size
            _require(isfinite(ratio) and ratio <= max_stations,
                     'planar network station budget exhausted')
            count = max(1, ceil(ratio))
            params = {Fraction(i, count) for i in range(count+1)}
            params.update(Fraction(*point[4]) for point in accounting.points
                          if point[2] == edge)
            params = tuple(sorted(params))
            _require(len(rows)+len(params) <= max_stations,
                     'planar network station budget exhausted')
            matches = [correspondence for correspondence in correspondences
                       if edge in {child for loop in correspondence.exterior_loops
                                   for _original, _forward, children in loop
                                   for child in children}
                       or edge in dict(correspondence.interior_incidence)]
            _require(matches, 'planar network edge lacks an owner material trace')
            first = None
            for correspondence in sorted(matches,
                    key=lambda item: item.authored_definition.face_id):
                request_pin = _pin(correspondence)
                proof = query(geometry, correspondence, edge, params,
                              cancellation_check=cancellation_check)
                proof_pin = _pin(proof)
                _require(type(proof) is AuthoredMaterialStations
                         and type(proof.edge_id) is int and proof.edge_id == edge
                         and _pin(correspondence) == request_pin
                         and _pin(proof.correspondence) == request_pin,
                         'owner material station proof differs from the '
                         'requested edge or correspondence')
                _require(all(_pin(prior) == prior_pin
                             for prior, prior_pin in proofs),
                         'planar network station proof changed during later callbacks')
                _require(tuple(Fraction(*t) for t in proof.parameters) == params
                         and all(len(values) == len(params) for values in
                                 (proof.authored_uv, proof.authored_points,
                                  proof.current_points)),
                         'owner material station proof lost sequence or count')
                coordinates = tuple(tuple(float(Fraction(*v)) for v in xyz)
                                   for xyz in proof.current_points)
                validate(geometry, proof, coordinates,
                         cancellation_check=cancellation_check)
                _require(all(_pin(prior) == prior_pin
                             for prior, prior_pin in proofs),
                         'planar network station proof changed during later callbacks')
                _require(_pin(proof) == proof_pin,
                         'planar network station proof changed')
                proofs.append((proof, proof_pin))
                _require(_pin(accounting) == pinned,
                         'planar network accounting changed during station proof')
                if first is None:
                    first = proof
                    selected_root = correspondence.authored_definition.face_id
                    trace_kind = proof.trace_kind
                else:
                    _require(proof.current_points == first.current_points,
                             'shared planar network station coordinates disagree')
            for t, xyz in zip(params, first.current_points):
                vertices = {point[1] for point in accounting.points
                            if point[2] == edge and Fraction(*point[4]) == t}
                vertex = (record.current_definition.start if t == 0 else
                          record.current_definition.end if t == 1 else
                          next(iter(vertices)) if len(vertices) == 1 else None)
                rows.append((edge, (t.numerator, t.denominator), vertex, xyz,
                             selected_root, trace_kind))
            edge_parameters[edge] = params
        traversals = []
        for row in accounting.traversal:
            p, q = (Fraction(*t) for t in row[6])
            params = edge_parameters[row[3]]
            reverse = row[4] == 'reversed'
            ordered = tuple(reversed(params)) if reverse else params
            sequence = []
            for t in ordered:
                global_t = p+(q-p)*(1-t if reverse else t)
                sequence.append(((t.numerator, t.denominator),
                                 (global_t.numerator, global_t.denominator)))
            traversals.append((row, tuple(sequence)))
        anchors = []
        for contact in accounting.point_contacts:
            (attachment, member, _junction, _kind, face, _root, sheet, carrier,
             station, _uv, _source, _row) = contact
            p = Fraction(*station)
            uses = [row for row in accounting.traversal
                    if row[0] == member and row[3] == carrier]
            _require(len(uses) == 1,
                     'planar network point contact carrier is not a unique member use')
            u, v = (Fraction(*t) for t in uses[0][6])
            _require(u <= p <= v,
                     'planar network point contact leaves its carrier use')
            local = (p-u)/(v-u)
            if uses[0][4] == 'reversed':
                local = 1-local
            _require(local == 0 or local == 1,
                     'planar network point contact is not a canonical carrier endpoint')
            vertex = (records[carrier].current_definition.start if local == 0
                      else records[carrier].current_definition.end)
            anchors.append((attachment, member, vertex, face, sheet,
                            (p.numerator, p.denominator),
                            (local.numerator, local.denominator)))
        result = PreparedPlanarNetworkStations(PLANAR_STATION_SCHEMA, accounting,
            float(target_size), max_stations, tuple(rows), tuple(traversals),
            tuple(accounting.points), tuple(anchors))
        pin = _pin(result)
        validate_prepared_planar_network_accounting(geometry, accounting,
                                                   cancellation_check=cancellation_check)
        _require(all(_pin(proof) == proof_pin for proof, proof_pin in proofs),
                 'planar network station proof changed during later callbacks')
        _require(_pin(accounting) == pinned and _pin(result) == pin,
                 'planar network station closing proof changed')
        return result
    except GeometryError as error:
        raise MeshError('prepared planar network station owner refusal: '
                        + str(error)) from error


def validate_prepared_planar_network_stations(geometry, result, *, max_stations=None,
        cancellation_check=None):
    _require(type(result) is PreparedPlanarNetworkStations
             and result.schema == PLANAR_STATION_SCHEMA,
             'wrong planar network station receipt')
    pinned = _pin(result)
    if max_stations is not None:
        _require(type(max_stations) is int and 0 <= max_stations <= result.max_stations
                 and len(result.stations) <= max_stations,
                 'planar network validation budget may only decrease')
    expected = query_prepared_planar_network_stations(
        geometry, result.accounting, target_size=result.target_size,
        max_stations=result.max_stations, cancellation_check=cancellation_check)
    _require(_pin(result) == pinned and _pin(expected) == pinned,
             'planar network station contents changed')


def _mesh_content(result):
    """Deterministic JSON view of the provisional shell/beam connectivity."""
    mesh = result.shell.mesh
    return _json({
        'geometry_model_id': str(mesh.geometry_model_id),
        'geometry_revision': mesh.geometry_revision,
        'order': mesh.order,
        'nodes': {int(node): [float(value) for value in position]
                  for node, position in sorted(mesh.nodes.items())},
        'quads': {int(item): [int(node) for node in body]
                  for item, body in sorted(mesh.quads.items())},
        'tris': {int(item): [int(node) for node in body]
                 for item, body in sorted(mesh.tris.items())},
        'beams': {int(item): [int(node) for node in body]
                  for item, body in sorted(mesh.beams.items())},
        'node_of_vertex': {int(key): int(value)
                           for key, value in sorted(mesh.node_of_vertex.items())},
        'nodes_of_edge': {int(key): [int(node) for node in value]
                          for key, value in sorted(mesh.nodes_of_edge.items())},
        'elements_of_face': {int(key): [int(item) for item in value]
                             for key, value in sorted(mesh.elements_of_face.items())},
        'elements_of_sheet': {int(key): [int(item) for item in value]
                              for key, value in sorted(mesh.elements_of_sheet.items())},
        'elements_of_edge': {int(key): [int(item) for item in value]
                             for key, value in sorted(mesh.elements_of_edge.items())},
        'declared_plate_junction_edges': mesh.declared_plate_junction_edges,
        'coupling_ids': sorted(mesh.couplings),
        'thickness_of_face': mesh.thickness_of_face,
        'elements_of_member': {int(key): [int(item) for item in value]
                               for key, value in sorted(mesh.elements_of_member.items())},
        'nodes_of_member': {int(key): [int(node) for node in value]
                            for key, value in sorted(mesh.nodes_of_member.items())},
    })


def _mesh_result_pin(result):
    import json
    options = json.loads(result.options_json)
    return (_pin((result.schema, result.stations, result.order, result.member_chains,
                  result.beams, result.contact_anchors, result.member_joints,
                  result.certification_mode)), _mesh_content(result), options)


def generate_prepared_planar_network_mesh(geometry, stations, *, order='linear',
        options=None, certification_mode='none', cancellation_check=None):
    """Generate the provisional shell first, then B2/B3 beams from its chains.

    The shell runs through the existing quad-first route with private canonical
    station overrides on Member carrier edges, so shell boundary nodes on
    those edges are exactly the owner-certified stations.  Beam bodies reuse
    the shell's canonical edge chains, preserving shared corner and midside
    node IDs; Member groups are rebuilt by authoritative use orientation.
    Contact anchors use the owner canonical junction vertex and its face
    association.  The return is atomic: any refusal or cancellation leaves no
    partially published mesh, and no solver/load/publication authority is
    granted.
    """
    import numpy as np

    from .hybrid import CertificationMode, _quad_first_execute
    from .quad.options import QuadMeshingOptions
    from anygeometry import to_dict
    def source_binding():
        return (geometry.model_id, geometry.revision, _json(to_dict(geometry)))
    source_before = source_binding()
    _require(type(stations) is PreparedPlanarNetworkStations
             and stations.schema == PLANAR_STATION_SCHEMA,
             'wrong planar network station receipt')
    _require(order in ('linear', 'quadratic'),
             'planar network mesh order must be linear or quadratic')
    try:
        mode = CertificationMode(str(certification_mode))
    except ValueError as error:
        raise MeshError('unknown planar network certification mode') from error
    selected = options if options is not None else QuadMeshingOptions(
        quality_model='shape_jacobian')
    _require(type(selected) is QuadMeshingOptions,
             'planar network mesh options must be QuadMeshingOptions')
    pinned = _pin(stations)
    validate_prepared_planar_network_stations(geometry, stations,
                                              cancellation_check=cancellation_check)
    _require(_pin(stations) == pinned,
             'planar network stations changed before mesh generation')
    _require(source_binding() == source_before,
             'planar network source changed before mesh generation')
    canonical = {}
    for row in stations.stations:
        canonical.setdefault(row[0], []).append(float(Fraction(*row[1])))
    canonical = {edge: tuple(values) for edge, values in canonical.items()}
    accounting = stations.accounting
    shell = _quad_first_execute(
        geometry,
        face_ids=tuple(accounting.owner_receipt.current_face_ids),
        target_size=stations.target_size,
        certification_mode=mode,
        options=selected,
        capabilities=None,
        order=order,
        overrides={edge: len(values)-1 for edge, values in canonical.items()},
        _canonical_stations=canonical,
        cancellation_check=cancellation_check,
    )
    mesh = shell.mesh
    _require(source_binding() == source_before,
             'planar network source changed during mesh generation')
    _require(shell.certifiable, 'planar network shell result is not certifiable')
    for face_id in accounting.owner_receipt.current_face_ids:
        _require(mesh.elements_of_face.get(face_id),
                 'planar network shell lost a current face')
    expected_chain = (lambda count: 2*count-1) if order == 'quadratic' else (lambda count: count)
    by_member = {}
    for row in accounting.traversal:
        by_member.setdefault(row[0], []).append(row)
    member_chains, beams, joint_rows = [], [], []
    elements_of_member, nodes_of_member = {}, {}
    _require(not mesh.beams and not mesh.couplings,
             'planar network shell unexpectedly contains structural bodies')
    next_beam_id = max((*mesh.quads, *mesh.tris), default=-1)+1
    for member in sorted(by_member):
        chain = []
        for index, row in enumerate(by_member[member]):
            edge, orientation = row[3], row[4]
            sequence = mesh.nodes_of_edge.get(edge)
            _require(sequence,
                     'planar network carrier edge is not a shell boundary chain')
            _require(len(sequence) == expected_chain(len(canonical[edge])),
                     'shell boundary chain does not match the owner stations')
            piece = list(sequence) if orientation == 'forward' else list(reversed(sequence))
            if index == 0:
                chain.extend(piece)
            else:
                _require(chain[-1] == piece[0],
                         'planar network member chain lacks a shared canonical junction node')
                chain.extend(piece[1:])
        step = 3 if order == 'quadratic' else 2
        _require(len(chain) >= step, 'planar network member chain is too short')
        for start in range(0, len(chain)-1, step-1):
            body = tuple(int(node) for node in chain[start:start+step])
            _require(len(body) == step, 'planar network beam body is incomplete')
            beams.append((next_beam_id, member, body))
            elements_of_member.setdefault(member, []).append(next_beam_id)
            next_beam_id += 1
        member_chains.append((member, tuple(int(node) for node in chain)))
        nodes_of_member[member] = [int(node) for node in chain]
    for element_id, _member, body in beams:
        mesh.beams[element_id] = body
    mesh.elements_of_member = elements_of_member
    mesh.nodes_of_member = nodes_of_member
    anchors = []
    for anchor in stations.contact_anchors:
        attachment, member, vertex, face, sheet, station, _local = anchor
        node = mesh.node_of_vertex.get(vertex)
        _require(node is not None,
                 'planar network contact vertex has no shell node')
        anchors.append((attachment, member, int(node), face, sheet, station))
    for joint in accounting.member_joints:
        joint_rows.append((joint[0], joint[1], joint[2], joint[3], joint[4]))
    if order == 'quadratic':
        _require(mesh.is_quadratic,
                 'planar network quadratic order was not promoted')
        _require(all(len(body) == 8 for body in mesh.quads.values()),
                 'planar network Q8 body is malformed')
        _require(all(len(body) == 6 for body in mesh.tris.values()),
                 'planar network T6 body is malformed')
        _require(all(len(body) == 3 for body in mesh.beams.values()),
                 'planar network B3 body is malformed')
    else:
        _require(all(len(body) == 4 for body in mesh.quads.values()),
                 'planar network Q4 body is malformed')
        _require(all(len(body) == 3 for body in mesh.tris.values()),
                 'planar network T3 body is malformed')
        _require(all(len(body) == 2 for body in mesh.beams.values()),
                 'planar network B2 body is malformed')
    coordinates = np.asarray(list(mesh.nodes.values()), dtype=float)
    _require(coordinates.size and bool(np.isfinite(coordinates).all()),
             'planar network mesh is not finite')
    result = PreparedPlanarNetworkMesh(PLANAR_MESH_SCHEMA, stations, shell,
        str(order), _json({field.name: getattr(selected, field.name)
                           for field in fields(selected)}),
        str(mode.value), tuple(member_chains), tuple(beams), tuple(anchors),
        tuple(joint_rows))
    _require(_pin(stations) == pinned,
             'planar network stations changed during mesh generation')
    if cancellation_check is not None and cancellation_check('planar network closing'):
        raise MeshError('planar network mesh cancelled')
    _require(source_binding() == source_before and _pin(stations) == pinned,
             'planar network source or stations changed during closing publication')
    return result


def validate_prepared_planar_network_mesh(geometry, result, *,
        cancellation_check=None):
    """Rederive the provisional mesh; no digest-only approval."""
    import json

    from .quad.options import QuadMeshingOptions
    _require(type(result) is PreparedPlanarNetworkMesh
             and result.schema == PLANAR_MESH_SCHEMA,
             'wrong planar network mesh receipt')
    pin = _mesh_result_pin(result)
    expected = generate_prepared_planar_network_mesh(
        geometry, result.stations, order=result.order,
        options=QuadMeshingOptions(**json.loads(result.options_json)),
        certification_mode=result.certification_mode,
        cancellation_check=cancellation_check)
    _require(_mesh_result_pin(result) == pin and _mesh_result_pin(expected) == pin,
             'planar network mesh contents changed')
