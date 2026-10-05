"""Finite owner-relation accounting; no mesh generation or publication authority."""

from dataclasses import dataclass, fields
from fractions import Fraction
import json
from uuid import UUID

from .errors import MeshError

SCHEMA = "anymesher.prepared-member-accounting-v1"
_KINDS = ("members", "member_edge_uses", "attachments", "junctions")


def _require(condition, message):
    if not condition:
        raise MeshError("prepared member accounting: " + message)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _snapshot(value):
    """Pin plain immutable contents without invoking copy/equality/user hooks.

    Only actual owner dataclasses and this module's accounting dataclass are
    admissible. Containers and scalar subclasses cannot smuggle copy hooks into
    capture. The returned tree consists exclusively of builtin immutable values.
    """
    try:
        from anygeometry import (
            PreparedMemberSheetJointComponent, PreparedModelScope, PreparedFacePreimages,
            AuthoredFaceDefinition, PreparedEdgeSubcurvePreimages, EdgeSubcurvePreimage,
            PolynomialEdgeAncestor, PolynomialEdgeDefinition, PreparedAuthoredConstraintScope,
            AuthoredBoundaryCorrespondence,
        )
    except ImportError as error:
        raise MeshError("prepared member accounting snapshot capability unavailable") from error
    known = (PreparedMemberSheetJointComponent, PreparedModelScope,
             PreparedFacePreimages, AuthoredFaceDefinition, PreparedEdgeSubcurvePreimages,
             EdgeSubcurvePreimage, PolynomialEdgeAncestor, PolynomialEdgeDefinition,
             PreparedAuthoredConstraintScope, AuthoredBoundaryCorrespondence,
             PreparedMemberAccounting)
    # Inspect only these actual classes, never attributes of an unknown class.
    shapes = tuple(tuple(field.name for field in fields(cls)) for cls in known)
    active = set()

    def pin(item, depth):
        _require(depth <= 64, "malformed evidence nesting")
        cls = type(item)
        if any(cls is primitive for primitive in (type(None), bool, int, str)):
            return (cls.__name__, item)
        if cls is float:
            return ("float", item.hex())
        if cls is UUID:
            return ("UUID", pin(object.__getattribute__(item, "int"), depth+1))
        index = next((i for i, candidate in enumerate(known) if cls is candidate), None)
        _require(cls is tuple or index is not None, "evidence requires plain immutable fields")
        identity = id(item)
        _require(identity not in active, "cyclic evidence")
        active.add(identity)
        try:
            if cls is tuple:
                return ("tuple", tuple(pin(child, depth+1) for child in item))
            return ("owner-dataclass", index, tuple(
                (name, pin(object.__getattribute__(item, name), depth+1))
                for name in shapes[index]))
        finally:
            active.remove(identity)

    return pin(value, 0)


@dataclass(frozen=True, slots=True)
class PreparedMemberAccounting:
    schema: str
    owner_receipt: object
    constraint_receipt: object
    edge_preimages: object
    # All typed structural records, including unchanged and generated joints.
    accounted_records: tuple
    # In member traversal order: current use, current edge, orientation,
    # member parent interval, polynomial ancestor interval.
    member_traversal: tuple
    # source attachment, current carrier, exact original and local station.
    point_station: tuple
    # Junction ID, complete attachment IDs, Sheet IDs, member-use membership.
    junction_membership: tuple
    relation_json: str

    @property
    def beam_discretization_qualified(self):
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


def _derive(owner, constraints, ancestry):
    _require(owner.scope == constraints.scope, "different owner preparation")
    _require(tuple(owner.authored_face_ids) == constraints.selected_root_ids
             and tuple(owner.current_face_ids) == constraints.current_face_ids
             and not constraints.outside_root_ids, "incomplete component selection")
    _require(constraints.typed_inventory_complete
             and not constraints.semantic_mapping_qualified
             and not constraints.parameter_remapping_qualified
             and not constraints.publication_qualified, "unexpected inventory authority")
    _require(owner.bounded_relation_mapping_qualified
             and owner.occurrence_mapping_qualified
             and not owner.semantic_mapping_qualified
             and not owner.beam_discretization_qualified
             and not owner.external_reference_transfer_qualified
             and not owner.publication_qualified, "unexpected owner authority")
    _require(owner.edge_preimages == ancestry, "different polynomial ancestry")
    inventory = constraints.inventory
    accounted = []
    for side, records in (("original", owner.source_records),
                          ("current", owner.current_records)):
        complete = inventory[side]
        for kind in _KINDS:
            rows = complete["records"][kind]
            _require(sorted(rows, key=lambda row: row["id"]) ==
                     sorted(records[kind], key=lambda row: row["id"]),
                     f"unaccounted {side} {kind}")
            _require(len({row["id"] for row in rows}) == len(rows), "duplicate records")
            accounted.extend((side, kind, row["id"]) for row in rows)
        opaque = complete["opaque_unqualified"]
        _require(not opaque["groups"] and not opaque["extensions"]
                 and not opaque["construction_vertices"]
                 and not opaque["features"].get("records")
                 and not any(row["metadata"] for rows in opaque["metadata"].values()
                             for row in rows), "unqualified opaque references")
        # Generated decomposition tags describe owner edges, not a source remap.
        _require(not opaque["tags"] if side == "original" else
                 all(row["entity"][0] == "edge"
                     and row["entity"][1] in {edge["id"] for edge in records["edges"]}
                     and row["values"] == ["intersection_decomposition_seam"]
                     for row in opaque["tags"]), "unqualified tags")
    relation = owner.member_relation
    old = relation["source_member_uses"][0]
    carrier = old["edge_id"]
    mapped = {row.edge_id: row for row in ancestry.records
              if row.ancestor.definition.edge_id == carrier}
    uses = relation["current_member_uses"]
    _require(set(mapped) == {use["edge_id"] for use in uses}
             and not set(mapped).intersection(ancestry.unavailable_edge_ids),
             "incomplete member carrier ancestry")
    traversal = []
    for use in uses:
        record = mapped[use["edge_id"]]
        _require(Fraction(*record.squared_distance_bound) == 0,
                 "nonzero restriction error")
        traversal.append((use["id"], use["edge_id"], use["orientation"],
                          tuple(use["parent_range"]), record.interval))
    original = relation["source_attachment"]
    current = relation["current_attachment"]
    record = mapped[current["target_id"]]
    a, b = (Fraction(*value) for value in record.interval)
    source_station = Fraction(original["target_parameters"][0][0])
    local = (source_station-a)/(b-a)
    _require(current["target_parameters"] == [[float(local), float(local)]],
             "point station differs from owner mapping")
    # Vertex absence elsewhere must not masquerade as absence of a reference.
    for side in ("original", "current"):
        _require(set(inventory[side]["isolated_vertex_ids"]) <=
                 {relation[f'{"source" if side == "original" else "current"}_vertex']["id"]},
                 "unaccounted isolated vertices")
    joints = tuple((row["id"], tuple(row["attachment_ids"]), tuple(row["sheet_ids"]),
                    tuple(row["member_uses"]))
                   for row in sorted(owner.current_records["junctions"], key=lambda row: row["id"]))
    return PreparedMemberAccounting(SCHEMA, owner, constraints, ancestry,
        tuple(sorted(accounted)), tuple(traversal),
        (original["id"], current["target_id"],
         (source_station.numerator, source_station.denominator),
         (local.numerator, local.denominator)), joints, _json(relation))


def query_prepared_member_accounting(geometry, joint_edge_id, *, owner_receipt=None,
        constraint_receipt=None, edge_preimages=None, cancellation_check=None):
    """Account only the owner's two-Plane straight boundary-member case.

    Optional evidence is untrusted and revalidated; this function never builds
    beams, nodes, couplings or a mesh. Callback-bearing work is followed by a
    callback-free reproof of every supplied content and live owner binding.
    """
    try:
        from anygeometry import (
            GeometryError, PreparedMemberSheetJointComponent, to_dict,
            query_prepared_member_sheet_joint_component as query_owner,
            validate_prepared_member_sheet_joint_component_binding as validate_owner,
            query_prepared_authored_constraint_scope as query_constraints,
            validate_prepared_authored_constraint_scope_binding as validate_constraints,
            query_prepared_edge_subcurve_preimages as query_edges,
            validate_prepared_edge_subcurve_preimages_binding as validate_edges,
        )
    except ImportError as error:
        raise MeshError("prepared member accounting capability unavailable") from error
    supplied = (owner_receipt, constraint_receipt, edge_preimages)
    pinned = _snapshot(supplied)
    before = _json(to_dict(geometry))
    _require(_snapshot(supplied) == pinned, "supplied evidence changed during source serialization")
    try:
        owner, constraints, ancestry = supplied
        # Validate original evidence without callbacks before any user action.
        # No deepcopy is used: alias-returning or repairing hooks never execute.
        for value, validator in ((owner, validate_owner),
                                 (constraints, validate_constraints),
                                 (ancestry, validate_edges)):
            if value is not None:
                validator(geometry, value)
        _require(_snapshot(supplied) == pinned, "supplied evidence changed before accounting")
        if owner is None:
            owner = query_owner(geometry, joint_edge_id, cancellation_check=cancellation_check)
        owner_pin = _snapshot(owner)
        _require(type(owner) is PreparedMemberSheetJointComponent
                 and type(joint_edge_id) is int and owner.joint_edge_id == joint_edge_id,
                 "wrong receipt or joint")
        validate_owner(geometry, owner, cancellation_check=cancellation_check)
        if constraints is None:
            constraints = query_constraints(geometry, owner.authored_face_ids,
                                             cancellation_check=cancellation_check)
        constraint_pin = _snapshot(constraints)
        validate_constraints(geometry, constraints, cancellation_check=cancellation_check)
        if ancestry is None:
            ancestry = query_edges(geometry, cancellation_check=cancellation_check)
        ancestry_pin = _snapshot(ancestry)
        validate_edges(geometry, ancestry, cancellation_check=cancellation_check)
        _require(_snapshot(supplied) == pinned,
                 "supplied evidence changed before derivation")
        _require(_snapshot(owner) == owner_pin and
                 _snapshot(constraints) == constraint_pin and
                 _snapshot(ancestry) == ancestry_pin,
                 "acquired evidence changed before derivation")
        result = _derive(owner, constraints, ancestry)
        result_pin = _snapshot(result)
        validate_owner(geometry, owner)
        validate_constraints(geometry, constraints)
        validate_edges(geometry, ancestry)
        _require(_json(to_dict(geometry)) == before, "geometry changed during accounting")
        _require(_snapshot(supplied) == pinned, "supplied evidence changed during accounting")
        _require(_snapshot(result) == result_pin, "generated accounting changed during closing proof")
        return result
    except GeometryError as error:
        raise MeshError("prepared member accounting owner refusal: " + str(error)) from error


def validate_prepared_member_accounting(geometry, receipt, *, cancellation_check=None):
    """Rederive complete accounting; no omitted rows or digest-only approval."""
    _require(type(receipt) is PreparedMemberAccounting, "wrong accounting receipt")
    pinned = _snapshot(receipt)
    _require(receipt.schema == SCHEMA, "wrong accounting receipt")
    # Original receipt fields are already pinned, independently of aliases.
    expected = query_prepared_member_accounting(geometry, receipt.owner_receipt.joint_edge_id,
        owner_receipt=receipt.owner_receipt, constraint_receipt=receipt.constraint_receipt,
        edge_preimages=receipt.edge_preimages, cancellation_check=cancellation_check)
    _require(_snapshot(expected) == pinned and _snapshot(receipt) == pinned,
             "accounting contents changed")
