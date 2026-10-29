"""Source-relative outward winding; chart normals are not body normals."""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Callable, Iterable

import numpy as np

from .errors import MeshError
from .mesh import Mesh
from .serialize import mesh_from_dict, mesh_to_dict

BOUNDARY_ORIENTATION_CONTRACT = "anymesher.source-outward-t3-boundary-v1"


class BoundaryOrientationError(MeshError):
    """Typed refusal with a machine-readable category and details."""

    def __init__(self, category: str, message: str, **details: object):
        super().__init__(message)
        self.category, self.details = category, details


@dataclass(frozen=True)
class BoundaryOrientationResult:
    mesh: Mesh
    evidence: dict
    original_mesh: Mesh
    generation_evidence: dict | None = None


def _generation_audit_record(result: object, request: dict) -> dict:
    """Retain actual producer outcomes; missing evidence stays explicitly absent.

    A transport wheel SHA cannot be recovered from an import/version. The
    caller must authenticate that separately against this package-byte manifest.
    """
    from pathlib import Path
    package = Path(__file__).parent
    files = {str(p.relative_to(package)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(package.rglob("*"))
             if p.is_file() and p.suffix in (".py", ".pyd", ".so")}
    manifest_digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    audit = getattr(result, "audit_report", None)
    if audit is not None:
        if hasattr(audit, "to_dict"):
            audit = audit.to_dict()
        if not isinstance(audit, dict):
            raise BoundaryOrientationError("GENERATION_EVIDENCE", "audit report lacks its public JSON representation")
        # Strict JSON copying also refuses non-finite/opaque evidence.
        audit = json.loads(json.dumps(audit, allow_nan=False))
    certifiable = getattr(result, "certifiable", None)
    if type(certifiable) is not bool:
        certifiable = None
    audit_certifiable = None if audit is None else audit.get("certifiable")
    if type(audit_certifiable) is not bool:
        audit_certifiable = None
    mode = getattr(result, "certification_mode", None)
    if isinstance(mode, Enum):
        mode = mode.value
    return {
        "contract": "anymesher.displacement-generation-audit-v1",
        "status": "RETAINED" if certifiable is not None and audit_certifiable is not None else "MISSING_EVIDENCE",
        "certifiable": certifiable,
        "audit_report": audit,
        "audit_report_certifiable": audit_certifiable,
        "actual_certification_mode": mode,
        "actual_strategy_by_face": {str(f): value for f, value in getattr(result, "strategy_by_face", {}).items()},
        "generation_request": json.loads(json.dumps(request, allow_nan=False)),
        "generation_and_orientation_implementation": {
            "package_manifest_sha256": manifest_digest, "package_files_sha256": files,
            "wheel_sha256": None,
            "wheel_binding": "caller must authenticate the transport wheel against these package bytes",
        },
        "scope": "original public generation outcome, linked to orientation-only permutation; not a new audit of the oriented mesh",
    }


def _digest(mesh: Mesh) -> str:
    raw = json.dumps(mesh_to_dict(mesh), sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def orient_displacement_boundary(
    mesh: Mesh, geometry: object, *, face_ids: Iterable[int],
    cancellation_check: Callable[[], object] | None = None,
    owner_budget: object | None = None,
) -> BoundaryOrientationResult:
    """Return a new outward-wound mesh; never mutate input mesh or geometry.

    Supports one closed connected linear T3 boundary with complete source-face
    ownership and authoritative ``nodes_of_edge`` chains in source direction.
    Fresh ANYgeometry certified admission establishes source-loop embedding and
    outwardness. We verify every directed source trace, discrete topology, node
    support and local chart orientation. This does not certify global chord
    embedding, approximation error, physical quality or solver fitness.

    Cancellation callbacks may raise or return True. ``owner_budget`` is passed
    to the owner's fresh admission and final content/identity binding check.
    Owner admission/budget refusals propagate unchanged.
    """
    from anygeometry.certified_hydrostatics import (
        check_preparation, prepare_certified_hydrostatics,
    )
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.surfaces import Plane

    def poll():
        if cancellation_check is not None and cancellation_check():
            raise BoundaryOrientationError("CANCELLED", "boundary orientation cancelled")

    def owner_cancel():
        poll()
        return False

    def refuse(category, message, **details):
        raise BoundaryOrientationError(category, message, **details)

    poll()
    requested = []
    for f in face_ids:
        poll()
        if isinstance(f, bool) or not isinstance(f, (int, np.integer)):
            refuse("SOURCE_SCOPE", "face IDs must be integers")
        requested.append(int(f))
    if not requested or len(set(requested)) != len(requested):
        refuse("SOURCE_SCOPE", "nonempty unique source face IDs required")
    selected = tuple(sorted(requested))
    if (str(mesh.geometry_model_id) != str(geometry.model_id)
            or mesh.geometry_revision != geometry.revision):
        refuse("SOURCE_BINDING", "mesh/source identity or revision differs")
    if (mesh.order != "linear" or not mesh.tris or mesh.quads or mesh.beams
            or mesh.couplings or mesh.declared_plate_junction_edges):
        refuse("UNSUPPORTED", "requires pure closed linear T3 without structural junctions")
    try:
        before = _digest(mesh)
    except (ValueError, TypeError) as error:
        raise BoundaryOrientationError("GEOMETRY", "mesh is not finite serializable data") from error
    # Native generation may attach ephemeral owner handles that are not
    # pickleable. Clone the complete durable neutral contract, not live caches.
    snapshot = mesh_to_dict(mesh)
    original = mesh_from_dict(snapshot)
    if _digest(original) != before:
        refuse("INPUT_CHANGED", "input mesh changed while taking its snapshot")
    candidate = mesh_from_dict(snapshot)
    # Fresh admission, not a caller-supplied dataclass/digest-only attestation.
    prepared = prepare_certified_hydrostatics(
        geometry, selected, budget=owner_budget, cancellation_check=owner_cancel)
    poll()
    if (str(candidate.geometry_model_id) != prepared.model_id
            or candidate.geometry_revision != prepared.revision):
        refuse("SOURCE_BINDING", "source changed between mesh binding and fresh admission")
    if set(candidate.elements_of_face) != set(selected):
        refuse("ASSOCIATION", "face associations must equal the selected boundary")
    element_face = {}
    for f in selected:
        poll()
        if not candidate.elements_of_face[f]:
            refuse("ASSOCIATION", "selected face has no triangles", face_id=f)
        for e in candidate.elements_of_face[f]:
            if e in element_face or e not in candidate.tris:
                refuse("ASSOCIATION", "duplicate or unknown face element", element_id=e)
            element_face[e] = f
    if set(element_face) != set(candidate.tris):
        refuse("ASSOCIATION", "every triangle needs exactly one source face")
    incidence, links = defaultdict(list), defaultdict(list)
    face_incidence = {f: defaultdict(list) for f in selected}
    used_nodes, unordered_cells = set(), set()
    for e, tri in sorted(candidate.tris.items()):
        poll()
        if len(tri) != 3 or len(set(tri)) != 3:
            refuse("TOPOLOGY", "triangle needs three distinct corners", element_id=e)
        if frozenset(tri) in unordered_cells:
            refuse("TOPOLOGY", "duplicate triangle", element_id=e)
        unordered_cells.add(frozenset(tri))
        for n in tri:
            if n not in candidate.nodes:
                refuse("TOPOLOGY", "triangle references missing node", node_id=n)
            p = np.asarray(candidate.nodes[n])
            if p.shape != (3,) or not np.all(np.isfinite(p)):
                refuse("GEOMETRY", "node coordinates must be finite 3-vectors", node_id=n)
        used_nodes.update(tri)
        xyz = np.array([candidate.nodes[n] for n in tri])
        if not float(np.linalg.norm(np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0]))) > 0:
            refuse("GEOMETRY", "zero-area triangle", element_id=e)
        for a, b, c in zip(tri, tri[1:]+tri[:1], tri[2:]+tri[:2]):
            key = tuple(sorted((a, b)))
            direction = 1 if (a, b) == key else -1
            incidence[key].append((e, direction))
            face_incidence[element_face[e]][key].append((e, direction))
            links[a].append((b, c))
    if used_nodes != set(candidate.nodes):
        refuse("TOPOLOGY", "boundary contains unused nodes")
    for edge, uses in incidence.items():
        if len(uses) != 2:
            refuse("TOPOLOGY", "closed manifold requires two incidences", edge=edge, incidence=len(uses))
    for node, pairs in links.items():
        poll()
        graph = defaultdict(set)
        for a, b in pairs:
            graph[a].add(b)
            graph[b].add(a)
        if any(len(v) != 2 for v in graph.values()):
            refuse("TOPOLOGY", "vertex link is not a cycle", node_id=node)
        reached, pending = set(), [min(graph)]
        while pending:
            v = pending.pop()
            if v not in reached:
                reached.add(v)
                pending.extend(graph[v]-reached)
        if reached != set(graph):
            refuse("TOPOLOGY", "vertex link has multiple cycles", node_id=node)
    adjacency, original_bad = defaultdict(list), 0
    for uses in incidence.values():
        (a, da), (b, db) = uses
        original_bad += da == db
        parity = -da*db
        adjacency[a].append((b, parity))
        adjacency[b].append((a, parity))
    signs = {min(candidate.tris): 1}
    pending = deque(signs)
    while pending:
        poll()
        a = pending.popleft()
        for b, parity in adjacency[a]:
            desired = signs[a]*parity
            if b in signs and signs[b] != desired:
                refuse("TOPOLOGY", "boundary is not orientable")
            if b not in signs:
                signs[b] = desired
                pending.append(b)
    if set(signs) != set(candidate.tris):
        refuse("TOPOLOGY", "boundary has disconnected components")

    # All directed source coedges must agree with one global winding sign.
    global_sign, traces = None, []
    expected = {f: set() for f in selected}
    tolerance = min(1.0e-9, geometry.tolerance.surface_residual)
    edge_sequences = {}
    for edge_id, (start, end) in sorted(prepared.snapshot.edges.items()):
        poll()
        seq = tuple(candidate.nodes_of_edge.get(edge_id, ()))
        if len(seq) < 2 or len(set(seq)) != len(seq) or not set(seq) <= used_nodes:
            refuse("ASSOCIATION", "missing/invalid source-edge station chain", edge_id=edge_id)
        for vertex, node in ((start, seq[0]), (end, seq[-1])):
            if vertex in candidate.node_of_vertex and candidate.node_of_vertex[vertex] != node:
                refuse("ASSOCIATION", "source vertex and station disagree", edge_id=edge_id)
            if np.linalg.norm(candidate.nodes[node]-geometry.vertex_position(vertex)) > tolerance:
                refuse("SOURCE_SUPPORT", "edge endpoint differs from source vertex", edge_id=edge_id, node_id=node)
        parameters, maximum = [], 0.0
        for node in seq:
            poll()
            _, parameter, distance = geometry.closest_edge_point(edge_id, candidate.nodes[node])
            if not np.isfinite(distance) or distance > tolerance:
                refuse("SOURCE_SUPPORT", "station is off its source curve", edge_id=edge_id, node_id=node)
            parameters.append(parameter)
            maximum = max(maximum, float(distance))
        if not np.all(np.diff(parameters) > 0):
            refuse("ASSOCIATION", "source stations are not strictly ordered", edge_id=edge_id)
        edge_sequences[edge_id] = seq
        traces.append({"edge_id": edge_id, "node_ids": list(seq), "maximum_curve_residual": maximum})
    for face in prepared.snapshot.faces:
        poll()
        for loop in face.loops:
            for edge_id, forward in loop.edges:
                seq = edge_sequences[edge_id]
                if not forward:
                    seq = seq[::-1]
                for a, b in zip(seq, seq[1:]):
                    key = tuple(sorted((a, b)))
                    uses = face_incidence[face.id].get(key, ())
                    if len(uses) != 1 or key in expected[face.id]:
                        refuse("ASSOCIATION", "source trace needs one face-boundary incidence", face_id=face.id, edge_id=edge_id)
                    expected[face.id].add(key)
                    e, direction = uses[0]
                    desired = 1 if (a, b) == key else -1
                    anchor = desired*direction*signs[e]
                    if global_sign is not None and global_sign != anchor:
                        refuse("ORIENTATION", "source traces impose inconsistent winding", face_id=face.id, edge_id=edge_id)
                    global_sign = anchor
        actual = {k for k, v in face_incidence[face.id].items() if len(v) == 1}
        if actual != expected[face.id]:
            refuse("ASSOCIATION", "face boundary differs from complete source loops", face_id=face.id)
    if global_sign is None:
        refuse("ASSOCIATION", "no source trace anchors winding")
    flipped = []
    for e in sorted(candidate.tris):
        poll()
        if signs[e]*global_sign < 0:
            a, b, c = candidate.tris[e]
            candidate.tris[e] = (a, c, b)
            flipped.append(e)
    support = {}
    for f in selected:
        poll()
        face = geometry.faces[f]
        if face.parameterization is not None or not isinstance(face.surface, (Plane, NurbsSurface)):
            refuse("UNSUPPORTED", "requires Plane/NURBS support without alternate parameterization", face_id=f)
        elements = sorted(candidate.elements_of_face[f])
        nodes = sorted({n for e in elements for n in candidate.tris[e]})
        local, maximum = {}, 0.0
        for start in range(0, len(nodes), 128):
            poll()
            ids = nodes[start:start+128]
            xyz = np.array([candidate.nodes[n] for n in ids])
            if isinstance(face.surface, Plane):
                uv = np.array([face.surface.local_uv(p) for p in xyz])
                points = np.array([face.surface.evaluate(*p) for p in uv])
            else:
                uv = geometry.face_local_uv_many(f, xyz)
                points = geometry.evaluate_face_many(f, uv)
            residuals = np.linalg.norm(points-xyz, axis=1)
            if not np.all(np.isfinite(residuals)) or np.max(residuals) > tolerance:
                refuse("SOURCE_SUPPORT", "node is off its source support", face_id=f, maximum_residual=float(np.max(residuals)))
            maximum = max(maximum, float(np.max(residuals)))
            local.update(zip(ids, uv))
        chart_sign, minimum_alignment = None, 1.0
        for start in range(0, len(elements), 128):
            poll()
            ids = elements[start:start+128]
            xyz = np.array([[candidate.nodes[n] for n in candidate.tris[e]] for e in ids])
            uv = np.array([[local[n] for n in candidate.tris[e]] for e in ids])
            d1, d2 = uv[:,1]-uv[:,0], uv[:,2]-uv[:,0]
            area = d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0]
            normals = geometry.face_normal_many(f, np.mean(uv, axis=1))
            cross = np.cross(xyz[:,1]-xyz[:,0], xyz[:,2]-xyz[:,0])
            alignment = np.einsum('ij,ij->i', cross, normals)/np.linalg.norm(cross, axis=1)
            for e, a, value in zip(ids, area, alignment):
                sign = 1 if a > 0 else -1
                if (not np.isfinite(a) or a == 0 or not np.isfinite(value)
                        or sign*value <= 1.0e-12
                        or (chart_sign is not None and chart_sign != sign)):
                    refuse("ORIENTATION", "folded/degenerate local mapping", face_id=f, element_id=e)
                chart_sign = sign
            minimum_alignment = min(minimum_alignment, float(np.min(np.abs(alignment))))
        support[str(f)] = {"nodes": len(nodes), "triangles": len(elements),
                           "maximum_node_residual": maximum,
                           "outward_chart_sign": chart_sign,
                           "minimum_sampled_normal_alignment": minimum_alignment}
    poll()
    final = defaultdict(list)
    for tri in candidate.tris.values():
        poll()
        for a, b in zip(tri, tri[1:]+tri[:1]):
            final[tuple(sorted((a,b)))].append((a,b))
    if any(len(uses) != 2 or uses[0] != uses[1][::-1] for uses in final.values()):
        refuse("ORIENTATION", "final opposite-incidence verification failed")
    # Re-orienting a saved oriented mesh must still produce a digest verifiable
    # by excluding only the receipt present in the published output.
    candidate.hybrid_diagnostics.pop("boundary_orientation", None)
    evidence = {"contract": BOUNDARY_ORIENTATION_CONTRACT,
                "status": "SOURCE_RELATIVE_OUTWARD_WINDING",
                "owner_preparation": prepared.proof_identity(),
                "input_mesh_sha256": before,
                "oriented_mesh_sha256_before_receipt": _digest(candidate),
                "selected_face_ids": list(selected), "triangles": len(candidate.tris),
                "edges": len(final), "original_same_direction_edges": int(original_bad),
                "final_same_direction_edges": 0, "flipped_element_ids": flipped,
                "source_edge_traces": traces, "source_face_support": support,
                "support_tolerance": tolerance,
                "limits": ["not a global chord-embedding certificate",
                           "not approximation/physical-quality acceptance",
                           "not solver or consumer qualification"]}
    candidate.hybrid_diagnostics["boundary_orientation"] = deepcopy(evidence)
    # All cancellable work precedes publication binding. No user callback may
    # run after these final checks and silently invalidate the returned receipt.
    poll()
    check_preparation(geometry, prepared, budget=owner_budget, cancellation_check=owner_cancel)
    if (str(geometry.model_id) != prepared.model_id or geometry.revision != prepared.revision):
        refuse("SOURCE_BINDING", "source changed during final binding")
    if _digest(mesh) != before:
        refuse("INPUT_CHANGED", "input mesh changed during the read-only operation")
    return BoundaryOrientationResult(candidate, evidence, original)


def generate_displacement_boundary_mesh_result(
    geometry: object, *, target_size: float, face_ids: Iterable[int],
    cancellation_check: Callable[[], object] | None = None,
    owner_budget: object | None = None,
) -> BoundaryOrientationResult:
    """Generate strict/read-only native linear T3, then orient by owner loops.

    Both ``result.original_mesh`` and ``result.mesh`` retain the complete neutral
    mesh (including vertex, edge-station and face associations). The former is
    the unchanged generator output; the latter adds outward winding/provenance.
    No geometry preparation mutation, structural additions or silent fallback
    to another method/order is requested. No solver admission is implied.
    """
    from .hybrid import generate_hybrid_mesh_result

    selected = tuple(face_ids)
    def check(stage: str | None = None):
        if cancellation_check is not None and cancellation_check():
            raise BoundaryOrientationError("CANCELLED", "boundary generation cancelled")
    check()
    request = dict(target_size=target_size, strategy="native", recombine=False,
                   order="linear", face_ids=selected, member_ids=(), beam_edges=(),
                   structural_preparation=False, certification_mode="strict", mutation_policy="read_only")
    result = generate_hybrid_mesh_result(geometry, **request, cancellation_check=check)
    generation_mesh_digest = _digest(result.mesh)
    # Capture owner objects before final orientation/binding. No callback or
    # owner serialization is allowed after orient_displacement_boundary returns.
    generation = _generation_audit_record(result, request)
    check()
    oriented = orient_displacement_boundary(
        result.mesh, geometry, face_ids=selected,
        cancellation_check=cancellation_check, owner_budget=owner_budget,
    )
    if oriented.evidence["input_mesh_sha256"] != generation_mesh_digest:
        raise BoundaryOrientationError("GENERATION_EVIDENCE", "generated mesh changed before orientation binding")
    generation["original_generation_mesh_sha256"] = generation_mesh_digest
    generation["original_mesh_sha256"] = oriented.evidence["input_mesh_sha256"]
    generation["oriented_mesh_sha256_before_receipt"] = oriented.evidence["oriented_mesh_sha256_before_receipt"]
    evidence = deepcopy(oriented.evidence)
    evidence["generation_audit"] = generation
    oriented.mesh.hybrid_diagnostics["boundary_orientation"] = deepcopy(evidence)
    return BoundaryOrientationResult(oriented.mesh, evidence, oriented.original_mesh, generation)
