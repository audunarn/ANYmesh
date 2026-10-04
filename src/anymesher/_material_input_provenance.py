"""Bound pre-native material input origins; no final triangulation-row claims.

Owner path/registry identity is recorded before the surface triangulator runs.
This receipt grants no split permission, element validity or publication status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from hashlib import sha256
import json

import numpy as np

from .errors import MeshError


_TERMS = (
    "shared_or_analytic_supports", "all_native_faces", "not_overridden",
    "no_beam", "straight_curve", "default_seeding", "linear_order",
    "not_region_interior", "component_local_allowed",
)


def _fraction(value):
    if isinstance(value, (bool, np.bool_)):
        raise MeshError("material input has invalid source station")
    item = Fraction(float(value))
    if not 0 <= item <= 1:
        raise MeshError("material input source station is outside its edge")
    return (item.numerator, item.denominator)


def _payload(binding, mesh, registry, loops, pinned_ids, context, emitted,
             cancellation_check):
    from anygeometry.serialization import to_dict

    def frozen_array(value, shape_tail, label):
        array = np.asarray(value, dtype=np.float64)
        if (array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail
                or not np.isfinite(array).all()):
            raise MeshError(f"material input {label} has invalid shape or values")
        return array.copy()

    if not isinstance(emitted, dict):
        raise MeshError("material input emitted arrays are missing")
    transform = np.asarray(emitted.get("transform"), dtype=np.float64)
    if transform.shape != (2, 2) or not np.isfinite(transform).all():
        raise MeshError("material input chart transform is invalid")
    transform = transform.copy()
    chart_loops = tuple(frozen_array(item, (2,), "chart loop")
                        for item in emitted.get("chart_loops", ()))
    chart_constraints = tuple(frozen_array(item, (2,), "chart constraint")
                              for item in emitted.get("chart_constraints", ()))
    chart_pinned = frozen_array(emitted.get("chart_pinned"), (2,), "chart pins")
    loop_uv = tuple(frozen_array(loop.uv, (2,), "owner UV loop") for loop in loops)
    frozen_inputs = (transform, chart_loops, chart_constraints, chart_pinned, loop_uv)

    owner = binding.geometry
    if registry.view.source is not owner:
        raise MeshError("material input registry belongs to a different owner")
    registry.view.assert_current(owner)
    binding._validate_boundary_owner(cancellation_check)
    binding._validate_boundary_source_owner(mesh, registry)
    before = to_dict(owner)
    if (context.get("model_id") != str(owner.model_id)
            or context.get("revision") != owner.revision):
        raise MeshError("material input eligibility context has stale owner identity")
    edge_context = context.get("edges")
    if not isinstance(edge_context, dict):
        raise MeshError("material input eligibility context is missing")
    context_before = json.loads(json.dumps(context, sort_keys=True))
    if len(loops) != len(binding.region.boundaries):
        raise MeshError("material input loop occurrences are incomplete")
    all_paths = []
    boundary_rows = []
    owner_loops = []
    seen_tokens = set()
    used_edges = set()

    def bound_path_chain(path):
        try:
            return binding._boundary_source_chain(path, mesh, registry)
        except KeyError as error:
            raise MeshError("material input source path has a missing registered node") from error

    for loop_index, (paths, loop) in enumerate(zip(binding.region.boundaries, loops)):
        pair_owner = {}
        expected_nodes = []
        expected_uv = []
        for path_index, path in enumerate(paths):
            nodes, uv, values = bound_path_chain(path)
            if path.source_edge is None or len(nodes) < 2 or len(nodes) != len(values):
                raise MeshError("material input source path or stations are missing")
            edge = int(path.source_edge)
            used_edges.add(edge)
            expected_nodes.extend(int(node) for node in nodes[:-1])
            expected_uv.extend(np.asarray(row, dtype=np.float64) for row in uv[:-1])
            stations = tuple(_fraction(value) for value in values)
            all_paths.append({"kind": "boundary", "loop": loop_index,
                              "path": path_index, "source_edge": edge,
                              "curve_type": type(path.curve).__name__,
                              "global_nodes": tuple(int(node) for node in nodes),
                              "stations": stations,
                              "orientation": "forward" if Fraction(*stations[0]) < Fraction(*stations[-1])
                                             else "reverse"})
            for index, (a, b) in enumerate(zip(nodes, nodes[1:])):
                for pair, endpoints in (((int(a), int(b)), (stations[index], stations[index + 1])),
                                        ((int(b), int(a)), (stations[index + 1], stations[index]))):
                    if pair in pair_owner:
                        raise MeshError("material input boundary pair has multiple source occurrences")
                    pair_owner[pair] = (path_index, edge, endpoints)
        actual_nodes = tuple(int(node) for node in loop.node_ids)
        if actual_nodes == tuple(expected_nodes):
            required_uv = np.asarray(expected_uv, dtype=np.float64)
        elif actual_nodes == tuple(reversed(expected_nodes)):
            required_uv = np.asarray(expected_uv[::-1], dtype=np.float64)
        else:
            raise MeshError("material input loop lost owner path node order")
        if (required_uv.shape != loop_uv[loop_index].shape
                or required_uv.tobytes() != loop_uv[loop_index].tobytes()):
            raise MeshError("material input loop UV differs from owner path stations")
        owner_loops.append(required_uv)
        if len(actual_nodes) != len(loop.segment_specs):
            raise MeshError("material input loop has missing segment specifications")
        for index, node in enumerate(actual_nodes):
            next_node = actual_nodes[(index + 1) % len(actual_nodes)]
            identity = pair_owner.get((node, next_node))
            if identity is None:
                raise MeshError("material input segment has no bound owner occurrence")
            path_index, edge, stations = identity
            selected = edge_context.get(edge)
            if not isinstance(selected, dict) or set(_TERMS) - set(selected):
                raise MeshError("material input edge lacks complete route context")
            if any(type(selected[term]) is not bool for term in _TERMS):
                raise MeshError("material input route context has nonboolean terms")
            expected_selected = all(selected[term] for term in _TERMS)
            if type(selected.get("selected")) is not bool or selected["selected"] != expected_selected:
                raise MeshError("material input route context contradicts its predicate")
            specification = loop.segment_specs[index]
            if (specification is None) == expected_selected:
                raise MeshError("material input segment spec contradicts route context")
            if specification is not None and (
                    int(specification[0]) != edge
                    or (_fraction(specification[1]), _fraction(specification[2]))
                    != tuple(sorted(stations))):
                raise MeshError("material input segment spec changed source interval")
            token = ("boundary", loop_index, index)
            if token in seen_tokens:
                raise MeshError("material input has duplicate boundary token")
            seen_tokens.add(token)
            boundary_rows.append({"token": token, "global_node": node,
                                  "source_occurrence": (loop_index, path_index),
                                  "source_edge": edge, "interval_stations": stations,
                                  "eligible_in_current_route": expected_selected,
                                  "reasons": tuple(term for term in _TERMS if not selected[term])})
    boundary_nodes = {row["global_node"] for row in boundary_rows}
    interior_paths = []
    for path_index, path in enumerate(binding.region.interior_constraints):
        nodes, _uv, values = bound_path_chain(path)
        if path.source_edge is None or len(nodes) < 2 or len(nodes) != len(values):
            raise MeshError("material input internal source path is incomplete")
        edge = int(path.source_edge)
        used_edges.add(edge)
        interior_paths.append({"kind": "interior", "path": path_index,
                               "source_edge": edge, "curve_type": type(path.curve).__name__,
                               "global_nodes": tuple(int(node) for node in nodes),
                               "stations": tuple(_fraction(value) for value in values)})
        selected = edge_context.get(edge)
        if not isinstance(selected, dict) or selected.get("selected") is not False:
            raise MeshError("material input internal path has ambiguous split permission")
    pins = []
    if len(set(pinned_ids)) != len(pinned_ids):
        raise MeshError("material input has duplicate pinned node identities")
    path_nodes = {node for path in interior_paths for node in path["global_nodes"]}
    vertex_nodes = getattr(mesh, "node_of_vertex", {})
    if any(handle.id not in vertex_nodes or vertex_nodes[handle.id] not in mesh.nodes
           for handle in binding.region.retained_vertices):
        raise MeshError("material input retained vertex lacks an existing registered node")
    retained = {vertex_nodes[handle.id]: int(handle.id)
                for handle in binding.region.retained_vertices
                if handle.id in vertex_nodes}
    for index, raw_node in enumerate(pinned_ids):
        node = int(raw_node)
        if node in boundary_nodes or (node not in path_nodes and node not in retained):
            raise MeshError("material input pin lacks independent source identity")
        token = ("pin", index)
        if token in seen_tokens:
            raise MeshError("material input has duplicate pin token")
        seen_tokens.add(token)
        pins.append({"token": token, "global_node": node,
                     "retained_vertex": retained.get(node),
                     "internal_occurrences": tuple(path["path"] for path in interior_paths
                                                   if node in path["global_nodes"])})
    before_mesh_nodes = {int(node): np.asarray(point, dtype=np.float64).copy()
                         for node, point in mesh.nodes.items()}
    before_vertex_nodes = dict(vertex_nodes)
    required_constraints, required_ids, required_uv = binding.interior(
        mesh, registry,
        boundary_rows={row["global_node"]: owner_loops[row["token"][1]][row["token"][2]]
                       for row in boundary_rows})
    if (tuple(map(int, required_ids)) != tuple(map(int, pinned_ids))
            or dict(vertex_nodes) != before_vertex_nodes
            or set(mesh.nodes) != set(before_mesh_nodes)
            or any(not np.array_equal(mesh.nodes[node], point)
                   for node, point in before_mesh_nodes.items())):
        raise MeshError("material input pin set or owner nodes changed during capture")
    if len(chart_loops) != len(owner_loops) or any(
            chart.shape != owner.shape
            or chart.tobytes() != (owner @ transform).tobytes()
            for chart, owner in zip(chart_loops, owner_loops)):
        raise MeshError("material input emitted boundary chart differs from owner UV")
    if len(chart_constraints) != len(required_constraints) or any(
            chart.shape != expected.shape
            or chart.tobytes() != (np.asarray(expected, dtype=float) @ transform).tobytes()
            for chart, expected in zip(chart_constraints, required_constraints)):
        raise MeshError("material input emitted constraints differ from owner paths")
    if (chart_pinned.shape != required_uv.shape
            or chart_pinned.tobytes() != (required_uv @ transform).tobytes()):
        raise MeshError("material input emitted pins differ from complete owner pin set")
    entries = []
    for edge in sorted(used_edges):
        selected = edge_context.get(edge)
        if (not isinstance(selected, dict) or set(_TERMS) - set(selected)
                or any(type(selected[term]) is not bool for term in _TERMS)
                or type(selected.get("selected")) is not bool
                or selected["selected"] != all(selected[term] for term in _TERMS)):
            raise MeshError("material input source edge has incomplete route context")
        for entry in registry.entries(edge):
            entries.append({"source_edge": edge, "station": _fraction(entry.key.parameter),
                            "global_node": entry.node_id,
                            "owners": tuple(map(str, entry.owners))})
    used_nodes = {row["global_node"] for row in (*boundary_rows, *pins)}
    coordinates = []
    for node in sorted(used_nodes):
        if node not in mesh.nodes:
            raise MeshError("material input registered node has no mesh coordinate")
        point = np.asarray(mesh.nodes[node], dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise MeshError("material input node has invalid physical coordinates")
        coordinates.append((node, tuple(map(float, point))))
    if to_dict(owner) != before:
        raise MeshError("material input owner changed during provenance capture")
    if json.loads(json.dumps(context, sort_keys=True)) != context_before:
        raise MeshError("material input eligibility context changed during capture")
    now_inputs = (
        np.asarray(emitted["transform"], dtype=float),
        tuple(np.asarray(item, dtype=float) for item in emitted["chart_loops"]),
        tuple(np.asarray(item, dtype=float) for item in emitted["chart_constraints"]),
        np.asarray(emitted["chart_pinned"], dtype=float),
        tuple(np.asarray(loop.uv, dtype=float) for loop in loops),
    )
    if (now_inputs[0].tobytes() != frozen_inputs[0].tobytes()
            or now_inputs[3].tobytes() != frozen_inputs[3].tobytes()
            or any(len(now_inputs[index]) != len(frozen_inputs[index]) or any(
                a.tobytes() != b.tobytes() for a, b in zip(now_inputs[index], frozen_inputs[index]))
                   for index in (1, 2, 4))):
        raise MeshError("material input emitted arrays changed during owner validation")
    registry.view.assert_current(owner)
    owner_hash = sha256(json.dumps(before, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    context_hash = sha256(json.dumps(context_before, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()
    return {"schema": "anymesher.material-input-provenance-v1",
            "owner_model_id": str(owner.model_id), "owner_revision": owner.revision,
            "owner_content_sha256": owner_hash,
            "route_context_sha256": context_hash,
            "face_id": binding.representative,
            "boundary_paths": tuple(all_paths), "interior_paths": tuple(interior_paths),
            "boundary_inputs": tuple(boundary_rows), "pinned_inputs": tuple(pins),
            "owner_loop_uv": tuple(tuple(tuple(map(float, row)) for row in loop)
                                   for loop in owner_loops),
            "emitted_chart_transform": tuple(tuple(map(float, row)) for row in transform),
            "emitted_chart_loops": tuple(tuple(tuple(map(float, row)) for row in loop)
                                         for loop in chart_loops),
            "emitted_chart_constraints": tuple(tuple(tuple(map(float, row)) for row in item)
                                               for item in chart_constraints),
            "emitted_chart_pins": tuple(tuple(map(float, row)) for row in chart_pinned),
            "registry_entries": tuple(entries), "input_coordinates": tuple(coordinates),
            "eligibility_context": tuple((edge, tuple((name, edge_context[edge][name])
                                               for name in (*_TERMS, "selected")))
                                         for edge in sorted(used_edges))}


@dataclass(frozen=True)
class MaterialInputProvenanceReceipt:
    """Invocation-bound INPUT receipt; no post-triangulation row association."""

    payload: dict
    _binding: object = field(repr=False, compare=False)
    _mesh: object = field(repr=False, compare=False)
    _registry: object = field(repr=False, compare=False)
    _loops: tuple = field(repr=False, compare=False)
    _pins: tuple = field(repr=False, compare=False)
    _context: dict = field(repr=False, compare=False)
    _emitted: dict = field(repr=False, compare=False)
    _entry_bytes: bytes = field(repr=False, compare=False)

    def to_dict(self):
        return json.loads(self._entry_bytes)

    def validate(self, cancellation_check=None):
        entry_bytes = self._entry_bytes
        if json.dumps(self.payload, sort_keys=True, separators=(",", ":")).encode() != entry_bytes:
            raise MeshError("material input provenance payload changed since entry")
        current = _payload(self._binding, self._mesh, self._registry,
                           self._loops, self._pins, self._context, self._emitted,
                           cancellation_check)
        if (self._entry_bytes != entry_bytes
                or json.dumps(self.payload, sort_keys=True, separators=(",", ":")).encode() != entry_bytes
                or json.dumps(current, sort_keys=True, separators=(",", ":")).encode() != entry_bytes):
            raise MeshError("material input provenance changed since capture")
        return self


def make_material_input_provenance(binding, mesh, registry, loops, pinned_ids,
                                   context, emitted, cancellation_check=None):
    payload = _payload(binding, mesh, registry, loops, pinned_ids, context, emitted,
                       cancellation_check)
    entry_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    receipt = MaterialInputProvenanceReceipt(payload, binding, mesh, registry,
                                              tuple(loops), tuple(pinned_ids), context,
                                              emitted, entry_bytes)
    return receipt.validate(cancellation_check)
