"""Public mesh numbering and fixed-interface regional assembly.

Two owner operations live here.

``remap_mesh_ids`` translates every mesh-space ID of a :class:`Mesh` through
explicit temporary -> permanent maps and returns a detached mesh.  Anything the
function cannot translate with certainty (unknown fields, unknown metadata) is
refused rather than dropped or left stale.  Producer-temporary labels may be 0;
permanent targets are positive.

``assemble_regional_replacement`` swaps a staged replacement for the *affected*
elements of a previous mesh.  Everything outside the affected set is frozen
(coordinates, connectivity, order); the replacement meets it only at declared
interface nodes whose permanent IDs and coordinates are retained.  Conformity of
the seam is mandatory and quality limits must be stated explicitly.  A
replacement built against another geometry revision is accepted only with a
:class:`CrossRevisionAuthority` (explicit old/new owner baseline receipts per
component plus attested outside-association equivalence).

Metadata policy: ``structural_preparation`` / ``hybrid_diagnostics`` are accepted
only when they match the explicit producer schemas below (``hybrid._finalize_hybrid_candidate``
/ ``_preparation_payload`` and the ``s3_production`` qualified-S3 record, including
``NOT_APPLICABLE_NO_TRIANGLES``).  A FULL renumber translates the few mesh-ID
fields they hold (see ``MESH_ID_BEARING_METADATA``) through the explicit node/element
maps and retains the rest verbatim; geometry ids and solver indices are other
namespaces and are never translated.  An unknown field -- above all an unknown
numeric payload -- is refused, never passed silently.  A regional SPLICE edits the
shell, so ``SPLICE_RECOMPUTE_METADATA`` is stale and dropped (never translated).  A regional assembly invalidates the
whole-mesh qualification/quality evidence, so it is refused unless the caller
passes ``discard_stale_diagnostics=True``, and the discarded keys are listed in
the returned facts.

Limits (deliberate): no receipt, hash or certificate is *produced* here -- the
returned facts are plain statements.  Receipts supplied by the caller are checked
for identity consistency only; their cryptographic/geometry verification belongs
to the geometry owner that issued them.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .errors import MeshError
from .mesh import Coupling, Mesh
from .quality import MeshQuality, verify_mesh_quality
from .seeding import Seeding
from .serialize import mesh_from_dict, mesh_to_dict

__all__ = [
    "CrossRevisionAuthority",
    "OwnerBaselineReceipt",
    "RegionalAssembly",
    "RegionalQualityLimits",
    "STRUCTURED_QUALITY_COUNT_FIELDS",
    "MESH_ID_BEARING_METADATA",
    "SPLICE_RECOMPUTE_METADATA",
    "GEOMETRY_BOUND_METADATA",
    "assemble_regional_replacement",
    "remap_mesh_ids",
]

_INT64_MAX = 2**63 - 1

_KNOWN_FIELDS = frozenset(
    {
        "geometry_model_id", "geometry_revision", "nodes", "quads", "tris", "beams",
        "node_of_vertex", "nodes_of_edge", "offset_nodes_of_edge", "couplings",
        "grid_of_face", "block_grids_of_face", "elements_of_face", "elements_of_edge",
        "elements_of_sheet", "elements_of_member", "nodes_of_member", "activity",
        "thickness_of_face", "seeding", "order", "automatic_intersections",
        "declared_plate_junction_edges", "automatic_beam_connections",
        "automatic_shell_connections", "structural_preparation", "hybrid_diagnostics",
    }
)
_ELEMENT_FAMILIES = (("quads", (4, 8)), ("tris", (3, 6)), ("beams", (2, 3)))
_ELEMENTS_OF = ("elements_of_face", "elements_of_edge", "elements_of_sheet", "elements_of_member")
_KINDS = ("vertex", "edge", "face", "sheet", "member")

# Integer counters inside ``hybrid_diagnostics['structured_quality']`` that are
# explicitly known to be plain counts.  Any other integer there is an unknown
# numeric payload (it could be an element id) and is refused.
STRUCTURED_QUALITY_COUNT_FIELDS = frozenset({"growth_violation_count"})


# ----------------------------------------------------------------- validation


def _as_int(value: Any, what: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise MeshError(f"{what} must be an integer, not {value!r}")
    return int(value)


def _element_nodes(mesh: Mesh) -> Dict[int, Tuple[int, ...]]:
    conn: Dict[int, Tuple[int, ...]] = {}
    for family in ("quads", "tris", "beams"):
        for eid, nodes in getattr(mesh, family).items():
            conn[int(eid)] = tuple(int(n) for n in nodes)
    for eid, coupling in mesh.couplings.items():
        conn[int(eid)] = (int(coupling.beam_node),) + tuple(int(n) for n in coupling.plate_nodes)
    return conn


def _grid_nodes(grid: Any, zero_is_node: bool = False) -> Set[int]:
    return {int(v) for v in np.asarray(grid).ravel().tolist() if zero_is_node or int(v) != 0}


def _validate_structure(
    mesh: Mesh, label: str, allow_zero: bool = False
) -> Tuple[Set[int], Set[int]]:
    """Check that ``mesh`` is a plain, self-consistent Mesh; return node and element ids.

    ``allow_zero`` admits the label 0 for nodes and elements (producer-temporary
    numbering).  In grids 0 is the owner's "no node" marker (``Mesh.nodes_on``);
    it is a real node label only for a linear mesh that actually has node 0, and a
    quadratic mesh with node 0 plus a zero in a grid is ambiguous and refused.
    """

    if type(mesh) is not Mesh:
        raise MeshError(f"{label} must be exactly a Mesh, not {type(mesh).__name__}")
    present = {f.name for f in fields(mesh)}
    if present != _KNOWN_FIELDS:
        raise MeshError(
            f"{label} has fields this operation does not know how to renumber: "
            f"{sorted(present ^ _KNOWN_FIELDS)}"
        )
    if mesh.order not in ("linear", "quadratic"):
        raise MeshError(f"{label} has unsupported order {mesh.order!r}")
    quadratic = mesh.order == "quadratic"
    floor = 0 if allow_zero else 1

    node_ids: Set[int] = set()
    for nid, xyz in mesh.nodes.items():
        nid = _as_int(nid, f"{label} node id")
        if nid < floor:
            raise MeshError(f"{label} node id {nid} must be {'non-negative' if allow_zero else 'positive'}")
        arr = np.asarray(xyz, dtype=float)
        if arr.shape != (3,) or not np.all(np.isfinite(arr)):
            raise MeshError(f"{label} node {nid} needs a finite 3-vector coordinate")
        node_ids.add(nid)
    zero_is_node = 0 in node_ids and not quadratic

    element_ids: Set[int] = set()

    def claim(eid: Any) -> int:
        eid = _as_int(eid, f"{label} element id")
        if eid < floor:
            raise MeshError(f"{label} element id {eid} must be {'non-negative' if allow_zero else 'positive'}")
        if eid in element_ids:
            raise MeshError(f"{label} element id {eid} is used twice across element families")
        element_ids.add(eid)
        return eid

    for family, sizes in _ELEMENT_FAMILIES:
        want = sizes[1] if quadratic else sizes[0]
        for eid, nodes in getattr(mesh, family).items():
            eid = claim(eid)
            nodes = tuple(_as_int(n, f"{label} {family} {eid} node") for n in nodes)
            if len(nodes) != want:
                raise MeshError(f"{label} {family} element {eid} has {len(nodes)} nodes; {mesh.order} needs {want}")
            if len(set(nodes)) != len(nodes):
                raise MeshError(f"{label} element {eid} repeats a node")
            missing = [n for n in nodes if n not in node_ids]
            if missing:
                raise MeshError(f"{label} element {eid} references missing nodes {missing}")
    for eid, coupling in mesh.couplings.items():
        eid = claim(eid)
        if not isinstance(coupling, Coupling):
            raise MeshError(f"{label} coupling {eid} is not a Coupling")
        used = (int(coupling.beam_node),) + tuple(int(n) for n in coupling.plate_nodes)
        missing = [n for n in used if n not in node_ids]
        if missing:
            raise MeshError(f"{label} coupling {eid} references missing nodes {missing}")

    for name in ("node_of_vertex",):
        for key, nid in getattr(mesh, name).items():
            _as_int(key, f"{label} {name} key")
            if _as_int(nid, f"{label} {name} node") not in node_ids:
                raise MeshError(f"{label} {name}[{key}] references missing node {nid}")
    for name in ("nodes_of_edge", "offset_nodes_of_edge", "nodes_of_member"):
        for key, seq in getattr(mesh, name).items():
            _as_int(key, f"{label} {name} key")
            for nid in seq:
                if _as_int(nid, f"{label} {name} node") not in node_ids:
                    raise MeshError(f"{label} {name}[{key}] references missing node {nid}")
    for name in _ELEMENTS_OF:
        for key, seq in getattr(mesh, name).items():
            _as_int(key, f"{label} {name} key")
            for eid in seq:
                if _as_int(eid, f"{label} {name} element") not in element_ids:
                    raise MeshError(f"{label} {name}[{key}] references missing element {eid}")

    def check_grid(arr: np.ndarray, what: str) -> None:
        if arr.dtype.kind not in "iu":
            raise MeshError(f"{what} must hold integer node ids")
        if 0 in node_ids and quadratic and bool(np.any(arr == 0)):
            raise MeshError(
                f"{what}: 0 is both a node label and the empty-centre marker of a quadratic grid"
            )
        if not _grid_nodes(arr, zero_is_node) <= node_ids:
            raise MeshError(f"{what} references missing nodes")

    for key, grid in mesh.grid_of_face.items():
        _as_int(key, f"{label} grid_of_face key")
        check_grid(np.asarray(grid), f"{label} grid_of_face[{key}]")
    for key, blocks in mesh.block_grids_of_face.items():
        _as_int(key, f"{label} block_grids_of_face key")
        for block in blocks:
            check_grid(np.asarray(block), f"{label} block_grids_of_face[{key}]")
    for key in mesh.activity:
        if _as_int(key, f"{label} activity key") not in element_ids:
            raise MeshError(f"{label} activity key {key} is not an element id")
    for key in mesh.thickness_of_face:
        _as_int(key, f"{label} thickness_of_face key")
    for pair in mesh.declared_plate_junction_edges:
        if len(pair) != 2 or any(_as_int(n, f"{label} junction node") not in node_ids for n in pair):
            raise MeshError(f"{label} declared junction edge {pair!r} is invalid")
    return node_ids, element_ids


# ------------------------------------------------------------ metadata schemas


def _finite(value: Any, what: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise MeshError(f"{what} must be a number, not {value!r}")
    made = float(value)
    if not math.isfinite(made) or (positive and made <= 0.0):
        raise MeshError(f"{what} must be finite{' and positive' if positive else ''}")
    return made


def _string(value: Any, what: str, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str):
        raise MeshError(f"{what} must be a string, not {value!r}")


def _boolean(value: Any, what: str) -> None:
    if not isinstance(value, (bool, np.bool_)):
        raise MeshError(f"{what} must be a bool, not {value!r}")


def _mapping(value: Any, what: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise MeshError(f"{what} must be a mapping, not {type(value).__name__}")
    return value


def _geometry_face_key(key: Any, what: str) -> None:
    """Keys are *geometry* face ids (never mesh ids): an int, or its decimal text."""

    if isinstance(key, str):
        if not key.isdigit():
            raise MeshError(f"{what} key {key!r} is not a geometry face id")
        return
    _as_int(key, f"{what} key")


def _seconds(value: Any, what: str) -> None:
    for key, seconds in _mapping(value, what).items():
        _string(key, f"{what} key")
        _finite(seconds, f"{what}[{key}]")


def _v_strategy_by_face(value: Any) -> None:
    for key, strategy in _mapping(value, "hybrid_diagnostics.strategy_by_face").items():
        _geometry_face_key(key, "strategy_by_face")
        _string(strategy, f"strategy_by_face[{key}]")


class _Ids:
    """Mesh-identity context for metadata: membership checks plus optional translation.

    Only *mesh* IDs (nodes, elements) go through here.  Geometry entity IDs
    (vertex/edge/face/sheet/member/face-use) and solver indices are different
    namespaces and are never looked up or translated.  ``nm``/``em`` ``None``
    means "validate only" (identity).
    """

    def __init__(self, nodes: Set[int], elements: Set[int], tris: Set[int],
                 nm: Optional[Dict[int, int]] = None, em: Optional[Dict[int, int]] = None,
                 *, quads=()) -> None:
        self.nodes, self.elements, self.tris, self.nm, self.em = nodes, elements, tris, nm, em
        self.shells = self.tris.union(quads)

    def node(self, value: Any, what: str) -> int:
        made = _as_int(value, what)
        if made not in self.nodes:
            raise MeshError(f"{what} {made} is not a node of the mesh; metadata is stale")
        return made if self.nm is None else self.nm[made]

    def element(self, value: Any, what: str, triangle: bool = False) -> int:
        made = _as_int(value, what)
        if made not in (self.tris if triangle else self.elements):
            raise MeshError(f"{what} {made} is not a {'triangle' if triangle else 'element'} of the mesh; metadata is stale")
        return made if self.em is None else self.em[made]

    def shell(self, value, what):
        made = _as_int(value, what)
        if made not in self.shells:
            raise MeshError(f"{what} {made} is not a shell of the mesh; metadata is stale")
        return made if self.em is None else self.em[made]


def _list(value: Any, what: str) -> Sequence:
    if not isinstance(value, (list, tuple)):
        raise MeshError(f"{what} must be a list, not {type(value).__name__}")
    return value


def _count(value: Any, what: str) -> int:
    made = _as_int(value, what)
    if made < 0:
        raise MeshError(f"{what} must be non-negative")
    return made


def _geometry_only(node: Any, path: str) -> None:
    """Accept a JSON-safe subtree that was produced BEFORE any mesh existed.

    Used for structured-layout ``plan``/``options``: they are bound to a geometry
    revision and written before nodes/elements are numbered, so their integers
    are geometry IDs/settings and cannot be mesh IDs.
    """

    if node is None or isinstance(node, (str, bool, np.bool_, int, np.integer)):
        return
    if isinstance(node, (float, np.floating)):
        _finite(node, path)
        return
    if isinstance(node, Mapping):
        for key, item in node.items():
            if not isinstance(key, (str, int, np.integer)):
                raise MeshError(f"{path} has a non-scalar key {key!r}")
            _geometry_only(item, f"{path}.{key}")
        return
    if isinstance(node, (list, tuple)):
        for index, item in enumerate(node):
            _geometry_only(item, f"{path}[{index}]")
        return
    raise MeshError(f"{path} carries unsupported {type(node).__name__}")


_BACKEND_REQUIRED = frozenset({"requested_backend", "selected_backend", "actual_backend", "fallback_reason"})
_BACKEND_TOP_ONLY = frozenset({"structured_action", "working_face_ids", "working_face_diagnostics"})
_BACKEND_OPTIONAL = frozenset({"phase_seconds"}) | _BACKEND_TOP_ONLY


def _report_fields(names, validator):
    return dict.fromkeys(names.split(), validator)


def _native_seconds(value, path):
    try:
        _finite(value, path)
    except OverflowError as error:
        raise MeshError(f"{path} exceeds the finite timing range") from error
    if value < 0:
        raise MeshError(f"{path} must be nonnegative")


_COMPILED_TRIANGULATION_DIAGNOSTICS = {
    **_report_fields("domain_filter_seconds insertion_seconds segment_recovery_seconds", _native_seconds),
    "workspace_bytes": _count,
}


def _v_native_diagnostics(value, path):
    # Reference triangulation and older artifacts carry an empty diagnostics mapping.
    if _mapping(value, path):
        _validate_report(value, path, _COMPILED_TRIANGULATION_DIAGNOSTICS)


# surface_mesh emits this diagnostic before hybrid._publish_native_face_elements
# allocates any project node or element IDs. Its candidate indices, including
# poor_element_ids, identify rows in each candidate's own MeshCore. Keep them
# in that namespace. The enclosing whole-mesh structured_quality and S3
# records are different: their cell IDs require global translation.
_LOCAL_QUALITY = {
    **_report_fields("elements_above_aspect_ratio_5 elements_above_maximum_angle "
                     "elements_above_maximum_growth elements_below_minimum_angle "
                     "elements_below_minimum_scaled_jacobian invalid_element_count "
                     "quality_violation_count", _count),
    **_report_fields("max_angle max_aspect_ratio max_element_growth min_angle min_scaled_jacobian", _finite),
    "poor_element_ids": [_count], "repair_element_ids": [_count],
}
_LOCAL_WORK = _report_fields("full_triangulations optimization_noop_short_circuits quality_evaluations", _count)
_LOCAL_LIMITS = _report_fields("maximum_angle maximum_aspect_ratio maximum_warpage "
                               "minimum_angle minimum_scaled_jacobian", _finite)
_LOCAL_OPTIMIZATION = {
    **_report_fields("added_points attempted_added_points attempted_rounds candidate_count "
                     "flips initial_generated_points moved_nodes point_budget rounds", _count),
    **_report_fields("budget_exhausted target_met", _boolean),
    "selected_strategy": _string, "effective_target_size": _finite,
    "boundary_collar_skip_reason": lambda v, p: _string(v, p, optional=True),
    "boundary_collars": [], "published_alignment": None,
    "moved_node_rows": [_count], "work_totals": _LOCAL_WORK,
    "initial_quality": _LOCAL_QUALITY, "final_quality": _LOCAL_QUALITY,
    "candidates": [{"strategy": _string, "generated_points": _count, "target_met": _boolean,
                    "score": [_finite], "final_quality": _LOCAL_QUALITY, "work_statistics": _LOCAL_WORK}],
    "complex_geometry": {
        **_report_fields("boundary_segment_count candidate_budget candidate_count hole_count "
                         "hole_segment_count mandatory_segment_count", _count),
        "alignment_evaluation": _string, "declared_junction": _boolean,
    },
    "edge_guides": {
        **_report_fields("accepted_points distance_checks guide_count skipped_guides", _count),
        "guide_ids": [], "soft_symmetry": _boolean,
    },
    "lattice_statistics": _report_fields("accepted_points candidate_points distance_checks "
                                         "naive_distance_checks protected_segments", _count),
}
_NATIVE_FACE_REPORT = {
    "quality_optimization": _LOCAL_OPTIMIZATION,
    "quality_policy": {
        "accepted": _boolean, "poor_element_ids": [_count], "thresholds": _LOCAL_LIMITS,
        "worst": _LOCAL_LIMITS, "violation_counts": _report_fields(
            "aspect_ratio maximum_angle minimum_angle scaled_jacobian warpage", _count),
    },
    "lattice_alignment": _string, "metric_tensor": None, "max_lattice_points": _count,
    "recombination": {
        **_report_fields("candidate_count exchange_count exchange_work pair_count rejected_candidate_count", _count),
        "exchange_truncated": _boolean,
    },
    "native_diagnostics": _v_native_diagnostics,
}
_LOCAL_ALIGNMENT = {
    **_report_fields("edge_count quad_edge_count", _count),
    **_report_fields("maximum_normal_error_degrees mean_normal_error_degrees quad_fraction", _finite),
}
_LOCAL_COLLAR = {
    **_report_fields("accepted_points collar_points collar_segments removed_lattice_points requested_layers "
                     "shortened_rows transition_points transition_regions", _count),
    "preparation_cache_hit": _boolean, "holes": [],
    "outer": {**_report_fields("accepted_layers released_corner_stations skipped_layers", _count),
              "loop": _string, "stop_reason": lambda v, p: _string(v, p, optional=True)},
}
_LOCAL_RECOMBINED_CANDIDATE = {
    **_LOCAL_OPTIMIZATION["candidates"][0],
    **_report_fields("alignment_qualified mandatory_physical_quality physical_admissible quality_eligible", _boolean),
    "max_element_growth": _finite, "physical_triangle_retention": None,
    "hole_alignment": _LOCAL_ALIGNMENT, "outer_alignment": _LOCAL_ALIGNMENT,
    "published_growth": {"elements_above_maximum": _count, "maximum": _finite},
    "published_quality": _NATIVE_FACE_REPORT["quality_policy"],
}
_RECOMBINED_FACE_REPORT = {
    **_NATIVE_FACE_REPORT,
    "recombination": {**_NATIVE_FACE_REPORT["recombination"], "physical_triangle_retention": None},
    "quality_optimization": {
        **_LOCAL_OPTIMIZATION, "boundary_collars": [_LOCAL_COLLAR],
        "candidates": [_LOCAL_RECOMBINED_CANDIDATE], "triangulation_quality": _LOCAL_QUALITY,
        "final_quality": {**_LOCAL_QUALITY, "quality_scope": _string},
        "published_alignment": {"holes": _LOCAL_ALIGNMENT, "outer": _LOCAL_ALIGNMENT},
    },
}


def _validate_report(value, path, schema):
    if isinstance(schema, dict):
        record = _mapping(value, path)
        if set(record) != set(schema):
            raise MeshError(f"{path} has unknown or missing fields {sorted(set(record) ^ set(schema), key=str)}")
        for key, item in record.items():
            _validate_report(item, f"{path}.{key}", schema[key])
    elif isinstance(schema, list):
        items = _list(value, path)
        if not schema and items:
            raise MeshError(f"{path} has no documented nonempty schema")
        for item in items:
            _validate_report(item, f"{path} item", schema[0])
    elif schema is None:
        if value is not None:
            raise MeshError(f"{path} has no documented non-null schema")
    else:
        schema(value, path)


def _nullable(check):
    def validate(value, path):
        if value is not None:
            check(value, path)
    return validate


def _v_threadpools(value, path):
    # threadpoolctl introspection can be absent. Each supported library row
    # describes a loaded binary and its thread count, independently of mesh IDs.
    fields = _report_fields("user_api internal_api prefix filepath version threading_layer "
                            "architecture threading_backend", _nullable(_string))
    for row in _list(value, path):
        row = _mapping(row, path)
        if set(row) - (set(fields) | {"num_threads"}) or "num_threads" not in row:
            raise MeshError(f"{path} contains an unknown library record")
        _count(row["num_threads"], path)
        for key in set(row) & set(fields):
            fields[key](row[key], path)


def _v_worker_controls(value, path):
    schema = {
        **_report_fields("containment priority_requested priority_class priority_mechanism "
                         "priority_status priority_error", _nullable(_string)),
        **_report_fields("nice_before nice_requested nice_effective", _nullable(_as_int)),
        "priority_applied": _nullable(_boolean),
        "thread_env": _report_fields("OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS", _nullable(_string)),
        "threadpool": _nullable(_v_threadpools),
    }
    for key, worker in _mapping(value, path).items():
        if not isinstance(key, (int, np.integer, str)) or not str(key).isascii() or not str(key).isdecimal():
            raise MeshError(f"{path} has an invalid worker index")
        _validate_report(worker, f"{path}[{key}]", schema)


def _v_lifecycle(value, path):
    entries = {"workers-spawned": (), "pool-ready": (), "job-begin": (), "job-end": (),
               "cancel-signalled": (), "queued-cancelled": (_count,),
               "shutdown-begin": (_string,), "shutdown-end": (_count,)}
    for row in _list(value, path):
        event = _list(row, path)
        if not event or not isinstance(event[0], str) or event[0] not in entries:
            raise MeshError(f"{path} contains an unknown pool event")
        details = entries[event[0]]
        if len(event) != 2 + len(details):
            raise MeshError(f"{path} has an invalid {event[0]} event")
        _finite(event[1], path)
        for check, item in zip(details, event[2:]):
            check(item, path)


def _v_parallel(value):
    where = "hybrid_diagnostics.parallel"
    record = _mapping(value, where)
    if record.get("used") is False and record.get("route") == "serial":
        _validate_report(record, where, {"used": _boolean, "route": _string, "reason": _string})
        return
    if record.get("used") is not True or record.get("route") != "parallel":
        raise MeshError(f"{where} is not a serial fallback or parallel join")
    _validate_report(record, where, {
        "used": _boolean, "route": _string, "components": _count, "separation": _finite,
        "merge_reasons": _count,
        "runtime": {
            "pool": _string, "workers": _count, "worker_pids": [_count], "worker_controls": _v_worker_controls,
            **_report_fields("plan_seconds prepare_seconds mesh_seconds join_seconds", _finite),
            "component_seconds": [_finite], "component_estimated_elements": [_nullable(_count)],
            "component_estimate_basis": [_string], "worker_working_set_bytes": [_nullable(_count)],
            "lifecycle": _v_lifecycle,
        },
    })
    runtime = record["runtime"]
    workers = runtime["workers"]
    if (workers < 1 or len(runtime["worker_pids"]) != workers or 0 in runtime["worker_pids"]
            or len(set(runtime["worker_pids"])) != workers
            or {str(k) for k in runtime["worker_controls"]} != {str(i) for i in range(workers)}
            or len(runtime["worker_working_set_bytes"]) != workers):
        raise MeshError(f"{where} has inconsistent worker evidence")
    if record["components"] < 2 or any(len(runtime[key]) != record["components"] for key in (
            "component_seconds", "component_estimated_elements", "component_estimate_basis")):
        raise MeshError(f"{where} has inconsistent component evidence")


def _backend_record(record: Any, what: str, nested: bool) -> None:
    record = _mapping(record, what)
    allowed = _BACKEND_REQUIRED | (_BACKEND_OPTIONAL - (_BACKEND_TOP_ONLY if nested else frozenset()))
    keys = set(record)
    if keys.intersection(_NATIVE_FACE_REPORT):
        if not set(_NATIVE_FACE_REPORT) <= keys:
            raise MeshError(f"{what} has an incomplete native surface report")
        recombination = _mapping(record["recombination"], f"{what}.recombination")
        report = _RECOMBINED_FACE_REPORT if "physical_triangle_retention" in recombination else _NATIVE_FACE_REPORT
        for name, schema in report.items():
            _validate_report(record[name], f"{what}.{name}", schema)
        allowed = allowed.union(_NATIVE_FACE_REPORT)
    if not _BACKEND_REQUIRED <= keys or keys - allowed:
        raise MeshError(
            f"{what} is not the documented mapped-face record "
            f"(unknown or missing fields {sorted((keys - allowed) | (_BACKEND_REQUIRED - keys), key=str)}); "
            "only a full rebuild can carry it"
        )
    for name in ("requested_backend", "selected_backend", "actual_backend"):
        _string(record[name], f"{what}.{name}")
    _string(record["fallback_reason"], f"{what}.fallback_reason", optional=True)
    if "phase_seconds" in record:  # the persisted production record strips wall-clock samples
        _seconds(record["phase_seconds"], f"{what}.phase_seconds")
    if "structured_action" in record:
        _string(record["structured_action"], f"{what}.structured_action")
    if "working_face_ids" in record:  # geometry face ids of the working clone
        for item in _list(record["working_face_ids"], f"{what}.working_face_ids"):
            _as_int(item, f"{what}.working_face_ids item")
    if "working_face_diagnostics" in record:
        for index, item in enumerate(_list(record["working_face_diagnostics"], f"{what}.working_face_diagnostics")):
            _backend_record(item, f"{what}.working_face_diagnostics[{index}]", True)


def _v_backend_by_face(value: Any) -> None:
    for key, record in _mapping(value, "hybrid_diagnostics.triangulation_backend_by_face").items():
        _geometry_face_key(key, "triangulation_backend_by_face")
        _backend_record(record, f"triangulation_backend_by_face[{key}]", False)


def _quality_walk(node: Any, ids: _Ids, path: str, key: str = "", policy: bool = False) -> Any:
    """Validate (and translate) a structured-quality report; return the translated copy.

    The only mesh-ID payloads of the documented report are ``poor_element_ids`` and
    ``growth_violation_pairs`` ([element, element, ratio]).  Plain counters are
    ``growth_violation_count`` and the ``violation_counts`` mapping.  ``policy`` is
    configuration.  Any other integer or sequence is refused.
    """

    if node is None or isinstance(node, (str, bool, np.bool_)):
        return node
    if isinstance(node, (float, np.floating)):
        _finite(node, path)
        return float(node)
    if isinstance(node, (int, np.integer)):
        if policy:
            return int(node)
        if key in STRUCTURED_QUALITY_COUNT_FIELDS:
            return _count(node, path)
        raise MeshError(f"{path} carries an unknown integer payload {node!r}")
    if isinstance(node, Mapping):
        out: Dict[Any, Any] = {}
        for name, item in node.items():
            _string(name, f"{path} key")
            where = f"{path}.{name}"
            if not policy and name == "poor_element_ids":
                out[name] = sorted(ids.element(x, f"{where} item") for x in _list(item, where))
            elif not policy and name == "growth_violation_pairs":
                pairs = []
                for pair in _list(item, where):
                    pair = _list(pair, f"{where} item")
                    if len(pair) != 3:
                        raise MeshError(f"{where} item must be [element, element, ratio]")
                    ratio = _finite(pair[2], f"{where} ratio")
                    pairs.append([ids.element(pair[0], f"{where} element"), ids.element(pair[1], f"{where} element"), ratio])
                out[name] = pairs
            elif not policy and name == "violation_counts":
                out[name] = {
                    str(k): _count(v, f"{where}[{k}]") for k, v in _mapping(item, where).items()
                }
            else:
                out[name] = _quality_walk(item, ids, where, name, policy or name == "policy")
        return out
    raise MeshError(
        f"{path} carries an unknown {type(node).__name__} payload {node!r}; "
        "integers and sequences are not accepted unless explicitly known mesh-ID fields or counters"
    )


def _x_structured_quality(value: Any, ids: _Ids) -> Any:
    return _quality_walk(value, ids, "hybrid_diagnostics.structured_quality")


def _v_complex_geometry(value: Any, *, joined=False) -> None:
    where = "hybrid_diagnostics.complex_geometry"
    record = _mapping(value, where)
    counts = {"source_face_count", "final_face_count", "declared_junction_edge_count"}
    expected = {"strategy_ladder", *counts}
    if not joined:
        counts.add("component_candidate_budget")
        expected.update(("source_strategy_by_face", "component_candidate_budget"))
    if set(record) != expected:
        raise MeshError(f"{where} has unknown or missing fields {sorted(set(record) ^ expected, key=str)}")
    for name in counts:
        _count(record[name], f"{where}.{name}")
    for item in _list(record["strategy_ladder"], f"{where}.strategy_ladder"):
        _string(item, f"{where}.strategy_ladder item")
    for key, strategy in _mapping(record.get("source_strategy_by_face", {}), f"{where}.source_strategy_by_face").items():
        _geometry_face_key(key, where)
        _string(strategy, f"{where}.source_strategy_by_face[{key}]")


_S3_SUMMARY_KEYS = frozenset({"contract_id", "element_count", "formulation_id", "legacy_fallback", "status"})


def _v_s3_summary(value: Any) -> None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return
    record = _mapping(value, "hybrid_diagnostics.qualified_s3_preparation")
    if set(record) != _S3_SUMMARY_KEYS:
        raise MeshError(
            "hybrid_diagnostics.qualified_s3_preparation has unknown or missing fields "
            f"{sorted(set(record) ^ _S3_SUMMARY_KEYS, key=str)}"
        )
    _count(record["element_count"], "qualified_s3_preparation.element_count")
    for name in ("contract_id", "formulation_id", "legacy_fallback", "status"):
        _string(record[name], f"qualified_s3_preparation.{name}")


_RUNTIME_COUNTS = ("candidate_count", "connectivity_application_count")


def _v_runtime(value: Any) -> None:
    record = _mapping(value, "hybrid_diagnostics.runtime")
    allowed = {"operation_counts", "phase_seconds", *_RUNTIME_COUNTS}
    if set(record) - allowed:
        raise MeshError(f"hybrid_diagnostics.runtime has unknown fields {sorted(set(record) - allowed, key=str)}")
    for name in _RUNTIME_COUNTS:
        if name in record:
            _count(record[name], f"runtime.{name}")
    if "phase_seconds" in record:
        _seconds(record["phase_seconds"], "runtime.phase_seconds")
    for key, count in _mapping(record.get("operation_counts", {}), "runtime.operation_counts").items():
        _string(key, "runtime.operation_counts key")
        if _as_int(count, f"runtime.operation_counts[{key}]") < 0:
            raise MeshError(f"runtime.operation_counts[{key}] must be non-negative")


def _v_nonneg_int(what: str):
    def check(value: Any) -> None:
        if _as_int(value, what) < 0:
            raise MeshError(f"{what} must be non-negative")
    return check


def _v_revision(value: Any) -> None:
    if value is not None:
        _v_nonneg_int("hybrid_diagnostics.geometry_revision")(value)


def _v_model_id(value: Any) -> None:
    if value is not None and not isinstance(value, str) and type(value).__name__ != "UUID":
        raise MeshError(f"hybrid_diagnostics.geometry_model_id must be a string or UUID, not {value!r}")


_AUTOMATION_KEYS = frozenset({
    "attempts", "contract_id", "element_families", "mesh_validity", "quality_warnings",
    "selected_method", "solver_admission", "status",
})
_AUTOMATION_FAMILIES = frozenset({"beams", "couplings", "quads", "triangles"})


_ATTEMPT_SHAPES = {
    "selected": frozenset({"method", "status"}),
    "rejected": frozenset({"method", "status", "error_type", "reason", "problem_element_ids"}),
    "inspection_only": frozenset({"method", "status", "problem_element_ids", "quality_warnings", "reason"}),
}


def _x_automation(value: Any, ids: _Ids) -> Any:
    """recovery.AutomaticMeshResult.to_dict(): strings, per-family element counts and
    an attempt log whose only mesh ids are ``problem_element_ids`` (translated).
    Free-text ``reason``/warnings are retained verbatim: they are prose, not data."""

    where = "hybrid_diagnostics.automation"
    record = _mapping(value, where)
    if set(record) != _AUTOMATION_KEYS:
        raise MeshError(f"{where} is not the documented record ({sorted(set(record) ^ _AUTOMATION_KEYS, key=str)})")
    for name in ("contract_id", "mesh_validity", "selected_method", "solver_admission", "status"):
        _string(record[name], f"{where}.{name}")
    families = _mapping(record["element_families"], f"{where}.element_families")
    if set(families) != _AUTOMATION_FAMILIES:
        raise MeshError(f"{where}.element_families is not the documented record")
    for name, count in families.items():
        _count(count, f"{where}.element_families.{name}")
    attempts = []
    for index, attempt in enumerate(_list(record["attempts"], f"{where}.attempts")):
        here = f"{where}.attempts[{index}]"
        attempt = _mapping(attempt, here)
        if attempt.get("status") not in _ATTEMPT_SHAPES or set(attempt) != _ATTEMPT_SHAPES[attempt["status"]]:
            raise MeshError(f"{here} is not a documented attempt record")
        made = {}
        for name, item in attempt.items():
            if name == "problem_element_ids":
                made[name] = sorted(ids.element(x, f"{here}.problem_element_ids item") for x in _list(item, here))
            elif name == "quality_warnings":
                made[name] = [_checked_str(x) for x in _list(item, f"{here}.quality_warnings")]
            else:
                _string(item, f"{here}.{name}")
                made[name] = item
        attempts.append(made)
    warnings = [_checked_str(x) for x in _list(record["quality_warnings"], f"{where}.quality_warnings")]
    out = copy.deepcopy(dict(record))
    out["attempts"], out["quality_warnings"] = attempts, warnings
    return out


_HYBRID_SCHEMA = {
    "parallel": _v_parallel,
    "requested_target_size": lambda v: _finite(v, "requested_target_size", positive=True),
    "strategy_by_face": _v_strategy_by_face,
    "triangulation_backend_by_face": _v_backend_by_face,
    "geometry_model_id": _v_model_id,
    "geometry_revision": _v_revision,
    "certification_mode": lambda v: _string(v, "certification_mode"),
    "certifiable": lambda v: _boolean(v, "certifiable"),
    "preflight_count": _v_nonneg_int("preflight_count"),
    "structured_layout_status": lambda v: _string(v, "structured_layout_status", optional=True),
    "structured_plan_hash": lambda v: _string(v, "structured_plan_hash", optional=True),
    "complex_geometry": _v_complex_geometry,
    "structural_preparation_hash": lambda v: _string(v, "structural_preparation_hash", optional=True),
    "qualified_s3_preparation": _v_s3_summary,
    "reused_prepared_working_copy": lambda v: _boolean(v, "reused_prepared_working_copy"),
    "phase_seconds": lambda v: _seconds(v, "hybrid_diagnostics.phase_seconds"),
    "completed_phases": lambda v: [_string(x, "completed_phases item") for x in v]
    if isinstance(v, (list, tuple))
    else _string(None, "completed_phases (a list)"),
    "runtime": _v_runtime,
}
_HYBRID_ID_SCHEMA = {"structured_quality": _x_structured_quality, "automation": _x_automation}

# ---- structural_preparation (hybrid._preparation_payload + s3_production record)


def _x_lineage(value: Any, ids: _Ids, where: str) -> Any:
    """source_to_working_faces/edges: geometry ids -> geometry ids (no mesh ids)."""

    for key, values in _mapping(value, where).items():
        _geometry_face_key(key, where)
        for item in _list(values, f"{where}[{key}]"):
            _as_int(item, f"{where}[{key}] item")
    return copy.deepcopy(value)


_OPTION_BOOLS = (
    "automatic_face_connections", "automatic_member_connections",
    "automatic_member_sheet_connections", "declare_missing_owners",
)
_OPTION_LIMITS = ("maximum_candidate_pairs", "maximum_applications", "maximum_face_records", "maximum_edge_records")
_CLOSURE_COUNTS = (
    "candidate_queries", "applications", "face_connections", "member_connections",
    "member_sheet_connections",
)
_CLOSURE_KEYS = frozenset({
    "model_id", "source_revision", "working_revision", "options", "source_to_working_faces",
    "source_to_working_edges", "temporary_sheet_ids", "temporary_member_ids",
    "declared_face_connection_edges", "diagnostics", "status", "preparation_hash", *_CLOSURE_COUNTS,
})


def _x_closure(value: Any, ids: _Ids, where: str) -> Any:
    """StructuralPreparationReport.to_dict(): geometry ids, flags and counters only."""

    if value is None:
        return None
    record = _mapping(value, where)
    if set(record) != _CLOSURE_KEYS:
        raise MeshError(f"{where} is not the documented closure record "
                        f"({sorted(set(record) ^ _CLOSURE_KEYS, key=str)})")
    _string(record["model_id"], f"{where}.model_id")
    _count(record["source_revision"], f"{where}.source_revision")
    _count(record["working_revision"], f"{where}.working_revision")
    options = _mapping(record["options"], f"{where}.options")
    if set(options) != set(_OPTION_BOOLS) | set(_OPTION_LIMITS):
        raise MeshError(f"{where}.options is not the documented option record")
    for name in _OPTION_BOOLS:
        _boolean(options[name], f"{where}.options.{name}")
    for name in _OPTION_LIMITS:
        if options[name] is not None:
            _count(options[name], f"{where}.options.{name}")
    _x_lineage(record["source_to_working_faces"], ids, f"{where}.source_to_working_faces")
    _x_lineage(record["source_to_working_edges"], ids, f"{where}.source_to_working_edges")
    for name in ("temporary_sheet_ids", "temporary_member_ids", "declared_face_connection_edges"):
        for item in _list(record[name], f"{where}.{name}"):
            _as_int(item, f"{where}.{name} item")  # geometry ids
    for name in _CLOSURE_COUNTS:
        _count(record[name], f"{where}.{name}")
    for item in _list(record["diagnostics"], f"{where}.diagnostics"):
        _string(item, f"{where}.diagnostics item")
    _string(record["status"], f"{where}.status")
    _string(record["preparation_hash"], f"{where}.preparation_hash")
    return copy.deepcopy(value)


# structured.regularity_metrics(): the one integer counter not named ``*_count``.
_METRIC_COUNTERS = frozenset({"edges_outside_balanced_size_band"})


def _metrics_walk(node: Any, path: str, key: str = "") -> None:
    if node is None or isinstance(node, (str, bool, np.bool_)):
        return
    if isinstance(node, (float, np.floating)):
        _finite(node, path)
        return
    if isinstance(node, (int, np.integer)):
        if key.endswith("_count") or key in _METRIC_COUNTERS:
            _count(node, path)
            return
        raise MeshError(f"{path} carries an unknown integer payload {node!r}")
    if isinstance(node, Mapping):
        for name, item in node.items():
            _string(name, f"{path} key")
            _metrics_walk(item, f"{path}.{name}", name)
        return
    raise MeshError(f"{path} carries an unknown {type(node).__name__} payload; sequences are refused")


# StructuredLayoutReport.to_dict() members built from the plan classes
# (StructuredLayoutPlan/Block/Interface/SeedEquation) and the edge seed solution
# before any mesh exists: geometry ids, divisions and text only.
_LAYOUT_GEOMETRY_ONLY = frozenset({
    "plan", "options", "blocks", "interfaces", "seed_equations", "seed_solution",
})


def _x_layout(value: Any, ids: _Ids, where: str) -> Any:
    """StructuredLayoutReport.to_dict().

    ``plan``/``options`` precede meshing (geometry-only); ``quality`` is translated;
    ``metrics`` may only hold floats/strings/booleans/``*_count`` counters; any other
    field must be a scalar string/bool/float.  Everything else is refused.
    """

    if value is None:
        return None
    out: Dict[Any, Any] = {}
    for key, item in _mapping(value, where).items():
        _string(key, f"{where} key")
        here = f"{where}.{key}"
        if key in _LAYOUT_GEOMETRY_ONLY:
            _geometry_only(item, here)
            out[key] = copy.deepcopy(item)
        elif key in ("source_to_working_faces", "source_to_working_edges"):
            out[key] = _x_lineage(item, ids, here)
        elif key == "diagnostics":
            out[key] = [_checked_str(x) for x in _list(item, here)]
        elif key == "working_revision":
            out[key] = _count(item, here)
        elif key == "quality":
            out[key] = None if item is None else _quality_walk(item, ids, here)
        elif key == "metrics":
            _metrics_walk(item, here)
            out[key] = copy.deepcopy(item)
        elif item is None or isinstance(item, (str, bool, np.bool_)):
            out[key] = item
        elif isinstance(item, (float, np.floating)):
            _finite(item, here)
            out[key] = float(item)
        else:
            raise MeshError(f"{here} has no explicit schema; refusing to translate or retain it")
    return out


_S3_SCHEMA = "anymesher.qualified-s3-production-preparation-v2"
_S3_COMMON = frozenset({
    "admission", "authority_model", "contract_id", "element_ids", "formulation_id",
    "legacy_fallback", "quality_contract_id", "repair_contract_id", "schema", "status",
})
_S3_ADMITTED_ONLY = frozenset({
    "element_owner_normals", "element_owner_sources", "nodal_normals", "quality_policy",
    "quality_target", "repair",
})
_S3_QUALITY_FLOATS = (
    "edge_ratio", "maximum_angle_deg", "minimum_angle_deg", "minimum_scaled_jacobian",
    "normalized_area", "owner_normal_alignment", "signed_area_ratio",
)
_S3_POLICY_KEYS = frozenset({
    "maximum_angle_deg", "maximum_edge_ratio", "minimum_angle_deg", "minimum_normalized_area",
    "minimum_scaled_jacobian",
})
_S3_REPAIR_COUNTS = (
    "added_elements", "added_nodes", "edge_flip_attempts", "edge_flips", "refinement_attempts",
    "refinement_splits", "winding_repairs",
)


def _vec3(value: Any, where: str) -> List[float]:
    seq = _list(value, where)
    if len(seq) != 3:
        raise MeshError(f"{where} must be a 3-vector")
    return [_finite(x, where) for x in seq]


def _s3_policy(value: Any, where: str) -> Dict[str, float]:
    record = _mapping(value, where)
    if set(record) != _S3_POLICY_KEYS:
        raise MeshError(f"{where} is not the documented S3 policy record")
    return {k: _finite(record[k], f"{where}.{k}") for k in record}


def _x_qualified_s3(value: Any, ids: _Ids, where: str) -> Any:
    """Qualified-S3 admission record (``s3_production``): NOT_APPLICABLE_NO_TRIANGLES or ADMITTED.

    Mesh-ID fields are checked against, and translated through, the mesh:
    ``element_ids`` (must be exactly the triangles), ``admission.elements[].element_id``,
    ``admission.qualified_junction_edges`` (nodes), ``element_owner_normals``/
    ``element_owner_sources`` keys (elements), ``nodal_normals`` keys (nodes),
    ``quality_target.shortfall_element_ids`` and ``repair.attempts[].element_ids/edge``.
    Geometry ids (``face_id``, ``face_use_ids``, ``sheet_ids``) and revisions are retained.
    Repair attempts naming an element/node that no longer exists are refused (fail closed).
    """

    record = _mapping(value, where)
    status = record.get("status")
    if status == "NOT_APPLICABLE_NO_TRIANGLES":
        expected = _S3_COMMON
    elif status == "ADMITTED":
        expected = _S3_COMMON | _S3_ADMITTED_ONLY
    else:
        raise MeshError(f"{where}.status {status!r} is not a known qualified-S3 status")
    if set(record) != expected:
        raise MeshError(f"{where} is not the documented {status} record "
                        f"({sorted(set(record) ^ expected, key=str)})")
    if record["schema"] != _S3_SCHEMA:
        raise MeshError(f"{where}.schema {record['schema']!r} is not {_S3_SCHEMA}")
    for name in ("contract_id", "formulation_id", "legacy_fallback", "quality_contract_id", "repair_contract_id"):
        _string(record[name], f"{where}.{name}")
    authority = _mapping(record["authority_model"], f"{where}.authority_model")
    if not {"prepared_revision", "scope"} <= set(authority) or set(authority) - {
        "prepared_revision", "scope", "source_model_id", "source_revision"
    }:
        raise MeshError(f"{where}.authority_model is not the documented record")
    _count(authority["prepared_revision"], f"{where}.authority_model.prepared_revision")
    _string(authority["scope"], f"{where}.authority_model.scope")
    if "source_model_id" in authority:
        _string(authority["source_model_id"], f"{where}.authority_model.source_model_id")
    if "source_revision" in authority:
        _count(authority["source_revision"], f"{where}.authority_model.source_revision")

    raw_elements = [_as_int(x, f"{where}.element_ids item") for x in _list(record["element_ids"], f"{where}.element_ids")]
    if len(set(raw_elements)) != len(raw_elements) or set(raw_elements) != ids.tris:
        raise MeshError(f"{where}.element_ids do not describe exactly the triangles of this mesh; "
                        "the admission is stale and must be recomputed")
    out: Dict[str, Any] = {
        k: copy.deepcopy(record[k])
        for k in ("authority_model", "contract_id", "formulation_id", "legacy_fallback",
                  "quality_contract_id", "repair_contract_id", "schema", "status")
    }
    out["element_ids"] = sorted(ids.element(x, f"{where}.element_ids item", True) for x in raw_elements)

    admission = _mapping(record["admission"], f"{where}.admission")
    if set(admission) != {"elements", "qualified_junction_edges", "topology_violations"}:
        raise MeshError(f"{where}.admission is not the documented record")
    rows = []
    for index, item in enumerate(_list(admission["elements"], f"{where}.admission.elements")):
        item = _mapping(item, f"{where}.admission.elements[{index}]")
        if set(item) != {"element_id", "violations", *_S3_QUALITY_FLOATS}:
            raise MeshError(f"{where}.admission.elements[{index}] is not the documented quality record")
        row = {name: _finite(item[name], f"{where}.admission.elements[{index}].{name}") for name in _S3_QUALITY_FLOATS}
        row["element_id"] = ids.element(item["element_id"], f"{where}.admission.elements[{index}].element_id", True)
        row["violations"] = [str(v) if isinstance(v, str) else _string(v, "violation") for v in _list(item["violations"], "violations")]
        rows.append(row)
    junctions = []
    for pair in _list(admission["qualified_junction_edges"], f"{where}.admission.qualified_junction_edges"):
        pair = _list(pair, f"{where} junction edge")
        if len(pair) != 2:
            raise MeshError(f"{where} junction edge must be a node pair")
        a, b = ids.node(pair[0], f"{where} junction node"), ids.node(pair[1], f"{where} junction node")
        junctions.append([a, b])
    out["admission"] = {
        "elements": sorted(rows, key=lambda r: r["element_id"]),
        "qualified_junction_edges": junctions if ids.nm is None else sorted([min(p), max(p)] for p in junctions),
        "topology_violations": [x if isinstance(x, str) else _string(x, "topology violation")
                                for x in _list(admission["topology_violations"], "topology_violations")],
    }
    if status == "NOT_APPLICABLE_NO_TRIANGLES":
        if raw_elements or rows or junctions or out["admission"]["topology_violations"]:
            raise MeshError(f"{where}: NOT_APPLICABLE_NO_TRIANGLES must be empty")
        return out

    normals = {}
    for key, vec in _mapping(record["element_owner_normals"], f"{where}.element_owner_normals").items():
        normals[ids.shell(_idkey(key), f"{where}.element_owner_normals key")] = _vec3(vec, f"{where}.element_owner_normals[{key}]")
    sources = {}
    for key, src in _mapping(record["element_owner_sources"], f"{where}.element_owner_sources").items():
        src = _mapping(src, f"{where}.element_owner_sources[{key}]")
        if set(src) != {"face_id", "face_use_ids", "sheet_ids"}:
            raise MeshError(f"{where}.element_owner_sources[{key}] is not the documented record")
        _as_int(src["face_id"], "geometry face id")
        for name in ("face_use_ids", "sheet_ids"):
            for item in _list(src[name], f"{where}.{name}"):
                _as_int(item, f"{where}.{name} item")  # geometry ids
        sources[ids.shell(_idkey(key), f"{where}.element_owner_sources key")] = copy.deepcopy(dict(src))
    nodal = {}
    for key, vec in _mapping(record["nodal_normals"], f"{where}.nodal_normals").items():
        nodal[ids.node(_idkey(key), f"{where}.nodal_normals key")] = _vec3(vec, f"{where}.nodal_normals[{key}]")
    out["element_owner_normals"] = {str(k): normals[k] for k in sorted(normals)}
    out["element_owner_sources"] = {str(k): sources[k] for k in sorted(sources)}
    out["nodal_normals"] = {str(k): nodal[k] for k in sorted(nodal)}
    out["quality_policy"] = _s3_policy(record["quality_policy"], f"{where}.quality_policy")
    target = _mapping(record["quality_target"], f"{where}.quality_target")
    if set(target) != {"met", "policy", "shortfall_element_ids"}:
        raise MeshError(f"{where}.quality_target is not the documented record")
    _boolean(target["met"], f"{where}.quality_target.met")
    out["quality_target"] = {
        "met": bool(target["met"]),
        "policy": None if target["policy"] is None else _s3_policy(target["policy"], f"{where}.quality_target.policy"),
        "shortfall_element_ids": sorted(
            ids.element(x, f"{where}.quality_target.shortfall_element_ids item", True)
            for x in _list(target["shortfall_element_ids"], "shortfall_element_ids")
        ),
    }
    repair = _mapping(record["repair"], f"{where}.repair")
    if set(repair) != {"attempts", *_S3_REPAIR_COUNTS}:
        raise MeshError(f"{where}.repair is not the documented record")
    attempts = []
    for index, item in enumerate(_list(repair["attempts"], f"{where}.repair.attempts")):
        item = _mapping(item, f"{where}.repair.attempts[{index}]")
        if set(item) != {"action", "detail", "edge", "element_ids", "sequence", "status"}:
            raise MeshError(f"{where}.repair.attempts[{index}] is not the documented record")
        edge = _list(item["edge"], "attempt edge")
        attempts.append({
            "action": _checked_str(item["action"]), "detail": _checked_str(item["detail"]),
            "edge": [ids.node(x, f"{where} attempt edge node") for x in edge],
            "element_ids": [ids.element(x, f"{where} attempt element") for x in _list(item["element_ids"], "attempt elements")],
            "sequence": _count(item["sequence"], "attempt sequence"), "status": _checked_str(item["status"]),
        })
    out["repair"] = {name: _count(repair[name], f"{where}.repair.{name}") for name in _S3_REPAIR_COUNTS}
    out["repair"]["attempts"] = attempts
    return out


def _idkey(key: Any) -> Any:
    """JSON object keys carry mesh ids as decimal text; accept that and plain ints."""

    return int(key) if isinstance(key, str) and key.isdigit() else key


def _checked_str(value: Any) -> str:
    _string(value, "value")
    return value


def _x_text(value: Any, ids: _Ids, where: str) -> Any:
    _string(value, where)
    return value


_PREPARATION_FIELDS = {
    "schema": _x_text,
    "status": _x_text,
    "structural_closure": _x_closure,
    "structured_layout": _x_layout,
    "source_to_working_faces": _x_lineage,
    "source_to_working_edges": _x_lineage,
    "qualified_s3": _x_qualified_s3,
}


def _process_metadata(mesh: Mesh, label: str, ids: _Ids) -> Tuple[Dict[Any, Any], Dict[Any, Any], Tuple[str, ...]]:
    """Validate against the explicit schemas and return ``(structural_preparation,
    hybrid_diagnostics, evidence)`` translated through ``ids`` (fresh copies).

    Unknown fields -- above all unknown numeric/ID payloads -- are refused.
    """

    evidence: List[str] = []
    prep_out: Dict[Any, Any] = {}
    preparation = mesh.structural_preparation
    if preparation:
        record = _mapping(preparation, f"{label} structural_preparation")
        unknown = set(record) - set(_PREPARATION_FIELDS)
        if unknown:
            raise MeshError(
                f"{label} structural_preparation has fields with no explicit schema "
                f"{sorted(unknown, key=str)}"
            )
        for key in record:
            prep_out[key] = _PREPARATION_FIELDS[key](record[key], ids, f"{label} structural_preparation.{key}")
        evidence.extend(f"structural_preparation.{k}" for k in sorted(record))
    hybrid_out: Dict[Any, Any] = {}
    diagnostics = mesh.hybrid_diagnostics
    if diagnostics:
        record = _mapping(diagnostics, f"{label} hybrid_diagnostics")
        parallel = record.get("parallel")
        joined = isinstance(parallel, Mapping) and parallel.get("used") is True
        if joined and (record.get("certifiable") is not False
                       or record.get("certification_mode") not in ("none", "interactive")):
            raise MeshError(f"{label} parallel join cannot carry a certification")
        if joined:
            strategies = _mapping(record.get("strategy_by_face"), "parallel join face strategies")
            if ids.tris or not strategies or any(method != "mapped" for method in strategies.values()):
                # A join assembles component data, but its producer does not
                # publish a joined shell qualification for native triangles.
                raise MeshError(f"{label} full renumber supports mapped parallel plate joins only")
        from ._regional_quad_reports import quad_first_fields
        quad_report = quad_first_fields(record, ids, mesh)
        hybrid_out.update(quad_report)
        for key in record:
            if not isinstance(key, str) or (key not in _HYBRID_SCHEMA and key not in _HYBRID_ID_SCHEMA
                                            and key not in quad_report):
                raise MeshError(
                    f"{label} hybrid_diagnostics.{key!r} has no explicit schema; refusing to "
                    "translate or retain it (full rebuild required)"
                )
        for key in record:
            if key in quad_report:
                continue
            try:
                if key in _HYBRID_ID_SCHEMA:
                    hybrid_out[key] = _HYBRID_ID_SCHEMA[key](record[key], ids)
                else:
                    if key == "complex_geometry" and joined:
                        _v_complex_geometry(record[key], joined=True)
                    else:
                        _HYBRID_SCHEMA[key](record[key])
                    hybrid_out[key] = copy.deepcopy(record[key])
            except MeshError as error:
                raise MeshError(f"{label}: {error}") from None
        evidence.extend(f"hybrid_diagnostics.{k}" for k in sorted(record))
    return prep_out, hybrid_out, tuple(evidence)


def _validate_metadata(mesh: Mesh, label: str, nodes: Set[int], elements: Set[int]) -> Tuple[str, ...]:
    """Refuse metadata outside the explicit schemas; return the evidence keys present."""

    ids = _Ids(nodes, elements, {int(k) for k in mesh.tris}, quads=mesh.quads)
    return _process_metadata(mesh, label, ids)[2]


# What may survive a FULL renumber versus what a SPLICE must recompute.  A full
# renumber changes labels only: every field below is either label-free (retained
# verbatim) or listed in MESH_ID_BEARING_METADATA (translated through the maps).  A
# splice changes the shell itself, so whole-mesh admissions/quality are STALE and
# must be recomputed by their owners; only the geometry-bound request/lineage
# could be reused, and assemble_regional_replacement conservatively drops all.
MESH_ID_BEARING_METADATA = (
    "structural_preparation.qualified_s3",
    "structural_preparation.structured_layout.quality",
    "hybrid_diagnostics.structured_quality",
    "hybrid_diagnostics.automation",
    "hybrid_diagnostics.q5",
    "hybrid_diagnostics.quadratic_promotion",
    "hybrid_diagnostics.high_order_geometry",
)
SPLICE_RECOMPUTE_METADATA = MESH_ID_BEARING_METADATA + (
    "hybrid_diagnostics.parallel",
    "structural_preparation.structured_layout.metrics",
    "hybrid_diagnostics.qualified_s3_preparation",
    "hybrid_diagnostics.certifiable",
    "hybrid_diagnostics.structured_layout_status",
    "hybrid_diagnostics.triangulation_backend_by_face",
    "hybrid_diagnostics.front",
    "hybrid_diagnostics.q4",
    "hybrid_diagnostics.validation",
    "hybrid_diagnostics.t3_angle_rule",
    "hybrid_diagnostics.runtime",
    "structural_preparation.structured_layout.diagnostics",
    "hybrid_diagnostics.phase_seconds",
    "hybrid_diagnostics.completed_phases",
)
GEOMETRY_BOUND_METADATA = (
    "structural_preparation.structural_closure",
    "structural_preparation.source_to_working_faces",
    "structural_preparation.source_to_working_edges",
    "structural_preparation.structured_layout.plan",
    "hybrid_diagnostics.requested_target_size",
    "hybrid_diagnostics.strategy_by_face",
    "hybrid_diagnostics.complex_geometry",
    "hybrid_diagnostics.geometry_model_id",
    "hybrid_diagnostics.geometry_revision",
)


def _check_map(name: str, mapping: Any, required: Set[int]) -> Dict[int, int]:
    if not isinstance(mapping, Mapping):
        raise MeshError(f"{name} must be an explicit mapping")
    out: Dict[int, int] = {}
    count = 0
    for key, value in mapping.items():
        count += 1
        key = _as_int(key, f"{name} key")
        if key < 0:
            raise MeshError(f"{name} key {key} must be non-negative")
        value = _as_int(value, f"{name}[{key}]")
        if not 1 <= value <= _INT64_MAX:
            raise MeshError(f"{name}[{key}] = {value} is not a positive int64")
        out[key] = value
    if len(out) != count:
        raise MeshError(f"{name} has duplicate keys")
    if len(set(out.values())) != len(out):
        raise MeshError(f"{name} is not injective: two ids share one permanent target")
    absent = required - set(out)
    extra = set(out) - required
    if absent:
        raise MeshError(f"{name} does not cover mesh ids {sorted(absent)[:8]}")
    if extra:
        raise MeshError(f"{name} names ids not in the mesh {sorted(extra)[:8]}")
    return out


# ---------------------------------------------------------------------- remap


def _detached_identity(mesh: Mesh) -> Mesh:
    """Owner-codec copy (``mesh_to_dict``/``mesh_from_dict``) with codec-lossy fields restored."""

    # Metadata is carried separately (deepcopy below): the codec would stringify
    # integer geometry-face keys and reject non-JSON values such as UUIDs.
    bare = replace(mesh, structural_preparation={}, hybrid_diagnostics={})
    copied = mesh_from_dict(mesh_to_dict(bare))
    copied.geometry_model_id = mesh.geometry_model_id
    copied.seeding = copy.deepcopy(mesh.seeding)
    copied.declared_plate_junction_edges = tuple(
        (int(a), int(b)) for a, b in mesh.declared_plate_junction_edges
    )
    copied.structural_preparation = copy.deepcopy(mesh.structural_preparation)
    copied.hybrid_diagnostics = copy.deepcopy(mesh.hybrid_diagnostics)
    return copied


def _translate(
    mesh: Mesh, nm: Dict[int, int], em: Dict[int, int],
    preparation: Mapping[Any, Any], diagnostics: Mapping[Any, Any],
) -> Mesh:
    def n(x: Any) -> int:
        return nm[int(x)]

    def grid(g: Any) -> np.ndarray:
        arr = np.asarray(g)
        flat = [nm[int(v)] if (int(v) != 0 or 0 in nm) else 0 for v in arr.ravel().tolist()]
        return np.array(flat, dtype=np.int64).reshape(arr.shape)

    out = Mesh(
        geometry_model_id=mesh.geometry_model_id,
        geometry_revision=mesh.geometry_revision,
        order=mesh.order,
        seeding=copy.deepcopy(mesh.seeding),
        automatic_intersections=mesh.automatic_intersections,
        automatic_beam_connections=mesh.automatic_beam_connections,
        automatic_shell_connections=mesh.automatic_shell_connections,
        structural_preparation=dict(preparation),
        hybrid_diagnostics=dict(diagnostics),
    )
    out.nodes = {n(k): np.array(v, dtype=float) for k, v in mesh.nodes.items()}
    for family, _ in _ELEMENT_FAMILIES:
        setattr(out, family, {em[int(k)]: tuple(n(x) for x in v) for k, v in getattr(mesh, family).items()})
    out.couplings = {
        em[int(k)]: Coupling(
            beam_node=n(c.beam_node),
            plate_nodes=tuple(n(x) for x in c.plate_nodes),
            weights=tuple(float(w) for w in c.weights),
            eccentricity=tuple(float(e) for e in c.eccentricity),  # type: ignore[arg-type]
        )
        for k, c in mesh.couplings.items()
    }
    out.node_of_vertex = {int(k): n(v) for k, v in mesh.node_of_vertex.items()}
    for name in ("nodes_of_edge", "offset_nodes_of_edge", "nodes_of_member"):
        setattr(out, name, {int(k): [n(x) for x in v] for k, v in getattr(mesh, name).items()})
    for name in _ELEMENTS_OF:
        setattr(out, name, {int(k): [em[int(x)] for x in v] for k, v in getattr(mesh, name).items()})
    out.grid_of_face = {int(k): grid(g) for k, g in mesh.grid_of_face.items()}
    out.block_grids_of_face = {
        int(k): tuple(grid(b) for b in blocks) for k, blocks in mesh.block_grids_of_face.items()
    }
    out.activity = {em[int(k)]: float(v) for k, v in mesh.activity.items()}
    out.thickness_of_face = {int(k): float(v) for k, v in mesh.thickness_of_face.items()}
    out.declared_plate_junction_edges = tuple(
        (n(a), n(b)) for a, b in mesh.declared_plate_junction_edges
    )
    return out


def remap_mesh_ids(mesh: Mesh, node_ids: Mapping[int, int], element_ids: Mapping[int, int]) -> Mesh:
    """Return a detached copy of ``mesh`` with node/element ids translated.

    ``node_ids`` and ``element_ids`` are explicit mappings (a ``mappingproxy`` is
    fine) temporary -> permanent covering every node and every element (quads,
    tris, beams and couplings share one element namespace).  Temporary labels may
    be 0; targets are positive, unique int64 values.  Geometry entity ids
    (vertex/edge/face/sheet/member keys) are not mesh-space and are unchanged.

    ``structural_preparation``/``hybrid_diagnostics`` must match the explicit
    schemas of this module; they carry no mesh ids, so they are retained verbatim
    (``mesh_to_dict`` JSON-safety is enforced for the identity path).  Anything
    else is refused.
    """

    nodes, elements = _validate_structure(mesh, "mesh", allow_zero=True)
    nm = _check_map("node_ids", node_ids, nodes)
    em = _check_map("element_ids", element_ids, elements)
    ids = _Ids(nodes, elements, {int(k) for k in mesh.tris}, nm, em, quads=mesh.quads)
    preparation, diagnostics, _ = _process_metadata(mesh, "mesh", ids)  # validates even for identity
    if all(k == v for k, v in nm.items()) and all(k == v for k, v in em.items()):
        return _detached_identity(mesh)
    return _translate(mesh, nm, em, preparation, diagnostics)


# ------------------------------------------------------------ cross revision


@dataclass(frozen=True)
class OwnerBaselineReceipt:
    """A caller-supplied statement of one component's baseline in one revision.

    ``digest`` is the geometry owner's identity of that component's baseline; this
    module only compares digests, it cannot verify them.  ``verified`` must be
    ``True``: the issuing owner attests it checked the receipt.
    """

    owner: Any
    model_id: str
    revision: int
    digest: str
    verified: bool
    # "present": the owner exists in this revision (an ownership receipt).
    # "absent": the issuing owner attests the component does not exist in this
    # revision (the old side of a birth, the new side of a death).
    state: str = "present"


@dataclass(frozen=True)
class CrossRevisionAuthority:
    """Authority to join a replacement of one geometry revision to an older outside.

    ``old_receipts``/``new_receipts``: per component owner, verified baselines in the
    previous and replacement revisions.  ``changed_owners`` are the components whose
    elements are regenerated; every other owner must have equal old/new digests.
    ``added_owners`` are born in the replacement revision (old receipt: verified
    *absent* in the previous revision, new receipt: present); ``removed_owners`` die
    (old receipt present, new receipt verified *absent*).  A removed owner's elements
    are retired and it may own no replacement element.  ``replacement_element_owner``
    maps every replacement (temporary) element id to its owner and must cover exactly
    the changed and added owners.  ``unchanged_geometry_keys`` is the validated
    outside-association equivalence: per kind (vertex/edge/face/sheet/member) the
    geometry keys retained outside the region that denote the same entity in both
    revisions.  ``edge_owner`` maps geometry edge -> owner, to find the edges whose
    seed may change.
    """

    old_receipts: Mapping[Any, OwnerBaselineReceipt]
    new_receipts: Mapping[Any, OwnerBaselineReceipt]
    changed_owners: FrozenSet[Any]
    unchanged_geometry_keys: Mapping[str, FrozenSet[int]] = field(default_factory=dict)
    edge_owner: Optional[Mapping[int, Any]] = None
    added_owners: FrozenSet[Any] = frozenset()
    removed_owners: FrozenSet[Any] = frozenset()
    replacement_element_owner: Optional[Mapping[int, Any]] = None

    def __post_init__(self) -> None:
        changed = frozenset(self.changed_owners)
        added, removed = frozenset(self.added_owners), frozenset(self.removed_owners)
        if not (changed or added or removed):
            raise MeshError("cross-revision authority names no changed, added or removed owner")
        if changed & added or changed & removed or added & removed:
            raise MeshError("an owner cannot be changed, added and removed at once (ambiguous lineage)")
        for kind in self.unchanged_geometry_keys:
            if kind not in _KINDS:
                raise MeshError(f"unknown geometry key kind {kind!r}")
        mapping = self.replacement_element_owner
        if mapping is not None:
            if not isinstance(mapping, Mapping):
                raise MeshError("replacement_element_owner must be a mapping")
            stray = set(mapping.values()) - (changed | added)
            if stray:
                raise MeshError(
                    f"replacement elements are owned by owners that are neither changed nor added: "
                    f"{sorted(map(repr, stray))[:6]}"
                )
            if set(mapping.values()) != (changed | added):
                raise MeshError("replacement_element_owner must give every changed and added owner elements")
        elif added or removed:
            raise MeshError("an added or removed owner needs an explicit replacement_element_owner map")


def _check_authority(
    authority: CrossRevisionAuthority, previous: Mesh, replacement: Mesh, owners: Set[Any]
) -> None:
    changed, added, removed = (
        frozenset(authority.changed_owners), frozenset(authority.added_owners),
        frozenset(authority.removed_owners),
    )
    if not (changed | removed) <= owners:
        raise MeshError("changed_owners/removed_owners name owners that own no previous element")
    if added & owners:
        raise MeshError("added_owners already own previous elements; the lineage is ambiguous")
    for owner in owners | added:
        for side, receipts, mesh in (
            ("old", authority.old_receipts, previous),
            ("new", authority.new_receipts, replacement),
        ):
            receipt = receipts.get(owner)
            if not isinstance(receipt, OwnerBaselineReceipt):
                raise MeshError(f"no {side} baseline receipt for owner {owner!r}")
            if receipt.verified is not True or receipt.owner != owner:
                raise MeshError(f"{side} baseline receipt of owner {owner!r} is not verified for it")
            if (
                str(receipt.model_id) != str(mesh.geometry_model_id)
                or receipt.revision != mesh.geometry_revision
            ):
                raise MeshError(f"{side} baseline receipt of owner {owner!r} does not match the {side} mesh")
            if not isinstance(receipt.digest, str) or not receipt.digest:
                raise MeshError(f"{side} baseline receipt of owner {owner!r} has no digest")
            absent = (side == "old" and owner in added) or (side == "new" and owner in removed)
            if receipt.state != ("absent" if absent else "present"):
                raise MeshError(
                    f"{side} baseline receipt of owner {owner!r} must state "
                    f"{'absence' if absent else 'presence'}, not {receipt.state!r}"
                )
        if owner not in changed and owner not in added and owner not in removed and (
            authority.old_receipts[owner].digest != authority.new_receipts[owner].digest
        ):
            raise MeshError(f"retained owner {owner!r} changed between revisions; it must be in changed_owners")


def _merge_seeding(
    previous: Optional[Seeding],
    replacement: Optional[Seeding],
    changed_edges: Set[int],
    dropped_edges: Set[int],
) -> Tuple[Optional[Seeding], Tuple[int, ...]]:
    if replacement is None:
        merged = copy.deepcopy(previous)
        changed: Tuple[int, ...] = ()
    elif previous is None:
        raise MeshError("replacement carries seeding the previous mesh does not; refusing to guess a merge")
    else:
        try:
            same = bool(previous == replacement)
        except Exception:
            same = False
        if same:
            return _drop_seed_edges(copy.deepcopy(previous), dropped_edges), ()
        try:
            field_same = previous.size_field is replacement.size_field or bool(
                previous.size_field == replacement.size_field
            )
        except Exception:
            field_same = False
        if not field_same:
            from .refinement import SizeField
            old_field, new_field = previous.size_field, replacement.size_field
            field_same = (type(old_field) is SizeField and type(new_field) is SizeField
                          and old_field.is_uniform and new_field.is_uniform
                          and old_field.target_size == new_field.target_size)
        if not field_same:
            raise MeshError("replacement seeding was solved against a different size field")
        if (previous.divisions == replacement.divisions and previous.classes == replacement.classes
                and previous.sweeps == replacement.sweeps):
            return _drop_seed_edges(copy.deepcopy(previous), dropped_edges), ()
        if not changed_edges:
            raise MeshError(
                "replacement seeding differs from the previous mesh; name the changed owners' edges "
                "(seed_changed_edges or CrossRevisionAuthority.edge_owner) to allow it"
            )
        divisions = {int(k): int(v) for k, v in previous.divisions.items()}
        classes = {int(k): int(v) for k, v in previous.classes.items()}
        offset = max(classes.values(), default=-1) + 1
        touched: List[int] = []
        for edge, count in replacement.divisions.items():
            edge, count = int(edge), int(count)
            if divisions.get(edge) == count:
                continue
            if edge not in changed_edges:
                raise MeshError(f"replacement changes the seed of edge {edge} outside the changed owners")
            divisions[edge] = count
            touched.append(edge)
            if edge in replacement.classes:
                classes[edge] = offset + int(replacement.classes[edge])
        merged = Seeding(
            divisions=divisions,
            sweeps=max(int(previous.sweeps), int(replacement.sweeps)),
            classes=classes,
            size_field=previous.size_field,
        )
        changed = tuple(sorted(touched))
    return _drop_seed_edges(merged, dropped_edges), changed


def _drop_seed_edges(seeding: Optional[Seeding], dropped: Set[int]) -> Optional[Seeding]:
    if seeding is not None and dropped:
        for edge in dropped:
            seeding.divisions.pop(edge, None)
            seeding.classes.pop(edge, None)
    return seeding


# ------------------------------------------------------------------- assembly


@dataclass(frozen=True)
class RegionalQualityLimits:
    """Explicit quality limits for the newly created shell elements."""

    max_aspect_ratio: float | None = None
    max_warp: float | None = None
    min_scaled_jacobian: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_aspect_ratio", "max_warp", "min_scaled_jacobian"):
            if getattr(self, name) is not None and not np.isfinite(float(getattr(self, name))):
                raise MeshError(f"quality limit {name} must be finite")


@dataclass(frozen=True)
class RegionalAssembly:
    """The assembled mesh and plain facts about what this call did.

    These are statements, not receipts: nothing here is signed or hashed.
    """

    mesh: Mesh
    retained_node_ids: Tuple[int, ...]
    retained_element_ids: Tuple[int, ...]
    interface_node_ids: Tuple[int, ...]
    retired_node_ids: Tuple[int, ...]
    retired_element_ids: Tuple[int, ...]
    new_node_ids: Tuple[int, ...]
    new_element_ids: Tuple[int, ...]
    quality: Optional[MeshQuality]
    stations_validated: bool
    diagnostics_discarded: bool
    discarded_diagnostic_keys: Tuple[str, ...] = ()
    cross_revision: bool = False
    dropped_geometry_keys: Mapping[str, Tuple[int, ...]] = field(default_factory=dict)
    seeding_changed_edges: Tuple[int, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "retained_node_ids": list(self.retained_node_ids),
            "retained_element_ids": list(self.retained_element_ids),
            "interface_node_ids": list(self.interface_node_ids),
            "retired_node_ids": list(self.retired_node_ids),
            "retired_element_ids": list(self.retired_element_ids),
            "new_node_ids": list(self.new_node_ids),
            "new_element_ids": list(self.new_element_ids),
            "quality": None if self.quality is None else self.quality.as_dict(),
            "stations_validated": self.stations_validated,
            "diagnostics_discarded": self.diagnostics_discarded,
            "discarded_diagnostic_keys": list(self.discarded_diagnostic_keys),
            "cross_revision": self.cross_revision,
            "dropped_geometry_keys": {k: list(v) for k, v in self.dropped_geometry_keys.items()},
            "seeding_changed_edges": list(self.seeding_changed_edges),
        }


def _shell_edges(conn: Mapping[int, Tuple[int, ...]]):
    for eid, nodes in conn.items():
        n = 4 if len(nodes) in (4, 8) else 3
        quadratic = len(nodes) > n
        for i in range(n):
            mid = nodes[n + i] if quadratic else None
            yield eid, (frozenset((nodes[i], nodes[(i + 1) % n])), mid)


def _same_coords(a: Any, b: Any, tol: float) -> bool:
    return bool(np.all(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) <= tol))


def _drop_sets(retired_geometry_keys: Optional[Mapping[str, Iterable[int]]]) -> Dict[str, Set[int]]:
    drop: Dict[str, Set[int]] = {kind: set() for kind in _KINDS}
    if retired_geometry_keys is None:
        return drop
    if not isinstance(retired_geometry_keys, Mapping):
        raise MeshError("retired_geometry_keys must map a kind to geometry keys")
    for kind, keys in retired_geometry_keys.items():
        if kind not in drop:
            raise MeshError(f"unknown retired geometry key kind {kind!r}")
        drop[kind] = {_as_int(k, f"retired {kind} key") for k in keys}
    return drop


def assemble_regional_replacement(
    previous: Mesh,
    replacement: Mesh,
    *,
    quality: RegionalQualityLimits,
    node_ids: Mapping[int, int],
    element_ids: Mapping[int, int],
    affected_element_ids: Optional[Iterable[int]] = None,
    source_owner: Any = None,
    owner_of_element: Optional[Mapping[int, Any]] = None,
    interface_node_ids: Optional[Mapping[int, int]] = None,
    stations: Optional[Mapping[int, Sequence[float]]] = None,
    coordinate_tolerance: float = 0.0,
    discard_stale_diagnostics: bool = False,
    cross_revision: Optional[CrossRevisionAuthority] = None,
    retired_geometry_keys: Optional[Mapping[str, Iterable[int]]] = None,
    seed_changed_edges: Optional[Iterable[int]] = None,
) -> RegionalAssembly:
    """Replace the affected elements of ``previous`` by a staged ``replacement``.

    ``replacement`` is a self-contained mesh with temporary ids (0 allowed).
    ``node_ids`` / ``element_ids`` map its *fresh* nodes and all its elements to
    permanent ids that must not already exist in ``previous``.
    ``interface_node_ids`` maps replacement nodes to the retained previous node ids
    they coincide with; it must be exactly the set of previous nodes shared by
    affected and outside elements (empty for a disjoint region).  ``stations``
    optionally states the prescribed interface coordinates and is validated.

    The affected set comes from ``affected_element_ids`` or from
    ``owner_of_element`` + ``source_owner`` (the map must cover every element).
    With ``cross_revision`` the affected set is the elements of
    ``authority.changed_owners``; the replacement may then carry another geometry
    model/revision, the assembled mesh takes the replacement's, and retained
    associations must be attested in ``unchanged_geometry_keys``.

    ``retired_geometry_keys`` ({kind: keys}) names geometry entities of removed
    owners whose nodes/elements are entirely retired; their associations are
    dropped (verified to be fully retired and not rebuilt).  ``seed_changed_edges``
    names edges whose seed may differ in ``replacement.seeding``; every other seed
    is retained from ``previous``.
    """

    if not isinstance(quality, RegionalQualityLimits):
        raise MeshError("explicit RegionalQualityLimits are required")
    tol = float(coordinate_tolerance)
    if not np.isfinite(tol) or tol < 0.0:
        raise MeshError("coordinate_tolerance must be finite and non-negative")
    prev_nodes, prev_elements = _validate_structure(previous, "previous mesh")
    rep_nodes, rep_elements = _validate_structure(replacement, "replacement mesh", allow_zero=True)
    prev_evidence = _validate_metadata(previous, "previous mesh", prev_nodes, prev_elements)
    rep_evidence = _validate_metadata(replacement, "replacement mesh", rep_nodes, rep_elements)
    drop = _drop_sets(retired_geometry_keys)

    # -- geometry authority is copied, never invented
    revision_differs = False
    for attr in ("geometry_model_id", "geometry_revision"):
        a, b = getattr(previous, attr), getattr(replacement, attr)
        if (a is None) != (b is None) or (a is not None and str(a) != str(b)):
            revision_differs = True
    if revision_differs and cross_revision is None:
        raise MeshError(
            "replacement geometry model/revision differs from the previous mesh; "
            "a CrossRevisionAuthority with verified owner baseline receipts is required"
        )
    if cross_revision is not None and not isinstance(cross_revision, CrossRevisionAuthority):
        raise MeshError("cross_revision must be a CrossRevisionAuthority")
    if previous.order != replacement.order:
        raise MeshError("replacement element order differs from previous mesh")

    # -- ownership
    if cross_revision is not None:
        if not isinstance(owner_of_element, Mapping):
            raise MeshError("a cross-revision update needs owner_of_element for every component")
        if source_owner is not None and source_owner not in cross_revision.changed_owners:
            raise MeshError("source_owner is not among the changed owners")
        if {_as_int(k, "owner key") for k in owner_of_element} != prev_elements:
            raise MeshError("owner_of_element must cover exactly every element of the previous mesh")
        _check_authority(cross_revision, previous, replacement, set(owner_of_element.values()))
        creators = frozenset(cross_revision.changed_owners) | frozenset(cross_revision.added_owners)
        reo = cross_revision.replacement_element_owner
        if reo is None and len(creators) > 1:
            raise MeshError("several changed/added owners need an explicit replacement_element_owner map")
        if reo is not None and {_as_int(k, "replacement owner key") for k in reo} != rep_elements:
            raise MeshError("replacement_element_owner must cover exactly every replacement element")
        if not creators and rep_elements:
            raise MeshError("a pure removal must have an empty replacement")
        if creators and not rep_elements:
            raise MeshError("the replacement has no elements for the changed/added owners")
        owned = {
            int(k) for k, o in owner_of_element.items()
            if o in cross_revision.changed_owners or o in cross_revision.removed_owners
        }
        if affected_element_ids is not None and {_as_int(e, "affected id") for e in affected_element_ids} != owned:
            raise MeshError("affected_element_ids disagree with the changed owners' elements")
        affected = owned
    elif owner_of_element is not None:
        if not isinstance(owner_of_element, Mapping):
            raise MeshError("owner_of_element must be a mapping")
        if source_owner is None:
            raise MeshError("owner_of_element needs a source_owner")
        if {_as_int(k, "owner key") for k in owner_of_element} != prev_elements:
            raise MeshError("owner_of_element must cover exactly every element of the previous mesh")
        owned = {int(k) for k, o in owner_of_element.items() if o == source_owner}
        if affected_element_ids is not None and {_as_int(e, "affected id") for e in affected_element_ids} != owned:
            raise MeshError("affected_element_ids disagree with the source owner's elements")
        affected = owned
    else:
        if source_owner is not None or affected_element_ids is None:
            raise MeshError("give affected_element_ids, or owner_of_element together with source_owner")
        affected = {_as_int(e, "affected id") for e in affected_element_ids}
    pure_add = cross_revision is not None and bool(cross_revision.added_owners)
    if not affected and not pure_add:
        raise MeshError("the affected selection is empty")
    if not affected <= prev_elements:
        raise MeshError(f"affected elements not in previous mesh: {sorted(affected - prev_elements)[:8]}")

    # -- seeding: retained, except for explicitly changed edges
    changed_edges = {_as_int(e, "seed_changed_edges item") for e in (seed_changed_edges or ())}
    if cross_revision is not None and cross_revision.edge_owner is not None:
        changed_edges |= {
            int(e) for e, o in cross_revision.edge_owner.items()
            if o in cross_revision.changed_owners or o in cross_revision.added_owners
        }

    # -- topology outside the region
    prev_conn = _element_nodes(previous)
    outside = prev_elements - affected
    aff_nodes: Set[int] = set().union(*(prev_conn[e] for e in affected))
    out_nodes: Set[int] = set().union(*(prev_conn[e] for e in outside)) if outside else set()
    boundary = aff_nodes & out_nodes
    retired_nodes = aff_nodes - out_nodes
    closure = retired_nodes | boundary

    iface = {} if interface_node_ids is None else _check_map("interface_node_ids", interface_node_ids, set(interface_node_ids))
    if set(iface.values()) != boundary:
        raise MeshError(
            "interface_node_ids must be exactly the nodes shared by affected and retained elements; "
            f"expected {sorted(boundary)[:8]}, got {sorted(iface.values())[:8]}"
        )
    for key in iface:
        if key not in replacement.nodes:
            raise MeshError(f"interface node {key} is not in the replacement mesh")
    stations_ok = False
    if stations is not None:
        if {_as_int(k, "station key") for k in stations} != boundary:
            raise MeshError("stations must cover exactly the interface nodes")
        for nid, xyz in stations.items():
            arr = np.asarray(xyz, dtype=float)
            if arr.shape != (3,) or not np.all(np.isfinite(arr)):
                raise MeshError(f"station {nid} needs a finite 3-vector")
            if not _same_coords(arr, previous.nodes[int(nid)], tol):
                raise MeshError(f"station {nid} differs from the retained interface coordinate")
        stations_ok = True
    for key, perm in iface.items():
        if not _same_coords(replacement.nodes[key], previous.nodes[perm], tol):
            raise MeshError(f"interface node {perm} moved in the replacement (frozen coordinates)")
        if stations is not None and not _same_coords(replacement.nodes[key], stations[perm], tol):
            raise MeshError(f"interface node {perm} does not sit at its prescribed station")

    # -- permanent numbering of the staged replacement
    if set(node_ids) & set(iface):
        raise MeshError("node_ids must not renumber interface nodes")
    nmap = {**dict(node_ids), **iface}
    counters = [
        name
        for name in ("automatic_intersections", "automatic_beam_connections", "automatic_shell_connections")
        if getattr(previous, name) or getattr(replacement, name)
    ]
    discarded_keys = tuple(
        [f"previous.{k}" for k in prev_evidence] + [f"replacement.{k}" for k in rep_evidence] + counters
    )
    stale = bool(discarded_keys)
    if stale and not discard_stale_diagnostics:
        raise MeshError(
            "qualification/quality diagnostics and automatic-connection counters describe the whole "
            "previous mesh and cannot be rebuilt for a regional update; pass "
            f"discard_stale_diagnostics=True to drop them explicitly: {list(discarded_keys)}"
        )
    staged = replacement
    if stale:
        # Only replace the whole-mesh diagnostic fields on this view. The
        # remapper below builds detached topology and associations itself.
        staged = copy.copy(replacement)
        staged.structural_preparation = {}
        staged.hybrid_diagnostics = {}
        staged.automatic_intersections = 0
        staged.automatic_beam_connections = 0
        staged.automatic_shell_connections = 0
    rep = remap_mesh_ids(staged, nmap, element_ids)
    interface_perm = set(iface.values())
    fresh_nodes = [n for n in rep.nodes if n not in interface_perm]
    clash = [n for n in fresh_nodes if n in prev_nodes]
    if clash:
        raise MeshError(f"new node ids reuse existing ids {sorted(clash)[:8]}")
    rep_conn = _element_nodes(rep)
    pure_removal = (
        cross_revision is not None and bool(cross_revision.removed_owners)
        and not cross_revision.changed_owners and not cross_revision.added_owners
    )
    if not rep_conn and not pure_removal:
        raise MeshError("the replacement has no elements")
    clash = [e for e in rep_conn if e in prev_elements]
    if clash:
        raise MeshError(f"new element ids reuse existing ids {sorted(clash)[:8]}")
    used = set().union(*rep_conn.values())
    added_keys: Dict[str, Set[int]] = {kind: set() for kind in _KINDS}
    if cross_revision is not None and cross_revision.added_owners:
        reo = cross_revision.replacement_element_owner or {}
        owner_of_new = {element_ids[int(t)]: o for t, o in reo.items()}
        born = {e for e, o in owner_of_new.items() if o in cross_revision.added_owners}
        born_nodes = set().union(*(rep_conn[e] for e in born)) if born else set()
        other_nodes = set().union(*(rep_conn[e] for e in rep_conn if e not in born)) if len(born) < len(rep_conn) else set()
        if born_nodes & set(iface.values()) or born_nodes & other_nodes or born_nodes & prev_nodes:
            raise MeshError("an added owner shares nodes with retained or regenerated mesh; it must be disjoint")
        keep = np.array([previous.nodes[n] for n in prev_nodes - retired_nodes], dtype=float).reshape(-1, 3)
        for node in born_nodes:
            if keep.size and bool(np.any(np.all(np.abs(keep - np.asarray(rep.nodes[node], dtype=float)) <= tol, axis=1))):
                raise MeshError(f"added owner node {node} coincides with a retained node; it is not disjoint")
        for name, kind in (("elements_of_face", "face"), ("elements_of_edge", "edge"), ("elements_of_sheet", "sheet")):
            for key, seq in getattr(rep, name).items():
                holders = {owner_of_new[int(e)] for e in seq if int(e) in owner_of_new}
                if not holders & cross_revision.added_owners:
                    continue
                if holders - cross_revision.added_owners:
                    raise MeshError(f"{name}[{key}] is shared by an added and another owner; lineage is ambiguous")
                if key in getattr(previous, name):
                    raise MeshError(f"added owner reuses {kind} key {key} that the previous mesh already holds")
                added_keys[kind].add(int(key))
    if not interface_perm <= used:
        raise MeshError("the replacement leaves interface nodes unused; topology would be incomplete")

    # -- assemble; outside nodes/elements keep their order and values untouched
    authority_source = replacement if revision_differs else previous
    asm = Mesh(
        geometry_model_id=authority_source.geometry_model_id,
        geometry_revision=authority_source.geometry_revision,
        order=previous.order,
    )
    asm.nodes = {n: np.array(x, dtype=float) for n, x in previous.nodes.items() if n not in retired_nodes}
    for n in fresh_nodes:
        asm.nodes[n] = np.array(rep.nodes[n], dtype=float)
    for family, _ in _ELEMENT_FAMILIES:
        setattr(asm, family, {
            e: tuple(v) for e, v in getattr(previous, family).items() if e not in affected
        })
        getattr(asm, family).update(getattr(rep, family))
    asm.couplings = {e: c for e, c in previous.couplings.items() if e not in affected}
    asm.couplings.update(rep.couplings)

    # -- geometry metadata
    retained: Dict[str, Set[int]] = {kind: set() for kind in _KINDS}
    seen_drop: Dict[str, Set[int]] = {kind: set() for kind in _KINDS}

    def retire(kind: str, key: int, fully: bool, supplied: bool) -> bool:
        """True when ``key`` is a declared retired key (and verified so)."""

        if key not in drop[kind]:
            return False
        if supplied:
            raise MeshError(f"retired {kind} key {key} is also supplied by the replacement")
        if not fully:
            raise MeshError(f"{kind} key {key} is not fully retired: it still has retained mesh")
        seen_drop[kind].add(key)
        return True

    nv = {}
    for vertex, nid in previous.node_of_vertex.items():
        if nid in retired_nodes:
            if retire("vertex", vertex, True, vertex in rep.node_of_vertex):
                continue
            if vertex not in rep.node_of_vertex:
                raise MeshError(f"vertex {vertex} lost its node and the replacement does not rebuild it")
        else:
            if retire("vertex", vertex, False, False):
                continue
            nv[vertex] = nid
            retained["vertex"].add(vertex)
    for vertex, nid in rep.node_of_vertex.items():
        if vertex in nv and nv[vertex] != nid:
            raise MeshError(f"vertex {vertex} maps to conflicting nodes")
        nv[vertex] = nid
    asm.node_of_vertex = nv

    def rebuild_nodes(name: str, kind: str):
        prev_map, rep_map = getattr(previous, name), getattr(rep, name)
        out: Dict[int, List[int]] = {}
        for key, seq in prev_map.items():
            seq = [int(x) for x in seq]
            touched = any(x in retired_nodes for x in seq)
            inside = all(x in closure for x in seq)
            if retire(kind, key, all(x in retired_nodes for x in seq), key in rep_map):
                continue
            if key in rep_map:
                if not inside:
                    raise MeshError(
                        f"{name}[{key}] spans retained mesh outside the region and the replacement "
                        "also supplies it; the merge order is ambiguous"
                    )
                out[key] = [int(x) for x in rep_map[key]]
            elif touched:
                raise MeshError(f"{name}[{key}] loses retired nodes and the replacement does not rebuild it")
            else:
                out[key] = seq
                retained[kind].add(key)
        for key, seq in rep_map.items():
            if key not in prev_map:
                out[key] = [int(x) for x in seq]
        setattr(asm, name, out)

    rebuild_nodes("nodes_of_edge", "edge")
    rebuild_nodes("offset_nodes_of_edge", "edge")
    rebuild_nodes("nodes_of_member", "member")

    def rebuild_grids(name: str, kind: str, nodes_of):
        prev_map, rep_map = getattr(previous, name), getattr(rep, name)
        out: Dict[int, Any] = {}
        for key, g in prev_map.items():
            nodes = nodes_of(g)
            if retire(kind, key, nodes <= retired_nodes, key in rep_map):
                continue
            if key in rep_map:
                if not nodes <= closure:
                    raise MeshError(f"{name}[{key}] spans retained mesh outside the region; cannot merge")
                out[key] = copy.deepcopy(rep_map[key])
            elif nodes & retired_nodes:
                raise MeshError(f"{name}[{key}] loses retired nodes and the replacement does not rebuild it")
            else:
                out[key] = copy.deepcopy(g)
                retained[kind].add(key)
        for key, g in rep_map.items():
            if key not in prev_map:
                out[key] = copy.deepcopy(g)
        setattr(asm, name, out)

    rebuild_grids("grid_of_face", "face", _grid_nodes)
    rebuild_grids(
        "block_grids_of_face", "face",
        lambda blocks: set().union(*(_grid_nodes(b) for b in blocks)) if blocks else set(),
    )
    kind_of = {
        "elements_of_face": "face", "elements_of_edge": "edge",
        "elements_of_sheet": "sheet", "elements_of_member": "member",
    }
    for name in _ELEMENTS_OF:
        kind = kind_of[name]
        merged: Dict[int, List[int]] = {}
        rep_map = getattr(rep, name)
        for key, seq in getattr(previous, name).items():
            seq = [int(e) for e in seq]
            if seq and retire(kind, key, all(e in affected for e in seq), key in rep_map):
                continue
            kept = [e for e in seq if e not in affected]
            if seq and not kept and key not in rep_map:
                raise MeshError(f"{name}[{key}] loses all its elements and the replacement does not rebuild it")
            merged[key] = kept
            if kept:
                retained[kind].add(key)
        for key, seq in rep_map.items():
            merged.setdefault(key, []).extend(int(e) for e in seq)
        setattr(asm, name, merged)
    for kind, keys in drop.items():
        unseen = keys - seen_drop[kind]
        if unseen:
            raise MeshError(f"retired {kind} keys {sorted(unseen)[:8]} are not in the previous mesh associations")
    asm.activity = {e: v for e, v in previous.activity.items() if e not in affected}
    asm.activity.update(rep.activity)
    thickness = {k: v for k, v in previous.thickness_of_face.items() if k not in drop["face"]}
    for key, value in rep.thickness_of_face.items():
        if key in thickness and thickness[key] != value:
            raise MeshError(f"thickness_of_face[{key}] conflicts with the previous mesh")
        thickness[key] = value
    asm.thickness_of_face = thickness
    junctions = [tuple(p) for p in previous.declared_plate_junction_edges if not (set(p) & retired_nodes)]
    junctions += [tuple(p) for p in rep.declared_plate_junction_edges if tuple(p) not in junctions]
    asm.declared_plate_junction_edges = tuple(junctions)

    # -- seeding and cross-revision equivalence of everything retained outside
    asm.seeding, seed_touched = _merge_seeding(
        previous.seeding, replacement.seeding, changed_edges, set(drop["edge"])
    )
    if cross_revision is not None:
        for kind in _KINDS:
            contradiction = drop[kind] & set(cross_revision.unchanged_geometry_keys.get(kind, ()))
            if contradiction:
                raise MeshError(f"retired {kind} keys {sorted(contradiction)[:8]} are also attested as unchanged")
            if drop[kind] & added_keys[kind]:
                raise MeshError(f"retired {kind} keys are reused by an added owner; lineage is ambiguous")
        for kind, keys in retained.items():
            attested = set(cross_revision.unchanged_geometry_keys.get(kind, ()))
            missing = keys - attested
            if missing:
                raise MeshError(
                    f"retained {kind} associations {sorted(missing)[:8]} are not attested as "
                    "unchanged between revisions"
                )

    # -- mandatory conformity
    prev_out = {e: prev_conn[e] for e in outside if e in previous.quads or e in previous.tris}
    prev_aff = {e: prev_conn[e] for e in affected if e in previous.quads or e in previous.tris}
    new_shell = {e: v for e, v in rep_conn.items() if e in rep.quads or e in rep.tris}
    prev_seam = {k for _, k in _shell_edges(prev_out)} & {k for _, k in _shell_edges(prev_aff)}
    new_seam = {k for _, k in _shell_edges(prev_out)} & {k for _, k in _shell_edges(new_shell)}
    if prev_seam != new_seam:
        raise MeshError("the replacement does not conform to the retained shell edges along the interface")
    declared = {tuple(sorted(p)) for p in asm.declared_plate_junction_edges}
    counts: Dict[Any, int] = {}
    for _, key in _shell_edges({e: v for e, v in _element_nodes(asm).items() if e in asm.quads or e in asm.tris}):
        counts[key] = counts.get(key, 0) + 1
    for _, key in _shell_edges(new_shell):
        corners = tuple(sorted(key[0]))
        if counts[key] > 2 and corners not in declared:
            raise MeshError(f"replacement creates a non-manifold shell edge {corners}")

    # -- explicit quality over the new shells only
    q: Optional[MeshQuality] = None
    if new_shell:
        probe = Mesh(order=asm.order)
        probe.nodes = asm.nodes
        probe.quads = {e: v for e, v in rep.quads.items()}
        probe.tris = {e: v for e, v in rep.tris.items()}
        q = verify_mesh_quality(probe)
        if not np.isfinite((q.min_scaled_jacobian, q.max_aspect_ratio, q.max_warp)).all() or q.min_scaled_jacobian <= 0:
            raise MeshError("replacement contains a degenerate or inverted shell")
        if quality.max_aspect_ratio is not None and q.max_aspect_ratio > quality.max_aspect_ratio:
            raise MeshError(f"replacement aspect ratio {q.max_aspect_ratio:.4g} exceeds {quality.max_aspect_ratio}")
        if quality.max_warp is not None and q.max_warp > quality.max_warp:
            raise MeshError(f"replacement warp {q.max_warp:.4g} exceeds {quality.max_warp}")
        if quality.min_scaled_jacobian is not None and q.min_scaled_jacobian < quality.min_scaled_jacobian:
            raise MeshError(
                f"replacement scaled Jacobian {q.min_scaled_jacobian:.4g} below {quality.min_scaled_jacobian}"
            )

    if cross_revision is not None:
        # Certify the whole assembled result, retained elements included.
        _validate_structure(asm, "assembled mesh")
        shells = Mesh(order=asm.order)
        shells.nodes = asm.nodes
        shells.quads = dict(asm.quads)
        shells.tris = dict(asm.tris)
        if shells.quads or shells.tris:
            full = verify_mesh_quality(shells)
            if (not np.isfinite((full.min_scaled_jacobian, full.max_aspect_ratio, full.max_warp)).all()
                    or full.min_scaled_jacobian <= 0):
                raise MeshError("the assembled mesh contains a degenerate, inverted or non-finite shell")
    return RegionalAssembly(
        mesh=asm,
        retained_node_ids=tuple(sorted(set(prev_nodes) - retired_nodes)),
        retained_element_ids=tuple(sorted(outside)),
        interface_node_ids=tuple(sorted(boundary)),
        retired_node_ids=tuple(sorted(retired_nodes)),
        retired_element_ids=tuple(sorted(affected)),
        new_node_ids=tuple(sorted(fresh_nodes)),
        new_element_ids=tuple(sorted(rep_conn)),
        quality=q,
        stations_validated=stations_ok,
        diagnostics_discarded=stale,
        discarded_diagnostic_keys=discarded_keys,
        cross_revision=cross_revision is not None,
        dropped_geometry_keys={k: tuple(sorted(v)) for k, v in seen_drop.items() if v},
        seeding_changed_edges=seed_touched,
    )
