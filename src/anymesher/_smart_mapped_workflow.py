"""Private, opt-in adapter: hidden smart-mapped prototype as the quad-first seed.

``hybrid._quad_first_execute`` resolves ``build_planar_quad_seed`` as a module
global for every face and then runs its unchanged downstream dataflow: Q4 MCF,
the PQ-M1 front, residual flips, Q5 optimisation, ``validate_planar_quad_result``,
beam assembly, couplings, quadratic promotion and publication.  Inside
:class:`SmartMappedSeedContext` that single name is temporarily replaced by a
wrapper that returns a genuine :class:`PlanarQuadSeed` generated from
``_smart_mapped_prototype.build_smart_mapped``.  Nothing else is patched, no
certificate is fabricated, no validation is skipped and no public default or
registration changes.

Seed contract kept (same as ``build_planar_quad_seed``):

* rows ``0..n_outer-1`` are the registry's outer stations in registry order,
  then each hole's stations, then every other node; ``station_to_node`` is the
  canonical ``BoundaryStationKey -> row`` map and equals the baseline seed's;
* outer/hole coordinates are the exact ``domain.project(station.position)``
  chart points; the prototype's returned boundary rows are compared with
  ``numpy.array_equal`` and any difference is a refusal;
* ``triangulation`` covers the same area (T3 cells plus each Q4 split in two),
  with ``boundary_segments`` the station loops and no mandatory segments;
* the ``QuadMeshState`` holds Q4 and T3 cells, protected nodes/edges are exactly
  the boundary stations/edges (what the validator checks), and the initial
  front is the boundary of the residual-T3 region (or empty on request).

Source edges: ``Straight`` and ``Arc`` edges of an exact ``PlanarQuadDomain``
are adopted (rounded-rectangle access holes are planar faces with Arc edges).
The registry samples every edge through ``geometry.sample_edge``; an Arc is
therefore a canonical station chain and the seed boundary is the exact
projection of those stations, like the baseline seed.  No curved-surface
support is added and no tolerance is relaxed: the seed stays a straight-chord
chart, and the owner curve midpoint of each boundary span is produced later by
the existing quadratic promotion.  When any Arc is present the adapter also
fails closed if a sampled station leaves the face plane or if the sampled
station loops touch or cross (the domain only validates the vertex chords).

Unsupported requests are refused with :class:`SmartMappedRefusal` (a
``MeshError`` that is deliberately *not* ``QuadPublicUnsupported``, so no
public fallback treats it as "try another route"): non-``PlanarQuadDomain``
charts, any source edge that is neither ``Straight`` nor ``Arc``, non-uniform
size fields, ``source_aligned_cut`` seeds and unknown layout policies.

Truthfulness: the seed *is* prototype output (``build_smart_mapped`` may apply
its own pairing/smoothing/refinement, reported per call), but Q4 MCF, the front,
residual flips and Q5 run afterwards on the seed state and may modify it, so no
"pure unchanged prototype" claim is ever made (``pure_prototype_claim`` is
always ``False``).  The optional callback receives immutable copies of the seed
as created, for an independent parent audit.

Safety: the context is process-exclusive (non-blocking lock), refuses nesting,
a second thread and a pre-patched ``hybrid.build_planar_quad_seed``, and always
restores the original in ``finally`` on success, error or cancellation.  A
successful exit that swallowed a recorded refusal/failure, or never called the
seed (``require_calls``), raises instead of reporting a silent fallback.
"""
from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from anygeometry.curves import Arc, Straight

from . import hybrid as _hybrid
from ._smart_mapped_prototype import SmartMappedPrototype, build_smart_mapped
from .errors import MeshError
from .quad import seed as _seed_module
from .quad.boundary import BoundaryStationKey
from .quad.domain import PlanarQuadDomain
from .quad.seed import PlanarQuadSeed
from .quad.state import QuadMeshState
from .triangulation import PlanarTriangulation

__all__ = [
    "ADAPTER_ID",
    "SmartMappedAdapterError",
    "SmartMappedRefusal",
    "SmartMappedSeedRecord",
    "SmartMappedSeedContext",
    "build_smart_mapped_planar_seed",
    "smart_mapped_seed_context",
]

ADAPTER_ID = "smart_mapped_seed/1"
_BACKEND = "smart-mapped-prototype"
_FRONT_POLICIES = ("interface", "none")
_LAYOUT_POLICIES = ("existing", "adaptive")
_PROTOTYPE_OPTION_NAMES = (
    "max_grid_cells", "max_work", "enable_pairing", "enable_smoothing",
    "improvement_budget", "enable_refinement", "refinement_budget",
)

_LOCK = threading.Lock()
_ACTIVE: "SmartMappedSeedContext | None" = None


class SmartMappedAdapterError(MeshError):
    """Unsafe use of the adapter (nesting, threads, tampering, silent fallback)."""


class SmartMappedRefusal(SmartMappedAdapterError):
    """The requested face/options are outside the adapter's supported scope."""


@dataclass(frozen=True)
class SmartMappedSeedRecord:
    """Immutable per-call snapshot of the seed exactly as created."""

    call_index: int
    face_id: int
    target_size: float
    prototype: SmartMappedPrototype
    seed: PlanarQuadSeed
    seed_points: np.ndarray
    seed_quads: np.ndarray
    seed_triangles: np.ndarray
    seed_state_digest: str
    boundary_keys: tuple[BoundaryStationKey, ...]
    diagnostics: Mapping[str, Any]


def _readonly(values: Any, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True).reshape(shape)
    array.setflags(write=False)
    return array


def _edge_ids(uses: Sequence[Any]):
    for item in uses:
        if len(item) == 2 and isinstance(item[1], (bool, np.bool_)):
            yield int(item[0])
        else:
            yield from _edge_ids(item)


def _assert_stations_in_plane(domain: PlanarQuadDomain, positions: Sequence[Sequence[float]]) -> None:
    """Refuse stations that the planar chart would silently flatten."""
    extent = max(
        1.0,
        *(abs(float(v)) for loop in (domain.outer_chart, *domain.hole_charts) for p in loop for v in p),
    )
    tol = 1.0e-10 * extent
    for position in positions:
        flat = domain.lift(domain.project(position))
        distance = abs(float(np.dot(np.asarray(position) - domain.origin, domain.normal)))
        if not np.isfinite(distance) or distance > tol:
            raise SmartMappedRefusal(
                "a sampled boundary station is not in the face plane "
                f"(out-of-plane distance {distance:.3e} > {tol:.3e})"
            )


def _assert_simple_loops(loops: Sequence[Sequence[Sequence[float]]]) -> None:
    """Refuse station loops whose non-adjacent segments touch or cross.

    Float orientation screening plus bounding-box overlap; the prototype subsequently uses its exact predicates; adjacent segments
    of one loop (which share a station) are exempt.  Fail closed.
    """
    starts, ends, loop_of, index_of, size_of = [], [], [], [], []
    for number, loop in enumerate(loops):
        points = np.asarray(loop, dtype=float).reshape((-1, 2))
        count = len(points)
        starts.append(points)
        ends.append(np.roll(points, -1, axis=0))
        loop_of.extend([number] * count)
        index_of.extend(range(count))
        size_of.extend([count] * count)
    p = np.concatenate(starts)
    q = np.concatenate(ends)
    loop_of_a = np.asarray(loop_of)
    index_of_a = np.asarray(index_of)
    size_of_a = np.asarray(size_of)
    low = np.minimum(p, q)
    high = np.maximum(p, q)

    def orient(a, b, c):
        return np.sign((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
                       - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))

    for i in range(len(p) - 1):
        j = slice(i + 1, None)
        gap = (index_of_a[j] - index_of_a[i]) % size_of_a[j]
        adjacent = (loop_of_a[j] == loop_of_a[i]) & ((gap == 1) | (gap == size_of_a[j] - 1))
        overlap = (
            (low[j, 0] <= high[i, 0]) & (high[j, 0] >= low[i, 0])
            & (low[j, 1] <= high[i, 1]) & (high[j, 1] >= low[i, 1])
        )
        o1 = orient(p[i], q[i], p[j])
        o2 = orient(p[i], q[i], q[j])
        o3 = orient(p[j], q[j], p[i])
        o4 = orient(p[j], q[j], q[i])
        hit = overlap & (o1 * o2 <= 0) & (o3 * o4 <= 0) & ~adjacent
        if hit.any():
            other = int(np.flatnonzero(hit)[0]) + i + 1
            raise SmartMappedRefusal(
                "sampled boundary station loops touch or cross "
                f"(segments {i} and {other})"
            )


def _validated_prototype_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    values = dict(options or {})
    unknown = sorted(set(values) - set(_PROTOTYPE_OPTION_NAMES))
    if unknown:
        raise SmartMappedRefusal(f"unknown smart mapped prototype option(s): {unknown}")
    return values


def _layout_from_prototype(
    prototype: SmartMappedPrototype,
    outer: Sequence[Sequence[float]],
    holes: Sequence[Sequence[Sequence[float]]],
    *,
    initial_front: str = "interface",
) -> tuple[PlanarTriangulation, QuadMeshState, tuple[int, ...]]:
    """Canonical-row triangulation and state from prototype output.

    Returns ``(triangulation, state, order)`` where ``order[new] = old`` prototype
    row.  Pure: no geometry, registry or global state is touched.
    """
    if initial_front not in _FRONT_POLICIES:
        raise SmartMappedRefusal(f"initial_front must be one of {_FRONT_POLICIES}")
    charts = (np.asarray(outer, dtype=float), *(np.asarray(h, dtype=float) for h in holes))
    loops = (prototype.outer_loop, *prototype.hole_loops)
    if len(loops) != len(charts) or any(len(l) != len(c) for l, c in zip(loops, charts)):
        raise SmartMappedRefusal("prototype boundary loops differ from the canonical station loops")
    order = [int(row) for loop in loops for row in loop]
    boundary_count = len(order)
    if len(set(order)) != boundary_count:
        raise SmartMappedRefusal("prototype boundary loops share or repeat nodes")
    total = len(prototype.points)
    used = set(order)
    order.extend(row for row in range(total) if row not in used)
    if sorted(order) != list(range(total)):
        raise SmartMappedRefusal("prototype boundary rows are outside its point table")
    new_of_old = {old: new for new, old in enumerate(order)}
    points = np.asarray(prototype.points, dtype=float)[order]
    cursor = 0
    for chart in charts:
        if not np.array_equal(points[cursor:cursor + len(chart)], chart):
            raise SmartMappedRefusal(
                "prototype changed a canonical boundary station coordinate"
            )
        cursor += len(chart)

    quads = [tuple(new_of_old[int(n)] for n in body) for body in prototype.quads]
    tris = [tuple(new_of_old[int(n)] for n in body) for body in prototype.triangles]
    coverage = list(tris)
    for a, b, c, d in quads:
        coverage.extend(((a, b, c), (a, c, d)))
    boundary: list[tuple[int, int]] = []
    holes_rows: list[np.ndarray] = []
    cursor = 0
    for index, chart in enumerate(charts):
        count = len(chart)
        boundary.extend((cursor + i, cursor + (i + 1) % count) for i in range(count))
        if index:
            holes_rows.append(np.arange(cursor, cursor + count, dtype=np.int64))
        cursor += count
    boundary_array = np.asarray(boundary, dtype=np.int64).reshape((-1, 2))
    triangulation = PlanarTriangulation(
        points=points,
        triangles=np.asarray(coverage, dtype=np.int64).reshape((-1, 3)),
        segments=boundary_array,
        boundary_segments=boundary_array,
        mandatory_segments=np.empty((0, 2), dtype=np.int64),
        outer_loop=np.arange(len(charts[0]), dtype=np.int64),
        hole_loops=tuple(holes_rows),
        backend=_BACKEND, requested_backend=_BACKEND,
        selected_backend=_BACKEND, actual_backend=_BACKEND,
    )
    cells: dict[int, tuple[int, ...]] = {}
    kinds: dict[int, str] = {}
    for body in quads:
        kinds[len(cells)] = "Q4"
        cells[len(cells)] = body
    for body in tris:
        kinds[len(cells)] = "T3"
        cells[len(cells)] = body
    front: list[tuple[int, int]] = []
    if initial_front == "interface":
        incident: dict[tuple[int, int], list[int]] = {}
        for cid, body in cells.items():
            for i in range(len(body)):
                a, b = body[i], body[(i + 1) % len(body)]
                incident.setdefault((min(a, b), max(a, b)), []).append(cid)
        for edge, owners in sorted(incident.items()):
            t3 = [cid for cid in owners if kinds[cid] == "T3"]
            if len(t3) == 1 and (len(owners) == 1 or kinds[next(c for c in owners if c != t3[0])] == "Q4"):
                front.append(edge)
    state = QuadMeshState(
        {row: tuple(map(float, point)) for row, point in enumerate(points)},
        cells, kinds, initial_front=front,
        protected_nodes=range(boundary_count),
        protected_edges=tuple(tuple(map(int, edge)) for edge in boundary),
    )
    return triangulation, state, tuple(order)


def build_smart_mapped_planar_seed(
    geometry: Any,
    face_id: int,
    target_size: float,
    *,
    domain: PlanarQuadDomain | None = None,
    registry: Any = None,
    size_field: Any = None,
    source_aligned_cut: tuple[float, float] | None = None,
    layout_policy: str = "existing",
    cancellation_check: Callable[[str], None] | None = None,
    prototype_options: Mapping[str, Any] | None = None,
    initial_front: str = "interface",
    prototype_builder: Callable[..., SmartMappedPrototype] | None = None,
) -> tuple[PlanarQuadSeed, SmartMappedPrototype, dict[str, Any]]:
    """Seed one planar face from the prototype; no global state is touched.

    Same keyword contract as ``build_planar_quad_seed`` plus adapter options.
    Returns ``(seed, prototype, diagnostics)``.
    """
    from .quad.boundary import BoundaryStationRegistry

    options = _validated_prototype_options(prototype_options)
    if initial_front not in _FRONT_POLICIES:
        raise SmartMappedRefusal(f"initial_front must be one of {_FRONT_POLICIES}")
    if layout_policy not in _LAYOUT_POLICIES:
        raise SmartMappedRefusal("layout_policy must be 'existing' or 'adaptive'")
    if source_aligned_cut is not None:
        raise SmartMappedRefusal("smart mapped seed does not support source-aligned cut seeds")
    if size_field is not None and not bool(getattr(size_field, "is_uniform", False)):
        raise SmartMappedRefusal("smart mapped seed supports only uniform target size")
    if domain is not None and type(domain) is not PlanarQuadDomain:
        raise SmartMappedRefusal(
            f"smart mapped seed supports only PlanarQuadDomain, got {type(domain).__name__}"
        )
    h = float(target_size)
    if not np.isfinite(h) or h <= 0.0:
        raise SmartMappedRefusal("target_size must be positive and finite")
    start_revision, start_model_id = int(geometry.revision), str(geometry.model_id)
    domain = domain or PlanarQuadDomain.from_geometry(geometry, face_id)
    if domain.face_id != int(face_id):
        raise MeshError("domain face does not match requested face")
    domain.assert_current(geometry)
    if size_field is not None and abs(float(size_field.target_size) - h) > 1.0e-12 * max(1.0, abs(h)):
        raise MeshError("provided size field target_size differs from seed target_size")
    has_arc = False
    for edge_id in (*_edge_ids(domain.edge_uses), *_edge_ids(domain.hole_edge_uses)):
        curve_type = type(geometry.edges[edge_id].curve)
        if curve_type is Arc:
            has_arc = True
        elif curve_type is not Straight:
            raise SmartMappedRefusal(
                "smart mapped seed supports only Straight and Arc source edges "
                f"(edge {edge_id} is {curve_type.__name__})"
            )
    registry = registry or BoundaryStationRegistry.for_domain(
        geometry, domain, h, size_field=size_field)
    registry.assert_current(geometry)
    if abs(registry.target_size - h) > 1.0e-15 * max(1.0, abs(h)):
        raise MeshError("boundary registry target_size differs from seed target_size")
    stations = registry.outer_stations(domain)
    outer = tuple(domain.project(station.position) for station in stations)
    hole_stations = [registry.hole_stations(domain, i) for i in range(len(domain.hole_edge_uses))]
    holes = []
    for group in hole_stations:
        if len(group) < 3:
            raise MeshError("hole boundary has fewer than three stations")
        holes.append(tuple(domain.project(station.position) for station in group))
    if has_arc:
        _assert_stations_in_plane(
            domain, [station.position for group in (stations, *hole_stations) for station in group])
        _assert_simple_loops((outer, *holes))
    station_to_node: dict[BoundaryStationKey, int] = {}
    cursor = 0
    for group in (stations, *hole_stations):
        for station in group:
            if station.key in station_to_node:
                raise MeshError("canonical boundary station identities are not unique on the face")
            station_to_node[station.key] = cursor
            cursor += 1
    builder = prototype_builder or build_smart_mapped
    prototype = builder(outer, tuple(holes), (), target_size=h,
                        cancellation_check=cancellation_check, **options)
    triangulation, state, _order = _layout_from_prototype(
        prototype, outer, holes, initial_front=initial_front)
    domain.assert_current(geometry)
    registry.assert_current(geometry)
    if int(geometry.revision) != start_revision or str(geometry.model_id) != start_model_id:
        raise MeshError("smart mapped seed mutated source geometry")
    seed = PlanarQuadSeed(domain, registry, h, tuple(station_to_node),
                          triangulation, state, station_to_node)
    diagnostics = {
        "face": int(face_id),
        "faces_seeded": 1,
        "seed_quads": len(prototype.quads),
        "seed_triangles": len(prototype.triangles),
        "seed_nodes": len(prototype.points),
        "boundary_stations": len(station_to_node),
        "initial_front_edges": len(state.front),
        "initial_front": initial_front,
        "layout_policy_requested": layout_policy,
        "prototype_options": dict(options),
        "prototype_diagnostics": {
            key: copy.deepcopy(value)
            for key, value in prototype.diagnostics.items()
            if key != "retained_grid_indices"
        },
    }
    return seed, prototype, diagnostics


class SmartMappedSeedContext:
    """Temporarily route ``hybrid`` quad-first seeding through the prototype.

    Use as ``with SmartMappedSeedContext(...) as adapter:`` around one
    ``generate_hybrid_mesh_result(..., quad_options=...)`` call (the ANYfem
    ``project.generate_mesh`` path reaches the same function).  ``adapter.report()``
    returns the truthful JSON-friendly diagnostics; ``adapter.annotate(mesh)``
    stores that report in ``mesh.hybrid_diagnostics``.
    """

    def __init__(
        self,
        *,
        prototype_options: Mapping[str, Any] | None = None,
        initial_front: str = "interface",
        seed_callback: Callable[[SmartMappedSeedRecord], None] | None = None,
        require_calls: bool = True,
        prototype_builder: Callable[..., SmartMappedPrototype] | None = None,
    ) -> None:
        self._options = _validated_prototype_options(prototype_options)
        if initial_front not in _FRONT_POLICIES:
            raise SmartMappedRefusal(f"initial_front must be one of {_FRONT_POLICIES}")
        if seed_callback is not None and not callable(seed_callback):
            raise SmartMappedRefusal("seed_callback must be callable")
        if type(require_calls) is not bool:
            raise SmartMappedRefusal("require_calls must be bool")
        self._front = initial_front
        self._callback = seed_callback
        self._require_calls = require_calls
        self._builder = prototype_builder
        self._owner: int | None = None
        self._original: Callable[..., PlanarQuadSeed] | None = None
        self._entered = False
        self._wrapper: Callable[..., PlanarQuadSeed] | None = None
        self._records: list[SmartMappedSeedRecord] = []
        self._failures: list[dict[str, str]] = []
        self.restored = False

    # -- state ------------------------------------------------------------
    @property
    def active(self) -> bool:
        return _ACTIVE is self

    @property
    def calls(self) -> int:
        return len(self._records)

    @property
    def faces(self) -> tuple[int, ...]:
        return tuple(record.face_id for record in self._records)

    @property
    def records(self) -> tuple[SmartMappedSeedRecord, ...]:
        return tuple(self._records)

    @property
    def failures(self) -> tuple[dict[str, str], ...]:
        return tuple(dict(item) for item in self._failures)

    # -- context protocol -------------------------------------------------
    def __enter__(self) -> "SmartMappedSeedContext":
        global _ACTIVE
        if self._entered:
            raise SmartMappedAdapterError("a SmartMappedSeedContext instance is single-use")
        if not _LOCK.acquire(blocking=False):
            raise SmartMappedAdapterError(
                "smart mapped seed context is already active (nesting or concurrent use refused)"
            )
        try:
            if _hybrid.build_planar_quad_seed is not _seed_module.build_planar_quad_seed:
                raise SmartMappedAdapterError(
                    "hybrid.build_planar_quad_seed is already replaced; refusing to stack patches"
                )
            self._entered = True
            self._owner = threading.get_ident()
            self._original = _hybrid.build_planar_quad_seed
            self._wrapper = self._make_wrapper()
            _hybrid.build_planar_quad_seed = self._wrapper
            _ACTIVE = self
        except BaseException:
            _ACTIVE = None
            _LOCK.release()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        global _ACTIVE
        tampered = False
        try:
            if _hybrid.build_planar_quad_seed is not self._wrapper:
                tampered = True
            _hybrid.build_planar_quad_seed = self._original
            self.restored = _hybrid.build_planar_quad_seed is self._original
        finally:
            _ACTIVE = None
            _LOCK.release()
        if not self.restored:
            raise SmartMappedAdapterError("hybrid.build_planar_quad_seed could not be restored")
        if exc_type is not None:
            return False
        if tampered:
            raise SmartMappedAdapterError(
                "hybrid.build_planar_quad_seed was modified while the context was active"
            )
        if self._failures:
            raise SmartMappedAdapterError(
                "a smart mapped seed call failed but generation completed; "
                f"refusing a silent fallback: {self._failures[0]}"
            )
        if self._require_calls and not self._records:
            raise SmartMappedAdapterError(
                "smart mapped seed was never called; the quad-first route did not use it"
            )
        return False

    # -- seed wrapper -----------------------------------------------------
    def _make_wrapper(self) -> Callable[..., PlanarQuadSeed]:
        def smart_mapped_build_planar_quad_seed(
            geometry, face_id, target_size, *, domain=None, registry=None,
            size_field=None, source_aligned_cut=None, layout_policy="existing",
            cancellation_check=None,
        ) -> PlanarQuadSeed:
            try:
                if _ACTIVE is not self or threading.get_ident() != self._owner:
                    raise SmartMappedAdapterError(
                        "smart mapped seed called outside its owning thread/context"
                    )
                seed, prototype, diagnostics = build_smart_mapped_planar_seed(
                    geometry, face_id, target_size, domain=domain, registry=registry,
                    size_field=size_field, source_aligned_cut=source_aligned_cut,
                    layout_policy=layout_policy, cancellation_check=cancellation_check,
                    prototype_options=self._options, initial_front=self._front,
                    prototype_builder=self._builder,
                )
                diagnostics["call_index"] = len(self._records)
                record = SmartMappedSeedRecord(
                    call_index=len(self._records), face_id=int(face_id),
                    target_size=float(target_size), prototype=prototype, seed=seed,
                    seed_points=_readonly(seed.triangulation.points, float, (-1, 2)),
                    seed_quads=_readonly(prototype_quads(seed), np.int64, (-1, 4)),
                    seed_triangles=_readonly(prototype_triangles(seed), np.int64, (-1, 3)),
                    seed_state_digest=seed.state.digest(),
                    boundary_keys=tuple(seed.boundary_keys),
                    diagnostics=copy.deepcopy(diagnostics),
                )
                self._records.append(record)
                if self._callback is not None:
                    self._callback(record)
                return seed
            except BaseException as error:
                self._failures.append({"face": str(face_id), "type": type(error).__name__,
                                       "message": str(error)})
                raise

        return smart_mapped_build_planar_quad_seed

    # -- reporting --------------------------------------------------------
    def report(self) -> dict[str, Any]:
        return {
            "adapter": ADAPTER_ID,
            "prototype_module": "anymesher._smart_mapped_prototype",
            "prototype_function": "build_smart_mapped",
            "replaced": "anymesher.hybrid.build_planar_quad_seed",
            "calls": self.calls,
            "faces": list(self.faces),
            "seed_quads": sum(len(r.seed_quads) for r in self._records),
            "seed_triangles": sum(len(r.seed_triangles) for r in self._records),
            "failures": [dict(item) for item in self._failures],
            "initial_front": self._front,
            "prototype_options": dict(self._options),
            "per_face": [copy.deepcopy(dict(r.diagnostics)) for r in self._records],
            "context_active": bool(self.active),
            "restored": bool(self.restored),
            "supported_source_edges": ["Straight", "Arc"],
            "downstream_stages_unchanged": True,
            "downstream_may_modify_seed": True,
            "downstream_stages": [
                "q4_mcf", "pq_m1_front", "residual_flips", "q5_optimization",
                "validate_planar_quad_result", "beam_assembly", "couplings",
                "quadratic_promotion", "publication",
            ],
            "pure_prototype_claim": False,
        }

    def annotate(self, mesh: Any) -> dict[str, Any]:
        """Store an independent copy of :meth:`report` on a quad-first ``mesh``.

        Refused (never a false claim) when the context holds a recorded
        failure, never produced a seed, or ``mesh`` is not a quad-first mesh
        covering every face this adapter seeded.  The report is a snapshot: a
        call made *inside* the ``with`` block reports ``restored: False`` and
        ``context_active: True`` truthfully; call after exit to record the
        restoration.  The returned report is not aliased to the stored copy.
        """
        if not self._entered:
            raise SmartMappedAdapterError("the context was never entered; nothing to annotate")
        if self._failures:
            raise SmartMappedAdapterError(
                f"cannot annotate: a smart mapped seed call failed: {self._failures[0]}"
            )
        if not self._records:
            raise SmartMappedAdapterError("cannot annotate: smart mapped seed was never called")
        diagnostics = getattr(mesh, "hybrid_diagnostics", None)
        if not isinstance(diagnostics, dict):
            raise SmartMappedAdapterError("mesh has no hybrid_diagnostics dictionary to annotate")
        if diagnostics.get("route") != "quad-first":
            raise SmartMappedAdapterError(
                "cannot annotate: mesh was not produced by the quad-first route"
            )
        mesh_faces = {int(face) for face in getattr(mesh, "elements_of_face", {})}
        if not set(self.faces) <= mesh_faces:
            raise SmartMappedAdapterError(
                "cannot annotate: mesh does not contain every face seeded by the adapter"
            )
        report = self.report()
        diagnostics["smart_mapped_seed_adapter"] = copy.deepcopy(report)
        return report


def prototype_quads(seed: PlanarQuadSeed) -> list[tuple[int, ...]]:
    """Q4 bodies of a freshly built seed state in cell-id order."""
    state = seed.state
    return [state.cell(cid) for cid in sorted(state.cells) if state.cell_kind(cid) == "Q4"]


def prototype_triangles(seed: PlanarQuadSeed) -> list[tuple[int, ...]]:
    """T3 bodies of a freshly built seed state in cell-id order."""
    state = seed.state
    return [state.cell(cid) for cid in sorted(state.cells) if state.cell_kind(cid) == "T3"]


def smart_mapped_seed_context(**options: Any) -> SmartMappedSeedContext:
    """Convenience constructor for :class:`SmartMappedSeedContext`."""
    return SmartMappedSeedContext(**options)

