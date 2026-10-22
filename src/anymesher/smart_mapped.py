"""Public SmartMapped meshing method: owner module and cheap eligibility helper.

``generate_hybrid_mesh_result(..., strategy="smart_mapped")`` runs the real
planar quad-first pipeline (boundary-station registry, Q4 MCF, PQ-M1 front,
residual flips, Q5, ``validate_planar_quad_result``, quality repair, beam merge,
quadratic promotion, publication) with exactly one difference: every face is
seeded by the SmartMapped prototype through
:func:`build_smart_mapped_planar_seed`.  The seed builder is handed to
``hybrid._quad_first_execute`` as an explicit argument (:class:`SmartMappedRun`);
no module global is replaced, so nothing needs restoring and concurrent or
nested ordinary calls are unaffected.

Regular-rectangle preservation
------------------------------
Every Q4 of the seed whose four chart corners form an exact rectangle is a
*regular footprint*.  It is audited (cells present, node coordinates bit-equal)
after front/flips/Q5, before and after the quality repair and after quadratic
promotion.  The repair receives those nodes as protected and those cells as
non-splittable, so it can only act inside the irregular cavity; its angle,
Jacobian and normal gates are unchanged.  Any loss, move or blocked quality is
a typed :class:`QuadQualityRejected` that names the stage and counts - never a
silent drop.  Diagnostics: ``mesh.hybrid_diagnostics["smart_mapped"]`` and
``["meshing_strategy"] == "smart_mapped"``; ``strategy_by_face`` values are
``"smart_mapped"``.

Automatic eligibility
---------------------
:func:`smart_mapped_eligibility` is the cheap, side-effect free admission test
for a caller that wants to choose SmartMapped automatically (ANYfem).  It builds
no registry, no mesh, no trial mesh and starts no native worker; it only reads
the geometry (planar domain construction) and applies cheap, conservative
checks.  It does not qualify the geometry or prove work bounds: its cell/work
figures are heuristic scale estimates, and the generation path keeps its own
typed refusals (``SmartMappedRefusal``, ``QuadPublicUnsupported``, the curved
beam guard) as the authoritative unsupported guard.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from anygeometry.curves import Arc, Straight
from anygeometry.errors import GeometryError
from anygeometry.surfaces import CoonsSurface, Plane, RuledSurface

from ._smart_mapped_workflow import (
    ADAPTER_ID,
    SmartMappedAdapterError,
    SmartMappedRefusal,
    _edge_ids,
    build_smart_mapped_planar_seed,
)
from .errors import MeshError
from .quad.domain import PlanarQuadDomain
from .quad.options import QuadMeshingOptions
from .quad.validate import QuadQualityRejected
from .refinement import SizeField
from .seeding import edge_demand

__all__ = [
    "ARC_RADIAL_TOLERANCE",
    "BOUNDARY_PROFILE",
    "DEFAULT_MAX_CELLS",
    "DEFAULT_MAX_WORK",
    "SmartMappedAdapterError",
    "SmartMappedEligibility",
    "SmartMappedRefusal",
    "SmartMappedRun",
    "arc_divisions_for_tolerance",
    "arc_interpolation_radial_error",
    "arc_max_span",
    "build_smart_mapped_planar_seed",
    "prepare_smart_mapped_request",
    "smart_mapped_eligibility",
]

# Heuristic admission ceilings, sized so that a 15 m and a 60 m barge at their
# working target sizes are not refused.  They compare against scale estimates,
# not proven bounds, and make no promise about pipeline memory or time.
DEFAULT_MAX_CELLS = 250_000
DEFAULT_MAX_WORK = 5_000_000
_RECT_TOLERANCE = 1.0e-9

# ---------------------------------------------------------------------------
# SmartMapped boundary approximation profile (NOT a geometry-owner tolerance)
# ---------------------------------------------------------------------------
# Quadratic promotion interpolates every source Arc span by a parabola through
# its two stations and the owner midpoint.  On a circle of radius R and span
# delta, the local parabola Q(t) = (1 - (1 - cos(delta/2)) t^2, sin(delta/2) t)
# has maximum radial error  R * (1 - sqrt(1 - (1 - cos(delta/2))^2 / 4)).
# SmartMapped adopts a 5e-5 m bound on that analytic interpolation error as its
# own boundary approximation profile.  It is an analytic bound for the
# interpolation only; it is not a geometry tolerance, a certificate or a
# qualification of the mesh.  The same canonical station rule is used for the
# linear and the quadratic order so both share one corner topology.
ARC_RADIAL_TOLERANCE = 5.0e-5
BOUNDARY_PROFILE = "smart_mapped_arc_radial_interpolation/1"
_MAX_ARC_DIVISIONS = 100_000


def _positive_finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise SmartMappedRefusal(f"{name} must be a finite positive number") from None
    if not math.isfinite(number) or number <= 0.0:
        raise SmartMappedRefusal(f"{name} must be a finite positive number")
    return number


def arc_max_span(radius: float, tolerance: float = ARC_RADIAL_TOLERANCE) -> float:
    """Largest arc span (radians) whose quadratic radial error is <= ``tolerance``.

    ``epsilon = min(tolerance / R, 1)``; the closed form
    ``2 acos(clamp(1 - 2 sqrt(2 epsilon - epsilon^2), -1, 1))`` is evaluated in
    the equivalent small-angle-stable form ``4 asin(sqrt(sqrt(epsilon (2 -
    epsilon))))`` (``1 - cos x = 2 sin^2(x / 2)``).
    """
    r = _positive_finite(radius, "arc radius")
    tol = _positive_finite(tolerance, "arc radial tolerance")
    epsilon = min(tol / r, 1.0)
    half_c = math.sqrt(epsilon * (2.0 - epsilon))  # c / 2 with c = 1 - cos(delta / 2)
    return 4.0 * math.asin(min(1.0, math.sqrt(half_c)))


def arc_interpolation_radial_error(radius: float, span: float) -> float:
    """Maximum radial error of the quadratic interpolation of one circular span.

    ``R (1 - sqrt(1 - c^2 / 4))`` with ``c = 1 - cos(span / 2) = 2 sin^2(span/4)``,
    evaluated as ``R x / (1 + sqrt(1 - x))``, ``x = sin^4(span / 4)``, so that
    small spans lose no precision.  ``span`` must lie in ``[0, 2 pi]``.
    """
    r = _positive_finite(radius, "arc radius")
    delta = float(span)
    if not math.isfinite(delta) or delta < 0.0 or delta > 2.0 * math.pi * (1.0 + 1e-12):
        raise SmartMappedRefusal("arc span must be finite and within [0, 2 pi]")
    x = math.sin(0.25 * delta) ** 4
    return r * x / (1.0 + math.sqrt(max(0.0, 1.0 - x)))


def arc_divisions_for_tolerance(radius: float, sweep: float,
                                tolerance: float = ARC_RADIAL_TOLERANCE) -> int:
    """Smallest equal division count whose spans meet the radial bound.

    ``ceil(|sweep| / arc_max_span)``; refuses invalid/non-finite input and
    counts above a fixed bound instead of returning a plausible number.
    """
    total = abs(float(sweep)) if isinstance(sweep, (int, float)) else float("nan")
    if not math.isfinite(total) or total <= 0.0 or total > 2.0 * math.pi * (1.0 + 1e-6):
        raise SmartMappedRefusal("arc sweep must be finite, positive and at most 2 pi")
    count = int(math.ceil(total / arc_max_span(radius, tolerance)))
    if count > _MAX_ARC_DIVISIONS:
        raise SmartMappedRefusal(
            f"arc needs {count} divisions, above the bound {_MAX_ARC_DIVISIONS}")
    return max(1, count)


def _arc_radius_sweep(geometry: Any, edge_id: int) -> tuple[float, float]:
    """Radius and sweep from the geometry owner's canonical arc frame."""
    from anygeometry.curves import arc_frame

    try:
        edge = geometry.edges[int(edge_id)]
        frame = arc_frame(geometry.vertex_position(edge.start),
                          geometry.vertex_position(edge.curve.via_vertex),
                          geometry.vertex_position(edge.end))
        radius, sweep = float(frame.radius), abs(float(frame.sweep))
    except (GeometryError, MeshError, ValueError, TypeError, ArithmeticError) as error:
        raise SmartMappedRefusal(f"arc edge {edge_id} has no valid owner frame: {error}") from error
    if not (math.isfinite(radius) and radius > 0.0 and math.isfinite(sweep) and sweep > 0.0):
        raise SmartMappedRefusal(f"arc edge {edge_id} is degenerate or non-finite")
    return radius, sweep


# ---------------------------------------------------------------------------
# Cheap eligibility (no meshes, no workers)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SmartMappedEligibility:
    """Conservative admission verdict; ``eligible`` implies no ``reasons``.

    ``codes`` are the stable, deterministic machine-readable refusal kinds in
    discovery order (the face loop is sorted); ``reasons`` the matching
    messages.  ``estimated_cells``/``estimated_work`` are heuristic scale
    estimates (``estimate_kind``), not proven upper bounds.
    """

    eligible: bool
    reasons: tuple[str, ...]
    estimated_cells: int
    estimated_work: int
    max_cells: int
    max_work: int
    faces: tuple[Mapping[str, Any], ...] = ()
    adapter: str = ADAPTER_ID
    codes: tuple[str, ...] = ()
    estimate_kind: str = "heuristic_scale_estimate_not_a_proven_bound"

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "eligible": bool(self.eligible),
            "reasons": list(self.reasons),
            "codes": list(self.codes),
            "estimate_kind": self.estimate_kind,
            "estimated_cells": int(self.estimated_cells),
            "estimated_work": int(self.estimated_work),
            "max_cells": int(self.max_cells),
            "max_work": int(self.max_work),
            "faces": [dict(item) for item in self.faces],
        }


def _polygon_area(points: Sequence[Sequence[float]]) -> float:
    pts = np.asarray(points, dtype=float).reshape((-1, 2))
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def smart_mapped_eligibility(
    geometry: Any,
    face_ids: Iterable[int] | None = None,
    target_size: float | None = None,
    *,
    order: str = "linear",
    refinements: Iterable[Any] = (),
    overrides: Mapping[int, int] | None = None,
    beam_edges: Iterable[int] = (),
    member_ids: Iterable[int] | None = None,
    quad_face_ids: Iterable[int] | None = None,
    seeding: Any | None = None,
    max_cells: int = DEFAULT_MAX_CELLS,
    max_work: int = DEFAULT_MAX_WORK,
) -> SmartMappedEligibility:
    """Whether ``strategy="smart_mapped"`` is conservatively safe to choose.

    Admits only: planar faces (``Plane``/``Coons``/``Ruled`` support that yields
    an exact ``PlanarQuadDomain``) whose boundary edges are all ``Straight`` or
    ``Arc``; a uniform ``target_size`` (no refinements, overrides/pins or
    Seeding); no source-aligned cuts (member-through-face point attachments);
    and estimated cell and work counts within ``max_cells`` / ``max_work``.
    Beams and members are admitted: a linear beam on any edge, a straight
    quadratic beam, and a quadratic ``Arc`` beam only when it is a
    source-boundary edge of a selected planar face (the quad-first
    straight-sided B3 chord contract); any other quadratic curved beam is the
    typed refusal ``curved_beam_unsupported``.

    An unsupported size field, pin, Seeding, cut or unknown entity returns
    immediately, before any chart is built.  The cell/work figures are
    *heuristic scale estimates* from chart area, perimeter and arc length - not
    a proven upper bound (many projected stations can exceed them) - and are
    labelled so (``estimate_kind``).  No registry, mesh, trial mesh, worker or
    process-global cache is used.  Never raises for unsupported input: it
    returns ``eligible=False`` with every reason and its stable ``code``.
    """
    found: list[tuple[str, str]] = []

    def refuse(code: str, message: str) -> None:
        found.append((code, message))

    def verdict(cells: int = 0, work: int = 0, details: Sequence[Mapping[str, Any]] = ()):
        unique = list(dict.fromkeys(found))
        return SmartMappedEligibility(
            eligible=not unique,
            reasons=tuple(message for _code, message in unique),
            estimated_cells=int(cells),
            estimated_work=int(work),
            max_cells=int(max_cells),
            max_work=int(max_work),
            faces=tuple(details),
            codes=tuple(dict.fromkeys(code for code, _message in unique)),
        )

    # ---- cheap request screen: no geometry beyond dictionary lookups --------
    h = float("nan") if target_size is None else float(target_size)
    if not math.isfinite(h) or h <= 0.0:
        refuse("target_size_invalid", "target_size must be positive and finite")
    if order not in ("linear", "quadratic"):
        refuse("unsupported_order", f"unsupported element order {order!r}")
    refinements = tuple(refinements)
    # Plate/line entity sizes (mode="override") are admitted: each plate is
    # seeded at its own uniform target against the shared canonical stations.
    # Spatial refine-only zones still make the field non-uniform.
    if any(getattr(item, "mode", "refine") != "override" for item in refinements):
        refuse("non_uniform_size_field", "refinements make the size field non-uniform")
    face_target: dict[int, float] = {}
    for item in refinements:
        ref = getattr(item, "ref", None)
        if getattr(item, "mode", "refine") == "override" and ref is not None and ref.kind == "face":
            face_target[int(ref.id)] = min(float(item.size), face_target.get(int(ref.id), float("inf")))
    if overrides:
        refuse("edge_division_pins", "edge division overrides are not supported")
    if seeding is not None:
        refuse("precomputed_seeding", "a precomputed Seeding is not supported")
    if quad_face_ids is not None:
        refuse("mixed_quad_faces", "quad_face_ids (mixed quad/legacy faces) is not supported")
    selected: tuple[int, ...] = ()
    try:
        selected = (
            tuple(sorted(int(item) for item in geometry.faces))
            if face_ids is None
            else tuple(sorted(dict.fromkeys(int(item) for item in face_ids)))
        )
    except (TypeError, ValueError):
        refuse("face_ids_invalid", "face_ids must be integers")
    if not selected:
        refuse("no_faces", "no faces selected")
    missing = [item for item in selected if item not in geometry.faces]
    if missing:
        refuse("unknown_face", f"unknown face {missing[0]}")
        selected = tuple(item for item in selected if item in geometry.faces)
    selected_set = set(selected)
    for attachment in getattr(geometry, "attachments", {}).values():
        try:
            if (
                str(attachment.target_kind) == "face"
                and int(attachment.target_id) in selected_set
                and str(attachment.kind) == "member_through_face"
                and len(attachment.target_parameters) == 2
                and all(p.is_point and 0.0 < p.start < 1.0 for p in attachment.target_parameters)
            ):
                refuse(
                    "source_aligned_cut",
                    f"face {int(attachment.target_id)} carries an interior member "
                    "attachment point (source-aligned cut)",
                )
        except (AttributeError, TypeError, ValueError):
            refuse("attachment_unclassified", "an attachment could not be classified")

    # Beams: explicit beam edges plus the edges of the selected members.  They
    # are admitted (the quad-first beam merge supports them); only edges that
    # do not exist, or quadratic curved beams outside the scoped planar
    # boundary contract (checked after the face domains), are refused.
    beam_set: set[int] = set()
    try:
        beam_set.update(int(item) for item in beam_edges)
    except (TypeError, ValueError):
        refuse("beam_edges_invalid", "beam edges must be integers")
    try:
        members = getattr(geometry, "members", None) or {}
        member_selection = (
            tuple(sorted(int(item) for item in members))
            if member_ids is None
            else tuple(sorted(dict.fromkeys(int(item) for item in member_ids)))
        )
        for member_id in member_selection:
            member = members.get(member_id)
            if member is None:
                refuse("unknown_member", f"unknown structural member {member_id} (beam)")
                continue
            for use_id in member.edge_use_ids:
                beam_set.add(int(geometry.member_edge_uses[use_id].edge_id))
    except (AttributeError, KeyError, TypeError, ValueError):
        refuse("member_unclassified", "structural members could not be classified (beam)")
    for edge_id in sorted(beam_set):
        if edge_id not in geometry.edges:
            refuse("unknown_beam_edge", f"beam edge {edge_id} does not exist")
    # Unsupported size field, pins, seeding, cuts or unknown entities: stop
    # here, before any chart construction.
    if found:
        return verdict()

    # ---- planar charts and the heuristic work estimate -----------------------
    can_estimate = math.isfinite(h) and h > 0.0
    details: list[dict[str, Any]] = []
    cells_total = work_total = 0
    boundary_edges: set[int] = set()
    for face_id in selected:
        face_reasons: list[tuple[str, str]] = []
        surface = geometry.faces[face_id].surface
        if surface is not None and not (
            type(surface) is Plane or isinstance(surface, (CoonsSurface, RuledSurface))
        ):
            face_reasons.append((
                "face_not_planar",
                f"face {face_id} surface {type(surface).__name__} is not planar",
            ))
        cells = work = 0
        if not face_reasons:
            try:
                domain = PlanarQuadDomain.from_geometry(geometry, face_id)
                if type(domain) is not PlanarQuadDomain:
                    face_reasons.append(
                        ("no_planar_chart", f"face {face_id} has no exact planar chart"))
                else:
                    edge_ids = (*_edge_ids(domain.edge_uses), *_edge_ids(domain.hole_edge_uses))
                    boundary_edges.update(int(item) for item in edge_ids)
                    perimeter = 0.0
                    arc_sq = 0.0
                    for edge_id in edge_ids:
                        curve_type = type(geometry.edges[edge_id].curve)
                        length = float(geometry.edge_length(edge_id))
                        perimeter += length
                        if curve_type is Arc:
                            arc_sq += length * length
                        elif curve_type is not Straight:
                            face_reasons.append((
                                "unsupported_edge_curve",
                                f"face {face_id} edge {edge_id} is {curve_type.__name__}, "
                                "not Straight or Arc",
                            ))
                    area = _polygon_area(domain.outer_chart) - sum(
                        _polygon_area(hole) for hole in domain.hole_charts
                    )
                    if can_estimate:
                        # Heuristic scale estimate (quads of size h plus a
                        # boundary band and an arc allowance).  It is NOT a
                        # proven upper bound: many projected stations or a
                        # strongly graded chart can exceed it.  A plate with
                        # its own entity size is estimated at that size.
                        h_face = min(h, face_target.get(int(face_id), h))
                        cells = int(math.ceil((max(area, 0.0) + 0.25 * arc_sq) / (h_face * h_face)
                                              + perimeter / h_face)) + 4
                        stations = int(math.ceil(perimeter / h_face)) + 4 * len(edge_ids)
                        work = 12 * cells + 64 * stations
            except (MeshError, GeometryError, ValueError, ArithmeticError) as error:
                face_reasons.append((
                    "unsupported_planar_domain",
                    f"face {face_id} is not a supported planar domain: {error}",
                ))
        cells_total += cells
        work_total += work
        found.extend(face_reasons)
        details.append({
            "face": int(face_id),
            "estimated_cells": int(cells),
            "estimated_work": int(work),
            "reasons": [message for _code, message in face_reasons],
            "codes": [code for code, _message in face_reasons],
        })
    for edge_id in sorted(beam_set):
        curve_type = type(geometry.edges[edge_id].curve)
        if can_estimate:
            work_total += 16 * (int(math.ceil(float(geometry.edge_length(edge_id)) / h)) + 1)
        if order == "quadratic" and curve_type is not Straight and not (
            curve_type is Arc and edge_id in boundary_edges
        ):
            refuse(
                "curved_beam_unsupported",
                f"beam edge {edge_id} is {curve_type.__name__}: a quadratic curved beam is "
                "admitted only on an Arc that is a source-boundary edge of a selected "
                "planar face (straight-sided B3 chord contract)",
            )
    if cells_total > int(max_cells):
        refuse("cell_cap", f"estimated cells {cells_total} exceed the bound {int(max_cells)}")
    if work_total > int(max_work):
        refuse("work_cap", f"estimated work {work_total} exceeds the bound {int(max_work)}")
    return verdict(cells_total, work_total, details)


# ---------------------------------------------------------------------------
# Public strategy request preparation (called by hybrid)
# ---------------------------------------------------------------------------
def prepare_smart_mapped_request(
    *,
    quad_options: Any,
    quad_face_ids: Any,
    structured_options: Any,
    seeding: Any,
    refinements: Iterable[Any],
    overrides: Mapping[int, int] | None,
    deferred: bool,
) -> tuple[QuadMeshingOptions, "SmartMappedRun"]:
    """Validate an explicit SmartMapped request; fail closed on anything else."""
    if deferred:
        raise SmartMappedRefusal("smart_mapped cannot produce a deferred fallback candidate")
    if quad_face_ids is not None:
        raise SmartMappedRefusal("smart_mapped does not support mixed quad_face_ids")
    if structured_options is not None:
        raise SmartMappedRefusal("smart_mapped cannot be combined with structured_options")
    if seeding is not None:
        raise SmartMappedRefusal("smart_mapped does not support a precomputed Seeding")
    if tuple(refinements):
        raise SmartMappedRefusal("smart_mapped supports only a uniform target size (no refinements)")
    if overrides:
        raise SmartMappedRefusal("smart_mapped does not support edge division overrides")
    options = (
        QuadMeshingOptions(quality_model="shape_jacobian")
        if quad_options is None
        else QuadMeshingOptions.coerce(quad_options)
    )
    if options is None:
        raise SmartMappedRefusal("smart_mapped options must not coerce to None")
    return options, SmartMappedRun()


# ---------------------------------------------------------------------------
# Per-call run: explicit seed builder + regular footprint guard
# ---------------------------------------------------------------------------
@dataclass
class _Footprint:
    body: tuple[int, ...]
    positions: np.ndarray
    element_id: int | None = None
    global_nodes: tuple[int, ...] = ()
    xyz: np.ndarray | None = None


@dataclass
class _FaceRun:
    diagnostics: dict[str, Any]
    seed_quads: int
    seed_triangles: int
    footprints: list[_Footprint] = field(default_factory=list)
    # (geometry protected nodes, geometry protected edges, exact extra nodes)
    # while optimization-only pins are held; ``None`` otherwise.
    pins: tuple[frozenset, frozenset, frozenset] | None = None
    pin_report: dict[str, int] = field(default_factory=dict)


def _is_rectangle(points: np.ndarray) -> bool:
    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        return False
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 0.0):
        return False
    scale = float(lengths.max())
    for i in range(4):
        j = (i + 1) % 4
        if abs(float(np.dot(edges[i], edges[j]))) > _RECT_TOLERANCE * lengths[i] * lengths[j]:
            return False
    return (abs(lengths[0] - lengths[2]) <= _RECT_TOLERANCE * scale
            and abs(lengths[1] - lengths[3]) <= _RECT_TOLERANCE * scale)


class SmartMappedRun:
    """Single-use state of one explicit SmartMapped generation."""

    def __init__(self, *, prototype_options: Mapping[str, Any] | None = None,
                 initial_front: str = "interface") -> None:
        from ._smart_mapped_workflow import _FRONT_POLICIES, _validated_prototype_options

        self._options = _validated_prototype_options(prototype_options)
        if initial_front not in _FRONT_POLICIES:
            raise SmartMappedRefusal(f"initial_front must be one of {_FRONT_POLICIES}")
        self._front = initial_front
        self._faces: dict[int, _FaceRun] = {}
        self._stages: dict[str, dict[int, dict[str, int]]] = {}
        self._arc_requirements: dict[int, dict[str, Any]] = {}
        self._arc_report: dict[int, dict[str, Any]] = {}

    # -- boundary fidelity profile ---------------------------------------
    def boundary_overrides(self, geometry: Any, domains: Iterable[Any], target_size: float,
                           *, size_field: Any | None = None,
                           overrides: Mapping[int, int] | None = None,
                           tolerance: float = ARC_RADIAL_TOLERANCE) -> Mapping[int, int] | None:
        """Internally derived arc division counts, before any seeding.

        For every source ``Arc`` of the selected domains the required count is
        ``arc_divisions_for_tolerance(R, sweep)``; it is pinned only when it
        exceeds the ordinary target-sized count, so an arc that is already fine
        keeps the ordinary route bit-identically.  Explicit caller pins win.
        The registry applies the same canonical rule for both element orders.
        Returns ``overrides`` unchanged (the same object) when nothing is added.
        """
        h = _positive_finite(target_size, "target_size")
        field_ = size_field if size_field is not None else SizeField(geometry, h)
        user = {int(k): int(v) for k, v in (overrides or {}).items()}
        derived: dict[int, int] = {}
        self._arc_requirements = {}
        for domain in domains:
            for loop in (domain.edge_uses, *domain.hole_edge_uses):
                for raw_edge, _forward in loop:
                    edge_id = int(raw_edge)
                    if edge_id in self._arc_requirements:
                        continue
                    if type(geometry.edges[edge_id].curve) is not Arc:
                        continue
                    radius, sweep = _arc_radius_sweep(geometry, edge_id)
                    needed = arc_divisions_for_tolerance(radius, sweep, tolerance)
                    target_count = max(1, int(round(edge_demand(geometry, edge_id, field_))))
                    self._arc_requirements[edge_id] = {
                        "radius": radius, "sweep": sweep, "required_divisions": needed,
                        "target_sized_divisions": target_count,
                    }
                    if needed > target_count and edge_id not in user:
                        derived[edge_id] = needed
        if not derived:
            return overrides
        return {**derived, **user}

    def record_boundary_divisions(self, registry: Any) -> None:
        """Verify and report the registry's final arc divisions (fail closed)."""
        report: dict[int, dict[str, Any]] = {}
        for edge_id, item in sorted(self._arc_requirements.items()):
            final = int(registry.edge_divisions(edge_id))
            if final < item["required_divisions"]:
                raise SmartMappedRefusal(
                    f"arc edge {edge_id} has {final} division(s) but the SmartMapped boundary "
                    f"profile needs {item['required_divisions']}")
            span = item["sweep"] / final
            report[edge_id] = {
                **item, "divisions": final,
                "analytic_radial_error": arc_interpolation_radial_error(item["radius"], span),
            }
        self._arc_report = report

    # -- optimization-only pins ------------------------------------------
    def protect_optimization_pins(self, face_id: int, state: Any) -> None:
        """Temporarily protect every seed rectangle node (explicit state, no patch).

        Only nodes that are not already geometry-protected are added; they are
        recorded exactly so that :meth:`release_optimization_pins` removes
        those and nothing else before the unchanged ownership validator runs.
        """
        face = int(face_id)
        record = self._faces[face]
        if record.pins is not None:
            raise SmartMappedAdapterError(f"optimization pins already held for face {face}")
        geometry_nodes = frozenset(int(n) for n in state.protected_nodes)
        geometry_edges = frozenset(state.protected_edges)
        extras = frozenset(int(n) for item in record.footprints for n in item.body) - geometry_nodes
        if extras:
            transaction = state.transaction()
            try:
                for node in sorted(extras):
                    transaction.protect_node(node)
                transaction.commit()
            finally:
                transaction.close()
        record.pins = (geometry_nodes, geometry_edges, extras)
        record.pin_report = {"held": len(extras), "released": 0}

    def _check_pins(self, face: int, state: Any, stage: str) -> None:
        pins = self._faces[face].pins
        if pins is None:
            return
        nodes, edges, extras = pins
        if (frozenset(int(n) for n in state.protected_nodes) != nodes | extras
                or frozenset(state.protected_edges) != edges):
            raise QuadQualityRejected(
                f"smart mapped optimization pins of face {face} changed before {stage}")

    def release_optimization_pins(self, face_id: int, state: Any) -> None:
        """Verify pins unchanged, then release exactly the recorded extras."""
        face = int(face_id)
        record = self._faces[face]
        pins = record.pins
        if pins is None:
            return
        self._check_pins(face, state, "release")
        nodes, edges, extras = pins
        if extras:
            transaction = state.transaction()
            try:
                for node in sorted(extras):
                    transaction.release_node(node)
                transaction.commit()
            finally:
                transaction.close()
        if (frozenset(int(n) for n in state.protected_nodes) != nodes
                or frozenset(state.protected_edges) != edges):
            raise QuadQualityRejected(
                f"smart mapped face {face} does not carry exactly its geometry protection "
                "after releasing optimization pins")
        record.pins = None
        record.pin_report["released"] = len(extras)

    # -- seeding ---------------------------------------------------------
    def seed_builder(self, geometry, face_id, target_size, *, domain=None, registry=None,
                     size_field=None, source_aligned_cut=None, layout_policy="existing",
                     cancellation_check=None):
        """``build_planar_quad_seed``-compatible seed from the prototype."""
        face = int(face_id)
        if face in self._faces:
            raise SmartMappedAdapterError(f"smart mapped seed requested twice for face {face}")
        seed, prototype, diagnostics = build_smart_mapped_planar_seed(
            geometry, face, target_size, domain=domain, registry=registry,
            size_field=size_field, source_aligned_cut=source_aligned_cut,
            layout_policy=layout_policy, cancellation_check=cancellation_check,
            prototype_options=self._options, initial_front=self._front,
        )
        state = seed.state
        footprints: list[_Footprint] = []
        for cid in sorted(state.cells):
            if state.cell_kind(cid) != "Q4":
                continue
            body = tuple(int(n) for n in state.cell(cid))
            positions = np.asarray([state.position(n) for n in body], dtype=float)
            if _is_rectangle(positions):
                footprints.append(_Footprint(body=body, positions=positions.copy()))
        diagnostics = copy.deepcopy(diagnostics)
        diagnostics["regular_footprints"] = len(footprints)
        self._faces[face] = _FaceRun(
            diagnostics=diagnostics, seed_quads=len(prototype.quads),
            seed_triangles=len(prototype.triangles), footprints=footprints)
        return seed

    # -- audits ----------------------------------------------------------
    def _record(self, stage: str, face: int, total: int, lost: int, moved: int) -> None:
        self._stages.setdefault(stage, {})[face] = {
            "regular_footprints": total, "lost": lost, "moved": moved,
            "preserved": total - max(lost, moved),
        }

    def _raise(self, stage: str, face: int, total: int, lost: int, moved: int, first: Any) -> None:
        raise QuadQualityRejected(
            f"smart mapped regular footprint not preserved after {stage}: face {face} "
            f"lost {lost}, moved {moved} of {total} regular rectangle(s); first: {first}"
        )

    @staticmethod
    def _quad_index(state) -> dict[frozenset, int]:
        return {frozenset(int(n) for n in state.cell(cid)): cid
                for cid in state.cells if state.cell_kind(cid) == "Q4"}

    def audit_state(self, face_id: int, state: Any, stage: str) -> None:
        """Every seed rectangle is still a Q4 with bit-equal chart corners."""
        face = int(face_id)
        record = self._faces[face]
        self._check_pins(face, state, stage)  # held pins must be exactly unchanged
        index = self._quad_index(state)
        lost = moved = 0
        first = None
        for footprint in record.footprints:
            present = frozenset(footprint.body) in index
            try:
                current = np.asarray([state.position(n) for n in footprint.body], dtype=float)
                equal = bool(np.array_equal(current, footprint.positions))
            except MeshError:
                equal = False
            lost += 0 if present else 1
            moved += 0 if equal else 1
            if first is None and not (present and equal):
                first = {"nodes": footprint.body, "present": present, "unmoved": equal}
        self._record(stage, face, len(record.footprints), lost, moved)
        if lost or moved:
            self._raise(stage, face, len(record.footprints), lost, moved, first)

    def bind_face(self, face_id: int, state: Any, element_of_local_cell: Mapping[int, int],
                  local_to_global: Mapping[int, int], global_nodes: Mapping[int, Any]) -> None:
        """Attach final element and global node ids to each regular footprint."""
        face = int(face_id)
        index = self._quad_index(state)
        for footprint in self._faces[face].footprints:
            cid = index.get(frozenset(footprint.body))
            if cid is None:
                raise SmartMappedAdapterError(
                    f"regular footprint of face {face} vanished before element assembly")
            footprint.element_id = int(element_of_local_cell[cid])
            footprint.global_nodes = tuple(int(local_to_global[int(n)]) for n in state.cell(cid))
            footprint.xyz = np.asarray([global_nodes[n] for n in footprint.global_nodes],
                                       dtype=float).copy()

    def audit_mesh(self, stage: str, mesh: Any) -> None:
        """Every regular footprint is a live quad of its face at unmoved nodes."""
        index = {frozenset(int(n) for n in body[:4]): int(eid)
                 for eid, body in mesh.quads.items()}
        for face, record in sorted(self._faces.items()):
            live = {int(e) for e in mesh.elements_of_face.get(face, ())}
            lost = moved = 0
            first = None
            for footprint in record.footprints:
                eid = index.get(frozenset(footprint.global_nodes))
                present = eid is not None and eid in live
                try:
                    current = np.asarray([mesh.nodes[n] for n in footprint.global_nodes], dtype=float)
                    equal = bool(np.array_equal(current, footprint.xyz))
                except KeyError:
                    equal = False
                lost += 0 if present else 1
                moved += 0 if equal else 1
                if first is None and not (present and equal):
                    first = {"element": footprint.element_id, "nodes": footprint.global_nodes,
                             "present": present, "unmoved": equal}
            self._record(stage, face, len(record.footprints), lost, moved)
            if lost or moved:
                self._raise(stage, face, len(record.footprints), lost, moved, first)

    # -- repair constraints ---------------------------------------------
    def protected_nodes(self) -> frozenset[int]:
        return frozenset(n for r in self._faces.values() for f in r.footprints
                         for n in f.global_nodes)

    def regular_cells(self) -> frozenset[int]:
        return frozenset(int(f.element_id) for r in self._faces.values()
                         for f in r.footprints if f.element_id is not None)

    # -- reporting -------------------------------------------------------
    def report(self) -> dict[str, Any]:
        seeded = {face: len(r.footprints) for face, r in sorted(self._faces.items())}
        stages: dict[str, dict[str, int]] = {}
        for stage, per_face in self._stages.items():
            stages[stage] = {
                "regular_footprints": sum(v["regular_footprints"] for v in per_face.values()),
                "preserved": sum(v["preserved"] for v in per_face.values()),
                "lost": sum(v["lost"] for v in per_face.values()),
                "moved": sum(v["moved"] for v in per_face.values()),
            }
        final = stages.get("final")
        return {
            "adapter": ADAPTER_ID,
            "strategy": "smart_mapped",
            "dispatch": "explicit_seed_builder_argument",
            "global_patch": False,
            "prototype_module": "anymesher._smart_mapped_prototype",
            "prototype_function": "build_smart_mapped",
            "faces": sorted(self._faces),
            "seed_quads": sum(r.seed_quads for r in self._faces.values()),
            "seed_triangles": sum(r.seed_triangles for r in self._faces.values()),
            "regular_footprints": {
                "seeded": sum(seeded.values()),
                "per_face": seeded,
                "guard": "frozen_nodes_and_cells_in_repair",
                "stages": stages,
                "final_preserved": None if final is None else int(final["preserved"]),
                "final_lost": None if final is None else int(final["lost"]),
            },
            "boundary_fidelity": {
                "profile": BOUNDARY_PROFILE,
                "arc_radial_tolerance_m": ARC_RADIAL_TOLERANCE,
                "kind": "analytic_quadratic_interpolation_bound_on_source_arcs",
                "geometry_tolerance_changed": False,
                "qualification_claim": False,
                "same_station_rule_for_linear_and_quadratic": True,
                "arc_divisions": {int(e): int(v["divisions"])
                                  for e, v in sorted(self._arc_report.items())},
                "arcs": {int(e): dict(v) for e, v in sorted(self._arc_report.items())},
            },
            "optimization_pins": {
                "scope": "temporary_extra_protected_nodes_released_before_validator",
                "per_face": {int(f): dict(r.pin_report) for f, r in sorted(self._faces.items())},
            },
            "downstream_stages_unchanged": True,
            "downstream_may_modify_seed": True,
            "pure_prototype_claim": False,
            "per_face": [copy.deepcopy(r.diagnostics) for _f, r in sorted(self._faces.items())],
        }
