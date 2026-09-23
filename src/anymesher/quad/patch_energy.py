"""Quad-first Q5: composite Q4-patch re-optimization energy (pure Python).

This module owns the *documented* composite energy that the C++ TinyAD
worker (``third_party/quad/worker/quad_tinyad_optimizer.cc``) minimizes,
plus the neighbor-halo validity guard it must never cross, plus a
central finite-difference gradient for the analytic-derivative test.

The module is **deterministic** and **library-free** (no numpy; uses
``math`` only) so it is a stable cross-check against the TinyAD+Eigen
active-differentiation code.  The C++ worker and this module must agree
to machine precision on both the energy and the analytic first
derivative.

Energy
------

For a patch whose nodes are ``P = [(px0, py0), ..., (pxN-1, pyN-1)]``
and quads ``Q = [[i0, i1, i2, i3], ...]`` (four distinct nodes each) with
target edge length ``h > 0`` and (unit) reference axis ``u = (ux, uy)``:
set ``v = (-uy, ux)``.  For each quad with edges

    e_k = P[i_{k+1}] - P[i_k]        for k in {0, 1, 2, 3}
    n_k = e_k . e_k

the four energy components are

* **size** (edge-length deviation)::

        size = (1/4) * sum_k (n_k / h^2 - 1)^2

* **alignment** (each edge should be parallel to the corresponding
  reference direction: even-indexed edges to ``u``, odd-indexed edges
  to ``v``)::

        align_k = 1 - (e_k . d_k)^2 / (n_k + eps)
        align   = (1/4) * sum_k align_k

  where ``d_0 = d_2 = u`` and ``d_1 = d_3 = v``.  The ``+ eps`` in the
  denominator is present so that the energy is continuous at zero-length
  edges — it changes the value at measure-zero configurations and has no
  effect on the interior optimum.

* **shape** (successive edges should be perpendicular)::

        shape_k = (e_k . e_{k+1})^2 / (n_k n_{k+1} + eps)
        shape   = (1/4) * sum_k shape_k

  (with indices mod 4)

* **jacobian** (both triangles of the quad should carry the expected
  area / orientation — this is the guard against a non-planar
  degenerate quad or a self-intersecting one)::

        detA    = (P1 - P0) x (P2 - P0)          # = e0 x e1  (2D cross, a scalar)
        detB    = (P2 - P1) x (P3 - P1)          # = e1 x e2
        jacobian = (1/2) * ((detA / h^2 - 1)^2 + (detB / h^2 - 1)^2)

The total patch energy is the sum of these four components over all
quads.  All arithmetic is in the same operation order as the C++ worker
(multiplication and division order within each scalar expression is
identical, but summations are grouped the same way).

Validity (neighbor-halo guard)
------------------------------

A patch is **valid** iff for *every* quad ``(i0, i1, i2, i3)``:

    (P1 - P0) x (P2 - P0) > 0   (strictly positive)
    (P2 - P1) x (P3 - P1) > 0   (strictly positive)

i.e. both triangle orientants of the quad are strictly positive.  This
is a necessary and sufficient condition for a simple quad with the
given vertex ordering and no self-intersection at the diagonal.

Finite-difference gradient
--------------------------

``fd_gradient(patch, coord_index, step)`` computes a forward/central
finite-difference estimate of the total patch energy along the given
scalar coordinate, returning the same value as
``patch_energy_energy_1d`` but with the requested component perturbed.
``fd_gradient_all(patch, coords, base, step)`` returns the full vector
for a list of coordinate indices.

The step is recommended to be ``h * 1e-5`` (i.e. ``1e-5`` times the
target edge length), which gives a relative error ~1e-10 in
double precision for the composite energy used here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

from ..errors import MeshError

__all__ = [
    # Constants
    "Eps",
    "Q5_REQUEST_SCHEMA",
    "Q5_RESPONSE_SCHEMA",

    # Typed exceptions
    "PatchRejected",
    "InvalidSolutionQ4Patch",
    "ObjectiveRegression",
    "SolutionOutsideBox",
    "WorkerErrorQ5",

    # Data types
    "PatchSpec",
    "Q5SolveReport",

    # Energy components
    "norm_axis",
    "unit_axis",
    "energy",
    "size_term",
    "align_term",
    "shape_term",
    "jacobian_term",
    "is_valid_patch",
    "quad_valid_dets",

    # Finite-difference gradient
    "fd_gradient",
    "fd_gradient_all",
    "energy_and_gradient_fd",

    # Worker response decoding
    "decode_q5_response",
]

# Epsilon in alignment / shape denominators.  Must match the C++ worker
# (``kEps``) exactly.
Eps: float = 1e-12

# Worker protocol tags (request / response schema identifiers).
Q5_REQUEST_SCHEMA = "anymesher.quad-tinyad-request/1"
Q5_RESPONSE_SCHEMA = "anymesher.quad-tinyad-response/1"


class PatchRejected(MeshError):
    """Raised when the patch specification is out of domain for Q5.

    (zero nodes, ``h <= 0``, duplicate quad corners, axis norm too small,
    missing / non-finite positions, free indices out of range, etc.)
    """


class InvalidSolutionQ4Patch(MeshError):
    """Raised when the worker returned a solution that fails re-validation.

    (objective regressed below the initial value, free nodes moved outside
    the finite box, or the free-node positions do not yield a *valid*
    patch, i.e. a quad became self-intersecting / zero-area.)
    """

    def __init__(self, message: str, *, worker_message: str | None = None) -> None:
        super().__init__(message)
        self.worker_message = worker_message


class ObjectiveRegression(InvalidSolutionQ4Patch):
    """The worker reported a final objective *higher* than the initial one
    (violates the documented safeguard)."""


class SolutionOutsideBox(InvalidSolutionQ4Patch):
    """A returned free-node coordinate is outside the finite clamp box
    ``[init - h/2, init + h/2]`` (violates the documented safeguard)."""


class WorkerErrorQ5(MeshError):
    """The worker reported status ``"ERROR"`` (or the response failed schema
    checks in a way that is *not* a lifecycle problem).

    Carries the worker's free-text ``message`` (if any) in ``.message``.
    """

    def __init__(self, message: str, *, worker_message: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.worker_message = worker_message
def _is_finite_float(v: Any) -> bool:
    if not isinstance(v, (int, float)):
        return False
    return math.isfinite(float(v))


def _strict_real_sequence(seq: Any, name: str, *, count: int | None = None) -> list[float]:
    if not isinstance(seq, (list, tuple)) or len(seq) != (count if count is not None else len(seq)):
        pass
    if not isinstance(seq, (list, tuple)):
        raise PatchRejected(f"{name} must be a list/tuple")
    out: list[float] = []
    for v in seq:
        if not _is_finite_float(v):
            raise PatchRejected(f"{name} contains a non-finite value")
        out.append(float(v))
    return out


def _strict_int(v: Any, name: str) -> int:
    if isinstance(v, bool) or not isinstance(v, Integral):
        raise PatchRejected(f"{name} must be an integer")
    return int(v)


# ---------------------------------------------------------------------------
# Patch spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatchSpec:
    """A fully-specified Q4 patch re-optimization request.

    Attributes
    ----------
    h:            target edge length (strictly > 0, finite)
    ux, uy:       reference axis (unit after normalization, non-zero)
    nx, ny:       node x/y positions, length N
    free:         sorted, distinct node indices in ``[0, N)``
    quads:        list of 4-element tuples of distinct node indices in ``[0, N)``
    max_iter:     clamp to ``[1, 1000]`` (worker clamps its own copy)

    The invariant: ``nx`` and ``ny`` are finite reals; ``quads`` is a
    list of 4-element tuples of integers in ``[0, N)`` with four
    distinct corners each; ``free`` is sorted, distinct, in range.
    """

    h: float
    ux: float
    uy: float
    nx: tuple[float, ...]
    ny: tuple[float, ...]
    free: tuple[int, ...]
    quads: tuple[tuple[int, int, int, int], ...]
    max_iter: int = 32

    def __post_init__(self) -> None:
        if not isinstance(self.nx, (tuple, list)):
            raise PatchRejected("nx must be a sequence")
        if not isinstance(self.ny, (tuple, list)):
            raise PatchRejected("ny must be a sequence")
        if not _is_finite_float(self.h) or self.h <= 0.0:
            raise PatchRejected("h must be strictly positive and finite")
        if not _is_finite_float(self.ux) or not _is_finite_float(self.uy):
            raise PatchRejected("axis must be finite")
        # unitize
        nrm = math.hypot(self.ux, self.uy)
        if nrm <= 1e-15:
            raise PatchRejected("axis norm must be > 1e-15")
        object.__setattr__(self, "ux", self.ux / nrm)
        object.__setattr__(self, "uy", self.uy / nrm)
        n = len(self.nx)
        if n < 1:
            raise PatchRejected("n_nodes must be >= 1")
        if len(self.ny) != n:
            raise PatchRejected("nx and ny lengths must match")
        for x in self.nx:
            if not _is_finite_float(x):
                raise PatchRejected("node x must be finite")
        for y in self.ny:
            if not _is_finite_float(y):
                raise PatchRejected("node y must be finite")
        # free: distinct, in [0, n)
        seen = set()
        ordered = []
        if not isinstance(self.free, (tuple, list)):
            raise PatchRejected("free must be a sequence of int")

        def _strict_int_spec(v: Any) -> int:
            # Local: the module-level name is later shadowed by the decode
            # section's WorkerErrorQ5 variant, but spec construction must
            # raise PatchRejected.
            if isinstance(v, bool) or not isinstance(v, Integral):
                raise PatchRejected(f"free index must be an integer")
            return int(v)

        for f in self.free:
            fi = _strict_int_spec(f)
            if fi < 0 or fi >= n:
                raise PatchRejected(f"free index {fi} out of range [0, {n})")
            if fi in seen:
                raise PatchRejected(f"duplicate free index {fi}")
            seen.add(fi)
            ordered.append(fi)
        object.__setattr__(self, "free", tuple(sorted(ordered)))
        # quads: 4 distinct corners each, in range
        if not isinstance(self.quads, (tuple, list)):
            raise PatchRejected("quads must be a sequence")
        qout: list[tuple[int, int, int, int]] = []
        for q in self.quads:
            if isinstance(q, (int, float, str)) or len(q) != 4:
                raise PatchRejected("each quad must have 4 corners")
            corners: list[int] = []
            s2 = set()
            for ci in q:
                cii = _strict_int(ci, "quad corner")
                if cii < 0 or cii >= n:
                    raise PatchRejected(f"quad corner {cii} out of range [0, {n})")
                if cii in s2:
                    raise PatchRejected(f"duplicate quad corner {cii}")
                s2.add(cii)
                corners.append(cii)
            qout.append(tuple(corners))
        object.__setattr__(self, "quads", tuple(qout))
        mi = _strict_int(self.max_iter, "max_iter")
        mi = max(1, min(1000, mi))
        object.__setattr__(self, "max_iter", mi)

        # --- True-interior free-node validation (Q5 invariant) -----------
        # Every free node is the *only* node that may move.  For Q5 to be
        # well-posed the free set must be a set of *true interior* nodes of
        # the patch, i.e. each free node (a) belongs to at least one quad and
        # (b) is incident to no *boundary* edge (an edge shared by
        # exactly one quad).  A boundary node carries fixed outer geometry and
        # must remain stationary, so it must not be declared free; a free node
        # that belongs to no quad contributes nothing to the energy and cannot
        # be optimized.
        participating = {i for q in qout for i in q}
        edge_count: dict[tuple[int, int], int] = {}
        for q in qout:
            seen_edges = set()
            for k in range(4):
                a = q[k]
                b = q[(k + 1) % 4]
                edge = (a, b) if a < b else (b, a)
                # A well-formed quad has no repeated corner, so each undirected
                # edge appears once; guard anyway against a degenerate listing.
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                edge_count[edge] = edge_count.get(edge, 0) + 1
        for edge, c in edge_count.items():
            if c > 2:
                raise PatchRejected(
                    f"non-manifold edge ({edge[0]}, {edge[1]}) shared by {c} quads"
                )
        boundary_edges = {e for e, c in edge_count.items() if c == 1}
        free_set = set(self.free)
        if free_set and not qout:
            raise PatchRejected(
                "free nodes are declared but the patch has no quads; "
                "a free node with no quads contributes nothing to the energy"
            )
        for fi in self.free:
            if fi not in participating:
                raise PatchRejected(
                    f"free node {fi} is not part of any quad"
                )
            for (a, b) in boundary_edges:
                if fi in (a, b):
                    raise PatchRejected(
                        f"free node {fi} lies on the patch boundary "
                        f"(edge ({a}, {b})); boundary nodes must be stationary"
                    )

    @property
    def n_nodes(self) -> int:
        return len(self.nx)

    def to_request_dict(self, *, self_test: str | None = None) -> dict[str, Any]:
        """Serialize to the worker request JSON (matches
        ``anymesher.quad-tinyad-request/1``)."""
        out: dict[str, Any] = {
            "schema": Q5_REQUEST_SCHEMA,
            "h": self.h,
            "axis": [self.ux, self.uy],
            "n_nodes": self.n_nodes,
            "nodes": [
                v for pair in zip(self.nx, self.ny) for v in pair
            ],
            "free": list(self.free),
            "quads": [list(q) for q in self.quads],
            "max_iter": self.max_iter,
        }
        if self_test is not None:
            out["worker_self_test"] = self_test
        return out

    def node(self, i: int) -> tuple[float, float]:
        return (self.nx[i], self.ny[i])


# ---------------------------------------------------------------------------
# Solve report (typed worker response)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Q5SolveReport:
    """A validated worker response for a Q5 re-optimization request.

    ``free_final`` is parallel to ``PatchSpec.free``: a tuple of
    ``(x, y)`` position pairs, one per free node, in the *same order* as
    the free list.  ``status`` is one of ``"CONVERGED"``, ``"NOIMPROVE"``,
    or ``"ERROR"``.

    The invariant: re-computed energy and validity have *already been
    checked* by :func:`anymesher.quad.quad_tinyad_worker.solve_q5_patch`;
    this dataclass is the *validated* record for reporting.
    """

    status: str
    objective_initial: float
    objective_final: float
    iterations: int
    free_final: tuple[tuple[float, float], ...]
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema": Q5_RESPONSE_SCHEMA,
            "status": self.status,
            "objective_initial": self.objective_initial,
            "objective_final": self.objective_final,
            "iterations": self.iterations,
            "free_nodes": [list(p) for p in self.free_final],
        }
        if self.message is not None:
            out["message"] = self.message
        return out



# ---------------------------------------------------------------------------
# Energy and validity (pure Python reference — same operation order as C++)
# ---------------------------------------------------------------------------

def _quad_edges(px: Sequence[float], py: Sequence[float], quad: Sequence[int]) -> list[tuple[float, float]]:
    """Return the 4 directed edges ``e_k = P[i_{k+1}] - P[i_k]`` of the quad."""
    pts = [(px[i], py[i]) for i in quad]
    return [
        (pts[(k + 1) % 4][0] - pts[k][0], pts[(k + 1) % 4][1] - pts[k][1])
        for k in range(4)
    ]


def size_term(px: Sequence[float], py: Sequence[float], quad: Sequence[int], h: float) -> float:
    """size term: ``(1/4) * sum_k ((n_k/h^2 - 1)^2)`` for a single quad."""
    h2 = h * h
    s = 0.0
    for ek in _quad_edges(px, py, quad):
        n = ek[0] * ek[0] + ek[1] * ek[1]
        q = n / h2 - 1.0
        s += q * q
    return 0.25 * s


def align_term(px: Sequence[float], py: Sequence[float], quad: Sequence[int],
               ux: float, uy: float) -> float:
    """alignment term: ``(1/4) * sum_k (1 - (e_k . d_k)^2 / (n_k + eps))``."""
    v0x, v0y = -uy, ux
    ax, ay = 0.0, 0.0
    for k, ek in enumerate(_quad_edges(px, py, quad)):
        if k % 2 == 0:
            w = ek[0] * ux + ek[1] * uy
        else:
            w = ek[0] * v0x + ek[1] * v0y
        n = ek[0] * ek[0] + ek[1] * ek[1]
        ax += w * w / (n + Eps)
    return 0.25 * (4.0 - ax)


def shape_term(px: Sequence[float], py: Sequence[float], quad: Sequence[int]) -> float:
    """shape term: ``(1/4) * sum_k (e_k . e_{k+1})^2 / (n_k n_{k+1} + eps)``."""
    edges = list(_quad_edges(px, py, quad))
    norms = [ek[0] * ek[0] + ek[1] * ek[1] for ek in edges]
    s = 0.0
    for k in range(4):
        j = (k + 1) % 4
        d = edges[k][0] * edges[j][0] + edges[k][1] * edges[j][1]
        s += d * d / (norms[k] * norms[j] + Eps)
    return 0.25 * s


def _cross2i(px: Sequence[float], py: Sequence[float], quad: Sequence[int]) -> tuple[float, float]:
    i0, i1, i2, i3 = list(quad)
    e0x = px[i1] - px[i0]; e0y = py[i1] - py[i0]
    e1x = px[i2] - px[i1]; e1y = py[i2] - py[i1]
    detA = e0x * e1y - e0y * e1x
    detB = e1x * (py[i3] - py[i2]) - e1y * (px[i3] - px[i2])
    return detA, detB


def jacobian_term(px: Sequence[float], py: Sequence[float], quad: Sequence[int],
                  h: float) -> float:
    """jacobian term: ``(1/2) * ((A/h^2 - 1)^2 + (B/h^2 - 1)^2)``."""
    h2 = h * h
    detA, detB = _cross2i(px, py, quad)
    j0 = detA / h2 - 1.0
    j1 = detB / h2 - 1.0
    return 0.5 * (j0 * j0 + j1 * j1)


def energy(px: Sequence[float], py: Sequence[float], quads: Sequence[Sequence[int]],
           h: float, ux: float, uy: float) -> float:
    """Total patch energy (size + align + shape + jacobian) over ``quads``.

    ``px``/``py`` are flat node coordinate sequences (length N).
    ``quads`` is a sequence of 4-element index tuples.
    """
    ux, uy = unit_axis(ux, uy)
    s = 0.0
    for q in quads:
        s += size_term(px, py, q, h) + align_term(px, py, q, ux, uy) \
           + shape_term(px, py, q) + jacobian_term(px, py, q, h)
    return s


def unit_axis(ux: float, uy: float) -> tuple[float, float]:
    nrm = math.hypot(ux, uy)
    if nrm <= 1e-15:
        raise PatchRejected("axis norm must be > 1e-15")
    return (ux / nrm, uy / nrm)


def norm_axis(ux: float, uy: float) -> float:
    n = math.hypot(ux, uy)
    if not math.isfinite(n):
        raise PatchRejected("axis norm must be finite")
    return n


def quad_valid_dets(px: Sequence[float], py: Sequence[float],
                    quad: Sequence[int]) -> tuple[float, float]:
    """Both triangle orientants of the quad (in the given vertex order)."""
    detA, detB = _cross2i(px, py, quad)
    return (detA, detB)


def is_valid_patch(px: Sequence[float], py: Sequence[float],
                   quads: Sequence[Sequence[int]]) -> bool:
    """True iff every quad has strictly positive triangle orientants."""
    for q in quads:
        d0, d1 = quad_valid_dets(px, py, q)
        if d0 <= 0.0 or d1 <= 0.0:
            return False
    return True


# ---------------------------------------------------------------------------
# Finite-difference gradient
# ---------------------------------------------------------------------------

def fd_gradient(spec: PatchSpec, coord_index: int, step: float = 1e-6) -> float:
    """Central FD gradient of :func:`energy` w.r.t. ``spec``, along the
    single coordinate ``coord_index`` (0 = px0, 1 = py0, 2 = px1, ...).

    ``step`` is the finite-difference step (units of the coordinate).  Use
    ``step = h * 1e-5`` for a good trade-off between truncation and
    round-off error in the composite energy.
    """
    if step <= 0.0 or not math.isfinite(step):
        raise PatchRejected("step must be strictly positive and finite")
    if coord_index < 0 or coord_index >= 2 * spec.n_nodes:
        raise PatchRejected(f"coord_index {coord_index} out of range")
    px_plus = list(spec.nx); py_plus = list(spec.ny)
    px_minus = list(spec.nx); py_minus = list(spec.ny)
    if coord_index % 2 == 0:
        node_i = coord_index // 2
        px_plus[node_i] += step
        px_minus[node_i] -= step
    else:
        node_i = coord_index // 2
        py_plus[node_i] += step
        py_minus[node_i] -= step
    e_plus = energy(px_plus, py_plus, spec.quads, spec.h, spec.ux, spec.uy)
    e_minus = energy(px_minus, py_minus, spec.quads, spec.h, spec.ux, spec.uy)
    return (e_plus - e_minus) / (2.0 * step)


def fd_gradient_all(spec: PatchSpec, coord_indices: Sequence[int],
                    step: float = 1e-6) -> dict[int, float]:
    """Central FD gradient along a set of scalar coordinates of the patch.

    The returned mapping is ``{coord_index: dE/d(coord)}``.
    For testing purposes only — the worker uses analytic TinyAD gradients.
    """
    return {i: fd_gradient(spec, i, step=step) for i in coord_indices}


def energy_and_gradient_fd(spec: PatchSpec, step: float = 1e-6) -> tuple[float, dict[int, float]]:
    """``(energy, {i: dE/d(coord_i)})`` for all ``2N`` coordinates of the
    patch (full gradient, central finite difference)."""
    n = spec.n_nodes
    all_indices = list(range(2 * n))
    e = energy(list(spec.nx), list(spec.ny), spec.quads, spec.h, spec.ux, spec.uy)
    return (e, fd_gradient_all(spec, all_indices, step=step))


# ---------------------------------------------------------------------------
# Response decoding (pure Python, shared by the subprocess adapter)
# ---------------------------------------------------------------------------

_Q5_STATUSES = frozenset({"CONVERGED", "NOIMPROVE", "ERROR"})


def _strict_float(v: Any, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise WorkerErrorQ5(f"{name} must be a real number, got {type(v).__name__}")
    fv = float(v)
    if not math.isfinite(fv):
        raise WorkerErrorQ5(f"{name} must be finite")
    return fv


def _strict_int(v: Any, name: str) -> int:
    if isinstance(v, bool) or not isinstance(v, Integral):
        raise WorkerErrorQ5(f"{name} must be an integer, got {type(v).__name__}")
    return int(v)


def decode_q5_response(raw: Mapping[str, Any], *, free_count: int) -> Q5SolveReport:
    """Decode a raw ``anymesher.quad-tinyad-response/1`` mapping into a
    :class:`Q5SolveReport`.

    Raises
    ------
    WorkerErrorQ5
        on any schema mismatch (unknown/missing field, wrong type, bad
        status tag, bad free_nodes count) — *not* a lifecycle problem.
    """
    if not isinstance(raw, Mapping):
        raise WorkerErrorQ5("response is not a JSON object")
    schema = raw.get("schema")
    if schema != Q5_RESPONSE_SCHEMA:
        raise WorkerErrorQ5(
            f"unknown response schema {schema!r}, expected {Q5_RESPONSE_SCHEMA!r}"
        )
    status = raw.get("status")
    if not isinstance(status, str) or status not in _Q5_STATUSES:
        raise WorkerErrorQ5(f"unknown status {status!r}")
    fi = _strict_float(raw.get("objective_initial"), "objective_initial")
    ff = _strict_float(raw.get("objective_final"), "objective_final")
    it = _strict_int(raw.get("iterations"), "iterations")
    if it < 0:
        raise WorkerErrorQ5(f"iterations must be >= 0, got {it}")
    free_nodes = raw.get("free_nodes")
    if not isinstance(free_nodes, (list, tuple)):
        raise WorkerErrorQ5("free_nodes must be a list")
    if len(free_nodes) != free_count:
        raise WorkerErrorQ5(
            f"free_nodes length {len(free_nodes)} != request free length {free_count}"
        )
    pairs: list[tuple[float, float]] = []
    for i, p in enumerate(free_nodes):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise WorkerErrorQ5(f"free_nodes[{i}] must be a 2-element array")
        pairs.append((_strict_float(p[0], f"free_nodes[{i}][0]"),
                      _strict_float(p[1], f"free_nodes[{i}][1]")))
    msg = raw.get("message", None)
    if msg is not None and not isinstance(msg, str):
        raise WorkerErrorQ5("message (if present) must be a string")
    known = {"schema", "status", "objective_initial", "objective_final",
             "iterations", "free_nodes", "message"}
    extra = set(raw.keys()) - known
    if extra:
        # Tolerate unknown keys (forward compatibility); surface them as a
        # note in the report rather than erroring.
        pass
    return Q5SolveReport(
        status=status,
        objective_initial=fi,
        objective_final=ff,
        iterations=it,
        free_final=tuple(pairs),
        message=msg,
    )
