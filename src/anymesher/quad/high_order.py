"""Shared interpolation and strict validity for Q4/Q8/T3/T6 shell geometry.

The validity certificate bounds the *signed* surface Jacobian polynomial with
Bernstein coefficients.  Q4/Q8 use tensor-product degree-3 bounds on a square;
T3/T6 use total-degree-2 Bernstein bounds on a triangle.  A positive set of
samples is never a certificate: ambiguous coefficient boxes are subdivided and
become ``UNRESOLVED`` at the depth/budget limit.  A negative status always has
an actually evaluated witness.

All arithmetic is float64.  Coefficient bounds are widened by a scale-aware
roundoff guard; near-zero cases therefore fail closed instead of being falsely
certified.  This is intentionally independent of geometry-owner APIs so CH2/CH3
can supply their own expected reference normal later.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from math import comb, copysign
from time import perf_counter
from typing import Callable, Iterable

import numpy as np

from .timing import record_quad_stage



class ElementFamily(str, Enum):
    Q4 = "Q4"
    Q8 = "Q8"
    T3 = "T3"
    T6 = "T6"


class ValidityStatus(str, Enum):
    CERTIFIED_POSITIVE = "CERTIFIED_POSITIVE"
    INVALID = "INVALID"
    UNRESOLVED = "UNRESOLVED"


_CURVATURE_CLASSES = frozenset(("straight", "analytic_curved", "sampled"))


@dataclass(frozen=True)
class HighOrderBoundaryMidside:
    canonical_edge: tuple[object, ...]
    source_edge_id: int
    station_interval: tuple[float, float]
    parameter: float
    node_id: int
    residual: float
    curvature_class: str

    def __post_init__(self) -> None:
        lower, upper = (float(v) for v in self.station_interval)
        parameter = float(self.parameter)
        residual = float(self.residual)
        if not np.all(np.isfinite((lower, upper, parameter, residual))):
            raise ValueError("high-order boundary provenance must be finite")
        if lower > upper or parameter < lower - 1.0e-12 or parameter > upper + 1.0e-12:
            raise ValueError("high-order midpoint parameter is outside its station interval")
        if residual < 0.0:
            raise ValueError("high-order boundary residual must be non-negative")
        if str(self.curvature_class) not in _CURVATURE_CLASSES:
            raise ValueError(f"unsupported high-order edge curvature class: {self.curvature_class!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical_edge": list(self.canonical_edge),
            "source_edge": int(self.source_edge_id),
            "station_interval": [float(value) for value in self.station_interval],
            "midpoint_parameter": float(self.parameter),
            "midpoint_node": int(self.node_id),
            "residual": float(self.residual),
            "curvature_class": str(self.curvature_class),
        }


class HighOrderCertificationCancelled(RuntimeError):
    """Raised when strict high-order certification is cancelled at a safe checkpoint."""


_NODE_COUNT = {ElementFamily.Q4: 4, ElementFamily.Q8: 8, ElementFamily.T3: 3, ElementFamily.T6: 6}

@dataclass(frozen=True)
class MappingEvaluation:
    points: np.ndarray
    dxi: np.ndarray
    deta: np.ndarray
    jacobian_vector: np.ndarray
    jacobian_magnitude: np.ndarray
    signed_jacobian: np.ndarray
    normalized_quality: np.ndarray
    reference_normal: np.ndarray


@dataclass(frozen=True)
class ValidityReport:
    status: ValidityStatus
    method: str
    lower_bound: float
    upper_bound: float
    tolerance: float
    subdivisions: int
    max_depth_reached: int
    witness_value: float | None
    witness_parameter: tuple[float, float] | None

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-safe representation of the certificate."""
        return {
            "status": self.status.value,
            "method": self.method,
            "lower_bound": float(self.lower_bound),
            "upper_bound": float(self.upper_bound),
            "tolerance": float(self.tolerance),
            "subdivisions": int(self.subdivisions),
            "max_depth_reached": int(self.max_depth_reached),
            "witness_value": None if self.witness_value is None else float(self.witness_value),
            "witness_parameter": None if self.witness_parameter is None else [float(v) for v in self.witness_parameter],
        }


@dataclass(frozen=True)
class HighOrderGeometryReport:
    model_id: str
    revision: int
    face_id: int
    geometry_family: str
    chart_kind: str
    chart_origin: tuple[float, float, float]
    boundary_projection: str
    interior_projection: str
    q8_count: int
    t6_count: int
    certified_elements: int
    total_elements: int
    boundary_midsides: tuple[HighOrderBoundaryMidside, ...]
    edge_curvature_classes: tuple[str, ...]
    interior_midside_count: int
    max_geometry_residual: float
    validity_status: str = "CERTIFIED_POSITIVE"
    invalid_elements: int = 0

    def __post_init__(self) -> None:
        counts = (self.q8_count, self.t6_count, self.certified_elements, self.total_elements, self.interior_midside_count, self.invalid_elements)
        if any(int(value) < 0 for value in counts):
            raise ValueError("high-order geometry report counts must be non-negative")
        if int(self.total_elements) != int(self.q8_count) + int(self.t6_count):
            raise ValueError("high-order geometry report shell counts do not match total_elements")
        if int(self.certified_elements) > int(self.total_elements) or int(self.invalid_elements) > int(self.total_elements):
            raise ValueError("high-order geometry report certification counts are inconsistent")
        origin = np.asarray(self.chart_origin, dtype=np.float64)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("high-order chart origin must be a finite 3D point")
        if not np.isfinite(float(self.max_geometry_residual)) or float(self.max_geometry_residual) < 0.0:
            raise ValueError("high-order geometry residual must be finite and non-negative")
        classes = tuple(sorted(set(str(value) for value in self.edge_curvature_classes)))
        if classes != tuple(self.edge_curvature_classes) or any(value not in _CURVATURE_CLASSES for value in classes):
            raise ValueError("high-order edge curvature classes must be sorted unique supported values")
        boundary_classes = tuple(sorted({item.curvature_class for item in self.boundary_midsides}))
        if boundary_classes != classes:
            raise ValueError("high-order face curvature classes do not match boundary provenance")
        if str(self.validity_status) == ValidityStatus.CERTIFIED_POSITIVE.value and (int(self.certified_elements) != int(self.total_elements) or int(self.invalid_elements)):
            raise ValueError("positive high-order face report must certify every element")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": str(self.model_id), "revision": int(self.revision),
            "face_id": int(self.face_id), "geometry_family": str(self.geometry_family),
            "chart_kind": str(self.chart_kind), "chart_origin": [float(value) for value in self.chart_origin],
            "boundary_projection": str(self.boundary_projection), "interior_projection": str(self.interior_projection),
            "q8_count": int(self.q8_count), "t6_count": int(self.t6_count),
            "certified_elements": int(self.certified_elements), "total_elements": int(self.total_elements),
            "boundary_midsides": [item.to_dict() for item in self.boundary_midsides],
            "edge_curvature_classes": list(self.edge_curvature_classes),
            "interior_midside_count": int(self.interior_midside_count),
            "max_geometry_residual": float(self.max_geometry_residual),
            "validity_status": str(self.validity_status), "invalid_elements": int(self.invalid_elements),
        }


@dataclass(frozen=True)
class HighOrderMeshCertificate:
    status: ValidityStatus | str
    model_id: str
    revision: int
    reports: tuple[HighOrderGeometryReport, ...]
    q8_count: int
    t6_count: int
    unique_midside_count: int
    unique_boundary_midside_count: int
    max_geometry_residual: float
    order: str = "quadratic"
    target_size: float = 0.0
    route: str = "quad-first"

    def _unique_boundary(self) -> dict[tuple[object, ...], dict[str, object]]:
        unique: dict[tuple[object, ...], dict[str, object]] = {}
        for face in self.reports:
            for item in face.boundary_midsides:
                payload = item.to_dict()
                key = tuple(payload["canonical_edge"])
                prior = unique.get(key)
                if prior is not None and prior != payload:
                    raise ValueError(f"conflicting high-order boundary midside provenance for {key!r}")
                unique[key] = payload
        return unique

    def __post_init__(self) -> None:
        try:
            status = self.status.value if isinstance(self.status, ValidityStatus) else ValidityStatus(str(self.status)).value
        except ValueError as error:
            raise ValueError(f"unsupported high-order certificate status: {self.status!r}") from error
        counts = (self.q8_count, self.t6_count, self.unique_midside_count, self.unique_boundary_midside_count)
        if any(int(value) < 0 for value in counts):
            raise ValueError("high-order certificate counts must be non-negative")
        if int(self.unique_boundary_midside_count) > int(self.unique_midside_count):
            raise ValueError("boundary midside count exceeds total midside count")
        if not np.isfinite(float(self.max_geometry_residual)) or float(self.max_geometry_residual) < 0.0:
            raise ValueError("high-order certificate residual must be finite and non-negative")
        if len({int(item.face_id) for item in self.reports}) != len(self.reports):
            raise ValueError("high-order certificate contains duplicate face reports")
        if any(str(item.model_id) != str(self.model_id) or int(item.revision) != int(self.revision) for item in self.reports):
            raise ValueError("high-order face report model/revision does not match certificate")
        if sum(int(item.q8_count) for item in self.reports) != int(self.q8_count) or sum(int(item.t6_count) for item in self.reports) != int(self.t6_count):
            raise ValueError("high-order certificate shell counts do not match face reports")
        unique = self._unique_boundary()
        if len(unique) != int(self.unique_boundary_midside_count):
            raise ValueError("high-order certificate boundary midside count is inconsistent")
        interior_count = sum(int(item.interior_midside_count) for item in self.reports)
        if interior_count + len(unique) != int(self.unique_midside_count):
            raise ValueError("high-order certificate total midside count is inconsistent")
        report_max = max((float(item.max_geometry_residual) for item in self.reports), default=0.0)
        if not np.isclose(report_max, float(self.max_geometry_residual), rtol=0.0, atol=1.0e-15):
            raise ValueError("high-order certificate residual does not match face reports")
        if status == ValidityStatus.CERTIFIED_POSITIVE.value:
            bad = [item.face_id for item in self.reports if item.validity_status != status or item.invalid_elements or item.certified_elements != item.total_elements]
            if bad:
                raise ValueError(f"positive high-order certificate contains invalid faces {bad}")

    def to_dict(self) -> dict[str, object]:
        status = self.status.value if isinstance(self.status, ValidityStatus) else str(self.status)
        unique = self._unique_boundary()
        return {
            "status": status, "model_id": str(self.model_id), "revision": int(self.revision),
            "order": str(self.order), "target_size": float(self.target_size), "route": str(self.route),
            "reports": [item.to_dict() for item in self.reports],
            "q8_count": int(self.q8_count), "t6_count": int(self.t6_count),
            "unique_midside_count": int(self.unique_midside_count),
            "unique_boundary_midside_count": int(self.unique_boundary_midside_count),
            "max_geometry_residual": float(self.max_geometry_residual),
            "boundary_midsides": [unique[key] for key in sorted(unique, key=repr)],
        }


@dataclass(frozen=True)
class GeometryErrorReport:
    maximum: float
    rms: float


@dataclass(frozen=True)
class NormalErrorReport:
    maximum_angle_rad: float
    rms_angle_rad: float

def _family(value: ElementFamily | str) -> ElementFamily:
    try:
        return value if isinstance(value, ElementFamily) else ElementFamily(str(value).upper())
    except ValueError as exc:
        raise ValueError(f"unsupported high-order element family: {value!r}") from exc


def _params_array(params: np.ndarray | Iterable[Iterable[float]] | Iterable[float]) -> tuple[np.ndarray, bool]:
    arr = np.asarray(params, dtype=np.float64)
    single = arr.ndim == 1
    if single:
        if arr.shape != (2,):
            raise ValueError("a reference parameter must have shape (2,)")
        arr = arr.reshape(1, 2)
    elif arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("reference parameters must have shape (n, 2)")
    if not np.isfinite(arr).all():
        raise ValueError("reference parameters must be finite")
    return arr, single


def _nodes_array(nodes: np.ndarray | Iterable[Iterable[float]], fam: ElementFamily) -> np.ndarray:
    arr = np.asarray(nodes, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != _NODE_COUNT[fam] or arr.shape[1] not in (2, 3):
        raise ValueError(f"{fam.value} nodes must have shape ({_NODE_COUNT[fam]}, 2|3)")
    if not np.isfinite(arr).all():
        raise ValueError("element nodes must be finite")
    if arr.shape[1] == 2:
        arr = np.column_stack((arr, np.zeros(arr.shape[0], dtype=np.float64)))
    return arr

def shape_values(family: ElementFamily | str, params: np.ndarray | Iterable) -> np.ndarray:
    """Evaluate shape values with frozen Q8/T6 corner-then-midside ordering."""
    fam = _family(family)
    p, single = _params_array(params)
    x, y = p[:, 0], p[:, 1]
    if fam is ElementFamily.Q4:
        out = 0.25 * np.column_stack(((1-x)*(1-y), (1+x)*(1-y), (1+x)*(1+y), (1-x)*(1+y)))
    elif fam is ElementFamily.Q8:
        out = np.column_stack((
            -0.25*(1-x)*(1-y)*(1+x+y), -0.25*(1+x)*(1-y)*(1-x+y),
            -0.25*(1+x)*(1+y)*(1-x-y), -0.25*(1-x)*(1+y)*(1+x-y),
            0.5*(1-x*x)*(1-y), 0.5*(1+x)*(1-y*y),
            0.5*(1-x*x)*(1+y), 0.5*(1-x)*(1-y*y),
        ))
    else:
        l0, l1, l2 = 1.0-x-y, x, y
        if fam is ElementFamily.T3:
            out = np.column_stack((l0, l1, l2))
        else:
            out = np.column_stack((
                l0*(2*l0-1), l1*(2*l1-1), l2*(2*l2-1),
                4*l0*l1, 4*l1*l2, 4*l2*l0,
            ))
    return out[0] if single else out

def shape_gradients(family: ElementFamily | str, params: np.ndarray | Iterable) -> np.ndarray:
    """Return analytic dN/d(xi,eta), shape ``(..., nnode, 2)``."""
    fam = _family(family)
    p, single = _params_array(params)
    x, y = p[:, 0], p[:, 1]
    if fam is ElementFamily.Q4:
        dx = 0.25*np.column_stack((-(1-y), (1-y), (1+y), -(1+y)))
        dy = 0.25*np.column_stack((-(1-x), -(1+x), (1+x), (1-x)))
    elif fam is ElementFamily.Q8:
        dx = np.column_stack((
            0.25*(1-y)*(2*x+y), 0.25*(1-y)*(2*x-y),
            0.25*(1+y)*(2*x+y), 0.25*(1+y)*(2*x-y),
            -x*(1-y), 0.5*(1-y*y), -x*(1+y), -0.5*(1-y*y),
        ))
        dy = np.column_stack((
            0.25*(1-x)*(x+2*y), 0.25*(1+x)*(-x+2*y),
            0.25*(1+x)*(x+2*y), 0.25*(1-x)*(-x+2*y),
            -0.5*(1-x*x), -(1+x)*y, 0.5*(1-x*x), -(1-x)*y,
        ))
    else:
        m = p.shape[0]
        if fam is ElementFamily.T3:
            base = np.array([[-1.,-1.],[1.,0.],[0.,1.]], dtype=np.float64)
            out = np.broadcast_to(base, (m, 3, 2)).copy()
            return out[0] if single else out
        l0, l1, l2 = 1.0-x-y, x, y
        dl = np.array([[-1.,-1.],[1.,0.],[0.,1.]], dtype=np.float64)
        out = np.empty((m, 6, 2), dtype=np.float64)
        for i, li in enumerate((l0,l1,l2)):
            out[:, i, :] = (4*li-1)[:,None]*dl[i]
        out[:,3,:] = 4*(l0[:,None]*dl[1] + l1[:,None]*dl[0])
        out[:,4,:] = 4*(l1[:,None]*dl[2] + l2[:,None]*dl[1])
        out[:,5,:] = 4*(l2[:,None]*dl[0] + l0[:,None]*dl[2])
        return out[0] if single else out
    out = np.stack((dx, dy), axis=2)
    return out[0] if single else out

def _corner_reference_normal(nodes: np.ndarray, fam: ElementFamily) -> np.ndarray:
    if fam in (ElementFamily.Q4, ElementFamily.Q8):
        raw = np.cross(nodes[1]-nodes[0], nodes[3]-nodes[0])
    else:
        raw = np.cross(nodes[1]-nodes[0], nodes[2]-nodes[0])
    norm = float(np.linalg.norm(raw))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("corner skeleton has no usable reference normal")
    return raw / norm


@lru_cache(maxsize=32)
def _mapping_reference_basis(fam: ElementFamily, parameter_bytes: bytes):
    """Read-only basis tables for at most 256 reference samples per entry.

    Exact parameter bytes preserve signed zeros and detach the cache from
    mutable caller arrays.  Only reference data is retained, never geometry,
    normals or mapping results.  The caller bounds both sample count and
    coordinates and retains the uncached path for floating error handling.
    """
    params = np.frombuffer(parameter_bytes, dtype=np.float64).reshape(-1, 2)
    values = np.asarray(shape_values(fam, params), dtype=np.float64)
    gradients = np.asarray(shape_gradients(fam, params), dtype=np.float64)
    values.flags.writeable = False
    gradients.flags.writeable = False
    return values, gradients


def evaluate_mapping(
    nodes: np.ndarray | Iterable[Iterable[float]],
    family: ElementFamily | str,
    params: np.ndarray | Iterable,
    *,
    reference_normal: np.ndarray | Iterable[float] | None = None,
) -> MappingEvaluation:
    fam = _family(family)
    xyz = _nodes_array(nodes, fam)
    p, _ = _params_array(params)
    if (len(p) <= 256 and np.geterr()["under"] == "ignore"
            and np.max(np.abs(p), initial=0.0) <= 1.0):
        # These bounded reference coordinates cannot overflow the basis
        # polynomials. Underflow remains observable on the uncached route
        # whenever NumPy is configured to report or raise it.
        vals, grads = _mapping_reference_basis(fam, p.tobytes())
    else:
        vals = np.asarray(shape_values(fam, p), dtype=np.float64)
        grads = np.asarray(shape_gradients(fam, p), dtype=np.float64)
    points = vals @ xyz
    dxi = grads[:,:,0] @ xyz
    deta = grads[:,:,1] @ xyz
    cross = np.cross(dxi, deta)
    mag = np.linalg.norm(cross, axis=1)
    if reference_normal is None:
        nref = _corner_reference_normal(xyz, fam)
    else:
        nref = np.asarray(reference_normal, dtype=np.float64)
        if nref.shape != (3,) or not np.isfinite(nref).all():
            raise ValueError("reference_normal must be a finite 3-vector")
        nn = float(np.linalg.norm(nref))
        if nn == 0.0:
            raise ValueError("reference_normal must be nonzero")
        nref = nref / nn
    signed = cross @ nref
    denom = np.linalg.norm(dxi, axis=1) * np.linalg.norm(deta, axis=1)
    quality = np.divide(signed, denom, out=np.zeros_like(signed), where=denom > 0.0)
    return MappingEvaluation(
        points=points,
        dxi=dxi,
        deta=deta,
        jacobian_vector=cross,
        jacobian_magnitude=mag,
        signed_jacobian=signed,
        normalized_quality=quality,
        reference_normal=nref.copy(),
    )


def physical_area(nodes: np.ndarray | Iterable[Iterable[float]], family: ElementFamily | str) -> float:
    fam = _family(family)
    if fam in (ElementFamily.Q4, ElementFamily.Q8):
        q, w = np.polynomial.legendre.leggauss(4)
        pts = np.array([(a,b) for a in q for b in q], dtype=np.float64)
        weights = np.array([wa*wb for wa in w for wb in w], dtype=np.float64)
    else:
        # Dunavant degree-5, weights normalized to reference-triangle area 1/2.
        a1, b1 = 0.470142064105115, 0.059715871789770
        a2, b2 = 0.101286507323456, 0.797426985353087
        pts = np.array([[1/3,1/3],[a1,a1],[a1,b1],[b1,a1],[a2,a2],[a2,b2],[b2,a2]], dtype=np.float64)
        weights = 0.5*np.array([0.225, 0.132394152788506, 0.132394152788506, 0.132394152788506, 0.125939180544827, 0.125939180544827, 0.125939180544827], dtype=np.float64)
    ev = evaluate_mapping(nodes, fam, pts)
    return float(np.dot(weights, ev.jacobian_magnitude))


def geometry_error(
    nodes: np.ndarray | Iterable[Iterable[float]],
    family: ElementFamily | str,
    params: np.ndarray | Iterable,
    reference_points: np.ndarray | Iterable[Iterable[float]],
) -> GeometryErrorReport:
    mapped = evaluate_mapping(nodes, family, params).points
    ref = np.asarray(reference_points, dtype=np.float64)
    if ref.shape != mapped.shape or not np.isfinite(ref).all():
        raise ValueError("reference_points must be finite and match mapped point shape")
    err = np.linalg.norm(mapped-ref, axis=1)
    return GeometryErrorReport(float(np.max(err, initial=0.0)), float(np.sqrt(np.mean(err*err))) if err.size else 0.0)


def normal_error(
    nodes: np.ndarray | Iterable[Iterable[float]],
    family: ElementFamily | str,
    params: np.ndarray | Iterable,
    reference_normals: np.ndarray | Iterable[Iterable[float]],
) -> NormalErrorReport:
    ev = evaluate_mapping(nodes, family, params)
    ref = np.asarray(reference_normals, dtype=np.float64)
    if ref.shape != ev.jacobian_vector.shape or not np.isfinite(ref).all():
        raise ValueError("reference_normals must be finite and match mapped normal shape")
    an = np.linalg.norm(ev.jacobian_vector, axis=1)
    rn = np.linalg.norm(ref, axis=1)
    if np.any(an == 0.0) or np.any(rn == 0.0):
        raise ValueError("normal vectors must be nonzero")
    dots = np.einsum('ij,ij->i', ev.jacobian_vector/an[:,None], ref/rn[:,None])
    ang = np.arccos(np.clip(dots, -1.0, 1.0))
    return NormalErrorReport(float(np.max(ang, initial=0.0)), float(np.sqrt(np.mean(ang*ang))) if ang.size else 0.0)

def _bernstein_row_3(t: float) -> np.ndarray:
    return np.array([comb(3,k)*(t**k)*((1.0-t)**(3-k)) for k in range(4)], dtype=np.float64)


_BERN_NODES = np.linspace(0.0, 1.0, 4)
_BERN_MATRIX = np.vstack([_bernstein_row_3(float(t)) for t in _BERN_NODES])
# Exact inverse of the cubic Bernstein collocation matrix at t=(0,1/3,2/3,1).
_BERN_MATRIX_INV = np.array([[1.,0.,0.,0.],[-5/6,3.,-3/2,1/3],[1/3,-3/2,3.,-5/6],[0.,0.,0.,1.]], dtype=np.float64)
_BERN_CHECK_ROWS = tuple(
    (t, _bernstein_row_3(t)) for t in (0.137, 0.389, 0.731)
)
for _t, _row in _BERN_CHECK_ROWS:
    _row.flags.writeable = False


def _square_coefficients(jfun, bounds: tuple[float,float,float,float], scale2: float) -> tuple[np.ndarray,float]:
    x0,x1,y0,y1 = bounds
    xs = x0 + (x1-x0)*_BERN_NODES
    ys = y0 + (y1-y0)*_BERN_NODES
    vals = np.array([[jfun(float(x),float(y)) for y in ys] for x in xs], dtype=np.float64)
    coeff = _BERN_MATRIX_INV @ vals @ _BERN_MATRIX_INV.T
    residual = 0.0
    for t, bt in _BERN_CHECK_ROWS:
        x = x0 + (x1-x0)*t
        for u, bu in _BERN_CHECK_ROWS:
            y = y0 + (y1-y0)*u
            residual = max(residual, abs(float(bt @ coeff @ bu) - jfun(float(x),float(y))))
    eps = np.finfo(np.float64).eps
    guard = residual + 4096.0*eps*max(scale2, float(np.max(np.abs(vals))), float(np.max(np.abs(coeff))), np.finfo(float).tiny)
    return coeff, float(guard)


def _square_witness(jfun, bounds: tuple[float,float,float,float]) -> tuple[float,tuple[float,float]]:
    x0,x1,y0,y1 = bounds
    best = (float("inf"), (x0,y0))
    for t in np.linspace(0.0,1.0,5):
        x = x0 + (x1-x0)*float(t)
        for u in np.linspace(0.0,1.0,5):
            y = y0 + (y1-y0)*float(u)
            value = jfun(float(x),float(y))
            if value < best[0]:
                best = (float(value),(float(x),float(y)))
    return best

def _tri_coefficients(jfun, tri: np.ndarray, scale2: float) -> tuple[np.ndarray,float]:
    v0,v1,v2 = tri
    m01,m12,m20 = 0.5*(v0+v1), 0.5*(v1+v2), 0.5*(v2+v0)
    f0,f1,f2 = (jfun(*map(float,v)) for v in (v0,v1,v2))
    f01,f12,f20 = (jfun(*map(float,v)) for v in (m01,m12,m20))
    b01 = 2.0*f01 - 0.5*(f0+f1)
    b12 = 2.0*f12 - 0.5*(f1+f2)
    b20 = 2.0*f20 - 0.5*(f2+f0)
    coeff = np.array([f0,f1,f2,b01,b12,b20], dtype=np.float64)
    c = (v0+v1+v2)/3.0
    recon = (coeff[0]+coeff[1]+coeff[2]+2*(coeff[3]+coeff[4]+coeff[5]))/9.0
    residual = abs(float(recon) - jfun(float(c[0]),float(c[1])))
    eps = np.finfo(np.float64).eps
    guard = residual + 4096.0*eps*max(scale2, float(np.max(np.abs(coeff))), np.finfo(float).tiny)
    return coeff, float(guard)


def _tri_witness(jfun, tri: np.ndarray) -> tuple[float,tuple[float,float]]:
    v0,v1,v2 = tri
    candidates = [v0,v1,v2,0.5*(v0+v1),0.5*(v1+v2),0.5*(v2+v0),(v0+v1+v2)/3.0]
    best = (float("inf"),(float(v0[0]),float(v0[1])))
    for p in candidates:
        value = float(jfun(float(p[0]),float(p[1])))
        if value < best[0]:
            best = (value,(float(p[0]),float(p[1])))
    return best


def _split_triangle(tri: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    a,b,c = tri
    ab,bc,ca = 0.5*(a+b),0.5*(b+c),0.5*(c+a)
    return (np.array([a,ab,ca]), np.array([ab,b,bc]), np.array([ca,bc,c]), np.array([ab,bc,ca]))

def _point_gradients(fam: ElementFamily, x: float, y: float) -> np.ndarray:
    """``shape_gradients(fam, (x, y))`` for one point, bit for bit.

    The same element-wise expressions in the same order as the vectorized
    function, returned as the same C-contiguous ``(nnode, 2)`` array, without
    the per-call ``column_stack``/``stack`` overhead.
    """
    if fam is ElementFamily.Q4:
        rows = (
            (0.25*(-(1-y)), 0.25*(-(1-x))), (0.25*(1-y), 0.25*(-(1+x))),
            (0.25*(1+y), 0.25*(1+x)), (0.25*(-(1+y)), 0.25*(1-x)),
        )
    elif fam is ElementFamily.Q8:
        rows = (
            (0.25*(1-y)*(2*x+y), 0.25*(1-x)*(x+2*y)),
            (0.25*(1-y)*(2*x-y), 0.25*(1+x)*(-x+2*y)),
            (0.25*(1+y)*(2*x+y), 0.25*(1+x)*(x+2*y)),
            (0.25*(1+y)*(2*x-y), 0.25*(1-x)*(-x+2*y)),
            (-x*(1-y), -0.5*(1-x*x)),
            (0.5*(1-y*y), -(1+x)*y),
            (-x*(1+y), 0.5*(1-x*x)),
            (-0.5*(1-y*y), -(1-x)*y),
        )
    elif fam is ElementFamily.T3:
        rows = ((-1., -1.), (1., 0.), (0., 1.))
    else:
        l0, l1, l2 = 1.0-x-y, x, y
        dl = ((-1., -1.), (1., 0.), (0., 1.))
        rows = tuple(
            ((4*li-1)*dl[i][0], (4*li-1)*dl[i][1]) for i, li in enumerate((l0, l1, l2))
        ) + (
            (4*(l0*dl[1][0] + l1*dl[0][0]), 4*(l0*dl[1][1] + l1*dl[0][1])),
            (4*(l1*dl[2][0] + l2*dl[1][0]), 4*(l1*dl[2][1] + l2*dl[1][1])),
            (4*(l2*dl[0][0] + l0*dl[2][0]), 4*(l2*dl[0][1] + l0*dl[2][1])),
        )
    return np.array(rows, dtype=np.float64)


@lru_cache(maxsize=4096)
def _certification_gradients(
    fam: ElementFamily, a: float, b: float, sign_a: float, sign_b: float
) -> np.ndarray:
    """Bounded, geometry-independent reference data shared by certificates.

    The sign arguments keep signed zeros distinct in the cache key.  No mesh
    coordinates, normals or Jacobian values survive a certification call.
    """
    gradients = _point_gradients(fam, a, b)
    gradients.flags.writeable = False
    return gradients


def _jacobian_function(xyz: np.ndarray, fam: ElementFamily, nref: np.ndarray):
    # Subdivision revisits parameters (the root patch is expanded twice and
    # neighbouring patches share nodes); the value is a pure function of them.
    memo: dict[tuple[float, float, float, float], float] = {}

    def evaluate(a: float, b: float) -> float:
        key = (a, b, copysign(1.0, a), copysign(1.0, b))  # keep -0.0 distinct
        value = memo.get(key)
        if value is not None:
            return value
        g = _certification_gradients(fam, float(a), float(b), key[2], key[3])
        dxi = g[:,0] @ xyz
        deta = g[:,1] @ xyz
        # ``np.cross`` of two 3-vectors: the same multiply-then-subtract.
        x0, x1, x2 = float(dxi[0]), float(dxi[1]), float(dxi[2])
        y0, y1, y2 = float(deta[0]), float(deta[1]), float(deta[2])
        cross = np.array((x1*y2 - x2*y1, x2*y0 - x0*y2, x0*y1 - x1*y0), dtype=np.float64)
        value = float(np.dot(cross, nref))
        memo[key] = value
        return value
    return evaluate


def _scale_tolerance(xyz: np.ndarray, fam: ElementFamily) -> tuple[float,float]:
    ncorner = 4 if fam in (ElementFamily.Q4,ElementFamily.Q8) else 3
    corners = xyz[:ncorner]
    diameter = max(float(np.linalg.norm(a-b)) for i,a in enumerate(corners) for b in corners[i+1:])
    if diameter == 0.0:
        raise ValueError("corner skeleton has zero diameter")
    scale2 = diameter*diameter
    return scale2, 4096.0*np.finfo(np.float64).eps*scale2


def _check_certification_cancellation(
    cancellation_check: Callable[[str], object] | None, stage: str
) -> None:
    if cancellation_check is not None and cancellation_check(stage):
        raise HighOrderCertificationCancelled(stage)


def certify_mapping_validity(
    nodes: np.ndarray | Iterable[Iterable[float]],
    family: ElementFamily | str,
    *,
    reference_normal: np.ndarray | Iterable[float] | None = None,
    max_depth: int = 6,
    max_subdivisions: int = 4096,
    cancellation_check: Callable[[str], object] | None = None,
) -> ValidityReport:
    """Conservatively certify orientation using Bernstein/subdivision bounds."""
    fam = _family(family)
    xyz = _nodes_array(nodes, fam)
    if max_depth < 0 or max_subdivisions < 0:
        raise ValueError("subdivision limits must be non-negative")
    _check_certification_cancellation(cancellation_check, "high-order validity start")
    if reference_normal is None:
        nref = _corner_reference_normal(xyz, fam)
    else:
        nref = np.asarray(reference_normal, dtype=np.float64)
        if nref.shape != (3,) or not np.isfinite(nref).all() or np.linalg.norm(nref) == 0.0:
            raise ValueError("reference_normal must be a finite nonzero 3-vector")
        nref = nref / np.linalg.norm(nref)
    scale2, tol = _scale_tolerance(xyz, fam)
    jfun = _jacobian_function(xyz, fam, nref)
    leaf_bounds: list[tuple[float, float]] = []
    unresolved = False
    subdivisions = 0
    depth_reached = 0
    witness_value = float("inf")
    witness_param: tuple[float, float] | None = None

    if fam in (ElementFamily.Q4, ElementFamily.Q8):
        root_bounds = (-1.0, 1.0, -1.0, 1.0)
        root_coeff, root_guard = _square_coefficients(jfun, root_bounds, scale2)
        global_lower = float(np.min(root_coeff) - root_guard)
        global_upper = float(np.max(root_coeff) + root_guard)
        stack: list[tuple[tuple[float, float, float, float], int]] = [(root_bounds, 0)]
        method = "bernstein-square-bicubic"
        while stack:
            _check_certification_cancellation(cancellation_check, "high-order validity square patch")
            bounds, depth = stack.pop()
            depth_reached = max(depth_reached, depth)
            coeff, guard = (
                (root_coeff, root_guard) if depth == 0
                else _square_coefficients(jfun, bounds, scale2)
            )
            lower = float(np.min(coeff) - guard)
            upper = float(np.max(coeff) + guard)
            direct, param = _square_witness(jfun, bounds)
            if direct < witness_value:
                witness_value, witness_param = direct, param
            if direct <= -tol or direct == 0.0:
                return ValidityReport(ValidityStatus.INVALID, method, global_lower, global_upper, float(tol), subdivisions, depth_reached, direct, param)
            if lower > tol:
                leaf_bounds.append((lower, upper))
                continue
            if depth >= max_depth or subdivisions >= max_subdivisions:
                unresolved = True
                leaf_bounds.append((lower, upper))
                continue
            x0, x1, y0, y1 = bounds
            xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            stack.extend([
                ((x0, xm, y0, ym), depth + 1),
                ((xm, x1, y0, ym), depth + 1),
                ((x0, xm, ym, y1), depth + 1),
                ((xm, x1, ym, y1), depth + 1),
            ])
            subdivisions += 1
    else:
        root = np.array([[0., 0.], [1., 0.], [0., 1.]], dtype=np.float64)
        root_coeff, root_guard = _tri_coefficients(jfun, root, scale2)
        global_lower = float(np.min(root_coeff) - root_guard)
        global_upper = float(np.max(root_coeff) + root_guard)
        stack_t: list[tuple[np.ndarray, int]] = [(root, 0)]
        method = "bernstein-triangle-degree2"
        while stack_t:
            _check_certification_cancellation(cancellation_check, "high-order validity triangle patch")
            tri, depth = stack_t.pop()
            depth_reached = max(depth_reached, depth)
            coeff, guard = (
                (root_coeff, root_guard) if depth == 0
                else _tri_coefficients(jfun, tri, scale2)
            )
            lower = float(np.min(coeff) - guard)
            upper = float(np.max(coeff) + guard)
            direct, param = _tri_witness(jfun, tri)
            if direct < witness_value:
                witness_value, witness_param = direct, param
            if direct <= -tol or direct == 0.0:
                return ValidityReport(ValidityStatus.INVALID, method, global_lower, global_upper, float(tol), subdivisions, depth_reached, direct, param)
            if lower > tol:
                leaf_bounds.append((lower, upper))
                continue
            if depth >= max_depth or subdivisions >= max_subdivisions:
                unresolved = True
                leaf_bounds.append((lower, upper))
                continue
            stack_t.extend((child, depth + 1) for child in _split_triangle(tri))
            subdivisions += 1

    if not leaf_bounds:
        raise RuntimeError("validity subdivision produced no terminal patches")
    status = ValidityStatus.UNRESOLVED if unresolved else ValidityStatus.CERTIFIED_POSITIVE
    if witness_param is None:
        witness_value, witness_param = jfun(0.0, 0.0), (0.0, 0.0)
    return ValidityReport(status, method, global_lower, global_upper, float(tol), subdivisions, depth_reached, float(witness_value), witness_param)



_BATCH_SCALAR_CERTIFIER = certify_mapping_validity
_BATCH_ROOT_FUNCTIONS = (
    _corner_reference_normal, _scale_tolerance, _jacobian_function,
    _square_coefficients, _tri_coefficients, _square_witness, _tri_witness,
    _certification_gradients, _point_gradients,
)
_CERTIFICATION_BATCH_SIZE = 256
_BATCH_NUMPY_ERRORS = dict(divide="warn", over="warn", under="ignore", invalid="warn")


def _batch_root_functions_unchanged() -> bool:
    return _BATCH_ROOT_FUNCTIONS == (
        _corner_reference_normal, _scale_tolerance, _jacobian_function,
        _square_coefficients, _tri_coefficients, _square_witness, _tri_witness,
        _certification_gradients, _point_gradients,
    )


@lru_cache(maxsize=2)
def _batch_root_samples(fam: ElementFamily):
    """Reference data only, with the scalar root's exact order and signed zeros."""
    rows = {}
    witnesses = []

    def collect(a, b):
        key = (float(a), float(b), copysign(1., a), copysign(1., b))
        rows.setdefault(key, len(rows))
        return 1.

    def witness(a, b):
        collect(a, b)
        key = (float(a), float(b), copysign(1., a), copysign(1., b))
        witnesses.append((rows[key], (float(a), float(b))))
        return 1.

    if fam is ElementFamily.Q8:
        root = (-1., 1., -1., 1.)
        _square_coefficients(collect, root, 1.)
        _square_witness(witness, root)
    else:
        root = np.array([[0., 0.], [1., 0.], [0., 1.]])
        _tri_coefficients(collect, root, 1.)
        _tri_witness(witness, root)
    return tuple(rows), rows, tuple(witnesses)


def _batch_root_values(xyz: np.ndarray, fam: ElementFamily, normals: np.ndarray):
    """Batch only the outer element loop of the same tiny vector products.

    Each gradient is still a strided (nnode,) vector and each coordinate core
    is the same C-contiguous (nnode, 3) matrix as scalar certification. No
    sample/node dimensions are merged into a larger BLAS matrix. Normal dots
    use the original scalar np.dot; the Python scalar cross is reproduced by
    separate multiply/subtract ufuncs, without a fused reduction.
    """
    started = perf_counter()
    keys, _, _ = _batch_root_samples(fam)
    values = np.empty((len(keys), len(xyz)), dtype=np.float64)
    for row, key in enumerate(keys):
        g = _certification_gradients(fam, *key)
        dx = g[:, 0] @ xyz
        dy = g[:, 1] @ xyz
        cross = np.column_stack((
            dx[:, 1]*dy[:, 2] - dx[:, 2]*dy[:, 1],
            dx[:, 2]*dy[:, 0] - dx[:, 0]*dy[:, 2],
            dx[:, 0]*dy[:, 1] - dx[:, 1]*dy[:, 0],
        ))
        values[row] = [float(np.dot(c, n)) for c, n in zip(cross, normals)]
    record_quad_stage("strict_validation_batched_root_products", perf_counter() - started)
    return values


def _root_report_from_batch(values, fam, scale2, tol):
    _, indices, witnesses = _batch_root_samples(fam)

    def jfun(a, b):
        return float(values[indices[(a, b, copysign(1., a), copysign(1., b))]])

    if fam is ElementFamily.Q8:
        coeff, guard = _square_coefficients(jfun, (-1., 1., -1., 1.), scale2)
        method = "bernstein-square-bicubic"
    else:
        coeff, guard = _tri_coefficients(
            jfun, np.array([[0., 0.], [1., 0.], [0., 1.]]), scale2,
        )
        method = "bernstein-triangle-degree2"
    lower = float(np.min(coeff) - guard)
    upper = float(np.max(coeff) + guard)
    direct, param = float("inf"), None
    for row, point in witnesses:
        value = float(values[row])
        if value < direct:
            direct, param = value, point
    if direct <= -tol or direct == 0.:
        status = ValidityStatus.INVALID
    elif lower > tol:
        status = ValidityStatus.CERTIFIED_POSITIVE
    else:
        return None  # Subdivision and its complete metadata remain scalar.
    return ValidityReport(status, method, lower, upper, float(tol), 0, 0, direct, param)


def _certify_mapping_validity_batch(
    nodes, family, *, max_depth=6, max_subdivisions=4096, cancellation_check=None,
):
    """Yield scalar-identical reports, retaining at most one 256-element chunk.

    This private final-gate route admits only native packed three-coordinate
    Q8/T6 arrays, default NumPy error handling, and no callbacks. Every other
    route calls the public scalar function in consumption order. Preparation
    failures are deferred to that same scalar order, so a later bad element
    cannot replace the caller's earlier invalid-certificate failure.
    """
    fam = _family(family)
    options = dict(max_depth=max_depth, max_subdivisions=max_subdivisions,
                   cancellation_check=cancellation_check)
    admitted = (
        certify_mapping_validity is _BATCH_SCALAR_CERTIFIER
        and _batch_root_functions_unchanged()
        and fam in (ElementFamily.Q8, ElementFamily.T6)
        and type(nodes) is np.ndarray and nodes.dtype == np.dtype(np.float64)
        and nodes.ndim == 3 and nodes.shape[1:] == (_NODE_COUNT[fam], 3)
        and nodes.flags.c_contiguous and cancellation_check is None
        and type(max_depth) is int and max_depth >= 0
        and type(max_subdivisions) is int and max_subdivisions >= 0
        and np.geterr() == _BATCH_NUMPY_ERRORS
    )
    if not admitted:
        for row in nodes:
            yield certify_mapping_validity(row, fam, **options)
        return
    for start in range(0, len(nodes), _CERTIFICATION_BATCH_SIZE):
        if (np.geterr() != _BATCH_NUMPY_ERRORS
                or certify_mapping_validity is not _BATCH_SCALAR_CERTIFIER
                or not _batch_root_functions_unchanged()):
            for row in nodes[start:]:
                yield certify_mapping_validity(row, fam, **options)
            return
        source = nodes[start:start + _CERTIFICATION_BATCH_SIZE]
        xyz = source.copy(order="C")
        safe = np.isfinite(xyz).all() and np.max(np.abs(xyz), initial=0.) <= 2.**100
        if safe:
            try:
                # Keep scalar normal, norm, diameter and tolerance reductions.
                normals = np.array([_corner_reference_normal(row, fam) for row in xyz])
                scales = [_scale_tolerance(row, fam) for row in xyz]
            except ValueError:
                safe = False
        if not safe:
            for row in source:
                yield certify_mapping_validity(row, fam, **options)
            continue
        values = _batch_root_values(xyz, fam, normals)
        for offset in range(len(source)):
            row = nodes[start + offset]
            # A caller may mutate an alias or error policy between yields.
            # Neither coordinates nor certificates are retained across calls.
            if (row.tobytes() != xyz[offset].tobytes()
                    or np.geterr() != _BATCH_NUMPY_ERRORS
                    or certify_mapping_validity is not _BATCH_SCALAR_CERTIFIER
                    or not _batch_root_functions_unchanged()):
                report = None
            else:
                report = _root_report_from_batch(values[:, offset], fam, *scales[offset])
            if report is None:
                report = certify_mapping_validity(row, fam, **options)
            yield report


__all__ = [
    "ElementFamily", "ValidityStatus", "HighOrderCertificationCancelled", "MappingEvaluation", "ValidityReport",
    "GeometryErrorReport", "NormalErrorReport", "shape_values", "shape_gradients",
    "evaluate_mapping", "physical_area", "geometry_error", "normal_error",
    "certify_mapping_validity", "HighOrderBoundaryMidside", "HighOrderGeometryReport",
    "HighOrderMeshCertificate",
]
