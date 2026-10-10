"""Point correspondence for sequential shell submodelling.

Implements ``docs/SUBMODEL_CORRESPONDENCE_CONTRACT.md`` for planar linear shells
(slice S3a) and for linear shells on a cylinder (slice S3b, the ``cylinder``
argument): for each submodel boundary point, find the source shell element and
natural coordinates that correspond to it, restricted to the source elements the
caller's lineage allows.  Displacements are not evaluated here; ANYsolver does
that from the returned :class:`TransferHit`.

Tolerances are separate:

* ``bounds_padding`` pads the AABB candidate query;
* ``physical_tolerance`` bounds the distance from the point to the element's own map
  (``x_p = sum N_i x_i``, the point ANYsolver evaluates);
* ``natural_tolerance`` is the slack of the inside test in natural coordinates;
* ``weight_tolerance`` bounds the disagreement of node-indexed weights between
  several hits (a shared edge or vertex).

On a cylinder the correspondence is made in the unwrapped parameter plane
``(s, z) = (R theta, axial)``: a conforming curved mesh tiles that plane, so the
same conformity rules hold, and each hit's distance to the actual target is checked
against a bound computed from the interpolation error of the cylinder map.

The planar inverse map is explicit about its outcome: a warped Q4, a
degenerate element or a non-converged inverse refuses rather than being skipped.
Several hits are admitted only when they share the same interpolating nodes
(topology; weights below ``natural_tolerance`` count as zero) and their
node-indexed weights agree within ``weight_tolerance``; the other hits are returned as ``alternatives`` so ANYsolver
can evaluate them and include their value spread in the interpolation allowance.  Coincident but disconnected sheets therefore refuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .errors import MeshError
from .mesh import Mesh
from .mesh_bvh import MeshElementBVH, shape_functions_q4, shape_functions_t3

__all__ = ["CorrespondenceError", "CylinderFrame", "TransferHit", "locate_for_transfer"]

_EPS = float(np.finfo(float).eps)
#: Relative out-of-plane warp above which a Q4 is not planar (planar slice S3a).
_WARP_LIMIT = 1.0e-9
#: Relative size below which an element (area or corner Jacobian) is degenerate.
_DEGENERATE = 1.0e-12


class CorrespondenceError(MeshError):
    """A typed, fail-closed correspondence refusal."""

    def __init__(self, code: str, message: str, *, node: Optional[int] = None, element: Optional[int] = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.node = node
        self.element = element


@dataclass(frozen=True)
class TransferHit:
    target_node: int
    source_element_id: int
    source_element_type: str
    source_node_ids: Tuple[int, ...]
    natural: Tuple[float, float]
    weights: Tuple[float, ...]
    x_p: Tuple[float, float, float]
    distance: float
    inverse_residual: float  # physical position uncertainty of x_p (parametric hits: mapped from the parameter plane)
    incident_source_faces: Tuple[int, ...]
    conforming: bool
    weight_disagreement: float
    candidates: int
    alternatives: Tuple[Tuple[int, Tuple[float, float], float], ...] = ()
    parametric: bool = False  # located in a cylinder's parameter plane (S3b)
    bound: float = 0.0  # computed bound on ``distance`` (parametric hits)
    a_priori_bound: float = 0.0  # R dtheta^2 / 8 of the hit element (parametric hits)
    float_allowance: float = 0.0  # floating-point allowance added to ``bound`` in the check


@dataclass(frozen=True)
class CylinderFrame:
    """The analytic cylinder a curved source and target share (slice S3b)."""

    origin: Tuple[float, float, float]
    axis: Tuple[float, float, float]
    radius: float
    surface_tolerance: float

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin, dtype=float).reshape(3)
        axis = np.asarray(self.axis, dtype=float).reshape(3)
        length = float(np.linalg.norm(axis))
        if not (np.all(np.isfinite(origin)) and np.all(np.isfinite(axis)) and np.isfinite(length) and length > 0.0):
            raise CorrespondenceError("CORRESPONDENCE_TOLERANCE", "the cylinder origin and axis must be finite")
        radius, tolerance = float(self.radius), float(self.surface_tolerance)
        if not (np.isfinite(radius) and radius > 0.0 and np.isfinite(tolerance) and tolerance >= 0.0):
            raise CorrespondenceError("CORRESPONDENCE_TOLERANCE", "the cylinder radius must be positive and the surface tolerance non-negative")
        object.__setattr__(self, "origin", tuple(origin.tolist()))
        object.__setattr__(self, "axis", tuple((axis / length).tolist()))
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "surface_tolerance", tolerance)

    def frame(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        axis = np.asarray(self.axis)
        helper = np.eye(3)[int(np.argmin(np.abs(axis)))]
        first = helper - (helper @ axis) * axis
        first /= np.linalg.norm(first)
        return np.asarray(self.origin), axis, first, np.cross(axis, first)

    def coordinates(self, points: np.ndarray) -> np.ndarray:
        """``(r, theta, z)`` of each row of ``points``."""

        origin, axis, first, second = self.frame()
        offset = np.atleast_2d(points) - origin
        z = offset @ axis
        radial = offset - np.outer(z, axis)
        return np.column_stack([np.linalg.norm(radial, axis=1), np.arctan2(radial @ second, radial @ first), z])


def _plane(coordinates: np.ndarray, element_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Origin, in-plane basis (2x3) and unit normal of a planar element."""

    size = float(np.max(np.ptp(coordinates, axis=0)))
    if not np.isfinite(size) or size <= 0.0:
        raise CorrespondenceError("CORRESPONDENCE_ELEMENT_DEGENERATE", f"element {element_id} has no extent", element=element_id)
    if len(coordinates) == 4:
        normal = np.cross(coordinates[2] - coordinates[0], coordinates[3] - coordinates[1])
    else:
        normal = np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])
    length = float(np.linalg.norm(normal))
    if length <= _DEGENERATE * size * size:
        raise CorrespondenceError("CORRESPONDENCE_ELEMENT_DEGENERATE", f"element {element_id} has zero area", element=element_id)
    normal = normal / length
    origin = coordinates.mean(axis=0)
    warp = float(np.max(np.abs((coordinates - origin) @ normal)))
    if warp > _WARP_LIMIT * size:
        raise CorrespondenceError(
            "CORRESPONDENCE_ELEMENT_NOT_PLANAR",
            f"element {element_id} is warped by {warp / size:.3e} of its size; the planar slice admits flat elements only",
            element=element_id,
        )
    first = coordinates[1] - coordinates[0]
    first = first - (first @ normal) * normal
    first /= np.linalg.norm(first)
    basis = np.vstack([first, np.cross(normal, first)])
    return origin, basis, normal, size


def _planar_inverse(kind: str, coordinates: np.ndarray, point: np.ndarray, element_id: int):
    """Natural coordinates, weights, plane projection and distance; typed refusals."""

    origin, basis, normal, size = _plane(coordinates, element_id)
    local = (coordinates - origin) @ basis.T  # (n, 2)
    target = (point - origin) @ basis.T
    natural, weights, residual = _inverse_2d(kind, local, target, size, element_id)
    mapped = weights @ coordinates  # the element's own map, as ANYsolver evaluates it
    distance = float(np.linalg.norm(point - mapped))
    return natural, weights, mapped, distance, residual


def _inverse_2d(kind: str, local: np.ndarray, target: np.ndarray, size: float, element_id: int):
    """Natural coordinates, weights and residual of ``target`` in a 2D element."""

    if kind == "T3":
        matrix = np.column_stack([local[1] - local[0], local[2] - local[0]])
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) <= _DEGENERATE * size * size:
            raise CorrespondenceError("CORRESPONDENCE_ELEMENT_DEGENERATE", f"element {element_id} is degenerate", element=element_id)
        natural = np.linalg.solve(matrix, target - local[0])
        weights = shape_functions_t3(*natural)
    else:
        corners = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], dtype=float)
        signs = []
        for xi, eta in corners:
            d_xi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
            d_eta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
            signs.append(float(np.linalg.det(np.column_stack([d_xi @ local, d_eta @ local]))))
        signs = np.array(signs)
        if np.any(np.abs(signs) <= _DEGENERATE * size * size) or not (np.all(signs > 0) or np.all(signs < 0)):
            raise CorrespondenceError("CORRESPONDENCE_ELEMENT_DEGENERATE", f"element {element_id} is degenerate or folded", element=element_id)
        natural = np.zeros(2)
        converged = False
        for _ in range(50):
            xi, eta = natural
            weights = shape_functions_q4(xi, eta)
            d_xi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
            d_eta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
            jacobian = np.column_stack([d_xi @ local, d_eta @ local])
            step = np.linalg.solve(jacobian, target - weights @ local)
            natural = natural + step
            if float(np.max(np.abs(step))) <= 64.0 * _EPS * max(1.0, float(np.max(np.abs(natural)))):
                converged = True
                break
        if not converged or not np.all(np.isfinite(natural)):
            raise CorrespondenceError("CORRESPONDENCE_NOT_CONVERGED", f"inverse map of element {element_id} did not converge", element=element_id)
        weights = shape_functions_q4(*natural)
    in_plane = weights @ local
    residual = float(np.linalg.norm(in_plane - target))
    return natural, weights, residual


#: Largest admitted angular span of one element on a cylinder (unambiguous unwrapping).
_MAX_SPAN = 0.5 * np.pi


def _element_map(weights: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """``x_p = sum N_i x_i``: the element's own map, as ANYsolver evaluates it."""

    return weights @ coordinates


def _expanded_domain_image(kind: str, parameters: np.ndarray, tolerance: float) -> np.ndarray:
    """Points whose convex hull contains the image of the natural domain the inside test admits."""

    t = float(tolerance)
    if kind == "T3":
        reference = np.array([(-t, -t), (1.0 + 2.0 * t, -t), (-t, 1.0 + 2.0 * t)])
        return np.array([shape_functions_t3(*xi) @ parameters for xi in reference])
    corners = (1.0 + t) * np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], dtype=float)
    return np.array([shape_functions_q4(*xi) @ parameters for xi in corners])


class _CylinderIndex:
    """Candidate elements of one cylinder mapped once to the parameter plane."""

    def __init__(self, mesh: Mesh, cylinder: CylinderFrame, element_ids: Sequence[int], natural_tolerance: float):
        self.cylinder = cylinder
        radius, tolerance = cylinder.radius, cylinder.surface_tolerance
        self.ids = np.asarray(sorted(int(e) for e in element_ids), dtype=int)
        self.data: Dict[int, Tuple[str, Tuple[int, ...], np.ndarray, np.ndarray, np.ndarray, float]] = {}
        boxes, means = [], []
        for element_id in self.ids.tolist():
            node_ids = tuple(int(n) for n in mesh.shells[element_id])
            kind = {4: "Q4", 3: "T3"}.get(len(node_ids))
            if kind is None:
                raise CorrespondenceError(
                    "CORRESPONDENCE_ELEMENT_NOT_ADMITTED",
                    f"source element {element_id} has {len(node_ids)} nodes; only Q4 and T3 are admitted", element=element_id,
                )
            coordinates = np.array([mesh.nodes[n] for n in node_ids], dtype=float)
            transformed = cylinder.coordinates(coordinates)
            if not (np.all(np.isfinite(coordinates)) and np.all(np.isfinite(transformed))):
                raise CorrespondenceError("CORRESPONDENCE_NONFINITE", f"element {element_id} has non-finite coordinates",
                                          element=element_id)
            r, theta, z = transformed.T
            deviation = np.abs(r - radius)
            if np.any(deviation > tolerance):
                raise CorrespondenceError(
                    "CORRESPONDENCE_OFF_SURFACE",
                    f"element {element_id} has nodes {float(np.max(deviation)):.3e} m off the cylinder (tolerance {tolerance:.3e})",
                    element=element_id,
                )
            unwrapped = theta[0] + np.angle(np.exp(1j * (theta - theta[0])))
            span = float(np.ptp(unwrapped))
            if span >= _MAX_SPAN:
                raise CorrespondenceError(
                    "CORRESPONDENCE_FACET_TOO_COARSE",
                    f"element {element_id} spans {np.degrees(span):.1f} degrees of the cylinder", element=element_id,
                )
            parameters = np.column_stack([radius * unwrapped, z])
            size = float(np.max(np.ptp(parameters, axis=0)))
            expanded = _expanded_domain_image(kind, parameters, natural_tolerance)
            pad = 64.0 * _EPS * (float(np.max(np.abs(parameters))) + size)
            boxes.append([expanded[:, 0].min() - pad, expanded[:, 0].max() + pad,
                          expanded[:, 1].min() - pad, expanded[:, 1].max() + pad])
            means.append(float(np.mean(unwrapped)))
            self.data[element_id] = (kind, node_ids, coordinates, parameters, deviation, size)
        self.boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        self.means = np.asarray(means, dtype=float)

    def nearby(self, theta: float, z: float, allowed: Set[int]) -> List[Tuple[int, float]]:
        """``(element, unwrapped point angle)`` for the elements whose padded box holds the point."""

        if not len(self.ids):
            return []
        branch = theta + 2.0 * np.pi * np.round((self.means - theta) / (2.0 * np.pi))
        s = self.cylinder.radius * branch
        inside = ((self.boxes[:, 0] <= s) & (s <= self.boxes[:, 1])
                  & (self.boxes[:, 2] <= z) & (z <= self.boxes[:, 3]))
        return [(int(e), float(a)) for e, a, keep in zip(self.ids, branch, inside) if keep and int(e) in allowed]


def _cylinder_hit(index: _CylinderIndex, element_id: int, point: np.ndarray, deviation_f: float,
                  angle: float, z: float, natural_tolerance: float):
    """Parameter-plane inverse of one candidate, with the computed facet bound."""

    kind, node_ids, coordinates, parameters, deviation, size = index.data[element_id]
    radius = index.cylinder.radius
    target = np.array([radius * angle, z])
    natural, weights, _residual = _inverse_2d(kind, parameters, target, size, element_id)
    if not _inside(kind, natural, natural_tolerance):
        return None
    if abs(float(np.sum(weights)) - 1.0) > 64.0 * _EPS * len(weights):
        raise CorrespondenceError("CORRESPONDENCE_FACET_BOUND", f"element {element_id}: weights do not sum to one",
                                  element=element_id)
    mapped = _element_map(weights, coordinates)
    distance = float(np.linalg.norm(point - mapped))
    reproduced = weights @ parameters
    residual = float(np.linalg.norm(reproduced - target))  # |p - p'|, from the weights actually used
    theta = parameters[:, 0] / radius
    theta_hit = reproduced[0] / radius
    absolute = np.abs(weights)
    bound = (deviation_f + residual + 0.5 * radius * float(absolute @ (theta - theta_hit) ** 2)
             + float(absolute @ deviation))
    a_priori = radius * float(np.ptp(theta)) ** 2 / 8.0
    if not all(np.isfinite(v) for v in (distance, bound, residual)) or not np.all(np.isfinite(mapped)):
        raise CorrespondenceError("CORRESPONDENCE_NONFINITE", f"element {element_id}: a non-finite hit measurement",
                                  element=element_id)
    origin = np.asarray(index.cylinder.origin)
    # Positions are stored and combined in global coordinates, so the rounding scale is global.
    allowance = 64.0 * _EPS * (float(np.linalg.norm(point)) + float(np.max(np.linalg.norm(coordinates, axis=1)))
                               + float(np.linalg.norm(origin)) + radius)
    if distance > bound + allowance:
        raise CorrespondenceError(
            "CORRESPONDENCE_FACET_BOUND",
            f"element {element_id}: distance {distance:.6e} exceeds the computed facet bound {bound:.6e} "
            f"plus the floating-point allowance {allowance:.3e}", element=element_id,
        )
    # S1 reads ``inverse_residual`` as a physical position uncertainty of the source
    # point, so the parameter-plane residual is carried to physical space with a
    # conservative stretch bound (see _stretch_bound).
    stretch = _stretch_bound(kind, parameters, coordinates, natural_tolerance, element_id)
    physical_residual = stretch * residual
    if residual > natural_tolerance * _parametric_floor(kind, parameters, natural_tolerance):
        raise CorrespondenceError("CORRESPONDENCE_NOT_CONVERGED",
                                  f"inverse map of element {element_id} leaves a residual {residual:.3e}", element=element_id)
    if not np.isfinite(physical_residual):
        raise CorrespondenceError("CORRESPONDENCE_NONFINITE", f"element {element_id}: non-finite position uncertainty",
                                  element=element_id)
    return natural, weights, mapped, distance, physical_residual, bound, a_priori, allowance


def _jacobian_parts(kind: str, values: np.ndarray):
    """``J(xi, eta) = J0 + eta A + xi B`` (columns d/dxi, d/deta) of an element map.

    The Jacobian of a bilinear Q4 map is affine in the natural coordinates (each
    column depends on one coordinate only); a T3 map's is constant.
    """

    if kind == "T3":
        j0 = np.column_stack([values[1] - values[0], values[2] - values[0]])
        zero = np.zeros(values.shape[1])
        return j0, zero, zero
    a, b, c, d = values
    j0 = 0.25 * np.column_stack([-a + b + c - d, -a - b + c + d])
    first_by_eta = 0.25 * (a - b + c - d)  # d(column xi)/d eta
    second_by_xi = 0.25 * (a - b + c - d)  # d(column eta)/d xi
    return j0, first_by_eta, second_by_xi


def _domain_corners(kind: str, tolerance: float) -> np.ndarray:
    t = 2.0 * float(tolerance)
    if kind == "T3":
        return np.array([(-t, -t), (1.0 + 2.0 * t, -t), (-t, 1.0 + 2.0 * t)])
    return (1.0 + t) * np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], dtype=float)


def _parametric_floor(kind: str, parameters: np.ndarray, tolerance: float) -> float:
    """Lower bound of the smallest singular value of the parametric Jacobian over the domain."""

    j0, a, b = _jacobian_parts(kind, parameters)
    reach = float(np.max(np.abs(_domain_corners(kind, tolerance))))
    return float(np.linalg.svd(j0, compute_uv=False)[-1]) - reach * (float(np.linalg.norm(a)) + float(np.linalg.norm(b)))


def _stretch_bound(kind: str, parameters: np.ndarray, coordinates: np.ndarray, tolerance: float, element_id: int) -> float:
    """Bound of |physical position error| per unit parameter-plane residual.

    With an affine Jacobian, ``F(xi_1) - F(xi_2) = J_F(mid) (xi_1 - xi_2)`` exactly for
    both the parametric map Psi and the physical map Phi.  Hence
    ``|Phi(xi_hit) - Phi(xi_true)| <= max |J_Phi| |d xi|`` and
    ``|d xi| <= residual / min sigma_min(J_Psi)`` over the (doubly slack-expanded)
    domain: ``max |J_Phi|`` is attained at a corner (the norm is convex and J affine)
    and ``sigma_min`` is bounded below by Weyl's inequality about the centre.
    """

    j0, a, b = _jacobian_parts(kind, coordinates)
    largest = max(float(np.linalg.norm(j0 + xi_eta[1] * a[:, None] * np.array([1.0, 0.0])
                                       + xi_eta[0] * b[:, None] * np.array([0.0, 1.0]), 2))
                  for xi_eta in _domain_corners(kind, tolerance))
    floor = _parametric_floor(kind, parameters, tolerance)
    if not (np.isfinite(largest) and np.isfinite(floor)) or floor <= 0.0:
        raise CorrespondenceError("CORRESPONDENCE_ELEMENT_DEGENERATE",
                                  f"element {element_id} is too distorted in the parameter plane for a position bound",
                                  element=element_id)
    return largest / floor


def _inside(kind: str, natural: Sequence[float], tolerance: float) -> bool:
    first, second = float(natural[0]), float(natural[1])
    if kind == "Q4":
        return abs(first) <= 1.0 + tolerance and abs(second) <= 1.0 + tolerance
    return first >= -tolerance and second >= -tolerance and first + second <= 1.0 + tolerance


def locate_for_transfer(
    source_mesh: Mesh,
    points: Mapping[int, Sequence[float]],
    *,
    candidate_elements: Mapping[int, Iterable[int]],
    physical_tolerance: Optional[float] = None,
    natural_tolerance: float = 1.0e-9,
    weight_tolerance: float = 1.0e-9,
    bounds_padding: Optional[float] = None,
    require_faces: bool = True,
    cylinder: Optional[CylinderFrame] = None,
) -> Dict[int, TransferHit]:
    """Locate each target point in its lineage-allowed source elements.

    Returns a dictionary keyed by target node, in ascending node order.  The
    planar default ``physical_tolerance`` is ``1e-9`` times the source mesh
    extent.  With ``require_faces`` (default) every hit element must belong to a
    source face, so incident-face data for the junction rule is complete.

    With ``cylinder`` every point and candidate element must lie on that
    cylinder within its ``surface_tolerance``; points are located in the
    unwrapped parameter plane over all lineage-allowed candidates (no 3D search,
    and ``physical_tolerance`` and ``bounds_padding`` are not used), and each
    hit's distance is checked against its computed facet bound.
    """

    if not points:
        return {}
    for name, value in (("natural_tolerance", natural_tolerance), ("weight_tolerance", weight_tolerance)):
        if not np.isfinite(value) or value < 0.0:
            raise CorrespondenceError("CORRESPONDENCE_TOLERANCE", f"{name} must be finite and non-negative")
    # Each target's candidates are read once (the interface admits one-shot iterables).
    allowed_of: Dict[int, Set[int]] = {}
    for target in points:
        if int(target) in candidate_elements:
            allowed_of[int(target)] = {int(e) for e in candidate_elements[int(target)]}
    if cylinder is None:
        # Planar route only: its distance check and its 3D search padding.
        positions = np.array([np.asarray(xyz, dtype=float) for xyz in source_mesh.nodes.values()])
        extent = float(np.linalg.norm(np.ptp(positions, axis=0))) if positions.size else 1.0
        physical = 1.0e-9 * max(extent, 1.0e-300) if physical_tolerance is None else float(physical_tolerance)
        padding = physical if bounds_padding is None else float(bounds_padding)
        for name, value in (("physical_tolerance", physical), ("bounds_padding", padding)):
            if not np.isfinite(value) or value < 0.0:
                raise CorrespondenceError("CORRESPONDENCE_TOLERANCE", f"{name} must be finite and non-negative")
        if padding < physical:
            raise CorrespondenceError("CORRESPONDENCE_TOLERANCE", "bounds_padding must be at least physical_tolerance")

    shells = source_mesh.shells
    faces_of: Dict[int, Set[int]] = {}
    for face_id, element_ids in source_mesh.elements_of_face.items():
        for element_id in element_ids:
            faces_of.setdefault(int(element_id), set()).add(int(face_id))
    if cylinder is None:
        bvh = MeshElementBVH(source_mesh, tolerance=padding)
        index = None
    else:
        bvh = None
        missing = sorted(int(n) for n in points if int(n) not in allowed_of)
        if missing:
            raise CorrespondenceError("CORRESPONDENCE_NO_LINEAGE", f"no candidate source elements for node {missing[0]}", node=missing[0])
        every = set().union(*allowed_of.values())
        unknown = sorted(e for e in every if e not in shells)
        if unknown:
            raise CorrespondenceError("CORRESPONDENCE_ELEMENT_NOT_ADMITTED", f"source element {unknown[0]} is not a shell", element=unknown[0])
        index = _CylinderIndex(source_mesh, cylinder, sorted(every), natural_tolerance)

    hits: Dict[int, TransferHit] = {}
    for target in sorted(int(n) for n in points):
        if target not in allowed_of:
            raise CorrespondenceError("CORRESPONDENCE_NO_LINEAGE", f"no candidate source elements for node {target}", node=target)
        allowed = allowed_of[target]
        point = np.asarray(points[target], dtype=float).reshape(3)
        if not np.all(np.isfinite(point)):
            raise CorrespondenceError("CORRESPONDENCE_POINT", f"node {target} has a non-finite position", node=target)
        found = []
        if index is not None:
            r_f, theta_f, z_f = index.cylinder.coordinates(point)[0]
            if not np.all(np.isfinite((r_f, theta_f, z_f))):
                raise CorrespondenceError("CORRESPONDENCE_NONFINITE", f"node {target} maps to non-finite cylinder coordinates",
                                          node=target)
            deviation_f = abs(float(r_f) - cylinder.radius)
            if deviation_f > cylinder.surface_tolerance:
                raise CorrespondenceError(
                    "CORRESPONDENCE_OFF_SURFACE",
                    f"node {target} is {deviation_f:.3e} m off the cylinder (tolerance {cylinder.surface_tolerance:.3e})",
                    node=target,
                )
            nearby = index.nearby(float(theta_f), float(z_f), allowed)
            for element_id, angle in nearby:
                located = _cylinder_hit(index, element_id, point, deviation_f, angle, float(z_f), natural_tolerance)
                if located is None:
                    continue
                natural, weights, projected, distance, residual, bound, a_priori, allowance = located
                kind, node_ids = index.data[element_id][0], index.data[element_id][1]
                found.append((distance, element_id, kind, node_ids, natural, weights, projected, residual, bound,
                              a_priori, allowance))
            if not found:
                raise CorrespondenceError(
                    "CORRESPONDENCE_NOT_LOCATED",
                    f"node {target} at {point.tolist()} lies in no allowed source element's parameter domain "
                    f"({len(nearby)} candidates near it)",
                    node=target,
                )
        if index is None:
            nearby = sorted(int(e) for e in bvh.candidates(point, padding) if int(e) in allowed)
        for element_id in (nearby if index is None else ()):
            node_ids = tuple(int(n) for n in shells[element_id])
            kind = {4: "Q4", 3: "T3"}.get(len(node_ids))
            if kind is None:
                raise CorrespondenceError(
                    "CORRESPONDENCE_ELEMENT_NOT_ADMITTED",
                    f"source element {element_id} has {len(node_ids)} nodes; only Q4 and T3 are admitted",
                    node=target, element=element_id,
                )
            coordinates = np.array([source_mesh.nodes[n] for n in node_ids], dtype=float)
            natural, weights, projected, distance, residual = _planar_inverse(kind, coordinates, point, element_id)
            if distance > physical or not _inside(kind, natural, natural_tolerance):
                continue
            found.append((distance, element_id, kind, node_ids, natural, weights, projected, residual, 0.0, 0.0, 0.0))
        if not found:
            raise CorrespondenceError(
                "CORRESPONDENCE_NOT_LOCATED",
                f"node {target} at {point.tolist()} is not within {physical:.3e} m of an allowed source element "
                f"({len(nearby)} candidates near it)",
                node=target,
            )
        found.sort(key=lambda item: (item[0], item[1]))
        indexed = []
        for item in found:
            weights_by_node: Dict[int, float] = {}
            for node, weight in zip(item[3], item[5]):
                weights_by_node[node] = weights_by_node.get(node, 0.0) + float(weight)
            if not any(abs(w) > natural_tolerance for w in weights_by_node.values()):
                raise CorrespondenceError("CORRESPONDENCE_EMPTY_SUPPORT", f"node {target} has no interpolating node", node=target)
            indexed.append(weights_by_node)
        supports = {frozenset(n for n, w in weights.items() if abs(w) > natural_tolerance) for weights in indexed}
        if len(supports) != 1:
            raise CorrespondenceError(
                "CORRESPONDENCE_NONCONFORMING",
                f"node {target} lies on {len(found)} allowed elements that do not share their interpolating nodes "
                "(coincident or disconnected sheets, or a non-conforming edge)",
                node=target,
            )
        union = set().union(*indexed)
        disagreement = max(
            (abs(a.get(n, 0.0) - b.get(n, 0.0)) for a in indexed for b in indexed for n in union), default=0.0
        )
        if disagreement > weight_tolerance:
            raise CorrespondenceError(
                "CORRESPONDENCE_NONCONFORMING",
                f"node {target} lies on {len(found)} allowed elements whose node weights disagree by {disagreement:.3e} "
                "(coincident or disconnected sheets, or a non-conforming edge)",
                node=target,
            )
        hit_faces: Set[int] = set()
        for item in found:
            membership = faces_of.get(item[1])
            if not membership:
                if require_faces:
                    raise CorrespondenceError(
                        "CORRESPONDENCE_FACE_MISSING", f"source element {item[1]} belongs to no source face",
                        node=target, element=item[1],
                    )
                continue
            hit_faces |= membership
        distance, element_id, kind, node_ids, natural, weights, projected, residual, bound, a_priori, allowance = found[0]
        hits[target] = TransferHit(
            target_node=target,
            source_element_id=element_id,
            source_element_type=kind,
            source_node_ids=node_ids,
            natural=(float(natural[0]), float(natural[1])),
            weights=tuple(float(w) for w in weights),
            x_p=tuple(float(v) for v in projected),
            distance=float(distance),
            inverse_residual=float(residual),
            incident_source_faces=tuple(sorted(hit_faces)),
            conforming=True,
            weight_disagreement=float(disagreement),
            candidates=len(nearby),
            alternatives=tuple(
                (int(item[1]), (float(item[4][0]), float(item[4][1])), float(item[7])) for item in found[1:]
            ),
            parametric=index is not None,
            bound=float(bound),
            a_priori_bound=float(a_priori),
            float_allowance=float(allowance),
        )
    return hits
