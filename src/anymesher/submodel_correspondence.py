"""Point correspondence for sequential shell submodelling.

Implements ``docs/SUBMODEL_CORRESPONDENCE_CONTRACT.md`` for planar linear shells
(slice S3a): for each submodel boundary point, find the source shell element and
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

The inverse map is planar and explicit about its outcome: a warped Q4, a
degenerate element or a non-converged inverse refuses rather than being skipped.
Several hits are admitted only when they share the same interpolating nodes
(topology; weights below ``natural_tolerance`` count as zero) and their
node-indexed weights agree within ``weight_tolerance``; the other hits are returned as ``alternatives`` so ANYsolver
can evaluate them and include their value spread in the interpolation allowance.  Coincident but disconnected sheets therefore refuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .errors import MeshError
from .mesh import Mesh
from .mesh_bvh import MeshElementBVH, shape_functions_q4, shape_functions_t3

__all__ = ["CorrespondenceError", "TransferHit", "locate_for_transfer"]

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
    inverse_residual: float
    incident_source_faces: Tuple[int, ...]
    conforming: bool
    weight_disagreement: float
    candidates: int
    alternatives: Tuple[Tuple[int, Tuple[float, float], float], ...] = ()


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
    mapped = weights @ coordinates  # the element's own map, as ANYsolver evaluates it
    distance = float(np.linalg.norm(point - mapped))
    return natural, weights, mapped, distance, residual


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
) -> Dict[int, TransferHit]:
    """Locate each target point in its lineage-allowed source elements.

    Returns a dictionary keyed by target node, in ascending node order.  The
    planar default ``physical_tolerance`` is ``1e-9`` times the source mesh
    extent.  With ``require_faces`` (default) every hit element must belong to a
    source face, so incident-face data for the junction rule is complete.
    """

    if not points:
        return {}
    positions = np.array([np.asarray(xyz, dtype=float) for xyz in source_mesh.nodes.values()])
    extent = float(np.linalg.norm(np.ptp(positions, axis=0))) if positions.size else 1.0
    physical = 1.0e-9 * max(extent, 1.0e-300) if physical_tolerance is None else float(physical_tolerance)
    padding = physical if bounds_padding is None else float(bounds_padding)
    for name, value in (("physical_tolerance", physical), ("bounds_padding", padding),
                        ("natural_tolerance", natural_tolerance), ("weight_tolerance", weight_tolerance)):
        if not np.isfinite(value) or value < 0.0:
            raise CorrespondenceError("CORRESPONDENCE_TOLERANCE", f"{name} must be finite and non-negative")
    if padding < physical:
        raise CorrespondenceError("CORRESPONDENCE_TOLERANCE", "bounds_padding must be at least physical_tolerance")

    shells = source_mesh.shells
    faces_of: Dict[int, Set[int]] = {}
    for face_id, element_ids in source_mesh.elements_of_face.items():
        for element_id in element_ids:
            faces_of.setdefault(int(element_id), set()).add(int(face_id))
    bvh = MeshElementBVH(source_mesh, tolerance=padding)

    hits: Dict[int, TransferHit] = {}
    for target in sorted(int(n) for n in points):
        if target not in candidate_elements:
            raise CorrespondenceError("CORRESPONDENCE_NO_LINEAGE", f"no candidate source elements for node {target}", node=target)
        allowed = {int(e) for e in candidate_elements[target]}
        point = np.asarray(points[target], dtype=float).reshape(3)
        if not np.all(np.isfinite(point)):
            raise CorrespondenceError("CORRESPONDENCE_POINT", f"node {target} has a non-finite position", node=target)
        nearby = sorted(int(e) for e in bvh.candidates(point, padding) if int(e) in allowed)
        found = []
        for element_id in nearby:
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
            found.append((distance, element_id, kind, node_ids, natural, weights, projected, residual))
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
        distance, element_id, kind, node_ids, natural, weights, projected, residual = found[0]
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
        )
    return hits
