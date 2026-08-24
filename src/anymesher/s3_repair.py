"""Bounded deterministic repair for the opt-in qualified S3 contract.

Repair is deliberately separate from admission.  Legacy meshes are never
changed merely because they contain triangles, and exhausting the bounded
operations raises a typed error instead of selecting a legacy formulation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from operator import index
from typing import Mapping, Sequence

import numpy as np

from .mesh import Mesh
from .s3_quality import (
    DEFAULT_S3_QUALITY_POLICY,
    S3AdmissionReport,
    S3QualityError,
    S3QualityPolicy,
    evaluate_s3_admission,
)

__all__ = [
    "DEFAULT_S3_REPAIR_POLICY",
    "S3_REPAIR_CONTRACT_ID",
    "S3RepairAttempt",
    "S3RepairError",
    "S3RepairPolicy",
    "S3RepairResult",
    "repair_s3_admission",
]


S3_REPAIR_CONTRACT_ID = "ANYMESHER_QUALIFIED_S3_REPAIR_V1"


def _bounded_integer(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        made = int(index(value))
    except TypeError as error:
        raise ValueError(f"{label} must be a non-negative integer") from error
    if made < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return made


@dataclass(frozen=True)
class S3RepairPolicy:
    """Hard limits for one qualified-S3 repair request."""

    maximum_winding_repairs: int = 4096
    maximum_edge_flips: int = 64
    maximum_edge_flip_attempts: int = 256
    maximum_refinement_splits: int = 4
    maximum_refinement_attempts: int = 64
    maximum_added_nodes: int = 4
    maximum_added_elements: int = 8
    minimum_flip_owner_alignment: float = 1.0 - 1.0e-10

    def __post_init__(self) -> None:
        for name in (
            "maximum_winding_repairs",
            "maximum_edge_flips",
            "maximum_edge_flip_attempts",
            "maximum_refinement_splits",
            "maximum_refinement_attempts",
            "maximum_added_nodes",
            "maximum_added_elements",
        ):
            object.__setattr__(self, name, _bounded_integer(getattr(self, name), name))
        alignment = float(self.minimum_flip_owner_alignment)
        if not np.isfinite(alignment) or not 0.0 < alignment <= 1.0:
            raise ValueError("minimum_flip_owner_alignment must lie in (0, 1]")
        object.__setattr__(self, "minimum_flip_owner_alignment", alignment)


DEFAULT_S3_REPAIR_POLICY = S3RepairPolicy()


@dataclass(frozen=True)
class S3RepairAttempt:
    """One stable, audit-friendly repair decision."""

    sequence: int
    action: str
    status: str
    element_ids: tuple[int, ...]
    edge: tuple[int, int] | tuple[()] = ()
    detail: str = ""


@dataclass(frozen=True)
class S3RepairResult:
    """An explicitly qualified repaired mesh and its complete audit."""

    mesh: Mesh
    element_ids: tuple[int, ...]
    owner_normals: tuple[tuple[int, tuple[float, float, float]], ...]
    admission: S3AdmissionReport
    attempts: tuple[S3RepairAttempt, ...]
    winding_repairs: int
    edge_flips: int
    edge_flip_attempts: int
    refinement_splits: int
    refinement_attempts: int
    added_nodes: int
    added_elements: int
    contract_id: str = S3_REPAIR_CONTRACT_ID

    def owner_normal_map(self) -> dict[int, tuple[float, float, float]]:
        """Return a fresh mapping suitable for admission and nodal normals."""

        return dict(self.owner_normals)


class S3RepairError(S3QualityError):
    """Bounded repair could not produce a qualified S3 mesh."""

    def __init__(
        self,
        message: str,
        *,
        attempts: Sequence[S3RepairAttempt],
        admission: S3AdmissionReport | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = tuple(attempts)
        self.admission = admission


def _unit_owner(value: Sequence[float], element_id: int) -> np.ndarray:
    made = np.asarray(value, dtype=float)
    if made.shape != (3,) or not np.all(np.isfinite(made)):
        raise ValueError(
            f"triangle {element_id} owner normal must be a finite three-vector"
        )
    length = float(np.linalg.norm(made))
    if length <= 0.0:
        raise ValueError(f"triangle {element_id} owner normal must be nonzero")
    return made / length


def _edge(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first < second else (second, first)


def _triangle_edges(connectivity: Sequence[int]) -> tuple[tuple[int, int], ...]:
    made = tuple(int(value) for value in connectivity)
    return tuple(_edge(made[index], made[(index + 1) % 3]) for index in range(3))


def _shell_incidence(mesh: Mesh) -> dict[tuple[int, int], list[tuple[str, int]]]:
    result: dict[tuple[int, int], list[tuple[str, int]]] = {}
    for kind, cells in (("quad", mesh.quads), ("tri", mesh.tris)):
        for element_id, connectivity in sorted(cells.items()):
            corners = tuple(int(value) for value in connectivity[: 4 if kind == "quad" else 3])
            for position, first in enumerate(corners):
                second = corners[(position + 1) % len(corners)]
                result.setdefault(_edge(first, second), []).append((kind, int(element_id)))
    return result


def _normal_tuple(value: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(component) for component in value)  # type: ignore[return-value]


def _oriented_triangle(
    mesh: Mesh,
    connectivity: Sequence[int],
    owner: np.ndarray,
) -> tuple[int, int, int] | None:
    made = tuple(int(value) for value in connectivity)
    corners = np.asarray([mesh.nodes[node] for node in made], dtype=float)
    cross = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    signed = float(np.dot(cross, owner))
    if not np.isfinite(signed) or signed == 0.0:
        return None
    return made if signed > 0.0 else (made[0], made[2], made[1])


def _membership_signature(mesh: Mesh, element_id: int) -> tuple[object, ...]:
    def owners(mapping: Mapping[int, Sequence[int]]) -> tuple[int, ...]:
        return tuple(
            sorted(int(owner) for owner, values in mapping.items() if element_id in values)
        )

    return (
        owners(mesh.elements_of_face),
        owners(mesh.elements_of_sheet),
        owners(mesh.elements_of_member),
        owners(mesh.elements_of_edge),
        mesh.activity.get(element_id),
    )


def _edge_is_geometry_bound(mesh: Mesh, edge: tuple[int, int]) -> bool:
    for sequence in mesh.nodes_of_edge.values():
        values = tuple(int(value) for value in sequence)
        if any(_edge(first, second) == edge for first, second in zip(values, values[1:])):
            return True
    return False


def _report_score(report: S3AdmissionReport) -> int:
    return len(report.topology_violations) + sum(
        len(element.violations) for element in report.elements
    )


def _record(
    attempts: list[S3RepairAttempt],
    action: str,
    status: str,
    element_ids: Sequence[int] = (),
    edge: Sequence[int] = (),
    detail: str = "",
) -> None:
    made_edge: tuple[int, int] | tuple[()]
    made_edge = () if not edge else _edge(int(edge[0]), int(edge[1]))
    attempts.append(
        S3RepairAttempt(
            sequence=len(attempts) + 1,
            action=action,
            status=status,
            element_ids=tuple(sorted(int(value) for value in element_ids)),
            edge=made_edge,
            detail=detail,
        )
    )


def _candidate_flip(
    mesh: Mesh,
    edge: tuple[int, int],
    element_ids: tuple[int, int],
    owners: Mapping[int, np.ndarray],
    minimum_owner_alignment: float,
) -> tuple[dict[int, tuple[int, int, int]] | None, str]:
    first_id, second_id = element_ids
    if _membership_signature(mesh, first_id) != _membership_signature(mesh, second_id):
        return None, "triangle association scopes differ"
    if _edge_is_geometry_bound(mesh, edge):
        return None, "shared edge is bound to a persistent geometry edge"
    alignment = float(np.dot(owners[first_id], owners[second_id]))
    if alignment < minimum_owner_alignment:
        return None, f"owner-normal alignment {alignment:.12g} is insufficient for a flip"

    first = tuple(int(value) for value in mesh.tris[first_id])
    second = tuple(int(value) for value in mesh.tris[second_id])
    opposite_first = next((value for value in first if value not in edge), None)
    opposite_second = next((value for value in second if value not in edge), None)
    if opposite_first is None or opposite_second is None or opposite_first == opposite_second:
        return None, "shared triangles do not form a four-node patch"
    new_edge = _edge(opposite_first, opposite_second)
    incidence = _shell_incidence(mesh)
    if incidence.get(new_edge):
        return None, f"replacement diagonal {new_edge} already exists"

    a, b = edge
    c, d = int(opposite_first), int(opposite_second)
    average_owner = owners[first_id] + owners[second_id]
    average_length = float(np.linalg.norm(average_owner))
    if average_length <= 0.0:
        return None, "owner normals cancel across the patch"
    normal = average_owner / average_length
    points = {
        node: np.asarray(mesh.nodes[node], dtype=float) for node in (a, b, c, d)
    }
    side_c = float(np.dot(np.cross(points[b] - points[a], points[c] - points[a]), normal))
    side_d = float(np.dot(np.cross(points[b] - points[a], points[d] - points[a]), normal))
    side_a = float(np.dot(np.cross(points[d] - points[c], points[a] - points[c]), normal))
    side_b = float(np.dot(np.cross(points[d] - points[c], points[b] - points[c]), normal))
    if side_c * side_d >= 0.0 or side_a * side_b >= 0.0:
        return None, "triangle pair is not a convex flippable four-node patch"

    raw_candidates = ((c, d, a), (d, c, b))
    original_centroids = tuple(
        np.mean(np.asarray([mesh.nodes[node] for node in mesh.tris[element_id]], dtype=float), axis=0)
        for element_id in element_ids
    )
    candidate_centroids = tuple(
        np.mean(np.asarray([mesh.nodes[node] for node in candidate], dtype=float), axis=0)
        for candidate in raw_candidates
    )
    direct = sum(
        float(np.dot(original_centroids[i] - candidate_centroids[i], original_centroids[i] - candidate_centroids[i]))
        for i in range(2)
    )
    swapped = sum(
        float(np.dot(original_centroids[i] - candidate_centroids[1 - i], original_centroids[i] - candidate_centroids[1 - i]))
        for i in range(2)
    )
    assigned = raw_candidates if direct <= swapped else raw_candidates[::-1]
    result: dict[int, tuple[int, int, int]] = {}
    for element_id, connectivity in zip(element_ids, assigned):
        oriented = _oriented_triangle(mesh, connectivity, owners[element_id])
        if oriented is None:
            return None, "replacement diagonal creates a degenerate or tangential triangle"
        result[element_id] = oriented
    return result, "accepted strict-envelope-improving diagonal flip"


def _gridded_parent(mesh: Mesh, element_id: int) -> bool:
    for face_id, elements in mesh.elements_of_face.items():
        if element_id in elements and (
            face_id in mesh.grid_of_face or face_id in mesh.block_grids_of_face
        ):
            return True
    return False


def _insert_geometry_midpoint(
    mesh: Mesh, edge: tuple[int, int], midpoint_id: int
) -> tuple[bool, str]:
    for geometry_edge, sequence in sorted(mesh.nodes_of_edge.items()):
        values = [int(value) for value in sequence]
        positions = [
            position
            for position, (first, second) in enumerate(zip(values, values[1:]))
            if _edge(first, second) == edge
        ]
        if positions:
            position = positions[0]
            values.insert(position + 1, midpoint_id)
            mesh.nodes_of_edge[geometry_edge] = values
        elif edge[0] in values and edge[1] in values:
            return False, f"geometry edge {geometry_edge} does not contain the shell edge consecutively"
    return True, ""


def _append_associations(mesh: Mesh, parent_id: int, child_id: int) -> None:
    for mapping in (
        mesh.elements_of_face,
        mesh.elements_of_sheet,
        mesh.elements_of_member,
    ):
        for owner, values in tuple(mapping.items()):
            if parent_id not in values:
                continue
            made = list(values)
            position = made.index(parent_id) + 1
            made.insert(position, child_id)
            mapping[owner] = made
    if parent_id in mesh.activity:
        mesh.activity[child_id] = mesh.activity[parent_id]


def _candidate_bisection(
    mesh: Mesh,
    edge: tuple[int, int],
    attached_ids: tuple[int, ...],
    owners: Mapping[int, np.ndarray],
    selected: frozenset[int],
    *,
    next_node_id: int,
    next_element_id: int,
) -> tuple[
    Mesh | None,
    dict[int, np.ndarray] | None,
    frozenset[int] | None,
    int,
    str,
]:
    if mesh.order != "linear":
        return None, None, None, 0, "local bisection is restricted to linear T3 meshes"
    incidence = _shell_incidence(mesh).get(edge, ())
    if not incidence or len(incidence) > 2:
        return None, None, None, 0, "edge incidence is missing or non-manifold"
    if any(kind != "tri" or element_id not in selected for kind, element_id in incidence):
        return None, None, None, 0, "edge touches an unselected shell; bisection would be nonconforming"
    incident_ids = tuple(sorted(element_id for _, element_id in incidence))
    if incident_ids != attached_ids:
        return None, None, None, 0, "edge incidence changed during deterministic repair"
    if any(
        parent_id in values
        for parent_id in incident_ids
        for values in mesh.elements_of_edge.values()
    ):
        return None, None, None, 0, "an incident triangle is owned by a geometry edge"
    if any(_gridded_parent(mesh, parent_id) for parent_id in incident_ids):
        return None, None, None, 0, "a structured face grid cannot represent local T3 bisection"

    candidate = deepcopy(mesh)
    candidate.nodes[next_node_id] = 0.5 * (
        np.asarray(candidate.nodes[edge[0]], dtype=float)
        + np.asarray(candidate.nodes[edge[1]], dtype=float)
    )
    inserted, reason = _insert_geometry_midpoint(candidate, edge, next_node_id)
    if not inserted:
        return None, None, None, 0, reason

    made_owners = {element_id: np.array(value, copy=True) for element_id, value in owners.items()}
    made_selected = set(selected)
    next_id = next_element_id
    for parent_id in incident_ids:
        connectivity = tuple(int(value) for value in candidate.tris[parent_id])
        start_position = next(
            (
                position
                for position in range(3)
                if _edge(connectivity[position], connectivity[(position + 1) % 3]) == edge
            ),
            None,
        )
        if start_position is None:
            return None, None, None, 0, "incident triangle does not contain the selected edge"
        first = connectivity[start_position]
        second = connectivity[(start_position + 1) % 3]
        opposite = connectivity[(start_position + 2) % 3]
        children = (
            (first, next_node_id, opposite),
            (next_node_id, second, opposite),
        )
        children = tuple(
            sorted(children, key=lambda value: (tuple(sorted(value)), tuple(value)))
        )
        candidate.tris[parent_id] = children[0]
        child_id = next_id
        next_id += 1
        candidate.tris[child_id] = children[1]
        _append_associations(candidate, parent_id, child_id)
        made_owners[child_id] = np.array(made_owners[parent_id], copy=True)
        made_selected.add(child_id)

    return candidate, made_owners, frozenset(made_selected), len(incident_ids), "accepted conforming midpoint bisection"


def _requested_scope(mesh: Mesh, element_ids: Sequence[int] | None) -> tuple[int, ...]:
    if element_ids is None:
        return tuple(sorted(int(value) for value in mesh.tris))
    requested = tuple(int(value) for value in element_ids)
    if len(set(requested)) != len(requested):
        raise ValueError("qualified S3 repair scope contains duplicate IDs")
    return tuple(sorted(requested))


def repair_s3_admission(
    mesh: Mesh,
    *,
    element_ids: Sequence[int] | None = None,
    element_owner_normals: Mapping[int, Sequence[float]] | None,
    quality_policy: S3QualityPolicy = DEFAULT_S3_QUALITY_POLICY,
    repair_policy: S3RepairPolicy = DEFAULT_S3_REPAIR_POLICY,
) -> S3RepairResult:
    """Return a separately copied qualified mesh or raise ``S3RepairError``.

    The operation order is fixed: winding, strict-envelope-improving diagonal
    flips, then bounded conforming midpoint bisection.  No operation changes a
    caller-owned mesh, and no exhausted request can return a legacy fallback.
    """

    if not isinstance(mesh, Mesh):
        raise TypeError("repair_s3_admission expects an anymesher.Mesh")
    attempts: list[S3RepairAttempt] = []
    try:
        requested = _requested_scope(mesh, element_ids)
    except (TypeError, ValueError) as error:
        _record(attempts, "validate", "rejected", detail=str(error))
        raise S3RepairError(str(error), attempts=attempts) from error
    if set(mesh.quads).intersection(mesh.tris):
        detail = "shell element IDs overlap between quadrilaterals and triangles"
        _record(attempts, "validate", "rejected", detail=detail)
        raise S3RepairError(detail, attempts=attempts)
    invalid = tuple(
        element_id
        for element_id in requested
        if element_id not in mesh.tris or len(mesh.tris[element_id]) != 3
    )
    if invalid:
        detail = f"qualified S3 repair requires exact T3 elements; invalid IDs={invalid}"
        _record(attempts, "validate", "rejected", invalid, detail=detail)
        raise S3RepairError(detail, attempts=attempts)
    if not requested:
        detail = "qualified S3 repair scope is empty"
        _record(attempts, "validate", "rejected", detail=detail)
        raise S3RepairError(detail, attempts=attempts)

    supplied_normals = {} if element_owner_normals is None else element_owner_normals
    owners: dict[int, np.ndarray] = {}
    for element_id in requested:
        if element_id not in supplied_normals:
            detail = f"triangle {element_id} authoritative owner normal is missing"
            _record(attempts, "validate", "rejected", (element_id,), detail=detail)
            raise S3RepairError(detail, attempts=attempts)
        try:
            owners[element_id] = _unit_owner(supplied_normals[element_id], element_id)
        except (TypeError, ValueError) as error:
            _record(attempts, "validate", "rejected", (element_id,), detail=str(error))
            raise S3RepairError(str(error), attempts=attempts) from error

    work = deepcopy(mesh)
    selected = frozenset(requested)
    winding_repairs = 0
    for element_id in requested:
        connectivity = tuple(int(value) for value in work.tris[element_id])
        try:
            oriented = _oriented_triangle(work, connectivity, owners[element_id])
        except KeyError as error:
            detail = f"triangle {element_id} references missing node {int(error.args[0])}"
            _record(attempts, "validate", "rejected", (element_id,), detail=detail)
            raise S3RepairError(detail, attempts=attempts) from error
        if oriented is None:
            detail = "triangle is degenerate or tangential to its owner normal"
            _record(attempts, "winding", "rejected", (element_id,), detail=detail)
            raise S3RepairError(detail, attempts=attempts)
        if oriented == connectivity:
            continue
        if winding_repairs >= repair_policy.maximum_winding_repairs:
            detail = "maximum_winding_repairs exhausted before authoritative orientation"
            _record(attempts, "winding", "limit", (element_id,), detail=detail)
            raise S3RepairError(detail, attempts=attempts)
        work.tris[element_id] = oriented
        winding_repairs += 1
        _record(
            attempts,
            "winding",
            "accepted",
            (element_id,),
            detail="swapped the final two T3 nodes to follow the authoritative normal",
        )

    def report_for(current: Mesh, scope: frozenset[int], made_owners: Mapping[int, np.ndarray]) -> S3AdmissionReport:
        return evaluate_s3_admission(
            current,
            element_ids=tuple(sorted(scope)),
            element_owner_normals={
                element_id: _normal_tuple(made_owners[element_id])
                for element_id in sorted(scope)
            },
            policy=quality_policy,
        )

    report = report_for(work, selected, owners)
    edge_flips = 0
    edge_flip_attempts = 0
    visited_flips: set[tuple[tuple[int, int], tuple[tuple[int, tuple[int, ...]], ...]]] = set()
    while not report.admitted:
        incidence = _shell_incidence(work)
        failing = {
            item.element_id for item in report.elements if not item.admitted
        }
        accepted = False
        for edge, attached in sorted(incidence.items()):
            if len(attached) != 2 or any(kind != "tri" for kind, _ in attached):
                continue
            ids = tuple(sorted(element_id for _, element_id in attached))
            if len(ids) != 2 or any(element_id not in selected for element_id in ids):
                continue
            if not failing.intersection(ids):
                continue
            state = (edge, tuple((element_id, tuple(work.tris[element_id])) for element_id in ids))
            if state in visited_flips:
                continue
            visited_flips.add(state)
            if edge_flip_attempts >= repair_policy.maximum_edge_flip_attempts:
                _record(
                    attempts,
                    "edge_flip",
                    "limit",
                    ids,
                    edge,
                    "maximum_edge_flip_attempts exhausted",
                )
                break
            edge_flip_attempts += 1
            if edge_flips >= repair_policy.maximum_edge_flips:
                _record(
                    attempts,
                    "edge_flip",
                    "limit",
                    ids,
                    edge,
                    "maximum_edge_flips exhausted",
                )
                break
            replacement, detail = _candidate_flip(
                work,
                edge,
                ids,  # type: ignore[arg-type]
                owners,
                repair_policy.minimum_flip_owner_alignment,
            )
            if replacement is None:
                _record(attempts, "edge_flip", "rejected", ids, edge, detail)
                continue
            candidate = deepcopy(work)
            candidate.tris.update(replacement)
            candidate_report = report_for(candidate, selected, owners)
            candidate_elements = {
                item.element_id: item for item in candidate_report.elements
            }
            if (
                any(not candidate_elements[element_id].admitted for element_id in ids)
                or _report_score(candidate_report) >= _report_score(report)
            ):
                _record(
                    attempts,
                    "edge_flip",
                    "rejected",
                    ids,
                    edge,
                    "replacement diagonal does not strictly reduce admission violations",
                )
                continue
            work = candidate
            report = candidate_report
            edge_flips += 1
            _record(attempts, "edge_flip", "accepted", ids, edge, detail)
            accepted = True
            break
        if not accepted:
            break

    refinement_splits = 0
    refinement_attempts = 0
    added_nodes = 0
    added_elements = 0
    visited_splits: set[tuple[int, tuple[int, int], tuple[int, ...]]] = set()
    while not report.admitted:
        failing = tuple(item.element_id for item in report.elements if not item.admitted)
        accepted = False
        incidence = _shell_incidence(work)
        candidates: list[tuple[float, tuple[int, int], int, tuple[int, ...]]] = []
        for element_id in sorted(failing):
            connectivity = tuple(int(value) for value in work.tris[element_id])
            for edge in _triangle_edges(connectivity):
                length = float(
                    np.linalg.norm(
                        np.asarray(work.nodes[edge[1]], dtype=float)
                        - np.asarray(work.nodes[edge[0]], dtype=float)
                    )
                )
                attached = tuple(
                    sorted(
                        owner_id
                        for kind, owner_id in incidence.get(edge, ())
                        if kind == "tri"
                    )
                )
                candidates.append((-length, edge, element_id, attached))
        for _, edge, element_id, attached in sorted(candidates):
            state = (element_id, edge, attached)
            if state in visited_splits:
                continue
            visited_splits.add(state)
            if refinement_attempts >= repair_policy.maximum_refinement_attempts:
                _record(
                    attempts,
                    "refinement",
                    "limit",
                    attached or (element_id,),
                    edge,
                    "maximum_refinement_attempts exhausted",
                )
                break
            refinement_attempts += 1
            if refinement_splits >= repair_policy.maximum_refinement_splits:
                _record(
                    attempts,
                    "refinement",
                    "limit",
                    attached or (element_id,),
                    edge,
                    "maximum_refinement_splits exhausted",
                )
                break
            if added_nodes >= repair_policy.maximum_added_nodes:
                _record(
                    attempts,
                    "refinement",
                    "limit",
                    attached or (element_id,),
                    edge,
                    "maximum_added_nodes exhausted",
                )
                break
            if added_elements + len(attached) > repair_policy.maximum_added_elements:
                _record(
                    attempts,
                    "refinement",
                    "limit",
                    attached or (element_id,),
                    edge,
                    "maximum_added_elements would be exceeded",
                )
                continue
            all_element_ids = set(work.quads) | set(work.tris) | set(work.beams) | set(work.couplings)
            next_node_id = max(work.nodes, default=0) + 1
            next_element_id = max(all_element_ids, default=0) + 1
            candidate, made_owners, made_selected, new_elements, detail = _candidate_bisection(
                work,
                edge,
                attached,
                owners,
                selected,
                next_node_id=next_node_id,
                next_element_id=next_element_id,
            )
            if candidate is None or made_owners is None or made_selected is None:
                _record(
                    attempts,
                    "refinement",
                    "rejected",
                    attached or (element_id,),
                    edge,
                    detail,
                )
                continue
            candidate_report = report_for(candidate, made_selected, made_owners)
            if _report_score(candidate_report) >= _report_score(report):
                _record(
                    attempts,
                    "refinement",
                    "rejected",
                    attached,
                    edge,
                    "conforming bisection does not strictly reduce admission violations",
                )
                continue
            work = candidate
            owners = made_owners
            selected = made_selected
            report = candidate_report
            refinement_splits += 1
            added_nodes += 1
            added_elements += new_elements
            _record(attempts, "refinement", "accepted", attached, edge, detail)
            accepted = True
            break
        if not accepted:
            break

    if not report.admitted:
        _record(
            attempts,
            "adjudication",
            "rejected",
            tuple(sorted(selected)),
            detail="; ".join(report.violations),
        )
        raise S3RepairError(
            "bounded qualified-S3 repair did not satisfy the admission contract",
            attempts=attempts,
            admission=report,
        )

    _record(
        attempts,
        "adjudication",
        "accepted",
        tuple(sorted(selected)),
        detail="all selected T3 elements satisfy the qualified-S3 admission contract",
    )
    return S3RepairResult(
        mesh=work,
        element_ids=tuple(sorted(selected)),
        owner_normals=tuple(
            (element_id, _normal_tuple(owners[element_id]))
            for element_id in sorted(selected)
        ),
        admission=report,
        attempts=tuple(attempts),
        winding_repairs=winding_repairs,
        edge_flips=edge_flips,
        edge_flip_attempts=edge_flip_attempts,
        refinement_splits=refinement_splits,
        refinement_attempts=refinement_attempts,
        added_nodes=added_nodes,
        added_elements=added_elements,
    )
