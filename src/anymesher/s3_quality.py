"""Admission checks for the qualified three-node shell formulation.

The existing :mod:`anymesher.quality_v2` metrics remain compatibility-facing
and deliberately accept a broad set of meshes.  The qualified E4-PL S3 shell
has a narrower, opt-in geometry contract.  This module keeps that contract
explicit so callers never silently fall back to a legacy triangle.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, sqrt
from typing import Mapping, Sequence

import numpy as np

from .errors import MeshError
from .mesh import Mesh

__all__ = [
    "DEFAULT_S3_QUALITY_POLICY",
    "S3_QUALITY_CONTRACT_ID",
    "S3AdmissionReport",
    "S3ElementQuality",
    "S3QualityError",
    "S3QualityPolicy",
    "assert_s3_admissible",
    "evaluate_s3_admission",
]


S3_QUALITY_CONTRACT_ID = "ANYMESHER_QUALIFIED_S3_ADMISSION_V1"


class S3QualityError(MeshError):
    """A mesh cannot use the qualified S3 formulation as supplied."""


@dataclass(frozen=True)
class S3QualityPolicy:
    """Scale-free admission limits for qualified S3 elements."""

    minimum_angle_deg: float = 30.0
    maximum_angle_deg: float = 150.0
    maximum_edge_ratio: float = 4.0
    minimum_scaled_jacobian: float = 0.20
    minimum_normalized_area: float = 0.60
    minimum_signed_area_ratio: float = 1.0e-14
    floating_area_factor: float = 64.0
    minimum_owner_normal_alignment: float = 1.0e-8
    require_authoritative_normals: bool = True

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.minimum_angle_deg,
                self.maximum_angle_deg,
                self.maximum_edge_ratio,
                self.minimum_scaled_jacobian,
                self.minimum_normalized_area,
                self.minimum_signed_area_ratio,
                self.floating_area_factor,
                self.minimum_owner_normal_alignment,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("S3 quality limits must be finite")
        if not 0.0 < self.minimum_angle_deg < self.maximum_angle_deg < 180.0:
            raise ValueError("S3 angle limits must lie strictly inside (0, 180)")
        if self.maximum_edge_ratio < 1.0:
            raise ValueError("S3 maximum edge ratio must be at least one")
        if not 0.0 < self.minimum_scaled_jacobian <= 1.0:
            raise ValueError("S3 minimum scaled Jacobian must lie in (0, 1]")
        if not 0.0 < self.minimum_normalized_area <= 1.0:
            raise ValueError("S3 minimum normalized area must lie in (0, 1]")
        if self.minimum_signed_area_ratio <= 0.0 or self.floating_area_factor <= 0.0:
            raise ValueError("S3 signed-area limits must be positive")
        if not 0.0 < self.minimum_owner_normal_alignment < 1.0:
            raise ValueError("S3 owner-normal alignment limit must lie in (0, 1)")


DEFAULT_S3_QUALITY_POLICY = S3QualityPolicy()


@dataclass(frozen=True)
class S3ElementQuality:
    element_id: int
    signed_area_ratio: float
    minimum_angle_deg: float
    maximum_angle_deg: float
    edge_ratio: float
    minimum_scaled_jacobian: float
    normalized_area: float
    owner_normal_alignment: float
    violations: tuple[str, ...]

    @property
    def admitted(self) -> bool:
        return not self.violations


@dataclass(frozen=True)
class S3AdmissionReport:
    elements: tuple[S3ElementQuality, ...]
    topology_violations: tuple[str, ...] = ()

    @property
    def admitted(self) -> bool:
        return not self.topology_violations and all(item.admitted for item in self.elements)

    @property
    def violations(self) -> tuple[str, ...]:
        result = list(self.topology_violations)
        for item in self.elements:
            result.extend(f"triangle {item.element_id}: {message}" for message in item.violations)
        return tuple(result)


def _unit(vector: Sequence[float], *, label: str) -> np.ndarray:
    made = np.asarray(vector, dtype=float)
    if made.shape != (3,) or not np.all(np.isfinite(made)):
        raise S3QualityError(f"{label} must be a finite three-vector")
    length = float(np.linalg.norm(made))
    if length <= 0.0:
        raise S3QualityError(f"{label} must be nonzero")
    return made / length


def _angles(corners: np.ndarray) -> np.ndarray:
    result = np.empty(3, dtype=float)
    for index in range(3):
        first = corners[(index - 1) % 3] - corners[index]
        second = corners[(index + 1) % 3] - corners[index]
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator <= 0.0:
            result[index] = 0.0
        else:
            cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
            result[index] = degrees(acos(cosine))
    return result


def _directed_edge_violations(
    mesh: Mesh, selected_triangles: frozenset[int]
) -> tuple[str, ...]:
    incidence: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for element_id, connectivity in sorted(mesh.shells.items()):
        corners = mesh.corners_of(element_id)
        for index, start in enumerate(corners):
            end = int(corners[(index + 1) % len(corners)])
            start = int(start)
            key = (min(start, end), max(start, end))
            incidence.setdefault(key, []).append((int(element_id), start, end))

    violations: list[str] = []
    for edge, attached in sorted(incidence.items()):
        if not any(item[0] in selected_triangles for item in attached):
            continue
        if len(attached) > 2:
            ids = tuple(item[0] for item in attached)
            violations.append(f"edge {edge} is non-manifold; owners={ids}")
        elif len(attached) == 2:
            first, second = attached
            if first[1:] == second[1:]:
                violations.append(
                    f"edge {edge} has equal traversal in elements {first[0]} and {second[0]}"
                )
    return tuple(violations)


def evaluate_s3_admission(
    mesh: Mesh,
    *,
    element_ids: Sequence[int] | None = None,
    element_owner_normals: Mapping[int, Sequence[float]] | None = None,
    policy: S3QualityPolicy = DEFAULT_S3_QUALITY_POLICY,
) -> S3AdmissionReport:
    """Evaluate the opt-in qualified-S3 geometry contract.

    ``element_owner_normals`` is keyed by stable triangle element ID.  It is
    intentionally caller supplied: a triangle's self-derived normal cannot
    distinguish a valid ordering from complete winding reversal.
    """

    if not isinstance(mesh, Mesh):
        raise TypeError("evaluate_s3_admission expects an anymesher.Mesh")
    normals = {} if element_owner_normals is None else element_owner_normals
    records: list[S3ElementQuality] = []
    topology_violations: list[str] = []
    if element_ids is None:
        requested = tuple(sorted(int(value) for value in mesh.tris))
    else:
        requested = tuple(int(value) for value in element_ids)
        if len(set(requested)) != len(requested):
            topology_violations.append("qualified S3 element scope contains duplicate IDs")
    selected: list[int] = []
    for element_id in sorted(set(requested)):
        connectivity = mesh.tris.get(element_id)
        if connectivity is None:
            topology_violations.append(
                f"qualified S3 element {element_id} is not a triangle in this mesh"
            )
        elif len(connectivity) != 3:
            topology_violations.append(
                f"qualified S3 element {element_id} has {len(connectivity)} nodes; exactly 3 required"
            )
        else:
            selected.append(element_id)
    epsilon_limit = policy.floating_area_factor * np.finfo(float).eps
    area_limit = max(policy.minimum_signed_area_ratio, epsilon_limit)

    for element_id in selected:
        connectivity = mesh.tris[element_id]
        violations: list[str] = []
        corners_ids = tuple(int(value) for value in connectivity[:3])
        if len(set(corners_ids)) != 3:
            violations.append("corner node IDs are not distinct")
        try:
            corners = np.asarray([mesh.nodes[node] for node in corners_ids], dtype=float)
        except KeyError as exc:
            raise S3QualityError(
                f"triangle {element_id} references missing node {int(exc.args[0])}"
            ) from exc
        if corners.shape != (3, 3) or not np.all(np.isfinite(corners)):
            violations.append("corner coordinates are not finite three-vectors")
            records.append(
                S3ElementQuality(
                    element_id=int(element_id),
                    signed_area_ratio=0.0,
                    minimum_angle_deg=0.0,
                    maximum_angle_deg=180.0,
                    edge_ratio=float(np.finfo(float).max),
                    minimum_scaled_jacobian=-1.0,
                    normalized_area=0.0,
                    owner_normal_alignment=0.0,
                    violations=tuple(violations),
                )
            )
            continue

        authoritative: np.ndarray | None = None
        if element_id in normals:
            authoritative = _unit(normals[element_id], label=f"triangle {element_id} owner normal")
        elif policy.require_authoritative_normals:
            violations.append("authoritative owner normal is missing")

        lengths = np.asarray(
            [np.linalg.norm(corners[(i + 1) % 3] - corners[i]) for i in range(3)],
            dtype=float,
        )
        lmax = float(np.max(lengths)) if lengths.size else 0.0
        lmin = float(np.min(lengths)) if lengths.size else 0.0
        edge_ratio = lmax / lmin if lmin > 0.0 else float("inf")
        cross = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        cross_norm = float(np.linalg.norm(cross))
        fallback = cross / cross_norm if cross_norm > 0.0 else np.zeros(3)
        reference = fallback if authoritative is None else authoritative
        owner_alignment = (
            0.0
            if authoritative is None
            else abs(float(np.dot(fallback, authoritative)))
        )
        signed_twice_area = float(np.dot(cross, reference))
        signed_ratio = signed_twice_area / (lmax * lmax) if lmax > 0.0 else 0.0
        area = 0.5 * cross_norm
        denominator = float(np.dot(lengths, lengths))
        normalized_area = 4.0 * sqrt(3.0) * area / denominator if denominator > 0.0 else 0.0
        angles = _angles(corners)

        orientation = 1.0 if signed_twice_area > 0.0 else -1.0
        jacobians = []
        for index in range(3):
            outgoing = corners[(index + 1) % 3] - corners[index]
            incoming = corners[(index - 1) % 3] - corners[index]
            scale = float(np.linalg.norm(outgoing) * np.linalg.norm(incoming))
            jacobians.append(
                float(orientation * np.linalg.norm(np.cross(outgoing, incoming)) / scale)
                if scale > 0.0
                else -1.0
            )
        minimum_jacobian = float(np.min(jacobians))
        minimum_angle = float(np.min(angles))
        maximum_angle = float(np.max(angles))

        if not np.all(np.isfinite(lengths)) or lmin <= 0.0:
            violations.append("contains a coincident or non-finite edge")
        if signed_ratio <= area_limit:
            violations.append(
                f"signed area ratio {signed_ratio:.12g} does not exceed {area_limit:.12g}"
            )
        if authoritative is not None and owner_alignment <= policy.minimum_owner_normal_alignment:
            violations.append(
                f"owner-normal alignment {owner_alignment:.12g} does not exceed "
                f"{policy.minimum_owner_normal_alignment:.12g}"
            )
        if minimum_angle < policy.minimum_angle_deg:
            violations.append(
                f"minimum angle {minimum_angle:.12g} is below {policy.minimum_angle_deg:.12g} degrees"
            )
        if maximum_angle > policy.maximum_angle_deg:
            violations.append(
                f"maximum angle {maximum_angle:.12g} exceeds {policy.maximum_angle_deg:.12g} degrees"
            )
        if edge_ratio > policy.maximum_edge_ratio:
            violations.append(
                f"edge ratio {edge_ratio:.12g} exceeds {policy.maximum_edge_ratio:.12g}"
            )
        if minimum_jacobian < policy.minimum_scaled_jacobian:
            violations.append(
                f"scaled Jacobian {minimum_jacobian:.12g} is below {policy.minimum_scaled_jacobian:.12g}"
            )
        if normalized_area < policy.minimum_normalized_area:
            violations.append(
                f"normalized area {normalized_area:.12g} is below {policy.minimum_normalized_area:.12g}"
            )

        records.append(
            S3ElementQuality(
                element_id=int(element_id),
                signed_area_ratio=signed_ratio,
                minimum_angle_deg=minimum_angle,
                maximum_angle_deg=maximum_angle,
                edge_ratio=edge_ratio,
                minimum_scaled_jacobian=minimum_jacobian,
                normalized_area=normalized_area,
                owner_normal_alignment=owner_alignment,
                violations=tuple(violations),
            )
        )

    topology_violations.extend(
        _directed_edge_violations(mesh, frozenset(selected))
    )
    return S3AdmissionReport(tuple(records), tuple(topology_violations))


def assert_s3_admissible(
    mesh: Mesh,
    *,
    element_ids: Sequence[int] | None = None,
    element_owner_normals: Mapping[int, Sequence[float]] | None = None,
    policy: S3QualityPolicy = DEFAULT_S3_QUALITY_POLICY,
) -> S3AdmissionReport:
    """Return the admission report or raise a stable typed rejection."""

    report = evaluate_s3_admission(
        mesh,
        element_ids=element_ids,
        element_owner_normals=element_owner_normals,
        policy=policy,
    )
    if not report.admitted:
        raise S3QualityError("; ".join(report.violations))
    return report
