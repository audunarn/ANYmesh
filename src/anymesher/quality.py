"""Mesh quality metrics.

Two numbers matter for shells: aspect ratio and warp.  A stretched element loses
accuracy in bending along its long direction, and a non-planar quadrilateral has
no single plane for its bending kinematics to refer to, so both show up as
accuracy loss rather than as a failure -- which is why they are worth reporting
before a solve rather than diagnosing after one.

The thresholds are advisory and are stated in the warnings rather than enforced.
A quality metric that refused to produce a mesh would just be worked around.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .mesh import Mesh

__all__ = ["MeshQuality", "ASPECT_RATIO_LIMIT", "WARP_LIMIT", "verify_mesh_quality"]

# Above this, bending accuracy along the long direction degrades noticeably.
ASPECT_RATIO_LIMIT = 5.0
# Out-of-plane corner deviation, normalized by mean edge length.
WARP_LIMIT = 0.05


@dataclass(frozen=True)
class MeshQuality:
    """Aspect ratio and warp over the shell elements of a mesh."""

    num_shell_elements: int
    max_aspect_ratio: float
    mean_aspect_ratio: float
    max_warp: float
    warnings: Tuple[str, ...] = ()
    min_scaled_jacobian: float = 1.0
    min_angle: float = 90.0
    max_angle: float = 90.0
    poor_element_ids: Tuple[int, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether every metric is inside its advisory limit."""

        return not self.warnings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "num_shell_elements": self.num_shell_elements,
            "max_aspect_ratio": self.max_aspect_ratio,
            "mean_aspect_ratio": self.mean_aspect_ratio,
            "max_warp": self.max_warp,
            "min_scaled_jacobian": self.min_scaled_jacobian,
            "min_angle": self.min_angle,
            "max_angle": self.max_angle,
            "poor_element_ids": list(self.poor_element_ids),
            "warnings": list(self.warnings),
        }


def verify_mesh_quality(mesh: Mesh) -> MeshQuality:
    """Measure aspect ratio and warp over every shell element."""

    aspect_ratios: List[float] = []
    warps: List[float] = []
    scaled_jacobians: List[float] = []
    minimum_angles: List[float] = []
    maximum_angles: List[float] = []
    element_metrics: List[Tuple[int, float, float, float]] = []
    shell_count = 0

    for element_id in mesh.quads:
        shell_count += 1
        # Corners only: a mid-side node does not describe the element's shape,
        # and including one would report a quadratic element as twice as fine.
        corner_coords = np.array([mesh.nodes[node] for node in mesh.corners_of(element_id)])

        e1 = corner_coords[1] - corner_coords[0]
        e2 = corner_coords[2] - corner_coords[1]
        e3 = corner_coords[3] - corner_coords[2]
        e4 = corner_coords[0] - corner_coords[3]

        lengths = [
            float(np.linalg.norm(e1)),
            float(np.linalg.norm(e2)),
            float(np.linalg.norm(e3)),
            float(np.linalg.norm(e4)),
        ]
        aspect_ratios.append(max(lengths) / max(min(lengths), 1.0e-15))

        # Warp is the fourth corner's distance from the plane of the first three,
        # normalized by mean edge length so it is dimensionless.
        n_raw = np.cross(e1, corner_coords[2] - corner_coords[0])
        n_norm = float(np.linalg.norm(n_raw))
        if n_norm > 1.0e-15:
            n = n_raw / n_norm
            d = abs(float(np.dot(corner_coords[3] - corner_coords[0], n)))
            warps.append(d / max(sum(lengths) / 4.0, 1.0e-15))
        else:
            warps.append(0.0)
        angles = []
        jacobians = []
        for corner in range(4):
            previous = corner_coords[(corner - 1) % 4] - corner_coords[corner]
            following = corner_coords[(corner + 1) % 4] - corner_coords[corner]
            denominator = max(float(np.linalg.norm(previous) * np.linalg.norm(following)), 1.0e-30)
            cosine = float(np.clip(np.dot(previous, following) / denominator, -1.0, 1.0))
            angles.append(float(np.degrees(np.arccos(cosine))))
            jacobians.append(float(np.linalg.norm(np.cross(following, previous))) / denominator)
        scaled_jacobians.append(min(jacobians))
        minimum_angles.append(min(angles))
        maximum_angles.append(max(angles))
        element_metrics.append((int(element_id), aspect_ratios[-1], warps[-1], scaled_jacobians[-1]))

    for element_id in mesh.tris:
        shell_count += 1
        corner_coords = np.array([mesh.nodes[node] for node in mesh.corners_of(element_id)])
        lengths = [
            float(np.linalg.norm(corner_coords[1] - corner_coords[0])),
            float(np.linalg.norm(corner_coords[2] - corner_coords[1])),
            float(np.linalg.norm(corner_coords[0] - corner_coords[2])),
        ]
        aspect_ratios.append(max(lengths) / max(min(lengths), 1.0e-15))
        # A triangle is planar by definition, so it contributes no warp.  It is
        # still appended, so the mean is over every shell element rather than
        # over an arbitrary subset.
        warps.append(0.0)
        angles = []
        jacobians = []
        for corner in range(3):
            previous = corner_coords[(corner - 1) % 3] - corner_coords[corner]
            following = corner_coords[(corner + 1) % 3] - corner_coords[corner]
            denominator = max(float(np.linalg.norm(previous) * np.linalg.norm(following)), 1.0e-30)
            cosine = float(np.clip(np.dot(previous, following) / denominator, -1.0, 1.0))
            angles.append(float(np.degrees(np.arccos(cosine))))
            jacobians.append(float(np.linalg.norm(np.cross(following, previous))) / denominator)
        scaled_jacobians.append(min(jacobians))
        minimum_angles.append(min(angles))
        maximum_angles.append(max(angles))
        element_metrics.append((int(element_id), aspect_ratios[-1], 0.0, scaled_jacobians[-1]))

    warnings: List[str] = []
    max_ar = float(np.max(aspect_ratios)) if aspect_ratios else 1.0
    mean_ar = float(np.mean(aspect_ratios)) if aspect_ratios else 1.0
    max_warp = float(np.max(warps)) if warps else 0.0
    min_jacobian = float(np.min(scaled_jacobians)) if scaled_jacobians else 1.0
    min_angle = float(np.min(minimum_angles)) if minimum_angles else 90.0
    max_angle = float(np.max(maximum_angles)) if maximum_angles else 90.0
    poor_ids = tuple(
        sorted(
            element_id
            for element_id, aspect, warp, jacobian in element_metrics
            if aspect > ASPECT_RATIO_LIMIT or warp > WARP_LIMIT or jacobian <= 0.0
        )
    )

    if max_ar > ASPECT_RATIO_LIMIT:
        warnings.append(
            f"High aspect ratio detected (max AR = {max_ar:.2f}). "
            "Highly stretched elements can reduce solver accuracy. Consider refining the mesh divisions."
        )
    if max_warp > WARP_LIMIT:
        warnings.append(
            f"Significant element warp detected (max warp = {max_warp:.4f}). "
            "Warped shell elements can lose accuracy. Ensure plate geometries are flat or sufficiently refined."
        )

    return MeshQuality(
        num_shell_elements=shell_count,
        max_aspect_ratio=max_ar,
        mean_aspect_ratio=mean_ar,
        max_warp=max_warp,
        warnings=tuple(warnings),
        min_scaled_jacobian=min_jacobian,
        min_angle=min_angle,
        max_angle=max_angle,
        poor_element_ids=poor_ids,
    )
