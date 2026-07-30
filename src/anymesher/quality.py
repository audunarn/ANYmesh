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
            "warnings": list(self.warnings),
        }


def verify_mesh_quality(mesh: Mesh) -> MeshQuality:
    """Measure aspect ratio and warp over every shell element."""

    aspect_ratios: List[float] = []
    warps: List[float] = []
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

    warnings: List[str] = []
    max_ar = float(np.max(aspect_ratios)) if aspect_ratios else 1.0
    mean_ar = float(np.mean(aspect_ratios)) if aspect_ratios else 1.0
    max_warp = float(np.max(warps)) if warps else 0.0

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
    )
