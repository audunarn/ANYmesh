"""Sampling a chain of oriented edges by arc length.

A face side may be made of several edges, so "the point 30% along this side" has
to be resolved across edge boundaries and weighted by length.  That is a geometry
question, not a meshing one, which is why these helpers live here.

They used to live with the mapped mesher, and the decomposition operations
imported them from there -- so the geometry package depended on the mesher while
the mesher depended on the geometry package.  Moving them removes that cycle
without changing any behaviour.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..errors import MeshError
from .entities import OrientedEdge
from .model import GeometryModel

__all__ = ["chain_breaks", "chain_point", "sample_chain"]


def chain_breaks(geometry: GeometryModel, side: Sequence[OrientedEdge]) -> np.ndarray:
    """Normalised arc-length positions of each edge junction along a chain."""

    lengths = np.array([geometry.edge_length(item.edge) for item in side], dtype=float)
    total = float(lengths.sum())
    if total <= 0.0:
        raise MeshError("cannot measure a zero-length chain")
    return np.concatenate(([0.0], np.cumsum(lengths) / total))


def chain_point(
    geometry: GeometryModel,
    side: Sequence[OrientedEdge],
    fraction: float,
    breaks: np.ndarray | None = None,
) -> np.ndarray:
    """The point at a normalised arc-length position along a chain of edges."""

    if breaks is None:
        breaks = chain_breaks(geometry, side)
    fraction = float(min(max(fraction, 0.0), 1.0))
    segment = int(np.searchsorted(breaks, fraction, side="right") - 1)
    segment = min(max(segment, 0), len(side) - 1)
    span = breaks[segment + 1] - breaks[segment]
    local = 0.0 if span <= 0.0 else (fraction - breaks[segment]) / span
    item = side[segment]
    parameter = local if item.forward else 1.0 - local
    return geometry.sample_edge(item.edge, np.array([parameter]))[0]


def sample_chain(
    geometry: GeometryModel, side: Sequence[OrientedEdge], count: int
) -> np.ndarray:
    """Sample ``count + 1`` points along a chain of edges, by arc length.

    Used for display tessellation, where the division counts are chosen for
    smoothness rather than by the seeding solver.
    """

    if count < 1:
        raise MeshError("a chain needs at least one division")

    breaks = chain_breaks(geometry, side)
    return np.array(
        [
            chain_point(geometry, side, fraction, breaks)
            for fraction in np.linspace(0.0, 1.0, count + 1)
        ]
    )
