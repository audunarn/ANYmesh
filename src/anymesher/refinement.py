"""Local mesh refinement: a size field over the model.

Uniform meshing forces one compromise everywhere -- fine enough for the stress
concentration means fine everywhere, which is what makes a model too big to
solve.  A refinement zone says "this small, here", and the element size grows
back to the global target with distance.

The zone is a *distance to a set of source points*.  Which points those are
comes from what the zone is bound to: a modelled point contributes itself, a
line contributes samples along it, a plate contributes samples over its
surface.  One distance function then serves point, line and patch refinement
rather than three special cases, and the sampling density is tied to the zone's
own size so the discretisation of the source is always finer than the mesh it
controls.

Outside the radius the size grows geometrically::

    size(d) = zone.size * growth ** ((d - radius) / zone.size)

capped at the global target.  ``growth`` is therefore the ratio between
neighbouring element sizes in the transition -- 1.0 would never return to the
target, so it is refused.

Like every other attribute, a zone binds to a topology entity and is resolved
at mesh time, so re-meshing or editing the geometry never loses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from anygeometry.chains import sample_chain
from anygeometry.entities import EntityRef
from anygeometry.model import GeometryModel


__all__ = [
    "Refinement",
    "SizeField",
    "refine_around",
    "refine_at",
]


def _finite_real(value: object, label: str) -> float:
    """Normalize one real modelling value without accepting boolean flags."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number")
    resolved = float(value)
    if not np.isfinite(resolved):
        raise ValueError(f"{label} must be a finite real number")
    return resolved


def _finite_center(values: Sequence[float], label: str) -> Tuple[float, float, float]:
    """Normalize an exact finite xyz coordinate."""

    try:
        components = tuple(values)
    except TypeError:
        raise ValueError(f"{label} needs exactly three finite real components") from None
    if len(components) != 3:
        raise ValueError(f"{label} needs exactly three finite real components")
    try:
        normalized = tuple(
            _finite_real(value, label) for value in components
        )
    except ValueError:
        raise ValueError(f"{label} needs exactly three finite real components") from None
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True)
class Refinement:
    """A region asking for a smaller element size.

    Exactly one of ``ref`` and ``center`` says where.  A ``ref`` binds to
    persistent topology and survives re-meshing and geometry edits; a
    ``center`` is a raw coordinate, which is what an impact zone needs since a
    sphere does not land on a modelled point.
    """

    size: float
    radius: float = 0.0
    ref: Optional[EntityRef] = None
    center: Optional[Sequence[float]] = None
    growth: float = 1.5
    name: str = "refinement"

    def __post_init__(self) -> None:
        size = _finite_real(
            self.size, f"refinement {self.name!r}: element size"
        )
        radius = _finite_real(
            self.radius, f"refinement {self.name!r}: radius"
        )
        growth = _finite_real(
            self.growth, f"refinement {self.name!r}: growth"
        )
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "growth", growth)

        if size <= 0.0:
            raise ValueError(
                f"refinement {self.name!r}: element size must be positive"
            )
        if radius < 0.0:
            raise ValueError(
                f"refinement {self.name!r}: radius must not be negative"
            )
        if growth <= 1.0:
            raise ValueError(
                f"refinement {self.name!r}: growth must exceed 1.0. It is the "
                "ratio between neighbouring element sizes in the transition, "
                "so 1.0 would never return to the global target size."
            )
        if (self.ref is None) == (self.center is None):
            raise ValueError(
                f"refinement {self.name!r}: give either ref (bound to "
                "geometry) or center (a raw coordinate), not both and not "
                "neither."
            )
        if self.center is not None:
            object.__setattr__(
                self,
                "center",
                _finite_center(
                    self.center, f"refinement {self.name!r}: center"
                ),
            )

    def sources(self, geometry: GeometryModel) -> np.ndarray:
        """The points distance is measured from, as an (n, 3) array.

        Sampled finer than the size the zone asks for, so the source is never
        the coarser of the two.
        """

        if self.center is not None:
            return np.asarray([self.center], dtype=float)

        ref = self.ref
        assert ref is not None  # guarded in __post_init__
        if ref.kind == "vertex":
            return np.asarray([geometry.vertex_position(ref.id)], dtype=float)
        if ref.kind == "edge":
            count = self._sample_count(geometry.edge_length(ref.id))
            return geometry.sample_edge(
                ref.id, np.linspace(0.0, 1.0, count)
            )
        if ref.kind == "face":
            return self._face_sources(geometry, ref.id)
        raise ValueError(
            f"refinement {self.name!r}: cannot refine around a {ref.kind}"
        )

    def _sample_count(self, length: float) -> int:
        # Two samples per requested element, so the source polyline is finer
        # than anything the zone will produce.
        return int(max(2, np.ceil(2.0 * length / self.size) + 1))

    def _face_sources(
        self, geometry: GeometryModel, face_id: int
    ) -> np.ndarray:
        from .mapped import coons_grid

        face = geometry.faces[face_id]
        sides = face.sides()
        spans = [
            sum(geometry.edge_length(item.edge) for item in side)
            for side in sides
        ]
        n_u = self._sample_count(max(spans[0], spans[2])) - 1
        n_v = self._sample_count(max(spans[1], spans[3])) - 1
        grid = coons_grid(
            sample_chain(geometry, sides[0], n_u),
            sample_chain(geometry, sides[1], n_v),
            sample_chain(geometry, sides[2], n_u)[::-1],
            sample_chain(geometry, sides[3], n_v)[::-1],
        )
        return grid.reshape(-1, 3)


class SizeField:
    """Desired element size anywhere, from the zones plus a global target.

    Built once per mesh and queried per edge, so the source points are
    gathered up front rather than re-derived for every sample.
    """

    def __init__(
        self,
        geometry: GeometryModel,
        target_size: float,
        refinements: Iterable[Refinement] = (),
    ) -> None:
        if target_size <= 0.0:
            raise ValueError("target element size must be positive")
        self.target_size = float(target_size)
        self.zones: Tuple[Refinement, ...] = tuple(refinements)
        self._sources: List[np.ndarray] = [
            zone.sources(geometry) for zone in self.zones
        ]
        for zone in self.zones:
            if zone.size > self.target_size:
                raise ValueError(
                    f"refinement {zone.name!r} asks for {zone.size:g} m "
                    f"elements, which is coarser than the global target size "
                    f"{self.target_size:g} m. A refinement zone refines; to "
                    "coarsen, raise the target size."
                )

    @property
    def is_uniform(self) -> bool:
        """Whether this field is just the global target everywhere.

        The uniform case takes the original code path exactly, so adding the
        size field cannot move a number in a model that does not use it.
        """

        return not self.zones

    def size_at(self, points: np.ndarray) -> np.ndarray:
        """Desired element size at each of an (n, 3) array of points."""

        points = np.atleast_2d(np.asarray(points, dtype=float))
        size = np.full(len(points), self.target_size, dtype=float)
        for zone, sources in zip(self.zones, self._sources):
            distance = _nearest_distance(points, sources)
            beyond = np.maximum(distance - zone.radius, 0.0)
            grown = zone.size * zone.growth ** (beyond / zone.size)
            size = np.minimum(size, np.minimum(grown, self.target_size))
        return size

    def summary(self) -> str:
        if self.is_uniform:
            return f"uniform {self.target_size:g} m"
        finest = min(zone.size for zone in self.zones)
        return (
            f"{self.target_size:g} m target, {len(self.zones)} refinement "
            f"zone(s), finest {finest:g} m"
        )

    def metric_spec(self):
        """Translate this compatibility field to the serializable metric model."""

        from .metric import MetricFieldSpec

        return MetricFieldSpec.from_size_field(self)


def _nearest_distance(points: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Distance from every point to the nearest source point.

    Chunked over the sources so a finely sampled plate does not build an
    n-by-m matrix large enough to matter.
    """

    nearest = np.full(len(points), np.inf, dtype=float)
    chunk = max(1, 4_000_000 // max(len(points), 1))
    for start in range(0, len(sources), chunk):
        block = sources[start : start + chunk]
        gaps = np.linalg.norm(points[:, None, :] - block[None, :, :], axis=2)
        nearest = np.minimum(nearest, gaps.min(axis=1))
    return nearest


def refine_around(
    ref: EntityRef, size: float, radius: float = 0.0, **options
) -> Refinement:
    """Refine to ``size`` within ``radius`` of a modelled entity."""

    return Refinement(size=size, radius=radius, ref=ref, **options)


def refine_at(
    center: Sequence[float], size: float, radius: float = 0.0, **options
) -> Refinement:
    """Refine to ``size`` within ``radius`` of a raw coordinate."""

    return Refinement(
        size=size, radius=radius, center=center, **options
    )
