"""Topology entities with persistent identity.

Identity is the load-bearing idea here.  Loads, boundary conditions, sections
and materials all reference entities by ``EntityRef``, never by coordinate or
by index into a mesh.  IDs are allocated monotonically per kind and are never
reused, so an attribute keeps pointing at the thing the user picked even after
the model is edited and re-meshed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple

import numpy as np

from .curves import CurveShape

__all__ = [
    "Edge",
    "EntityKind",
    "EntityRef",
    "Face",
    "OrientedEdge",
    "Vertex",
]

EntityKind = Literal["vertex", "edge", "face"]


@dataclass(frozen=True)
class EntityRef:
    """A stable reference to one geometry entity."""

    kind: EntityKind
    id: int

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.kind}{self.id}"


@dataclass
class Vertex:
    """A modelled point."""

    id: int
    position: np.ndarray

    @property
    def ref(self) -> EntityRef:
        return EntityRef("vertex", self.id)


@dataclass
class Edge:
    """A line between two vertices, with a curve shape."""

    id: int
    start: int
    end: int
    curve: CurveShape

    @property
    def ref(self) -> EntityRef:
        return EntityRef("edge", self.id)

    def other_vertex(self, vertex_id: int) -> int:
        """Return the far vertex of this edge."""

        if vertex_id == self.start:
            return self.end
        if vertex_id == self.end:
            return self.start
        raise ValueError(f"vertex {vertex_id} is not on edge {self.id}")


@dataclass(frozen=True)
class OrientedEdge:
    """One edge traversed in a stated direction within a face loop."""

    edge: int
    forward: bool


@dataclass
class Face:
    """A plate bounded by a closed loop of edges, organised into four sides.

    ``loop`` is the ordered, closed traversal of the boundary.  ``corners``
    holds four indices into ``loop`` marking where each side begins, so side
    ``k`` spans ``loop[corners[k]:corners[k+1]]`` (wrapping at the end).  A
    face with more than four edges is still mappable whenever its edges group
    into four such chains.
    """

    id: int
    loop: Tuple[OrientedEdge, ...]
    corners: Tuple[int, int, int, int]
    metadata: dict = field(default_factory=dict)

    @property
    def ref(self) -> EntityRef:
        return EntityRef("face", self.id)

    def side(self, index: int) -> Tuple[OrientedEdge, ...]:
        """Return the oriented edge chain forming side ``index`` (0..3)."""

        if not 0 <= index < 4:
            raise IndexError("a mapped face has exactly four sides (0..3)")
        start = self.corners[index]
        stop = self.corners[(index + 1) % 4]
        if stop > start:
            return self.loop[start:stop]
        # The last side wraps past the end of the loop.
        return self.loop[start:] + self.loop[:stop]

    def sides(self) -> Tuple[Tuple[OrientedEdge, ...], ...]:
        return tuple(self.side(k) for k in range(4))
