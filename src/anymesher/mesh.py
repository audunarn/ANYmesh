"""The neutral mesh container and its coupling records.

A mesh here is nodes, quadrilaterals, beams, coupling records, and the
**association** back to the geometry that produced it -- which node came from
which vertex, which nodes lie along which edge, which elements belong to which
face.  The association is the point of the container: it is what lets a load or a
restraint be named against geometry and survive a re-mesh, and what makes results
addressable by the thing the user drew rather than by node number.

Nothing here knows about elements, materials or constraints.  A coupling record
says where a beam node sits relative to the plating; deciding that this becomes
six multi-point constraints is the consuming solver's business.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from anygeometry.entities import EntityRef

from .errors import MeshError
from .seeding import Seeding

__all__ = ["Coupling", "Mesh"]


@dataclass(frozen=True)
class Coupling:
    """Where an eccentric beam node sits relative to the plating under it.

    ``plate_nodes`` and ``weights`` interpolate a point in the plating: the
    coupled location is ``sum(weight_i * position_i)``, and ``eccentricity`` is
    the vector from there to the beam node.  Weights sum to one.

    Two cases share this record.  A stiffener meshed on a plate edge has its beam
    nodes directly above plate nodes, so there is one master with weight one.  A
    stiffener crossing a panel mesh lands in the middle of a shell element, so
    there are four or eight masters carrying that element's shape-function values
    at the projected point.  Interpolating is what removes the older requirement
    that a beam node lie exactly on a shell node row -- the mesh no longer has to
    be aligned to the stiffeners for the coupling to be exact.

    This is a statement about geometry, not a constraint.  It does not know that
    a solver will turn it into six MPCs tying three translations and three
    rotations with rigid-offset terms.
    """

    beam_node: int
    plate_nodes: Tuple[int, ...]
    weights: Tuple[float, ...]
    eccentricity: Tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.plate_nodes) != len(self.weights):
            raise MeshError("a coupling needs one weight per plate node")
        if not self.plate_nodes:
            raise MeshError("a coupling needs at least one plate node")
        total = float(np.sum(np.asarray(self.weights, dtype=float)))
        if not np.isfinite(total) or abs(total - 1.0) > 1.0e-9:
            # Weights that do not sum to one describe an interpolation that does
            # not reproduce a rigid translation, which would make the coupling
            # add or remove displacement rather than transfer it.
            raise MeshError(f"coupling weights must sum to 1.0, not {total!r}")
        if len(self.eccentricity) != 3:
            raise MeshError("a coupling eccentricity is a 3-vector")

    @property
    def is_node_to_node(self) -> bool:
        """Whether the beam node couples to a single plate node."""

        return len(self.plate_nodes) == 1

    @property
    def plate_node(self) -> int:
        """The single master node, for the node-to-node case."""

        if not self.is_node_to_node:
            raise MeshError(
                f"coupling of beam node {self.beam_node} interpolates "
                f"{len(self.plate_nodes)} plate nodes and has no single master"
            )
        return int(self.plate_nodes[0])

    @classmethod
    def node_to_node(
        cls, beam_node: int, plate_node: int, eccentricity: Sequence[float]
    ) -> "Coupling":
        """A beam node standing directly off one plate node."""

        return cls(
            beam_node=int(beam_node),
            plate_nodes=(int(plate_node),),
            weights=(1.0,),
            eccentricity=tuple(float(value) for value in eccentricity),  # type: ignore[arg-type]
        )


@dataclass
class Mesh:
    """A mesh, with the association back to the geometry that made it."""

    nodes: Dict[int, np.ndarray] = field(default_factory=dict)
    # Four node IDs when linear, eight when quadratic: corners first, then the
    # mid-side nodes, which is the serendipity ordering solvers expect.
    quads: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    # Three node IDs when linear, six when quadratic, same corners-then-mid-sides
    # ordering.  The mapped mesher never produces one -- a triangle is not a
    # mapped element -- but an unstructured backend does, and dropping them to
    # keep the container tidy would silently delete part of the mesh.
    tris: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    # Two node IDs when linear, three when quadratic: start, mid, end.
    beams: Dict[int, Tuple[int, ...]] = field(default_factory=dict)

    node_of_vertex: Dict[int, int] = field(default_factory=dict)
    nodes_of_edge: Dict[int, List[int]] = field(default_factory=dict)
    # Stiffener nodes standing off the plating, when a beam has an
    # eccentricity.  Kept apart from ``nodes_of_edge`` because the two answer
    # different questions: a load on a line acts once, at the line; a restraint
    # on a line holds everything there, plating and stiffener alike.
    offset_nodes_of_edge: Dict[int, List[int]] = field(default_factory=dict)
    couplings: Dict[int, Coupling] = field(default_factory=dict)
    grid_of_face: Dict[int, np.ndarray] = field(default_factory=dict)
    elements_of_face: Dict[int, List[int]] = field(default_factory=dict)
    elements_of_edge: Dict[int, List[int]] = field(default_factory=dict)
    # Plate thickness per face, where the generator knows it.  A primitive does;
    # a face meshed from a geometry model gets its thickness from an attribute
    # somewhere else, so this stays empty there.
    thickness_of_face: Dict[int, float] = field(default_factory=dict)

    seeding: Seeding | None = None
    order: str = "linear"

    @property
    def is_quadratic(self) -> bool:
        return self.order == "quadratic"

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_elements(self) -> int:
        return len(self.quads) + len(self.tris) + len(self.beams) + len(self.couplings)

    @property
    def shells(self) -> Dict[int, Tuple[int, ...]]:
        """Every shell element, quadrilateral and triangular alike."""

        return {**self.quads, **self.tris}

    def corners_of(self, element_id: int) -> Tuple[int, ...]:
        """The corner nodes of a shell, whatever its order.

        Three for a triangle, four for a quadrilateral.  Anything that treats a
        shell as a polygon -- normals, drawing, element centroids -- wants the
        corners and not the mid-side nodes.
        """

        if element_id in self.quads:
            return tuple(int(node) for node in self.quads[element_id][:4])
        if element_id in self.tris:
            return tuple(int(node) for node in self.tris[element_id][:3])
        raise MeshError(f"no shell element {element_id}")

    def nodes_on(self, ref: EntityRef) -> List[int]:
        """Every node lying on one geometry entity, boundary included."""

        if ref.kind == "vertex":
            node_id = self.node_of_vertex.get(ref.id)
            return [] if node_id is None else [node_id]
        if ref.kind == "edge":
            return list(self.nodes_of_edge.get(ref.id, ()))
        if ref.kind == "face":
            grid = self.grid_of_face.get(ref.id)
            if grid is not None:
                # Zeros mark the unused centres of quadratic elements.
                return sorted({int(n) for n in grid.ravel().tolist() if n})
            # An imported or generated mesh has no structured grid behind its
            # groups, so fall back to whatever its elements are made of.
            nodes: set[int] = set()
            shells = self.shells
            for element_id in self.elements_of_face.get(ref.id, ()):
                nodes.update(shells.get(element_id, ()))
            return sorted(nodes)
        raise MeshError(f"unknown entity kind {ref.kind!r}")

    def constrained_nodes_on(self, ref: EntityRef) -> List[int]:
        """Every node a restraint on this entity should hold.

        A support applies to a physical location, so it takes the offset
        stiffener nodes as well as the plating.  Clamping the plate edge while
        leaving the stiffener free would be a different structure.
        """

        nodes = list(self.nodes_on(ref))
        if ref.kind == "edge":
            nodes.extend(self.offset_nodes_of_edge.get(ref.id, ()))
        elif ref.kind == "vertex":
            for offset_nodes, edge_nodes in (
                (self.offset_nodes_of_edge.get(edge_id, ()), sequence)
                for edge_id, sequence in self.nodes_of_edge.items()
            ):
                for offset_node, edge_node in zip(offset_nodes, edge_nodes):
                    if edge_node in nodes and offset_node not in nodes:
                        nodes.append(offset_node)
        return sorted(set(nodes))

    def elements_on(self, ref: EntityRef) -> List[int]:
        """Every element belonging to one geometry entity."""

        if ref.kind == "face":
            return list(self.elements_of_face.get(ref.id, ()))
        if ref.kind == "edge":
            return list(self.elements_of_edge.get(ref.id, ()))
        if ref.kind == "vertex":
            return []
        raise MeshError(f"unknown entity kind {ref.kind!r}")

    def node_positions(self) -> np.ndarray:
        """All node coordinates, ordered by node ID."""

        return np.array([self.nodes[key] for key in sorted(self.nodes)])
