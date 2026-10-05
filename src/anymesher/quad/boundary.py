"""Canonical target-size boundary stations for planar quad domains."""
from __future__ import annotations
from dataclasses import dataclass
from math import ceil, floor, isfinite
from numbers import Integral
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence
from anygeometry.model import GeometryModel
from anygeometry.curves import Arc
from ..errors import MeshError
from ..refinement import SizeField
from ..seeding import (
    _apply_face_boundary_minimums, _apply_face_shape_minimums,
    edge_demand, edge_distribution, solve_seeding,
)
from .domain import CylindricalQuadDomain, PlanarQuadDomain, Vec3

@dataclass(frozen=True, order=True)
class BoundaryStationKey:
    kind: str
    entity_id: int
    ordinal: int = 0
    divisions: int = 0

@dataclass(frozen=True)
class BoundaryStation:
    key: BoundaryStationKey
    parameter: float
    vertex_id: int | None
    position: Vec3

@dataclass(frozen=True)
class BoundaryStationRegistry:
    model_id: str
    revision: int
    target_size: float
    divisions: Mapping[int, int]
    chains: Mapping[int, tuple[BoundaryStation, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "divisions", MappingProxyType(dict(self.divisions)))
        object.__setattr__(self, "chains", MappingProxyType(dict(self.chains)))

    @classmethod
    def for_domains(
        cls,
        geometry: GeometryModel,
        domains: Iterable[PlanarQuadDomain],
        target_size: float,
        *,
        size_field: SizeField | None = None,
        overrides: Mapping[int, int] | None = None,
        _canonical_stations: Mapping[int, Sequence[float]] | None = None,
        _independent_refined_counts: bool = False,
        _adaptive_independent_counts: bool = False,
    ) -> "BoundaryStationRegistry":
        domains = tuple(domains)
        if not domains:
            raise MeshError("boundary station registry needs at least one domain")
        for domain in domains:
            domain.assert_current(geometry)
        edge_ids = sorted({edge_id for d in domains for loop in (d.edge_uses, *d.hole_edge_uses) for edge_id, _ in loop})
        field = size_field if size_field is not None else SizeField(geometry, float(target_size))
        if abs(float(field.target_size) - float(target_size)) > 1.0e-12 * max(1.0, abs(float(target_size))):
            raise MeshError("provided size field target_size differs from registry target_size")
        selected_overrides = {
            int(edge_id): count for edge_id, count in (overrides or {}).items()
            if int(edge_id) in edge_ids
        }
        for edge_id, count in selected_overrides.items():
            if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
                raise MeshError(f"edge {edge_id} override must be a positive integer division count")
        # Private opt-in: pin exact nonuniform station parameters on selected
        # edges (owner-certified member carrier stations).  Absent the option,
        # every route keeps its existing seeding behaviour unchanged.
        canonical_stations: dict[int, tuple[float, ...]] = {}
        for raw_edge_id, raw_parameters in (_canonical_stations or {}).items():
            edge_id = int(raw_edge_id)
            if edge_id not in edge_ids:
                raise MeshError(
                    f"canonical station override references unknown edge {edge_id}"
                )
            parameters = tuple(float(value) for value in tuple(raw_parameters))
            if (
                len(parameters) < 2
                or parameters[0] != 0.0
                or parameters[-1] != 1.0
                or any(not isfinite(value) for value in parameters)
                or any(not 0.0 <= value <= 1.0 for value in parameters)
                or any(second <= first for first, second in zip(parameters, parameters[1:]))
            ):
                raise MeshError(
                    f"edge {edge_id} canonical stations must be increasing finite "
                    "parameters from 0 to 1"
                )
            if edge_id in selected_overrides and int(selected_overrides[edge_id]) != len(parameters) - 1:
                raise MeshError(
                    f"edge {edge_id} division override conflicts with its canonical stations"
                )
            canonical_stations[edge_id] = parameters
        curved_edges = {
            edge_id for domain in domains if isinstance(domain, CylindricalQuadDomain)
            for loop in (domain.edge_uses, *domain.hole_edge_uses)
            for edge_id, _ in loop
            if isinstance(geometry.edges[edge_id].curve, Arc)
        }
        # A source arc just longer than one requested interval must advance
        # to two spans. Rounded edge demand otherwise holds the same Q8 arc
        # approximation at adjacent resolutions (the RA1 deck plateau).
        curved_minimums = {}
        for edge_id in curved_edges:
            if edge_id in selected_overrides:
                continue
            demand = edge_demand(geometry, edge_id, field)
            if 1e-12 < demand-floor(demand) <= 0.1:
                curved_minimums[edge_id] = ceil(demand - 1e-12)
        if _adaptive_independent_counts:
            # Irregular quad fronts do not require opposite mapped sides to
            # carry equal counts. Size each source edge in physical units and
            # retain one canonical chain for every incident face.
            counts = {
                edge_id: max(1, ceil(edge_demand(geometry, edge_id, field) - 1e-12),
                             curved_minimums.get(edge_id, 1))
                for edge_id in edge_ids
            }
            counts.update({edge_id: int(count) for edge_id, count in selected_overrides.items()})
        elif _independent_refined_counts and not field.is_uniform:
            # The quad-first seed can transition between unequal opposite
            # boundary counts. Mapped-face equalities otherwise carry a local
            # shared-edge refinement all the way to remote exterior edges.
            counts = {
                edge_id: max(1, int(round(edge_demand(geometry, edge_id, field))),
                             curved_minimums.get(edge_id, 1))
                for edge_id in edge_ids
            }
            if curved_minimums:
                # Match uniform face-local station balance around a newly
                # split narrow source arc, without propagating a local zone's
                # counts around the whole connected assembly.
                affected = [geometry.faces[domain.face_id] for domain in domains
                            if isinstance(domain, CylindricalQuadDomain)
                            and any(edge_id in curved_minimums
                                    for edge_id, _ in domain.edge_uses)]
                demands = {edge_id: edge_demand(geometry, edge_id, field)
                           for edge_id in edge_ids}
                _apply_face_boundary_minimums(
                    affected, desired=counts, demands=demands,
                    maximum_step_ratio=1.2,
                )
                _apply_face_shape_minimums(affected, desired=counts, demands=demands)
            counts.update({edge_id: int(count) for edge_id, count in selected_overrides.items()})
        else:
            seeding = solve_seeding(geometry, size_field=field, edge_ids=edge_ids,
                                    overrides=selected_overrides,
                                    minimum_divisions=curved_minimums)
            counts = dict(seeding.divisions)
            # Mapped opposite-side equalities only make sense when each mapped
            # side is a single source edge.  A face whose declared corners
            # group several edges into one side (e.g. an L-shaped polyline
            # face) would otherwise force one short edge to carry the whole
            # opposite chain, fanning thin slivers off it.  Relax that only on
            # edges owned exclusively by quad-first faces, so any legacy
            # neighbour keeps its own seeding contract.
            quad_faces = {int(domain.face_id) for domain in domains}
            composite = {face_id for face_id in quad_faces
                         if _has_composite_mapped_side(geometry, face_id)}
            for edge_id in edge_ids:
                if edge_id in selected_overrides:
                    continue
                users = {int(face) for face in geometry.faces_using_edge(edge_id)}
                if users and users <= quad_faces and users & composite:
                    counts[edge_id] = max(
                        1, int(round(edge_demand(geometry, edge_id, field))),
                        curved_minimums.get(edge_id, 1),
                    )
        for edge_id, parameters in canonical_stations.items():
            counts[edge_id] = len(parameters) - 1
        chains: dict[int, tuple[BoundaryStation, ...]] = {}
        for edge_id in edge_ids:
            edge = geometry.edges[edge_id]
            divisions = int(counts[edge_id])
            if edge_id in canonical_stations:
                params = canonical_stations[edge_id]
            else:
                interior = tuple(float(v) for v in edge_distribution(geometry, edge_id, divisions, field))
                params = (0.0, *interior, 1.0)
            xyz = geometry.sample_edge(edge_id, params)
            stations: list[BoundaryStation] = []
            for ordinal, (parameter, point) in enumerate(zip(params, xyz)):
                if ordinal == 0:
                    key = BoundaryStationKey("vertex", int(edge.start))
                    vertex_id = int(edge.start)
                elif ordinal == divisions:
                    key = BoundaryStationKey("vertex", int(edge.end))
                    vertex_id = int(edge.end)
                else:
                    key = BoundaryStationKey("edge", int(edge_id), ordinal, divisions)
                    vertex_id = None
                stations.append(BoundaryStation(key, parameter, vertex_id, tuple(map(float, point))))
            chains[edge_id] = tuple(stations)
        result = cls(
            str(geometry.model_id), int(geometry.revision), float(target_size),
            {eid: int(counts[eid]) for eid in edge_ids}, chains,
        )
        for domain in domains:
            domain.assert_current(geometry)
        return result

    @classmethod
    def for_domain(
        cls,
        geometry: GeometryModel,
        domain: PlanarQuadDomain,
        target_size: float,
        *,
        size_field: SizeField | None = None,
    ) -> "BoundaryStationRegistry":
        return cls.for_domains(geometry, (domain,), target_size, size_field=size_field)

    def loop_stations(self, domain: PlanarQuadDomain, loop_uses: tuple[tuple[int, bool], ...]) -> tuple[BoundaryStation, ...]:
        self._check_domain(domain)
        out: list[BoundaryStation] = []
        for edge_id, forward in loop_uses:
            if int(edge_id) not in self.chains:
                raise MeshError("boundary registry is missing a loop edge")
            chain = self.chain(int(edge_id), forward)
            out.extend(chain[:-1])
        if len(out) < 3:
            raise MeshError("loop boundary has fewer than three stations")
        return tuple(out)

    def outer_stations(self, domain: PlanarQuadDomain) -> tuple[BoundaryStation, ...]:
        return self.loop_stations(domain, domain.edge_uses)

    def hole_stations(self, domain: PlanarQuadDomain, index: int = 0) -> tuple[BoundaryStation, ...]:
        index = int(index)
        if index < 0 or index >= len(domain.hole_edge_uses):
            raise MeshError(f"domain has no hole {index}")
        return self.loop_stations(domain, domain.hole_edge_uses[index])

    def all_loop_stations(self, domain: PlanarQuadDomain) -> tuple[BoundaryStation, ...]:
        out: list[BoundaryStation] = []
        out.extend(self.outer_stations(domain))
        for index in range(len(domain.hole_edge_uses)):
            out.extend(self.hole_stations(domain, index))
        return tuple(out)

    def _check_domain(self, domain: PlanarQuadDomain) -> None:
        if domain.model_id != self.model_id or domain.revision != self.revision:
            raise MeshError("domain and boundary registry are from different geometry revisions")

    def assert_current(self, geometry: GeometryModel) -> None:
        if str(geometry.model_id) != self.model_id or int(geometry.revision) != self.revision:
            raise MeshError("source geometry changed after boundary-station capture")

    def edge_divisions(self, edge_id: int) -> int:
        return int(self.divisions[int(edge_id)])

    def chain(self, edge_id: int, forward: bool = True) -> tuple[BoundaryStation, ...]:
        chain = self.chains[int(edge_id)]
        return chain if forward else tuple(reversed(chain))

    def face_stations(self, domain: PlanarQuadDomain) -> tuple[BoundaryStation, ...]:
        if domain.model_id != self.model_id or domain.revision != self.revision:
            raise MeshError("domain and boundary registry are from different geometry revisions")
        out: list[BoundaryStation] = []
        for edge_id, forward in domain.edge_uses:
            chain = self.chain(edge_id, forward)
            out.extend(chain[:-1])
        if len(out) < 3:
            raise MeshError("face boundary has fewer than three stations")
        return tuple(out)


def _has_composite_mapped_side(geometry: GeometryModel, face_id: int) -> bool:
    """True when solve_seeding would equate a side made of several edges."""
    face = geometry.faces[int(face_id)]
    if len(face.corners) != 4 or face.holes:
        return False
    return any(len(side) != 1 for side in face.sides())
