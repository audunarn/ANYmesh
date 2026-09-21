"""Canonical target-size boundary stations for planar quad domains."""
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping
from anygeometry.model import GeometryModel
from ..errors import MeshError
from ..refinement import SizeField
from ..seeding import edge_distribution, solve_seeding
from .domain import PlanarQuadDomain, Vec3

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
    ) -> "BoundaryStationRegistry":
        domains = tuple(domains)
        if not domains:
            raise MeshError("boundary station registry needs at least one domain")
        for domain in domains:
            domain.assert_current(geometry)
        edge_ids = sorted({edge_id for d in domains for edge_id, _ in d.edge_uses})
        field = SizeField(geometry, float(target_size))
        seeding = solve_seeding(geometry, size_field=field, edge_ids=edge_ids)
        chains: dict[int, tuple[BoundaryStation, ...]] = {}
        for edge_id in edge_ids:
            edge = geometry.edges[edge_id]
            divisions = int(seeding[edge_id])
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
            {eid: int(seeding[eid]) for eid in edge_ids}, chains,
        )
        for domain in domains:
            domain.assert_current(geometry)
        return result

    @classmethod
    def for_domain(cls, geometry: GeometryModel, domain: PlanarQuadDomain, target_size: float) -> "BoundaryStationRegistry":
        return cls.for_domains(geometry, (domain,), target_size)

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
