from __future__ import annotations
import numpy as np
import pytest
from anygeometry.entities import OrientedEdge
from anygeometry.model import GeometryModel
from anymesher.errors import MeshError
from anymesher.quad.boundary import BoundaryStationRegistry
from anymesher.quad.domain import PlanarQuadDomain
from anymesher.quad.seed import build_planar_quad_seed
from anymesher.triangulation import orient2d
from .fixtures import p01_geometry, p02_geometry


def _signature(geometry: GeometryModel):
    return (
        str(geometry.model_id), int(geometry.revision),
        tuple((i, tuple(map(float, geometry.vertex_position(i)))) for i in sorted(geometry.vertices)),
        tuple((i, geometry.edges[i].start, geometry.edges[i].end) for i in sorted(geometry.edges)),
        tuple(sorted(geometry.faces)),
    )

@pytest.mark.parametrize("h,expected_t3", [(1.0,120),(0.5,480),(0.25,1920)])
def test_p01_target_size_seed_scales_exactly(h: float, expected_t3: int) -> None:
    geometry, face_id = p01_geometry(); before = _signature(geometry)
    domain = PlanarQuadDomain.from_geometry(geometry, face_id)
    registry = BoundaryStationRegistry.for_domain(geometry, domain, h)
    divisions = tuple(registry.edge_divisions(edge) for edge, _ in domain.edge_uses)
    scale = round(1.0/h)
    assert divisions == (10*scale, 6*scale, 10*scale, 6*scale)
    seed = build_planar_quad_seed(geometry, face_id, h, domain=domain, registry=registry)
    assert len(seed.triangulation.triangles) == expected_t3
    assert len(seed.triangulation.triangles)/2 == pytest.approx(60.0/(h*h))
    assert len(seed.state.nodes) > len(seed.boundary_keys)
    assert set(seed.state.cell_kinds.values()) == {"T3"}
    assert all(orient2d(seed.state.position(t[0]), seed.state.position(t[1]), seed.state.position(t[2])) > 0 for t in seed.state.cells.values())
    boundary = {tuple(sorted(map(int,e))) for e in seed.triangulation.boundary_segments}
    assert seed.state.front == frozenset(boundary)
    assert seed.state.protected_edges == frozenset(boundary)
    assert seed.state.protected_nodes == frozenset(seed.station_to_node.values())
    assert seed.state.next_node_id == len(seed.state.nodes)
    assert seed.state.next_cell_id == expected_t3
    assert _signature(geometry) == before


def test_p01_halving_ratios_and_boundary_lengths() -> None:
    seeds=[]
    for h in (1.0,0.5,0.25):
        geometry, face_id = p01_geometry(); seed=build_planar_quad_seed(geometry,face_id,h); seeds.append(seed)
        lengths=[]
        for a,b in seed.triangulation.boundary_segments:
            pa,pb=seed.triangulation.points[int(a)],seed.triangulation.points[int(b)]
            lengths.append(float(np.linalg.norm(pb-pa)))
        assert 0.75*h <= float(np.median(lengths)) <= 1.25*h
        assert max(lengths) <= 1.5*h + 1e-12
    counts=[len(s.triangulation.triangles)/2 for s in seeds]
    assert 3.5 <= counts[1]/counts[0] <= 4.5
    assert 3.5 <= counts[2]/counts[1] <= 4.5


def test_p02_rigid_transform_has_canonical_chart_and_counts() -> None:
    g1,f1=p01_geometry(); g2,f2=p02_geometry()
    d1=PlanarQuadDomain.from_geometry(g1,f1); d2=PlanarQuadDomain.from_geometry(g2,f2)
    assert np.allclose(np.asarray(d1.outer_chart), np.asarray(d2.outer_chart), atol=1e-12)
    s1=build_planar_quad_seed(g1,f1,0.5,domain=d1); s2=build_planar_quad_seed(g2,f2,0.5,domain=d2)
    assert len(s1.state.nodes)==len(s2.state.nodes)
    assert len(s1.state.cells)==len(s2.state.cells)==480
    assert tuple(s1.registry.divisions.values()) == tuple(s2.registry.divisions.values())
    for p in d2.outer_chart:
        assert np.allclose(d2.project(d2.lift(p)), p, atol=1e-12)


def test_station_registry_is_repeatable_and_reverses_shared_edge_identity() -> None:
    geometry, face_id=p01_geometry(); domain=PlanarQuadDomain.from_geometry(geometry,face_id)
    a=BoundaryStationRegistry.for_domain(geometry,domain,0.5); b=BoundaryStationRegistry.for_domain(geometry,domain,0.5)
    edge=domain.edge_uses[0][0]
    assert [s.key for s in a.chain(edge)] == [s.key for s in b.chain(edge)]
    assert [s.key for s in a.chain(edge,False)] == list(reversed([s.key for s in a.chain(edge,True)]))
    assert a.chain(edge)[0].vertex_id is not None and a.chain(edge)[-1].vertex_id is not None
    assert all(s.vertex_id is None for s in a.chain(edge)[1:-1])


def test_registry_shared_edge_is_one_canonical_chain_across_two_faces() -> None:
    g=GeometryModel(); v=g.add_points(((0,0,0),(2,0,0),(2,1,0),(0,1,0),(4,0,0),(4,1,0)))
    e01=g.add_line(v[0],v[1]); shared=g.add_line(v[1],v[2]); e23=g.add_line(v[2],v[3]); e30=g.add_line(v[3],v[0])
    e14=g.add_line(v[1],v[4]); e45=g.add_line(v[4],v[5]); e52=g.add_line(v[5],v[2])
    f1=g.add_face_from_loop((OrientedEdge(e01,True),OrientedEdge(shared,True),OrientedEdge(e23,True),OrientedEdge(e30,True)))
    f2=g.add_face_from_loop((OrientedEdge(e14,True),OrientedEdge(e45,True),OrientedEdge(e52,True),OrientedEdge(shared,False)))
    d1=PlanarQuadDomain.from_geometry(g,f1); d2=PlanarQuadDomain.from_geometry(g,f2)
    reg=BoundaryStationRegistry.for_domains(g,(d1,d2),0.25)
    use1=dict(d1.edge_uses)[shared]; use2=dict(d2.edge_uses)[shared]; assert use1 != use2
    assert [s.key for s in reg.chain(shared,use1)] == list(reversed([s.key for s in reg.chain(shared,use2)]))
    assert len(reg.chain(shared)) > 3


def test_invalid_face_rejects_without_source_mutation() -> None:
    geometry,_=p01_geometry(); before=_signature(geometry)
    with pytest.raises(MeshError): PlanarQuadDomain.from_geometry(geometry,99999)
    assert _signature(geometry)==before
