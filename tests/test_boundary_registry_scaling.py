import numpy as np
import pytest
from anygeometry import GeometryModel
from anymesher.boundary import GlobalEdgeBoundaryRegistry
from anymesher.errors import MeshError
from anymesher.meshing_view import GeometryMeshingView


def test_edge_local_lookup_preserves_exact_stations_updates_and_failed_writes():
    geometry = GeometryModel()
    points = geometry.add_points(((0,0,0),(1,0,0),(2,0,0)))
    edges = [geometry.add_line(points[i],points[i+1]) for i in range(2)]
    registry = GlobalEdgeBoundaryRegistry(GeometryMeshingView(geometry))
    values = (1., np.nextafter(.5, 1.), .5, 0.)
    for edge in reversed(edges):
        for value in values:
            registry.register(edge,value,owner='first')
    expected = registry.entries()
    assert [e.key for e in expected] == sorted(e.key for e in expected)
    class NoGlobalScan(dict):
        def __iter__(self):
            raise AssertionError('edge lookup scanned global registry')
    registry._entries = NoGlobalScan(registry._entries)
    edge = edges[0]
    assert registry.parameters(edge) == tuple(sorted(values))
    assert registry.entries(edge) == tuple(e for e in expected if e.key.edge_id == edge)
    updated = registry.register(edge,.5,node_id=10,owner='second')
    assert updated.owners == ('first','second')
    assert registry.require(edge,.5) is updated
    assert next(e for e in registry.entries(edge) if e.key.parameter == .5) is updated
    with pytest.raises(MeshError,match='already owns'):
        registry.register(edge,.5,node_id=11)
    with pytest.raises(MeshError,match='inconsistent points'):
        registry.register(edge,.5,(.7,0,0))
    assert registry.require(edge,.5) is updated
    assert next(e for e in registry.entries(edge) if e.key.parameter == .5) is updated
    assert registry.parameters(999) == ()
    assert registry.entries(999) == ()


def test_unstructured_transition_seeding_retains_shared_identity_and_pins():
    from anymesher.seeding import solve_seeding, SeedingConflict
    model=GeometryModel()
    face=model.add_plate(model.add_points(((0,0,0),(8,0,0),(8,.002,0),(0,.002,0))))
    edges=[use.edge for use in model.faces[face].loop]
    local=solve_seeding(model,target_size=.5,unstructured_face_ids=(face,))
    assert local[edges[0]] == local[edges[2]] == 16
    assert local[edges[1]] == local[edges[3]] == 1
    pinned=solve_seeding(model,target_size=.5,unstructured_face_ids=(face,),overrides={edges[0]:23})
    assert pinned[edges[0]] == 23
    assert pinned[edges[2]] == 16
    assert len(local.divisions)==len(model.edges)
    with pytest.raises(SeedingConflict,match='unknown faces'):
        solve_seeding(model,target_size=.5,unstructured_face_ids=(999,))


def test_native_face_retains_mapped_neighbour_shared_edge_constraint():
    from anymesher.seeding import solve_seeding
    model=GeometryModel()
    v=model.add_points(((0,0,0),(2,0,0),(2,1,0),(0,1,0),(4,0,0),(4,1,0)))
    mapped=model.add_plate((v[0],v[1],v[2],v[3]))
    from anygeometry.entities import OrientedEdge
    joint=model.faces[mapped].loop[1]
    outer=[model.add_line(v[a],v[b]) for a,b in ((1,4),(4,5),(5,2))]
    native=model.add_face_from_loop(tuple(OrientedEdge(e,True) for e in outer)+(OrientedEdge(joint.edge,not joint.forward),),corners=(0,1,2,3))
    shared=set(u.edge for u in model.faces[mapped].loop)&set(u.edge for u in model.faces[native].loop)
    assert len(shared)==1
    edge=shared.pop()
    mapped_opposite=model.faces[mapped].loop[3].edge
    result=solve_seeding(model,target_size=.5,unstructured_face_ids=(native,),overrides={mapped_opposite:7})
    assert result[edge]==result[mapped_opposite]==7
    uses=[next(u for u in model.faces[f].loop if u.edge==edge) for f in (mapped,native)]
    assert uses[0].forward != uses[1].forward
    view=GeometryMeshingView(model)
    registry=GlobalEdgeBoundaryRegistry(view)
    registry.register_many(edge,np.linspace(0.,1.,result[edge]+1),node_ids=range(1,result[edge]+2))
    assert [entry.node_id for entry in registry.entries(edge)]==list(range(1,9))
