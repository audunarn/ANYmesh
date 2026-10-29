"""Vertex supports follow declared offset pairs, including sparse edge coverage."""
import numpy as np
import pytest
from anygeometry import EntityRef
from anymesher.mesh import Mesh, Coupling
from anymesher.errors import MeshError
from anymesher.serialize import mesh_from_dict, mesh_to_dict


def mesh(offsets, pairs=()):
    return Mesh(nodes={n:np.array((float(n),0.,0.)) for n in (1,2,3,11,12,13,22)},
        node_of_vertex={101:1,102:2,103:3},nodes_of_edge={7:[1,2,3]},
        offset_nodes_of_edge={7:list(offsets)},
        couplings={i:Coupling.node_to_node(beam,axis,(0,0,1)) for i,(beam,axis) in enumerate(pairs)})


@pytest.mark.parametrize('offsets,pairs,expected',[
    ((11,12),((11,1),(12,2)),({1,11},{2,12},{3})),
    ((12,13),((12,2),(13,3)),({1},{2,12},{3,13})),
    ((11,12,22,13),((11,1),(12,2),(22,2),(13,3)),({1,11},{2,12,22},{3,13})),
])
def test_sparse_first_later_child_and_branch_vertex_scope_roundtrip(offsets,pairs,expected):
    original=mesh(offsets,pairs)
    for candidate in (original,mesh_from_dict(mesh_to_dict(original))):
        for vertex,wanted in zip((101,102,103),expected):
            assert set(candidate.constrained_nodes_on(EntityRef('vertex',vertex)))==wanted
        assert set(candidate.constrained_nodes_on(EntityRef('edge',7)))=={1,2,3,*offsets}


def test_aligned_legacy_fallback_and_exact_pair_precedence():
    candidate=mesh((11,12,13))
    assert candidate.constrained_nodes_on(EntityRef('vertex',101))==[1,11]
    assert candidate.constrained_nodes_on(EntityRef('vertex',103))==[3,13]
    candidate.couplings={1:Coupling.node_to_node(11,3,(0,0,1))}
    assert candidate.constrained_nodes_on(EntityRef('vertex',101))==[1]
    assert candidate.constrained_nodes_on(EntityRef('vertex',103))==[3,11,13]


def test_sparse_undefined_pairs_refuse_but_edge_scope_stays_valid():
    candidate=mesh((12,13))
    with pytest.raises(MeshError,match='no exact node-to-node coupling'):
        candidate.constrained_nodes_on(EntityRef('vertex',101))
    assert candidate.constrained_nodes_on(EntityRef('edge',7))==[1,2,3,12,13]


def test_interpolation_is_not_a_vertex_pair():
    candidate=mesh((11,12,13))
    candidate.couplings={1:Coupling(11,(1,2),(.5,.5),(0,0,1))}
    assert candidate.constrained_nodes_on(EntityRef('vertex',101))==[1]
    candidate.offset_nodes_of_edge[7]=[11]
    with pytest.raises(MeshError,match='no exact node-to-node coupling'):
        candidate.constrained_nodes_on(EntityRef('vertex',101))
