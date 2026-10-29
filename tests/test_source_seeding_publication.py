"""Algebra/association unit regressions; these helpers grant no owner authority."""
from fractions import Fraction as Q
from types import SimpleNamespace as NS
from copy import deepcopy
import pytest
from anymesher.seeding import Seeding
from anymesher.errors import MeshError
from anymesher._source_seeding_publication import project_source_seeding
from anymesher.refinement import SizeField


def context():
    source = NS(model_id='source', edges={10:None,20:None})
    parent = lambda key: NS(id=key,model_id='source')
    maps = {1:(parent(10),Q(0),Q(1,2)),2:(parent(10),Q(1,2),Q(1)),
            3:(parent(20),Q(0),Q(1,2)),4:(parent(20),Q(1,2),Q(1))}
    chains = {1:[100,101,102],2:[102,103,104,105],
              3:[200,201,202],4:[202,203,204,205]}
    entries = {edge:[NS(node_id=node,key=NS(parameter=Q(i,len(chain)-1)))
                    for i,node in enumerate(chain)] for edge,chain in chains.items()}
    mesh = NS(seeding=Seeding({1:2,2:3,3:2,4:3},classes={1:1,2:2,3:1,4:2}),
              nodes_of_edge=chains,nodes={n:(0,0,0) for chain in chains.values() for n in chain},
              boundary_registry=NS(entries=lambda edge: entries[edge]))
    source_nodes = {10:[100,101,102,103,104,105],20:[200,201,202,203,204,205]}
    return source,mesh,NS(source_trace_ownership=()),maps,source_nodes,entries


def test_identical_formal_partial_class_sums_share_deterministic_source_class():
    source,mesh,receipt,maps,nodes,_entries = context()
    original = deepcopy(mesh.seeding)
    projected = project_source_seeding(source,mesh,receipt,maps,nodes)
    assert projected.divisions == {10:5,20:5}
    assert projected.classes == {10:10,20:10}
    assert mesh.seeding == original and projected is not mesh.seeding


def test_equal_numeric_totals_do_not_merge_different_class_expressions():
    source,mesh,receipt,maps,nodes,_entries = context()
    mesh.seeding.classes = {1:1,2:2,3:3,4:4}
    projected = project_source_seeding(source,mesh,receipt,maps,nodes)
    assert projected.divisions == {10:5,20:5}
    assert projected.classes == {10:10,20:20}


def test_multiplicity_preserved_and_adaptive_station_counts_are_not_seed_counts():
    source,mesh,receipt,maps,nodes,_entries = context()
    mesh.seeding.classes = {1:1,2:1,3:1,4:1}
    mesh.seeding.divisions = {1:2,2:2,3:2,4:2}
    projected = project_source_seeding(source,mesh,receipt,maps,nodes)
    assert projected.divisions == {10:4,20:4}  # actual station segments remain five
    assert projected.classes == {10:10,20:10}


@pytest.mark.parametrize('attack',('unknown','missing','count','bool','infinite','class_ref',
                                  'partition','registry','joint','source_chain'))
def test_invalid_projection_refuses_before_mutating_original_seeding(attack):
    source,mesh,receipt,maps,nodes,entries = context()
    if attack == 'unknown':
        mesh.seeding.divisions[99] = 2
        mesh.seeding.classes[99] = 99
    elif attack == 'missing':
        del mesh.seeding.classes[2]
    elif attack == 'count':
        mesh.seeding.divisions[3] = 7
    elif attack == 'bool':
        mesh.seeding.divisions[1] = True
    elif attack == 'infinite':
        mesh.seeding.divisions[1] = float('inf')
    elif attack == 'class_ref':
        mesh.seeding.classes[2] = 99
    elif attack == 'partition':
        parent,_lo,hi = maps[2]
        maps[2] = parent,Q(3,4),hi
    elif attack == 'registry':
        entries[1][1].node_id = 999
    elif attack == 'joint':
        mesh.nodes_of_edge[2][0] = 999
        mesh.nodes[999] = (0,0,0)
        entries[2][0].node_id = 999
    elif attack == 'source_chain':
        nodes[10] = nodes[10][:-1]
    original = deepcopy(mesh.seeding)
    with pytest.raises(MeshError):
        project_source_seeding(source,mesh,receipt,maps,nodes)
    assert mesh.seeding == original


def test_uniform_size_field_reconstructed_and_references_refused():
    source,mesh,receipt,maps,nodes,_entries = context()
    mesh.seeding.size_field = SizeField(source,.25)
    original = mesh.seeding.size_field
    projected = project_source_seeding(source,mesh,receipt,maps,nodes)
    assert projected.size_field is not original
    assert type(projected.size_field) is SizeField and projected.size_field.target_size == .25
    original.edge_sizes[1] = .125
    with pytest.raises(MeshError,match='refinement references'):
        project_source_seeding(source,mesh,receipt,maps,nodes)
    mesh.seeding.size_field = NS(target_size=.25)
    with pytest.raises(MeshError,match='refinement references'):
        project_source_seeding(source,mesh,receipt,maps,nodes)


def test_reversed_child_parameterization_preserves_counts_and_joint_identity():
    source,mesh,receipt,maps,nodes,entries = context()
    parent,lo,hi = maps[2]
    maps[2] = parent,hi,lo
    mesh.nodes_of_edge[2].reverse()
    entries[2] = [NS(node_id=node,key=NS(parameter=Q(i,3)))
                  for i,node in enumerate(mesh.nodes_of_edge[2])]
    projected = project_source_seeding(source,mesh,receipt,maps,nodes)
    assert projected.divisions == {10:5,20:5}
    assert projected.classes == {10:10,20:10}


def test_empty_legacy_classes_remain_empty_without_inventing_equivalences():
    source,mesh,receipt,maps,nodes,_entries = context()
    mesh.seeding.classes = {}
    projected = project_source_seeding(source,mesh,receipt,maps,nodes)
    assert projected.divisions == {10:5,20:5}
    assert projected.classes == {}
    assert mesh.seeding.classes == {}


def test_equal_totals_and_class_support_do_not_merge_different_multiplicities():
    source,mesh,receipt,_maps,_nodes,_entries = context()
    maps,chains,source_nodes = {},{},{}
    for parent_id,edges,start in ((10,(1,2,5),100),(20,(3,4,6),200)):
        source_nodes[parent_id] = list(range(start,start+7))
        for index,edge in enumerate(edges):
            maps[edge] = NS(id=parent_id,model_id='source'),Q(index,3),Q(index+1,3)
            chains[edge] = list(range(start+2*index,start+2*index+3))
    mesh.seeding = Seeding({edge:2 for edge in maps},classes={1:1,2:1,5:4,3:1,4:4,6:4})
    mesh.nodes_of_edge = chains
    mesh.nodes = {node:(0,0,0) for chain in chains.values() for node in chain}
    entries = {edge:[NS(node_id=node,key=NS(parameter=Q(i,2)))
                    for i,node in enumerate(chain)] for edge,chain in chains.items()}
    mesh.boundary_registry = NS(entries=lambda edge:entries[edge])
    projected = project_source_seeding(source,mesh,receipt,maps,source_nodes)
    assert projected.divisions == {10:6,20:6}
    # Both use exactly variables1 and4. Multiplicities2/1 versus1/2 differ.
    assert projected.classes == {10:10,20:20}
