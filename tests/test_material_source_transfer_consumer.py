"""Actual owner cell receipts must bind actual live mesh coverage and callbacks."""
from copy import deepcopy
import pytest
import anygeometry as owner
from anymesher import validate_material_source_transfer_for_mesh
from anymesher.errors import MeshError
from anymesher.hybrid import generate_hybrid_mesh_result
from test_nurbs_curvilinear_consumer import patch_model, options


def prepared_mesh(*, recombine=True):
    source, face, _ = patch_model()
    prepared = owner.clone_prepared_geometry(source)
    result = generate_hybrid_mesh_result(prepared, mutation_policy='working_copy',
                                        recombine=recombine, **options())
    return source, prepared, result.mesh, face


def test_actual_mesh_source_transfer_uses_owner_and_preserves_both_models():
    source, prepared, mesh, face = prepared_mesh()
    before = owner.to_dict(source), owner.to_dict(prepared)
    receipt = validate_material_source_transfer_for_mesh(source, prepared, mesh)
    assert receipt.source_to_prepared_faces == (
        (source.handle('face', face), (prepared.handle('face', face),)),)
    assert tuple(cell.node_ids for cell in receipt.prepared_cells)
    assert all(error == .01 for _face, _nodes, error in receipt.surface_errors)
    assert (owner.to_dict(source), owner.to_dict(prepared)) == before


def test_copied_mesh_cannot_inherit_generation_source_transfer():
    source, prepared, mesh, _ = prepared_mesh()
    with pytest.raises(MeshError, match='generation-owned'):
        validate_material_source_transfer_for_mesh(source, prepared, deepcopy(mesh))


@pytest.mark.parametrize('change', ['empty_registry', 'edge_chain', 'registry_point'])
def test_source_transfer_requires_generation_boundary_station_authority(change):
    from anymesher.boundary import GlobalEdgeBoundaryRegistry
    from anymesher.meshing_view import GeometryMeshingView
    source, prepared, mesh, _ = prepared_mesh()
    if change == 'empty_registry':
        mesh.boundary_registry = GlobalEdgeBoundaryRegistry(GeometryMeshingView(prepared))
    elif change == 'edge_chain':
        edge = next(iter(mesh.nodes_of_edge))
        mesh.nodes_of_edge[edge] = list(reversed(mesh.nodes_of_edge[edge]))
    else:
        entry = next(iter(mesh.boundary_registry.entries()))
        object.__setattr__(entry, 'point', tuple(float(value)+.1 for value in entry.point))
    with pytest.raises(MeshError, match='boundary stations differ'):
        validate_material_source_transfer_for_mesh(source, prepared, mesh)


def test_source_transfer_true_cancellation_preserves_models_and_mesh():
    from anymesher.serialize import mesh_to_dict
    source, prepared, mesh, _ = prepared_mesh()
    before = owner.to_dict(source), owner.to_dict(prepared), mesh_to_dict(mesh)
    with pytest.raises((MeshError, owner.GeometryError), match='cancelled'):
        validate_material_source_transfer_for_mesh(source, prepared, mesh,
            cancellation_check=lambda _phase: True)
    assert (owner.to_dict(source), owner.to_dict(prepared), mesh_to_dict(mesh)) == before


@pytest.mark.parametrize('change', ['membership', 'xyz', 'callback_xyz'])
def test_mesh_changes_cannot_reuse_owner_source_cells(change):
    source, prepared, mesh, face = prepared_mesh()
    before = owner.to_dict(source), owner.to_dict(prepared)
    node = next(iter(mesh.nodes))
    def move():
        xyz = tuple(mesh.nodes[node])
        mesh.nodes[node] = (xyz[0]+.1, xyz[1], xyz[2])
    callback = None
    if change == 'membership':
        mesh.elements_of_face[face].pop()
    elif change == 'xyz':
        move()
    else:
        def callback(_phase):
            move()
            return False
    with pytest.raises(MeshError, match='coverage|XYZ|changed during qualification'):
        validate_material_source_transfer_for_mesh(source, prepared, mesh,
            cancellation_check=callback)
    assert (owner.to_dict(source), owner.to_dict(prepared)) == before
