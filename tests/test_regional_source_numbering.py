import numpy as np
import pytest
from types import SimpleNamespace

from anymesher import (publish_model_closure_mesh_to_source, remap_source_mesh_ids,
                       validate_numbered_source_mesh)
from anymesher.errors import MeshError
from anymesher.material_source_transfer import _validated_curvilinear_reference_uv
from anymesher.serialize import mesh_from_dict, mesh_to_dict
from test_model_closure_source_publication import qualified_closure_publication_operands


def test_curvature_zone_and_retry_diagnostics_are_counts_and_scales():
    from anymesher.regional_source_numbering import _face_reports

    report = {'initial_curvature_size_zones': 3, 'source_cell_resolution': {
        'refinement_retries': 1, 'localized_failure_zones': 2, 'interior_scale': .5}}
    assert _face_reports({7: report}, {1: 101, 2: 203, 3: 307}) == {7: report}
    with pytest.raises(MeshError):
        _face_reports({7: {'initial_curvature_size_zones': -1}}, {})
    report['source_cell_resolution']['node_id'] = 4
    with pytest.raises(MeshError):
        _face_reports({7: report}, {})


def test_automatic_approximation_report_keeps_only_its_geometry_namespace():
    from anymesher.approximation_policy import resolve_geometry_approximation
    from anymesher.regional_source_numbering import _approximation_report
    from test_regional_source_generation import patch

    source, face, _ = patch('nurbs')
    issued = resolve_geometry_approximation(source, (face,)).to_dict()
    copied = _approximation_report(issued)
    assert copied == issued and copied['face_ids'] == [face]
    issued['local_field']['unexpected_node_id'] = 3
    with pytest.raises(MeshError, match='undocumented|unknown|schema|fields'):
        _approximation_report(issued)


def test_curved_recovery_transports_only_source_handles_and_documented_options():
    from anymesher.recovery import _record_curvilinear_recovery
    from anymesher.regional_source_numbering import _recovery_report
    from anymesher.native_v2 import NativeMeshingOptions
    from anymesher.structured import StructuredMeshingOptions

    result = SimpleNamespace(mesh=SimpleNamespace(hybrid_diagnostics={}))
    _record_curvilinear_recovery(result, dict(requested_strategy='auto', reason='curved source',
        geometry_approximation_error=.02, source_faces=[3], nurbs_tensor_faces=[3]), 'native',
        dict(strategy='native', recombine=True, native_options=NativeMeshingOptions(),
             _native_surface_options=StructuredMeshingOptions()))
    report = result.mesh.hybrid_diagnostics['curvilinear_recovery']
    assert _recovery_report(report) == report
    report['native_surface_options']['quality_policy']['element_id'] = 3
    with pytest.raises(MeshError, match='unknown'):
        _recovery_report(report)


def test_structured_source_grid_requires_every_directed_cell_and_identity_chart():
    from anymesher.material_source_transfer import _source_face_grids

    source = SimpleNamespace(id=3)
    prepared = SimpleNamespace(id=7)
    receipt = SimpleNamespace(source_to_prepared_faces=[(source, [prepared])], chart_correspondence=[
        (source, prepared, ((1, 0, 0), (0, 1, 0)))])
    grid = np.asarray([[10, 90], [18, 27], [81, 54]])
    mesh = SimpleNamespace(grid_of_face={7: grid}, block_grids_of_face={},
                           elements_of_face={7: [99, 70]}, quads={99: (10, 18, 27, 90), 70: (18, 81, 54, 27)})
    np.testing.assert_array_equal(_source_face_grids(mesh, receipt)[3], grid)
    for bad in (grid.T, grid[:2], grid[::-1], grid.astype(float)):
        mesh.grid_of_face[7] = bad
        with pytest.raises(MeshError, match='structured'):
            _source_face_grids(mesh, receipt)
    mesh.grid_of_face[7] = grid
    receipt.chart_correspondence = [(source, prepared, ((0, 1, 0), (1, 0, 0)))]
    with pytest.raises(MeshError, match='structured'):
        _source_face_grids(mesh, receipt)


def test_live_curve_number_transport_preserves_cells_stations_and_physical_normals():
    source, closure, _, mesh, token = qualified_closure_publication_operands()
    publish_model_closure_mesh_to_source(source, closure, mesh, qualification=token)
    before = mesh_to_dict(mesh)
    original_uv = _validated_curvilinear_reference_uv(source, mesh)
    assert original_uv and set(original_uv) == set(mesh.tris)
    nodes = {n: 2**53 + 5*n for n in mesh.nodes}
    elements = {e: 2**53 + 9*e for e in mesh.tris}
    numbered = remap_source_mesh_ids(mesh, nodes, elements, source_geometry=source)
    validate_numbered_source_mesh(numbered, source_geometry=source)
    assert mesh_to_dict(mesh) == before
    assert numbered.tris == {elements[e]: tuple(nodes[n] for n in row) for e,row in mesh.tris.items()}
    assert _validated_curvilinear_reference_uv(source, numbered) == {elements[e]: v for e,v in original_uv.items()}
    assert set(numbered.structural_preparation['qualified_s3']['element_ids']) == set(elements.values())
    assert [(s.key, s.node_id) for s in numbered.boundary_registry.entries()] == [
        (s.key, nodes[s.node_id]) for s in mesh.boundary_registry.entries()]
    renumbered = remap_source_mesh_ids(numbered, {v:v+1 for v in nodes.values()},
                                       {v:v+1 for v in elements.values()}, source_geometry=source)
    assert set(_validated_curvilinear_reference_uv(source, renumbered)) == set(renumbered.tris)
    with pytest.raises(MeshError, match='live material source authority'):
        validate_numbered_source_mesh(mesh_from_dict(mesh_to_dict(numbered)), source_geometry=source)
    with pytest.raises(MeshError):
        validate_numbered_source_mesh(numbered, source_geometry=closure.working_model)
    n = next(iter(numbered.nodes))
    numbered.nodes[n] = numbered.nodes[n] + np.array([.001,0.,0.])
    with pytest.raises(MeshError, match='changed'):
        validate_numbered_source_mesh(numbered, source_geometry=source)
    validate_numbered_source_mesh(mesh, source_geometry=source)
    mesh.structural_preparation['qualified_s3']['element_ids'] = []
    with pytest.raises(MeshError, match='reports changed'):
        validate_numbered_source_mesh(renumbered, source_geometry=source)
