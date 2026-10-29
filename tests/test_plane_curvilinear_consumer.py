"""Plane section boundaries must use the same owner cell authority as NURBS."""
from dataclasses import replace
import numpy as np
import pytest
import anygeometry as owner
from anygeometry.nurbs import NurbsSurface
from anygeometry.support_curves import PlaneNurbsCurve
from anymesher.errors import MeshError
from anymesher.native_v2 import NativeMeshingOptions
from anymesher._material_region_binding import (
    MaterialRegionBinding, face_requires_curvilinear, prepare_material_regions,
    region_requires_curvilinear,
)


def plane_disk(*, basis=((1., 0., 0.), (0., 1., 0.))):
    model = owner.GeometryModel()
    heights = (1., -1., 1.)
    surface = NurbsSurface(2, 2, (0., 0., 0., 1., 1., 1.), (0., 0., 0., 1., 1., 1.),
        tuple(tuple((i-1., j-1., heights[i]+heights[j]) for j in range(3)) for i in range(3)))
    plane = owner.Plane((0., 0., .5), *basis)
    curve = PlaneNurbsCurve.from_plane(plane, surface)
    points = [model.add_point(*curve.evaluate(i/4)) for i in range(4)]
    edges = [model.add_curve(points[i], points[(i+1)%4],
        replace(curve, domain=(i/4, (i+1)/4))) for i in range(4)]
    face = model.add_face(edges, surface=plane)
    model.add_sheet((face,))
    return model, face, edges


def test_implicit_plane_selection_requires_owner_receipts_and_explicit_error():
    model, face, _edges = plane_disk()
    before = owner.to_dict(model)
    assert face_requires_curvilinear(model, face)
    options = NativeMeshingOptions(point_placement='frontal_delaunay', metric_mode='isotropic_spatial')
    with pytest.raises(MeshError, match='explicit geometry_approximation_error'):
        prepare_material_regions(model, (face,), {face:(face,)}, order='linear', native_options=options)
    bindings = prepare_material_regions(model, (face,), {face:(face,)}, order='linear',
        native_options=options, geometry_approximation_error=.01)
    assert bindings[face].requires_curvilinear
    assert region_requires_curvilinear(bindings[face].region)
    assert owner.to_dict(model) == before


@pytest.mark.parametrize('basis', [
    ((1., 0., 0.), (0., 1., 0.)),
    ((2., 0., 0.), (.5, .25, 0.)),
])
def test_implicit_plane_metric_uses_real_owner_anchor_and_literal_world_map(basis):
    from anymesher._analytic_metric_chart import AnalyticMetricChart
    model, face, edges = plane_disk(basis=basis)
    regions = owner.query_material_surface_regions(model, (face,))
    chords = owner.query_prepared_material_boundary_chords(model, regions,
        [(model.handle('edge', edge), 0., 1.) for edge in edges], chord_error=1.)
    binding = MaterialRegionBinding(model, regions, regions.regions[0], face,
        geometry_approximation_error=1.)
    binding._curvilinear_state['chords'] = chords
    chart = AnalyticMetricChart(model, face, region_binding=binding)
    point = np.asarray([chart._reference_uv])
    assert chart._reference_uv == binding.reference_uv()
    assert np.allclose(chart.evaluate(point @ chart.transform),
        [model.face_support_point(face, *point[0])], rtol=0., atol=1e-12)
    expected_jacobian = np.column_stack(basis) @ chart.inverse.T
    assert np.allclose(chart.jacobians(point @ chart.transform),
        expected_jacobian[None, :, :], rtol=1e-12, atol=1e-12)
    object.__setattr__(binding, 'geometry_approximation_error', .5)
    with pytest.raises(MeshError, match='construction authority'):
        chart.validate_reference()


def test_implicit_plane_public_native_mesh_uses_actual_owner_cell_proof(monkeypatch):
    from anymesher.hybrid import generate_hybrid_mesh_result
    model, face, _edges = plane_disk()
    before = owner.to_dict(model)
    actual = owner.validate_prepared_material_curvilinear_cells
    calls = []
    def observe(*args, **kwargs):
        receipt = actual(*args, **kwargs)
        calls.append(receipt)
        return receipt
    monkeypatch.setattr(owner, 'validate_prepared_material_curvilinear_cells', observe)
    result = generate_hybrid_mesh_result(model, target_size=.25, strategy='native',
        order='linear', native_backend='python', certification_mode='none',
        mutation_policy='working_copy', structural_preparation=False,
        geometry_approximation_error=.01,
        native_options=NativeMeshingOptions(point_placement='frontal_delaunay', metric_mode='isotropic_spatial'))
    assert result.mesh.elements_of_face[face] and calls
    assert all(receipt.surface_error == .01 for receipt in calls)
    assert owner.to_dict(model) == before
