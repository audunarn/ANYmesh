"""Development checks for source-bound linear extrusion region meshing."""
import numpy as np
import pytest

from anygeometry import (GeometryModel, EntityRef, to_dict, plan_intersections,
                         apply_intersections, query_material_surface_regions)
from anymesher._material_region_binding import prepare_material_regions, registered_core_rows
from anymesher._analytic_metric_chart import AnalyticMetricChart
from anymesher.native_v2 import NativeMeshingOptions
from anymesher.errors import MeshError


def interior_cut_model():
    model = GeometryModel()
    points = model.add_points(((0,0,0),(.5,.05,0),(1,.05,0),(1.5,0,0)))
    wall = model.extrude((model.add_spline(points[0],points[1:-1],points[-1]),), (0,0,1))[0]
    cut = model.add_plate(model.add_points(((.5,-.5,.5),(1,-.5,.5),(1,.5,.5),(.5,.5,.5))))
    return model, wall, cut


def split_model():
    model, wall, cut = interior_cut_model()
    applied = apply_intersections(model, plan_intersections(model,(wall,cut),policy='connect'),policy='connect')
    descendants = tuple(ref.id for ref in model.resolve_ref(EntityRef('face',wall)))
    return model, wall, descendants, applied


def controls():
    return NativeMeshingOptions(point_placement='frontal_delaunay', metric_mode='isotropic_spatial',
                                max_insertions=128, max_topology_operations=2048)


def test_region_requires_one_authored_owner_and_retains_common_support():
    model, wall, descendants, _ = split_model()
    kwargs = dict(order='linear', native_options=controls())
    assert not prepare_material_regions(model,descendants,{face:(face,) for face in descendants},**kwargs)
    bindings = prepare_material_regions(model,descendants,{wall:descendants},**kwargs)
    assert set(bindings) == set(descendants)
    binding = bindings[min(descendants)]
    assert binding.collection.source.charts == binding.region.sources
    assert binding.collection.regions == (binding.region,)
    binding.validate()
    chart = AnalyticMetricChart(model,binding.representative,region_binding=binding)
    uv = np.asarray(((.2,.3),(.8,.7)))
    xyz = binding.region.support.evaluate_many(uv)
    np.testing.assert_allclose(chart.evaluate(uv @ chart.transform),xyz,atol=1e-13)
    assert chart.jacobians(uv @ chart.transform).shape == (2,3,2)
    assert not prepare_material_regions(model,descendants,{wall:descendants},
                                        edge_overrides=(binding.region.cancelled_seams[0].id,),**kwargs)
    assert not prepare_material_regions(model,descendants,{wall:descendants},
                                        supplied_seeding=True,**kwargs)
    assert not prepare_material_regions(model,descendants,{wall:descendants},
                                        order='quadratic',native_options=controls())


def test_missing_optional_owner_region_api_keeps_existing_face_route(monkeypatch):
    import anygeometry
    model, wall, _ = interior_cut_model()
    monkeypatch.delattr(anygeometry,'query_material_surface_regions')
    assert prepare_material_regions(model,(wall,),{wall:(wall,)},
                                    order='linear',native_options=controls()) == {}


def test_registered_input_receipts_reject_missing_and_conflicting_identity():
    from anymesher.core import MeshCore
    core = MeshCore(((0,0,0),(1,0,0),(0,1,0)),((0,1,2),))
    assert registered_core_rows(core,(((0.,0.),17),)) == {0:17}
    with pytest.raises(MeshError,match='moved, lost or merged'):
        registered_core_rows(core,(((.1,0.),17),))
    with pytest.raises(MeshError,match='conflicting'):
        registered_core_rows(core,(((0.,0.),17),((0.,0.),18)))


def test_registered_station_row_must_belong_to_an_active_cell():
    from anymesher.core import MeshCore
    core = MeshCore(((0,0,0),(1,0,0),(0,1,0),(.3,.3,0)),((0,1,2),))
    with pytest.raises(MeshError, match='active connectivity'):
        registered_core_rows(core,(((.3,.3),17),))
    core = MeshCore(((0,0,0),(1,0,0),(0,1,0)),((0,1,2),), triangle_active=(False,))
    with pytest.raises(MeshError, match='active connectivity'):
        registered_core_rows(core,(((0.,0.),17),))


def test_final_region_validation_rejects_unused_retained_vertex():
    from types import SimpleNamespace
    from anymesher.core import MeshCore
    from anymesher._material_region_binding import MaterialRegionBinding
    region = SimpleNamespace(interior_constraints=(),retained_vertices=(EntityRef('vertex',17),))
    binding = MaterialRegionBinding(None,None,region,5)
    mesh = SimpleNamespace(node_of_vertex={17:14})
    core = MeshCore(((0,0,0),(1,0,0),(0,1,0),(.3,.3,0)),((0,1,2),))
    with pytest.raises(MeshError,match='retained vertex.*active connectivity'):
        binding.validate_constraints(core,mesh,None,{0:11,1:12,2:13,3:14})
    # Publication removes unused rows; the missing receipt must still fail.
    with pytest.raises(MeshError,match='retained vertex.*active connectivity'):
        binding.validate_constraints(core,mesh,None,{0:11,1:12,2:13})


def test_final_region_certification_forwards_cancellation(monkeypatch):
    from types import SimpleNamespace
    from anymesher._material_region_binding import MaterialRegionBinding
    from anymesher.structured import StructuredMeshingOptions
    region = SimpleNamespace(faces=(EntityRef('face',5),),interior_constraints=(),retained_vertices=())
    binding = MaterialRegionBinding(None,None,region,5)
    seen = []
    monkeypatch.setattr(MaterialRegionBinding,'validate',lambda self,check: seen.append(check))
    mesh = SimpleNamespace(elements_of_face={5:(1,)},tris={1:(11,12,13)},quads={},
                           nodes={11:(0,0,0),12:(1,0,0),13:(0,1,0)})
    class Cancelled(Exception):
        pass
    def cancel(stage):
        assert stage == 'material region published cells'
        raise Cancelled
    with pytest.raises(Cancelled):
        binding.certify_published(mesh,None,StructuredMeshingOptions(),cancel)
    assert seen == [cancel]


def test_physical_region_growth_scan_has_cancellation_checkpoints():
    from anymesher.core import MeshCore
    from anymesher.surface_mesh import SurfaceMeshOptions
    class Cancelled(Exception):
        pass
    chart = object.__new__(AnalyticMetricChart)
    def cancel(stage):
        if stage == 'analytic physical growth incidence':
            raise Cancelled
    chart.check = cancel
    core = MeshCore(((0,0,0),(1,0,0),(0,1,0)),((0,1,2),))
    with pytest.raises(Cancelled):
        chart.certify_physical_core(core,SurfaceMeshOptions())


def test_region_refuses_unqualified_source_attachment_semantics():
    from anygeometry import ParameterRange
    model, wall, descendants, _ = split_model()
    face = min(descendants)
    point = model.add_point(*model.face_point(face,.5,.5))
    model.add_attachment(None,'vertex_on_face','face',face,ParameterRange.point(0.),
                         (ParameterRange.point(.5),ParameterRange.point(.5)),
                         source_kind='vertex',source_id=point)
    with pytest.raises(MeshError,match='attachments require a qualified consumer'):
        prepare_material_regions(model,descendants,{wall:descendants},
                                 order='linear',native_options=controls())


def test_known_constraint_endpoint_ids_canonicalize_roundoff_without_welding(monkeypatch):
    from types import SimpleNamespace
    from anymesher._material_region_binding import MaterialRegionBinding
    first, second = object(), object()
    region = SimpleNamespace(interior_constraints=(first,second),retained_vertices=())
    binding = MaterialRegionBinding(None,None,region,5)
    def path_chain(self, path, mesh, registry):
        if path is first:
            return [1,20], np.asarray(((1e-14,0.),(.5,.5))), None
        return [20,3], np.asarray(((.5+1e-14,.5),(1.,1.-1e-14))), None
    monkeypatch.setattr(MaterialRegionBinding,'path_chain',path_chain)
    segments, ids, points = binding.interior(None,None,
        boundary_rows={1:np.asarray((0.,0.)),3:np.asarray((1.,1.))})
    assert ids == (20,)
    np.testing.assert_array_equal(segments[0],((0.,0.),(.5,.5)))
    np.testing.assert_array_equal(segments[1],((.5,.5),(1.,1.)))
    np.testing.assert_array_equal(points,((.5,.5),))
    # Equal coordinates with different registered IDs retain separate receipts.
    monkeypatch.setattr(MaterialRegionBinding,'path_chain',
        lambda self,path,mesh,registry: ([21,22],np.asarray(((.5,.5),(1.,.5))),None))
    _, ids, _ = binding.interior(None,None,boundary_rows={})
    assert ids == (21,22)


def test_automatic_region_route_preserves_authored_faces_and_physical_joint(monkeypatch):
    import anymesher.hybrid as hybrid
    from anymesher.hybrid import generate_hybrid_mesh_result
    from anymesher.structured import StructuredMeshingOptions, MeshQualityPolicy
    model, wall, cut = interior_cut_model()
    before = to_dict(model)
    original = hybrid._mesh_native_face
    observed = []
    def capture(geometry, mesh, face, **kwargs):
        binding = kwargs.get('_material_region_binding')
        value = original(geometry,mesh,face,**kwargs)
        if binding is not None:
            elements = mesh.elements_of_face[face]
            cells = [mesh.corners_of(element) for element in elements]
            used = {node for cell in cells for node in cell}
            cell_edges = {tuple(sorted((int(a),int(b)))) for cell in cells
                          for a,b in zip(cell,(*cell[1:],cell[0]))}
            for path in binding.region.interior_constraints:
                chain = mesh.nodes_of_edge[path.source_edge]
                assert set(chain).issubset(used)
                assert all(tuple(sorted((a,b))) in cell_edges for a,b in zip(chain[:-1],chain[1:]))
            for handle in binding.region.retained_vertices:
                node = mesh.node_of_vertex[handle.id]
                assert node in used
                np.testing.assert_array_equal(mesh.nodes[node],geometry.vertex_position(handle.id))
            assert all(handle.id not in mesh.nodes_of_edge for handle in binding.region.cancelled_seams)
            observed.append(binding.representative)
        return value
    monkeypatch.setattr(hybrid,'_mesh_native_face',capture)
    result = generate_hybrid_mesh_result(
        model, target_size=.25, strategy='native', recombine=False, native_backend='python',
        native_options=controls(), _native_surface_options=StructuredMeshingOptions(
            quality_policy=MeshQualityPolicy(minimum_angle=15.)))
    assert to_dict(model) == before
    assert set(result.mesh.elements_of_face) == {wall, cut}
    assert set(result.mesh.elements_of_face[wall]).isdisjoint(result.mesh.elements_of_face[cut])
    shared = set(result.mesh.nodes_on(EntityRef('face',wall))) & set(result.mesh.nodes_on(EntityRef('face',cut)))
    assert len(shared) >= 2
    # Every published node belongs to an element: cancelled seam stations do not leak.
    active_nodes = {node for cell in (*result.mesh.tris.values(),*result.mesh.quads.values()) for node in cell}
    assert active_nodes == set(result.mesh.nodes)
    records = result.mesh.hybrid_diagnostics['triangulation_backend_by_face']
    region_record = next(record['material_region'] for record in records.values() if 'material_region' in record)
    assert region_record['authored_face'] == wall
    assert region_record['cancelled_seams']
    assert region_record['interior_constraint_edges']
    assert region_record['retained_vertices']
    assert len(observed) == 1


def test_retained_vertex_reuses_its_registered_constraint_branch(monkeypatch):
    from types import SimpleNamespace
    from anymesher._material_region_binding import MaterialRegionBinding
    def wrong_inverse(*args):
        raise AssertionError('registered vertex must not be reprojected')
    path = object()
    region = SimpleNamespace(interior_constraints=(path,),
        retained_vertices=(EntityRef('vertex',17),),
        support=SimpleNamespace(local_uv_many=wrong_inverse))
    binding = MaterialRegionBinding(None,None,region,5)
    mesh = SimpleNamespace(node_of_vertex={17:20},nodes={20:(.5,.5,0)})
    monkeypatch.setattr(MaterialRegionBinding,'path_chain',
        lambda self,path,mesh,registry: ([1,20,3],np.asarray(((0.,0.),(.5,.5),(1.,1.))),None))
    _, ids, points = binding.interior(mesh,None,
        boundary_rows={1:np.asarray((0.,0.)),3:np.asarray((1.,1.))})
    assert ids == (20,)
    np.testing.assert_array_equal(points,((.5,.5),))
