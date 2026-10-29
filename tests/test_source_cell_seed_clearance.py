"""Retry free seeds follow the existing native protected insertion predicate."""
import numpy as np
import pytest
import anymesher.surface_mesh as surface
from anymesher.native_v2 import NativeMeshingOptions


def ring():
    return np.array(((0.,0.),(.5,0.),(1.,0.),(1.,.5),
                     (1.,1.),(.5,1.),(0.,1.),(0.,.5)))


def test_actual_failed_retry_boundary_points_are_not_admissible_free_seeds():
    # Root run35 capture quality-002 rows27/24 and an interior free point.
    # The physical fixture's xy coordinates are its exact source chart rows.
    points = np.array(((.24936314134595494,.12831529900107452),
                       (.25103848224048214,.8608329966031298),
                       (.4733774689586201,.4405820518580148)))
    before = points.copy()
    result = surface._protected_diametral_seed_filter(points, ring(), (), ())
    np.testing.assert_array_equal(result, points[2:])
    np.testing.assert_array_equal(points, before)


def test_strict_native_predicate_keeps_disk_boundary_and_respects_constraints():
    outer = np.array(((0.,0.),(2.,0.),(2.,2.),(0.,2.)))
    segment = np.array(((.5,1.),(1.5,1.)))
    points = np.array(((1.,1.),(1.,1.5)))
    # Isolate an interior fixed segment; points lie on the outer diametral
    # boundaries and the second lies exactly on the constraint's disk.
    result = surface._protected_diametral_seed_filter(points,
        np.empty((0,2)), (), (segment,))
    np.testing.assert_array_equal(result, points[1:])


def test_hole_boundary_uses_same_protected_disk_predicate():
    outer = np.array(((-2.,-2.),(0.,-2.),(2.,-2.),(2.,0.),
                      (2.,2.),(0.,2.),(-2.,2.),(-2.,0.)))
    hole = np.array(((-.5,-.5),(-.5,.5),(.5,.5),(.5,-.5)))
    points = np.array(((0.,.6),(0.,1.),(1.5,1.5)))
    result = surface._protected_diametral_seed_filter(points, outer, (hole,), ())
    # First point is in the material but encroaches the top hole segment.
    assert not np.any(np.all(result == points[0], axis=1))
    assert np.any(np.all(result == points[1], axis=1))


def test_movable_trial_guard_excludes_mandatory_rows_and_refuses_actual_row15():
    base = ring()
    free = np.array(((.7453119377033139,.8788971341506702),))
    points = np.vstack((base, free))
    edges = np.array([(i, (i+1)%8) for i in range(8)])
    assert not surface._protected_free_rows_admissible(points, edges, np.empty((0,2)))
    assert surface._protected_free_rows_admissible(points, edges, free)


@pytest.mark.parametrize('enabled', [False, True])
def test_optimizer_rejects_free_row_disk_reentry_only_when_requested(monkeypatch, enabled):
    from types import SimpleNamespace as NS
    original = np.vstack((ring(), ((.75,.7),)))
    moved = original.copy()
    moved[8] = (.7453119377033139,.8788971341506702)
    triangles = np.array(((4,5,8),))
    candidate = NS(points=original, triangles=triangles, score=(0,2),
        added_points=0, rounds=0)
    edges = np.array([(i,(i+1)%8) for i in range(8)])
    monkeypatch.setattr(surface, 'local_edge_flip', lambda *args,**kwargs:
        NS(triangles=triangles, flip_count=0, converged=True))
    monkeypatch.setattr(surface, 'constrained_smoothing', lambda *args,**kwargs:
        NS(points=moved, moved_nodes=(8,)))
    monkeypatch.setattr(surface, '_make_candidate', lambda points,triangles,**kwargs:
        NS(points=points, triangles=triangles, score=(0,1)))
    monkeypatch.setattr(surface, '_candidate_selection_key', lambda item,**kwargs:item.score)
    result = surface._optimize_candidate(candidate, edges, np.empty((0,2)),
        protect_free_rows=enabled)
    assert result.points is (original if enabled else moved)
    np.testing.assert_array_equal(original[:8], ring())


@pytest.mark.parametrize('enabled', [False, True])
def test_prefrontal_midpoint_stage_cannot_reintroduce_filtered_bad_node(monkeypatch, enabled):
    from dataclasses import dataclass
    from types import SimpleNamespace as NS
    @dataclass
    class Triangulation:
        points: object
        triangles: object
        segments: object
    generated = np.array(((.25,.3),(.5,.4),(.75,.3),(.3,.6),
                          (.6,.6),(.8,.6),(.5,.7)))
    bad = np.array(((.747090841,.917727874),))  # stage009 input row15
    calls = []
    edges = np.array([(i,(i+1)%8) for i in range(8)])
    def triangulate(outer, holes, constraints, *, interior_points, **kwargs):
        calls.append(interior_points.copy())
        return Triangulation(np.vstack((outer,interior_points)),
            np.array(((0,1,8),)), edges)
    def candidate(points, triangles, **kwargs):
        return NS(points=points, triangles=triangles, score=(0,),
            report={'invalid_element_count':0,
                    'poor_element_ids':[1] if len(points) == 15 else [],
                    'elements_above_aspect_ratio_5':0})
    monkeypatch.setattr(surface, 'triangulate_polygon', triangulate)
    monkeypatch.setattr(surface, '_make_candidate', candidate)
    monkeypatch.setattr(surface, '_optimize_candidate', lambda item,*args,**kwargs:item)
    monkeypatch.setattr(surface, '_candidate_selection_key',
        lambda item,**kwargs:(len(item.report['poor_element_ids']),))
    monkeypatch.setattr(surface, '_refinement_midpoints', lambda *args,**kwargs:bad.copy())
    path = surface._run_quality_path('staggered_chart', ring(), (), (),
        np.empty((0,2)), generated, NS(backend='python',prefer_quality_policy=True),
        None, protect_generated_seed_disks=enabled)
    assert len(calls) == (1 if enabled else 2)
    if enabled:
        assert path['triangulation'].points is path['best'].points
        assert path['triangulation'].triangles is path['best'].triangles
        # Observe the actual native entry, before any insertion or quality work.
        class NativeEntryReached(Exception):
            pass
        def frontal(actual, *args, **kwargs):
            np.testing.assert_array_equal(actual.points, path['best'].points)
            np.testing.assert_array_equal(actual.triangles, path['best'].triangles)
            raise NativeEntryReached()
        monkeypatch.setattr(surface, 'frontal_delaunay_refine', frontal)
        settings = surface.SurfaceMeshOptions(target_size=.25, native_options=NativeMeshingOptions(), min_angle=30.)
        with pytest.raises(NativeEntryReached):
            surface._run_frontal_quality_path(path, settings, None,
                model_uuid=None, geometry_revision=None, metric_to_physical=None,
                metric_jacobian=None, automatically_seeded_shared_segments=None,
                component_seed_registry=None, supplemental_metric_field=None,
                protect_free_rows=True)


@pytest.mark.parametrize('enabled', [False, True])
def test_retry_hook_filters_only_generated_seeds_and_preserves_mandatory_rows(monkeypatch, enabled):
    mandatory = np.array(((.25,.1),))  # deliberately inside a protected disk
    before = mandatory.copy()
    class SeedInputsReached(Exception):
        pass
    def inspect(name, outer, holes, constraints, explicit, generated, settings,
                cancellation_check, **kwargs):
        np.testing.assert_array_equal(outer, ring())
        np.testing.assert_array_equal(explicit, before)
        assert kwargs['protected_node_ids'] == {0: 7}
        filtered = surface._protected_diametral_seed_filter(generated, outer, holes, constraints)
        if enabled:
            np.testing.assert_array_equal(generated, filtered)
        else:
            assert len(filtered) < len(generated)
        raise SeedInputsReached()
    monkeypatch.setattr(surface, '_run_quality_path', inspect)
    with pytest.raises(SeedInputsReached):
        surface.mesh_planar_surface(ring(), interior_points=mandatory,
            target_size=.25, backend='python', native_options=NativeMeshingOptions(
                point_placement='frontal_delaunay', metric_mode='isotropic_spatial'),
            _boundary_is_seeded=True, _protected_node_ids={0:7},
            _protect_generated_seed_disks=enabled)
    np.testing.assert_array_equal(mandatory, before)


def test_protected_seed_filter_preserves_callback_exception_identity():
    failure = RuntimeError('cancel seed clearance')
    def cancel(phase):
        raise failure
    with pytest.raises(RuntimeError) as caught:
        surface._protected_diametral_seed_filter(np.array(((.5,.5),)),
            ring(), (), (), cancellation_check=cancel)
    assert caught.value is failure


def test_selected_state_rebuilds_real_triangulation_preserving_source_provenance(monkeypatch):
    from types import SimpleNamespace as NS
    from anymesher.triangulation import PlanarTriangulation
    outer = np.array(((0.,0.),(1.,0.),(1.,1.),(0.,1.)))
    generated = np.array(((.5,.3),))
    constraint = np.array(((.25,.6),(.75,.6)))
    points = np.vstack((outer, generated, constraint))
    triangles = np.array(((0,1,4),(1,2,6),(2,3,5),(3,0,5),
                          (0,4,5),(1,6,4),(4,6,5),(2,5,6)))
    boundary = np.array(((0,1),(1,2),(2,3),(3,0)))
    segments = np.vstack((boundary, ((5,6),)))
    provenance = ((0,101),(1,102),(2,103),(3,104),(5,201),(6,202))
    original = PlanarTriangulation(points, triangles, segments, boundary,
        segments, np.arange(4), (), backend='python', requested_backend='python',
        selected_backend='python', actual_backend='python',
        native_diagnostics={'source_receipt': 'unchanged-source-identity'},
        protected_node_rows=provenance)
    original_points, original_triangles = original.points.copy(), original.triangles.copy()
    selected_points = original.points.copy()
    selected_points[4] = (.5,.35)
    selected_triangles = original.triangles.copy()
    selected_triangles[4] = (0,4,6)
    selected_triangles[6] = (0,6,5)
    def candidate(rows, cells, **kwargs):
        return NS(points=rows, triangles=cells, score=(0,), report={
            'invalid_element_count':0, 'poor_element_ids':[],
            'elements_above_aspect_ratio_5':0})
    monkeypatch.setattr(surface, 'triangulate_polygon', lambda *args,**kwargs:original)
    monkeypatch.setattr(surface, '_make_candidate', candidate)
    monkeypatch.setattr(surface, '_optimize_candidate',
        lambda *args,**kwargs:candidate(selected_points,selected_triangles))
    monkeypatch.setattr(surface, '_candidate_selection_key',
        lambda item,**kwargs:(0 if item.points is selected_points else 1,))
    path = surface._run_quality_path('staggered_chart', outer, (), (constraint,),
        np.empty((0,2)), generated, NS(backend='python',prefer_quality_policy=True),
        None, protect_generated_seed_disks=True)
    updated = path['triangulation']
    assert type(updated) is PlanarTriangulation and updated is not original
    np.testing.assert_array_equal(updated.points, selected_points)
    np.testing.assert_array_equal(updated.triangles, selected_triangles)
    np.testing.assert_array_equal(original.points, original_points)
    np.testing.assert_array_equal(original.triangles, original_triangles)
    for name in ('segments','boundary_segments','mandatory_segments','outer_loop'):
        np.testing.assert_array_equal(getattr(updated,name),getattr(original,name))
    for name in ('hole_loops','backend','requested_backend','selected_backend',
                 'actual_backend','fallback_reason','protected_node_rows','native_diagnostics'):
        assert getattr(updated,name) == getattr(original,name)
    assert updated.protected_node_rows == provenance
    np.testing.assert_array_equal(updated.points[[row for row,_ in provenance]],
                                  original.points[[row for row,_ in provenance]])
    assert not updated.points.flags.writeable and not updated.triangles.flags.writeable
