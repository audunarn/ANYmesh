"""Actual saved physical operands; these confer no source/owner authority."""
from pathlib import Path
import json

import numpy as np
import pytest

from anymesher.core import MeshCore
from anymesher.recombine import recombine_triangles_with_report
from anymesher.surface_mesh import SurfaceMeshOptions, _qualified_recombination, _published_growth


def operands():
    data = json.loads((Path(__file__).parent/'data'/'nurbs_closed_loop_quality_run16.json').read_text())
    core = MeshCore(np.asarray(data['points']), np.asarray(data['triangles']))
    settings = SurfaceMeshOptions(**data['thresholds'])
    return core, np.asarray(data['protected_edges']), settings


def snapshot(core):
    return tuple(getattr(core, name).tobytes() for name in (
        'node_coordinates', 'node_ids', 'triangle_connectivity', 'triangle_ids',
        'node_owner_handles', 'triangle_owner_handles', 'triangle_active'))


def test_saved_closed_loop_matching_repairs_growth_and_preserves_partition():
    core, protected, settings = operands()
    before = snapshot(core)
    report, quality = _qualified_recombination(core, protected, settings, None,
                                               physical_growth=True)
    assert snapshot(core) == before
    assert report.pair_count > 0  # Requested recombination remains active.
    assert quality['violation_counts']['minimum_angle'] == 2
    growth, poor = _published_growth(report.mesh, settings.max_element_growth)
    assert not poor and growth <= 1.5
    assert report.exchange_work <= settings.max_recombination_work
    np.testing.assert_array_equal(report.mesh.node_coordinates, core.node_coordinates)
    np.testing.assert_array_equal(report.mesh.node_ids, core.node_ids)
    visible = set()
    for cells in (report.mesh.triangle_connectivity[report.mesh.triangle_active],
                  report.mesh.quad_connectivity[report.mesh.quad_active]):
        for cell in cells:
            visible.update(tuple(sorted((int(a), int(b)))) for a,b in zip(cell, np.roll(cell,-1)))
    assert all(tuple(sorted(map(int, edge))) in visible for edge in protected)


@pytest.mark.parametrize('id_offset', (0, 10000))
def test_actual_curved_uv_repair_satisfies_complete_mixed_quality(id_offset):
    from anygeometry.nurbs import NurbsSurface
    from anymesher._physical_mixed_repair import repair_mixed_coordinates
    from anymesher.quality_v2 import evaluate_quality
    from anymesher.surface_mesh import _quality_threshold_report
    core, protected, settings = operands()
    if id_offset:
        core = MeshCore(core.node_coordinates, core.triangle_connectivity,
            node_ids=core.node_ids+id_offset, triangle_ids=core.triangle_ids+id_offset)
    chart = json.loads((Path(__file__).parent/'data'/'nurbs_closed_loop_chart_run16.json').read_text())
    uv = np.asarray(chart['chart_points'])
    support = NurbsSurface(**chart['support'])
    def evaluate(points):
        return support.evaluate_many(points)
    np.testing.assert_allclose(evaluate(uv), core.node_coordinates, atol=1.e-14, rtol=0.)
    report, _ = _qualified_recombination(core, protected, settings, None, physical_growth=True)
    fixed = sorted(set(map(int, protected.ravel())))
    before = uv.tobytes(), snapshot(core)
    points, repaired = repair_mixed_coordinates(uv, core.triangle_connectivity,
        report, settings, evaluate, fixed)
    assert (uv.tobytes(), snapshot(core)) == before
    np.testing.assert_array_equal(points[fixed], uv[fixed])
    mesh = MeshCore(evaluate(points), repaired.mesh.triangle_connectivity,
        repaired.mesh.quad_connectivity, node_ids=repaired.mesh.node_ids,
        triangle_ids=repaired.mesh.triangle_ids, quad_ids=repaired.mesh.quad_ids,
        node_active=repaired.mesh.node_active, triangle_active=repaired.mesh.triangle_active,
        quad_active=repaired.mesh.quad_active)
    quality = _quality_threshold_report(evaluate_quality(mesh), settings)
    assert quality['accepted'], quality
    assert _published_growth(mesh, 1.5)[0] <= 1.5
    assert repaired.exchange_work <= settings.max_recombination_work
    np.testing.assert_array_equal(repaired.mesh.node_ids, core.node_ids)
    np.testing.assert_array_equal(repaired.mesh.triangle_ids, core.triangle_ids)


def test_private_physical_option_never_invokes_native_decisions(monkeypatch):
    import importlib
    module = importlib.import_module('anymesher.recombine')
    core, protected, _ = operands()
    def forbidden(*args, **kwargs):
        raise AssertionError('physical repair cannot consume a second native allowance')
    monkeypatch.setattr(module, 'native_recombination_decisions', forbidden)
    module.recombine_triangles_with_report(core, protected_edges=protected,
        _physical_growth_limit=1.5)


def test_actual_matching_cancel_does_not_mutate_input():
    core, protected, settings = operands()
    before = snapshot(core)
    error = KeyboardInterrupt('cancel complete physical matching')
    def cancel(phase):
        if phase == 'physical recombination alternating path':
            raise error
    with pytest.raises(KeyboardInterrupt) as caught:
        _qualified_recombination(core, protected, settings, cancel, physical_growth=True)
    assert caught.value is error
    assert snapshot(core) == before


def test_default_matching_keeps_legacy_saved_failure_and_input():
    core, protected, settings = operands()
    before = snapshot(core)
    report, quality = _qualified_recombination(core, protected, settings, None)
    assert snapshot(core) == before
    assert not quality['accepted']
    assert quality['violation_counts']['minimum_angle'] == 2
    growth, _ = _published_growth(report.mesh, settings.max_element_growth)
    assert growth > 1.5


def test_physical_matching_respects_exhausted_existing_exchange_allowance():
    core, protected, settings = operands()
    report = recombine_triangles_with_report(core, protected_edges=protected,
        max_exchange_work=1, _physical_growth_limit=1.5, _use_native=False)
    assert report.exchange_work <= 1


def test_private_matching_refuses_inactive_triangle_rows():
    from anymesher.errors import MeshError
    core, protected, _ = operands()
    active = core.triangle_active.copy()
    active[0] = False
    core = MeshCore(core.node_coordinates, core.triangle_connectivity, triangle_active=active)
    before = snapshot(core)
    with pytest.raises(MeshError, match='active linear triangles'):
        recombine_triangles_with_report(core, protected_edges=protected,
            _physical_growth_limit=1.5)
    assert snapshot(core) == before


def test_shape_phase_cancellation_preserves_inputs_and_exception_identity():
    from anygeometry.nurbs import NurbsSurface
    from anymesher._physical_mixed_repair import repair_mixed_coordinates
    core, protected, settings = operands()
    data = json.loads((Path(__file__).parent/'data'/'nurbs_closed_loop_chart_run16.json').read_text())
    uv = np.asarray(data['chart_points'])
    support = NurbsSurface(**data['support'])
    report, _ = _qualified_recombination(core, protected, settings, None, physical_growth=True)
    before = uv.tobytes(), snapshot(core), snapshot(report.mesh)
    error = KeyboardInterrupt('shape callback')
    count = 0
    def cancel(phase):
        nonlocal count
        count += 1
        if count == 6:
            raise error
    with pytest.raises(KeyboardInterrupt) as caught:
        repair_mixed_coordinates(uv, core.triangle_connectivity, report,
            settings, support.evaluate_many, protected.ravel(), cancellation_check=cancel)
    assert caught.value is error
    assert (uv.tobytes(), snapshot(core), snapshot(report.mesh)) == before


def preview_refusal_operands():
    # Deliberately forged private staging data: only refusal is under test;
    # this object carries no geometry receipt or source admission authority.
    from types import SimpleNamespace
    import importlib
    module = importlib.import_module('anymesher.surface_mesh')
    points = np.asarray(((0.,0.), (1.,0.), (0.,1.)))
    triangles = np.asarray(((0,1,2),))
    segments = np.asarray(((0,1),(1,2),(0,2)))
    settings = SurfaceMeshOptions()
    class Evaluator:
        def _current(self):
            pass
        def evaluate(self, rows):
            return np.column_stack((rows, np.zeros(len(rows))))
    evaluator = Evaluator()
    core = MeshCore(evaluator.evaluate(points), triangles)
    report = recombine_triangles_with_report(core)
    preview = dict(accepted=True, settings=settings, settings_key=repr(settings),
        points_key=points.tobytes(), triangles_key=triangles.tobytes(),
        segments_key=segments.tobytes(), fixed=(0,1,2), fixed_key=points.tobytes(),
        report=report, mesh_key=module._mixed_preview_key(report.mesh), quality={})
    path = dict(best=SimpleNamespace(points=points, triangles=triangles),
        triangulation=SimpleNamespace(segments=segments), _curvilinear_mixed_preview=preview)
    return module, path, module._plane_from_outer(points), points, settings, evaluator


@pytest.mark.parametrize('mutation', ('settings', 'uv', 'mesh'))
def test_private_preview_mutation_is_refused(mutation):
    from dataclasses import replace
    from anymesher.errors import MeshError
    module, path, plane, points, settings, evaluator = preview_refusal_operands()
    if mutation == 'settings':
        settings = replace(settings, min_angle=31.)
    elif mutation == 'uv':
        path['best'].points = points.copy()
        path['best'].points[0,0] = .01
    else:
        preview = path['_curvilinear_mixed_preview']
        old = preview['report']
        xyz = old.mesh.node_coordinates.copy()
        xyz[0,0] = .01
        preview['report'] = replace(old, mesh=MeshCore(xyz, old.mesh.triangle_connectivity))
    with pytest.raises(MeshError, match='preview changed'):
        module._prepare_recombined_path(path, plane, points, (), settings,
            None, physical_evaluator=evaluator.evaluate)


def test_failed_preview_budget_is_not_reset(monkeypatch):
    from anymesher.errors import MeshError
    module, path, plane, points, settings, evaluator = preview_refusal_operands()
    del path['_curvilinear_mixed_preview']
    path['_curvilinear_mixed_spent'] = settings.max_recombination_work
    def forbidden(*args, **kwargs):
        raise AssertionError('exhausted preview cannot start another matching')
    monkeypatch.setattr(module, '_qualified_recombination', forbidden)
    with pytest.raises(MeshError, match='exhausted recombination work'):
        module._prepare_recombined_path(path, plane, points, (), settings,
            None, physical_evaluator=evaluator.evaluate)


def test_noncurvilinear_physical_route_keeps_ordinary_matching(monkeypatch):
    module, path, plane, points, settings, evaluator = preview_refusal_operands()
    del path['_curvilinear_mixed_preview']
    class StopInspection(Exception):
        pass
    def inspect(*args, **kwargs):
        assert kwargs['physical_growth'] is False
        raise StopInspection()
    monkeypatch.setattr(module, '_qualified_recombination', inspect)
    with pytest.raises(StopInspection):
        module._prepare_recombined_path(path, plane, points, (), settings,
            None, physical_evaluator=evaluator.evaluate)


def test_shape_invalid_detached_quad_refuses_without_swallowing_owner_error(monkeypatch):
    from anygeometry.nurbs import NurbsSurface
    from anymesher._physical_mixed_repair import repair_mixed_coordinates
    import anymesher.quality_v2 as quality_module
    core, protected, settings = operands()
    data = json.loads((Path(__file__).parent/'data'/'nurbs_closed_loop_chart_run16.json').read_text())
    uv = np.asarray(data['chart_points'])
    support = NurbsSurface(**data['support'])
    report, _ = _qualified_recombination(core, protected, settings, None, physical_growth=True)
    before = uv.tobytes(), snapshot(core), snapshot(report.mesh)
    actual = quality_module.evaluate_quality
    calls = 0
    invalid_observed = False
    def score(mesh, **kwargs):
        nonlocal calls, invalid_observed
        calls += 1
        if calls == 2:
            xyz = mesh.node_coordinates.copy()
            cell = mesh.quad_connectivity[mesh.quad_active][0]
            xyz[cell[0]] = xyz[cell[1]]
            invalid = MeshCore(xyz, mesh.triangle_connectivity, mesh.quad_connectivity,
                node_ids=mesh.node_ids, triangle_ids=mesh.triangle_ids, quad_ids=mesh.quad_ids,
                triangle_active=mesh.triangle_active, quad_active=mesh.quad_active)
            result = actual(invalid, **kwargs)
            invalid_observed = bool(result.validity.errors)
            return result
        return actual(mesh, **kwargs)
    monkeypatch.setattr(quality_module, 'evaluate_quality', score)
    repair_mixed_coordinates(uv, core.triangle_connectivity, report, settings,
        support.evaluate_many, protected.ravel())
    assert invalid_observed
    assert (uv.tobytes(), snapshot(core), snapshot(report.mesh)) == before
    error = ValueError('owner evaluation failed')
    def failed_owner(points):
        raise error
    with pytest.raises(ValueError) as caught:
        repair_mixed_coordinates(uv, core.triangle_connectivity, report, settings,
            failed_owner, protected.ravel())
    assert caught.value is error
    assert (uv.tobytes(), snapshot(core), snapshot(report.mesh)) == before
