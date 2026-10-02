"""Owner gradient batches preserve ordered, bounded detached repair."""
from dataclasses import fields

import numpy as np
import pytest

from anymesher._joint_triangle_repair import repair_joint_triangle_quality


def fixture():
    return (np.asarray(((0., 0.), (1., 0.), (1., 1.), (0., 1.),
                        (.2, .35), (.75, .6))),
            np.asarray(((0, 1, 4), (1, 5, 4), (1, 2, 5),
                        (2, 3, 5), (3, 4, 5), (3, 0, 4))),
            np.asarray(((0, 1), (1, 2), (2, 3), (3, 0))))


def physical(rows):
    return np.column_stack((rows, .25 * rows[:, 0] ** 2 + .125 * rows[:, 1] ** 2))


class RecordedOwner:
    def __init__(self, count, evaluator=physical):
        self.count = count
        self.evaluator = evaluator
        self.row_counts = []
        self.candidates = []

    def __call__(self, rows):
        self.row_counts.append(len(rows))
        assert len(rows) % self.count == 0
        self.candidates.extend(row.copy() for row in rows.reshape(-1, self.count, 2))
        return self.evaluator(rows)


def repair(points, cells, protected, owner, *, budget=2048, batch_size=1, cancel=None):
    return repair_joint_triangle_quality(
        points, cells, protected, range(len(cells)), min_angle=30., max_growth=1.5,
        max_trials=budget, evaluate_coordinates=owner, coordinate_batch_size=batch_size,
        cancellation_check=cancel)


def assert_same_result(first, second):
    assert first.points.tobytes() == second.points.tobytes()
    for item in fields(first):
        if item.name != "points":
            assert getattr(first, item.name) == getattr(second, item.name)


@pytest.mark.parametrize("budget", (0, 1, 3, 7, 8, 9, 31, 2048))
def test_ordered_batches_match_scalar_probes_and_exact_budget(budget):
    points, cells, protected = fixture()
    before = points.tobytes(), cells.tobytes(), protected.tobytes()
    serial = RecordedOwner(len(points))
    batched = RecordedOwner(len(points))
    first = repair(points, cells, protected, serial, budget=budget)
    second = repair(points, cells, protected, batched, budget=budget, batch_size=8)
    assert_same_result(first, second)
    assert second.trials <= budget
    assert first.points[:4].tobytes() == points[:4].tobytes()
    assert (points.tobytes(), cells.tobytes(), protected.tobytes()) == before
    assert len(serial.candidates) == len(batched.candidates)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(serial.candidates, batched.candidates))
    assert max(batched.row_counts) <= 8 * len(points)
    if budget == 1:
        assert second.trials == 1 and second.budget_exhausted
        assert second.final_penalty < second.initial_penalty
        assert second.moved_nodes  # The unmatched first plus probe is retained.
    if budget >= 8:
        assert len(batched.row_counts) < len(serial.row_counts)


def test_equal_probe_merits_keep_original_best_point():
    points, cells, protected = fixture()
    owner = lambda rows: np.column_stack((np.round(rows, 3), np.zeros(len(rows))))
    serial = repair(points, cells, protected, owner, budget=8)
    batched = repair(points, cells, protected, owner, budget=8, batch_size=8)
    assert_same_result(serial, batched)
    assert batched.trials == 8
    assert batched.initial_penalty > 0
    assert batched.final_penalty == batched.initial_penalty
    assert batched.points.tobytes() == points.tobytes() and not batched.moved_nodes


def test_invalid_orientation_consumes_trial_without_owner_submission():
    points = np.asarray(((0., 0.), (1., 0.), (1., 1.), (0., 1.), (1e-8, .5)))
    cells = np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)))
    protected = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)))
    serial = RecordedOwner(len(points))
    batched = RecordedOwner(len(points))
    first = repair(points, cells, protected, serial, budget=3)
    second = repair(points, cells, protected, batched, budget=3, batch_size=8)
    assert_same_result(first, second)
    assert second.trials == 3
    assert len(serial.candidates) == len(batched.candidates) == 4
    assert all(row[4, 0] >= 0 for row in batched.candidates)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(serial.candidates, batched.candidates))


@pytest.mark.parametrize("count", (3000, 9000))
def test_flattened_temporary_rows_are_bounded_without_capping_model(count):
    points, cells, protected = fixture()
    points = np.vstack((points, np.zeros((count - len(points), 2))))
    owner = RecordedOwner(count)
    result = repair(points, cells, protected, owner, budget=8, batch_size=8)
    assert result.trials == 8
    assert max(owner.row_counts) <= max(8192, count)
    assert result.points[6:].tobytes() == points[6:].tobytes()
    if count > 8192:
        assert owner.row_counts == [count] * 10


@pytest.mark.parametrize("batch_size", (1, 4, 8))
def test_cancellation_at_later_trial_propagates_without_input_mutation(batch_size):
    points, cells, protected = fixture()
    before = points.tobytes(), cells.tobytes(), protected.tobytes()
    owner = RecordedOwner(len(points))
    error = RuntimeError("cancel admitted owner probes")
    calls = []
    def cancel():
        calls.append(True)
        if len(calls) == 5:
            raise error
    with pytest.raises(RuntimeError) as caught:
        repair(points, cells, protected, owner, budget=9, batch_size=batch_size, cancel=cancel)
    assert caught.value is error and len(calls) == 5
    assert (points.tobytes(), cells.tobytes(), protected.tobytes()) == before
    # Batch admission intentionally precedes owner work. Cancellation publishes
    # no partial result even when it interrupts admission of the first chunk.
    assert len(owner.candidates) == (2 if batch_size == 8 else 6)


@pytest.mark.parametrize("batch_size", (1, 8))
def test_owner_exception_is_not_retried_or_replaced(batch_size):
    points, cells, protected = fixture()
    before = points.tobytes()
    error = RuntimeError("owner binding refused")
    calls = []
    def owner(rows):
        calls.append(len(rows))
        if len(calls) == 3:
            raise error
        return physical(rows)
    with pytest.raises(RuntimeError) as caught:
        repair(points, cells, protected, owner, budget=8, batch_size=batch_size)
    assert caught.value is error and len(calls) == 3
    assert points.tobytes() == before


@pytest.mark.parametrize("failure", ("shape", "nonfinite"))
@pytest.mark.parametrize("batch_size", (1, 8))
def test_malformed_owner_coordinates_fail_closed(failure, batch_size):
    points, cells, protected = fixture()
    before = points.tobytes()
    calls = []
    def owner(rows):
        calls.append(len(rows))
        output = physical(rows)
        if len(calls) == 3:
            if failure == "shape":
                return output[:-1]
            output[-1, 2] = float("nan")
        return output
    with pytest.raises(ValueError, match="invalid owner coordinates"):
        repair(points, cells, protected, owner, budget=8, batch_size=batch_size)
    assert len(calls) == 3 and points.tobytes() == before


@pytest.mark.parametrize("batch_size", (False, 0, 9, 1.5))
def test_coordinate_batch_size_is_explicitly_bounded(batch_size):
    points, cells, protected = fixture()
    with pytest.raises(ValueError, match="invalid joint"):
        repair(points, cells, protected, physical, batch_size=batch_size)


def test_frontal_receipt_consumes_the_same_work_for_owner_batches():
    from anymesher._frontal_transition_quality import repair_frontal_transition
    from anymesher.native_v2 import NativeMeshingOptions
    from anymesher.surface_mesh import SurfaceMeshOptions, _make_candidate, _physical_quality_candidate
    points, cells, protected = fixture()
    settings = SurfaceMeshOptions(min_angle=30., max_element_growth=1.5,
        prefer_quality_policy=True,
        native_options=NativeMeshingOptions(max_topology_operations=16))
    candidate = _physical_quality_candidate(_make_candidate(points, cells, settings=settings),
                                             settings, physical)
    report = {"topology_operations": 9}
    serial, first = repair_frontal_transition(candidate, protected, settings, report,
                                              evaluate_coordinates=physical)
    batched, second = repair_frontal_transition(candidate, protected, settings, report,
        evaluate_coordinates=physical, coordinate_batch_size=8)
    assert first == second and serial.points.tobytes() == batched.points.tobytes()
    assert first["chart_transition_repair"]["trials"] == 7
    assert first["topology_operations"] == 16
    assert report == {"topology_operations": 9}


def analytic_chart():
    from anygeometry import GeometryModel, BezierDirectrix, ExtrudedSurface
    from anymesher._analytic_metric_chart import AnalyticMetricChart
    model = GeometryModel()
    vertices = model.add_points(((0., 0., 0.), (.5, 0., 0.), (1., .2, 0.)))
    edge = model.add_spline(vertices[0], vertices[1:-1], vertices[-1])
    face = model.extrude((edge,), (0., 0., 1.))[0]
    model.set_face_surface(face, ExtrudedSurface(
        BezierDirectrix(((0., 0., 0.), (.5, 0., 0.), (1., .2, 0.))), (0., 0., 1.)))
    return model, AnalyticMetricChart(model, face)


def test_real_analytic_owner_batches_preserve_scalar_result():
    from anygeometry import to_dict
    model, chart = analytic_chart()
    before = to_dict(model)
    points, cells, protected = fixture()
    points = points @ chart.transform
    serial = repair(points, cells, protected, chart.evaluate, budget=31)
    batched = repair(points, cells, protected, chart.evaluate, budget=31, batch_size=8)
    assert_same_result(serial, batched)
    assert to_dict(model) == before


def test_analytic_owner_mutation_during_batch_is_rejected_by_after_check(monkeypatch):
    from anygeometry import GeometryError
    model, chart = analytic_chart()
    points, cells, protected = fixture()
    before = points.tobytes()
    original = type(model).evaluate_face_many
    calls = []
    def mutate_after_evaluation(owner, face, rows, *args, **kwargs):
        result = original(owner, face, rows, *args, **kwargs)
        if owner is model:
            calls.append(len(rows))
            if len(calls) == 3:
                owner.add_point(2., 2., 2.)
        return result
    monkeypatch.setattr(type(model), "evaluate_face_many", mutate_after_evaluation)
    with pytest.raises(GeometryError, match="stale"):
        repair(points, cells, protected, chart.evaluate, budget=8, batch_size=8)
    assert calls[-1] == 8 * len(points)
    assert points.tobytes() == before


@pytest.mark.parametrize("batch_size", (None, 8))
def test_surface_path_forwards_explicit_batching_and_keeps_generic_default_serial(monkeypatch, batch_size):
    import anymesher.surface_mesh as surface
    from anymesher.native_v2 import NativeMeshingOptions
    observed = []
    original = surface._run_frontal_quality_path
    def capture(*args, **kwargs):
        observed.append(kwargs["coordinate_batch_size"])
        return original(*args, **kwargs)
    monkeypatch.setattr(surface, "_run_frontal_quality_path", capture)
    extra = {} if batch_size is None else {"_coordinate_batch_size": batch_size}
    core = surface.mesh_planar_surface(((0., 0.), (1., 0.), (1., 1.), (0., 1.)),
        options=surface.SurfaceMeshOptions(target_size=1., recombine=False, backend="python",
            min_angle=15., native_options=NativeMeshingOptions(
                point_placement="frontal_delaunay", metric_mode="isotropic_spatial",
                max_insertions=8, max_topology_operations=32)),
        _metric_to_physical=lambda rows: np.column_stack((rows, np.zeros(len(rows)))),
        _metric_jacobian=np.asarray(((1., 0.), (0., 1.), (0., 0.))),
        _preserve_spatial_refinement=True, **extra)
    assert core.num_triangles > 0
    assert observed == [1 if batch_size is None else batch_size]


@pytest.mark.parametrize("analytic", (False, True))
def test_native_hybrid_enables_batches_only_for_analytic_owner_evaluator(monkeypatch, analytic):
    from anygeometry import GeometryModel, to_dict
    from anymesher import hybrid
    from anymesher._analytic_metric_chart import AnalyticMetricChart
    from anymesher.native_v2 import NativeMeshingOptions
    from anymesher.structured import StructuredMeshingOptions, MeshQualityPolicy
    if analytic:
        model, _ = analytic_chart()
    else:
        model = GeometryModel()
        model.add_plate(model.add_points(((0., 0., 0.), (1., 0., 0.),
                                          (1., 1., 0.), (0., 1., 0.))))
    before = to_dict(model)
    original = hybrid.mesh_planar_surface
    observed = []
    def capture(*args, **kwargs):
        observed.append(kwargs["_coordinate_batch_size"])
        if analytic:
            assert isinstance(kwargs["_metric_to_physical"].__self__, AnalyticMetricChart)
        return original(*args, **kwargs)
    monkeypatch.setattr(hybrid, "mesh_planar_surface", capture)
    result = hybrid.generate_hybrid_mesh_result(model, target_size=.5, strategy="native",
        recombine=False, native_backend="python",
        native_options=NativeMeshingOptions(point_placement="frontal_delaunay",
            metric_mode="isotropic_spatial", max_insertions=32, max_topology_operations=512),
        _native_surface_options=StructuredMeshingOptions(
            quality_policy=MeshQualityPolicy(minimum_angle=15.)))
    assert result.mesh.tris and observed
    assert observed == [8 if analytic else 1] * len(observed)
    assert to_dict(model) == before


def scoped_material_region_chart():
    from test_material_region_integration import controls, split_model
    from anymesher._analytic_metric_chart import AnalyticMetricChart
    from anymesher._material_region_binding import prepare_material_regions
    model, wall, descendants, _ = split_model()
    binding = prepare_material_regions(model, descendants, {wall: descendants},
        order="linear", native_options=controls())[min(descendants)]
    assert len(binding.face_ids) > 1
    assert binding.collection.regions == (binding.region,)
    assert binding.collection.source.charts == binding.region.sources
    return model, AnalyticMetricChart(model, binding.representative, region_binding=binding)


@pytest.mark.parametrize("budget", (1, 7, 8, 31))
def test_scoped_material_region_batches_preserve_exact_scalar_result(budget):
    from anygeometry import to_dict
    model, chart = scoped_material_region_chart()
    before = to_dict(model)
    points, cells, protected = fixture()
    points = points @ chart.transform
    original = points.tobytes()
    serial = RecordedOwner(len(points), chart.evaluate)
    batched = RecordedOwner(len(points), chart.evaluate)
    first = repair(points, cells, protected, serial, budget=budget)
    second = repair(points, cells, protected, batched, budget=budget, batch_size=8)
    assert_same_result(first, second)
    assert second.trials == budget
    assert len(serial.candidates) == len(batched.candidates)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(serial.candidates, batched.candidates))
    assert points.tobytes() == original
    assert second.points[:4].tobytes() == points[:4].tobytes()
    assert to_dict(model) == before
    if budget >= 7:
        assert max(batched.row_counts) > len(points)
        assert len(batched.row_counts) < len(serial.row_counts)


@pytest.mark.parametrize("stage", ("before", "during"))
def test_scoped_material_region_batch_rejects_stale_binding(monkeypatch, stage):
    from anygeometry import GeometryError
    import anygeometry.material_regions as owner
    model, chart = scoped_material_region_chart()
    points, cells, protected = fixture()
    points = points @ chart.transform
    before = points.tobytes(), cells.tobytes(), protected.tobytes()
    recorded = RecordedOwner(len(points), chart.evaluate)
    validations = []
    if stage == "before":
        model.add_point(2., 2., 2.)
    else:
        original = owner.validate_material_surface_regions_binding
        def mutate_at_after_check(*args, **kwargs):
            if args[0] is model:
                validations.append(args[1])
                # Two baseline calls each validate before and after evaluation;
                # the sixth check follows the first multi-probe owner operation.
                if len(validations) == 6:
                    model.add_point(2., 2., 2.)
            return original(*args, **kwargs)
        monkeypatch.setattr(owner, "validate_material_surface_regions_binding", mutate_at_after_check)
    with pytest.raises(GeometryError, match="stale"):
        repair(points, cells, protected, recorded, budget=8, batch_size=8)
    assert (points.tobytes(), cells.tobytes(), protected.tobytes()) == before
    if stage == "during":
        assert validations == [chart.binding] * 6
        assert recorded.row_counts == [len(points), len(points), 8 * len(points)]
