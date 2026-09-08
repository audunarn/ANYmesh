"""Private, component-atomic LINEAR cylinder recombination; no public activation."""

from copy import copy, deepcopy
from dataclasses import dataclass, field

import numpy as np

from .core import MeshCore, corner_edges
from .errors import MeshError
from .metric import MetricFieldSpec, SpatialMetricField
from .quality_v2 import assert_valid_mesh, evaluate_quality
from . import surface_mesh


@dataclass
class _Component:
    binding: object
    faces: tuple
    phase: str = "collecting"
    settings: dict = field(default_factory=dict)
    staged: dict = field(default_factory=dict)
    attempts: dict = field(default_factory=dict)
    metric_fields: dict = field(default_factory=dict)


def assert_face_open(registry, face_id):
    for state in getattr(registry, "_deferred_cylindrical_components", {}).values():
        if face_id in state.faces and state.phase != "collecting":
            raise MeshError("late cylindrical split/refinement after component closure")


def register_face(geometry, mesh, face_id, binding, registry, settings, *,
                  effective_metric_field, supplemental_metric_field=None):
    from ._cylindrical_atlas import CylindricalAtlasBinding
    from ._cylindrical_patch import CylindricalPatchBinding

    if not isinstance(binding, (CylindricalAtlasBinding, CylindricalPatchBinding)):
        raise MeshError("deferred recombination requires an owner-certified cylinder binding")
    binding.validate()
    if settings.quadratic or not settings.recombine or settings.native_options.point_placement != "frontal_delaunay":
        raise MeshError("deferred cylindrical recombination is linear frontal only")
    if settings.native_options.experimental_metric_provider is not None:
        raise MeshError("deferred cylindrical recombination does not admit external metric providers")
    if not isinstance(effective_metric_field, MetricFieldSpec):
        raise MeshError("deferred cylindrical recombination requires the bound effective metric field")
    if supplemental_metric_field is not None:
        # Native supplemental fields require the excluded experimental provider.
        raise MeshError("deferred cylindrical supplemental metric requires an excluded external provider")
    faces = tuple(sorted(sector.face.id for sector in binding.face_records))
    if len(faces) != len(set(faces)) or face_id not in faces:
        raise MeshError("deferred cylindrical component has ambiguous face ownership")
    for member in faces:
        face = geometry.faces[member]
        for loop in (face.loop, *face.holes):
            for edge in loop:
                if (not isinstance(binding, CylindricalPatchBinding)
                        and set(geometry.faces_using_edge(edge.edge)).difference(faces)):
                    raise MeshError("deferred cylindrical component has incompatible external neighbours")
    states = getattr(registry, "_deferred_cylindrical_components", None)
    if states is None:
        states = {}
        registry._deferred_cylindrical_components = states
    key = (binding.model_id, binding.revision, faces)
    state = states.get(key)
    if state is None:
        if any(set(other.faces).intersection(faces) for other in states.values()):
            raise MeshError("overlapping deferred cylindrical components")
        state = _Component(binding, faces)
        states[key] = state
    assert_face_open(registry, face_id)
    if face_id in state.settings:
        raise MeshError("duplicate deferred cylindrical face registration")
    state.settings[face_id] = settings
    state.metric_fields[face_id] = {
        "effective": effective_metric_field,
        "supplemental": supplemental_metric_field,
        "receipt": {
            "effective_metric_field": effective_metric_field.to_dict(),
            "supplemental_metric_field": None,
            "native_metric_mode": settings.native_options.metric_mode,
            "model_id": str(geometry.model_id), "geometry_revision": geometry.revision,
            "origin": "hybrid bound effective field after SizeField/native composition",
        },
    }
    return state


def mark_staged(registry, face_id, mesh, diagnostics):
    assert_face_open(registry, face_id)
    states = [s for s in registry._deferred_cylindrical_components.values() if face_id in s.settings]
    if len(states) != 1 or face_id in states[0].staged:
        raise MeshError("duplicate or unregistered deferred cylindrical face")
    ids = tuple(mesh.elements_of_face.get(face_id, ()))
    if not ids or any(e not in mesh.tris or len(mesh.tris[e]) != 3 for e in ids):
        raise MeshError("deferred cylindrical staging requires linear T3 ownership")
    states[0].staged[face_id] = {
        "diagnostics": diagnostics,
        "node_ids": tuple(sorted({n for e in ids for n in mesh.tris[e]})),
        "published_insertions": int(diagnostics.get("native_v2", {}).get("published_insertions", 0)),
    }


def _checkpoint(callback, phase):
    if callback is not None:
        callback(phase)


def _active(core):
    return (
        core.triangle_connectivity[np.flatnonzero(core.triangle_active)],
        core.quad_connectivity[np.flatnonzero(core.quad_active)],
    )


def _incidence(triangles, quads):
    result = {}
    for cells in (triangles, quads):
        for cell in cells:
            for edge in corner_edges(cell):
                result[edge] = result.get(edge, 0) + 1
    return result


def _physical_size(core, effective_metric_field, geometry, callback):
    """Physical endpoint/midpoint measurements; no absolute publication limit."""
    triangles, quads = _active(core)
    edges = sorted(_incidence(triangles, quads))
    ends = core.node_coordinates[np.asarray(edges, dtype=np.int64)]
    delta = ends[:, 1] - ends[:, 0]
    provider = SpatialMetricField(effective_metric_field, model_uuid=str(geometry.model_id), geometry_revision=geometry.revision)
    ratios = []
    for fraction in (0.0, 0.5, 1.0):
        tensors = provider.evaluate(ends[:, 0] + fraction * delta, cancellation_check=callback)
        ratios.append(np.sqrt(np.einsum("ni,nij,nj->n", delta, tensors, delta)))
    maximum = float(np.max(ratios))
    return {
        "finite": bool(np.all(np.isfinite(ratios))),
        "maximum_sampled_edge_metric_length": maximum,
        "maximum_physical_edge_length": float(np.max(np.linalg.norm(delta, axis=1))),
        "edge_count": len(edges), "samples_per_edge": 3,
        "edges": [list(edge) for edge in edges],
        "sample_fractions": [0.0, 0.5, 1.0],
        "sampled_edge_metric_lengths": np.asarray(ratios).T.tolist(),
        "physical_edge_lengths": np.linalg.norm(delta, axis=1).tolist(),
    }


def _sizing_non_regression(before, after):
    """Requested relative check, using identical coordinates and metric samples."""
    reference = {tuple(edge): i for i, edge in enumerate(before["edges"])}
    if not before["finite"] or not after["finite"]:
        return False
    for i, edge in enumerate(after["edges"]):
        j = reference.get(tuple(edge))
        if j is None:
            return False
        old = np.asarray(before["sampled_edge_metric_lengths"][j])
        new = np.asarray(after["sampled_edge_metric_lengths"][i])
        if np.any(new > old + 1.e-12 * np.maximum(1.0, np.abs(old))):
            return False
        length = before["physical_edge_lengths"][j]
        if after["physical_edge_lengths"][i] > length + 1.e-12 * max(1.0, abs(length)):
            return False
    return bool(
        after["maximum_sampled_edge_metric_length"] <= before["maximum_sampled_edge_metric_length"] + 1.e-12
        and after["maximum_physical_edge_length"] <= before["maximum_physical_edge_length"] + 1.e-12
    )


def _cell_quality(core, settings, source_triangles):
    quality = evaluate_quality(core)
    rows = []
    for kind, cells, group in (("T3", _active(core)[0], quality.triangles),
                               ("Q4", _active(core)[1], quality.quadrilaterals)):
        for i, cell in enumerate(cells):
            values = {name: float(getattr(group, name)[i]) for name in (
                "scaled_jacobian", "aspect_ratio", "minimum_angle", "maximum_angle", "warpage")}
            violations = [name for name, failed in (
                ("scaled_jacobian", values["scaled_jacobian"] < settings.min_scaled_jacobian),
                ("aspect_ratio", values["aspect_ratio"] > settings.max_aspect_ratio),
                ("minimum_angle", values["minimum_angle"] < settings.min_angle),
                ("maximum_angle", values["maximum_angle"] > settings.max_angle),
                ("warpage", values["warpage"] > settings.max_warpage)) if failed]
            rows.append({"kind": kind, "connectivity": cell.tolist(), "values": values,
                         "violations": violations,
                         "source_triangle_id": source_triangles.get(tuple(int(n) for n in cell)),
                         "provenance": "unchanged_repaired_T3" if kind == "T3" else "new_proposed_Q4"})
    return rows


def _recombine_face(geometry, working, face_id, state, callback):
    settings = state.settings[face_id]
    effective_metric_field = state.metric_fields[face_id]["effective"]
    elements = tuple(sorted(working.elements_of_face[face_id]))
    if any(e not in working.tris or len(working.tris[e]) != 3 for e in elements):
        raise MeshError("deferred finalization requires authoritative repaired T3 faces")
    ids = tuple(sorted({n for e in elements for n in working.tris[e]}))
    local = {node: row for row, node in enumerate(ids)}
    xyz = np.asarray([working.nodes[n] for n in ids], dtype=np.float64)
    sector = next(s for s in state.binding.face_records if s.face.id == face_id)
    chart = state.binding.chart_for(sector.face_use)
    _, uv, distances = geometry.project_to_face_many(face_id, xyz)
    scale = np.asarray((chart.circumferential_length, chart.axial_length))
    tolerance = geometry.tolerance.effective_length(float(np.max(scale)))
    if not np.all(np.isfinite(distances)) or np.any(distances > tolerance):
        raise MeshError("deferred recombination left its certified owner chart")
    points = np.asarray(uv) * scale
    triangles = np.asarray([[local[n] for n in working.tris[e]] for e in elements], dtype=np.int64)
    before = _incidence(triangles, ())
    boundary = {edge for edge, count in before.items() if count == 1}
    groups = []
    face = geometry.faces[face_id]
    for loop in (face.loop, *face.holes):
        group = []
        for oriented in loop:
            sequence = tuple(working.nodes_of_edge[oriented.edge])
            if not oriented.forward:
                sequence = sequence[::-1]
            group.extend((local[a], local[b]) for a, b in zip(sequence, sequence[1:]))
        groups.append(tuple(group))
    protected = {tuple(sorted(edge)) for group in groups for edge in group}
    if boundary != protected:
        raise MeshError("deferred recombination boundary differs from protected owner stations")
    core = MeshCore(points, triangles)
    pre_physical = MeshCore(xyz, triangles)
    assert_valid_mesh(pre_physical)
    pre_chart_quality = surface_mesh._quality_threshold_report(evaluate_quality(core), settings)
    pre_physical_quality = surface_mesh._quality_threshold_report(evaluate_quality(pre_physical), settings)
    pre_size = _physical_size(pre_physical, effective_metric_field, geometry, callback)
    source_triangles = {tuple(int(n) for n in cell): element for cell, element in zip(triangles, elements)}
    report, chart_quality = surface_mesh._qualified_recombination(core, sorted(protected), settings, callback)
    active_triangles, active_quads = _active(report.mesh)
    after = _incidence(active_triangles, active_quads)
    used = {int(n) for cells in (active_triangles, active_quads) for cell in cells for n in cell}
    if (
        not np.array_equal(report.mesh.node_coordinates, core.node_coordinates)
        or used != set(range(len(ids)))
        or {edge for edge, count in after.items() if count == 1} != boundary
        or any(after.get(edge) != before[edge] for edge in protected)
    ):
        raise MeshError("deferred recombination changed published refinement or protected identity")
    physical = MeshCore(xyz, active_triangles, active_quads)
    assert_valid_mesh(physical)
    physical_quality = surface_mesh._quality_threshold_report(evaluate_quality(physical), settings)
    size = _physical_size(physical, effective_metric_field, geometry, callback)
    size["accepted"] = _sizing_non_regression(pre_size, size)
    size["policy"] = "repaired_pre_pair_endpoint_midpoint_non_regression"
    result = {
        "chart_quality": chart_quality, "physical_quality": physical_quality,
        "physical_size": size, "pair_count": report.pair_count,
        "exchange_truncated": report.exchange_truncated,
        "repaired_node_ids": ids, "protected_node_ids": tuple(sorted({ids[n] for edge in protected for n in edge})),
        "published_insertions": state.staged[face_id]["published_insertions"],
        "effective_physical_field_receipt": state.metric_fields[face_id]["receipt"],
        "requested_settings": {name: getattr(settings, name) for name in (
            "enforce_quality", "prefer_quality_policy", "min_scaled_jacobian", "max_aspect_ratio",
            "min_angle", "max_angle", "max_warpage", "target_size")},
        "repaired_pre_pair": {
            "source_element_ids": elements, "source_node_ids": ids,
            "physical_coordinates": xyz.tolist(), "chart_coordinates": points.tolist(),
            "triangles": triangles.tolist(), "quads": [],
            "chart_quality": pre_chart_quality, "physical_quality": pre_physical_quality,
            "physical_size": pre_size, "physical_cells": _cell_quality(pre_physical, settings, source_triangles)},
        "post_recombination_proposal": {
            "source_node_ids": ids, "physical_coordinates": xyz.tolist(),
            "chart_coordinates": points.tolist(), "triangles": active_triangles.tolist(),
            "quads": active_quads.tolist(), "chart_quality": chart_quality,
            "physical_quality": physical_quality, "physical_size": size,
            "physical_cells": _cell_quality(physical, settings, source_triangles)},
        "outer_alignment": surface_mesh._boundary_alignment(report.mesh, points, groups[0]),
        "hole_alignment": surface_mesh._boundary_alignment(report.mesh, points, tuple(e for g in groups[1:] for e in g)),
    }
    state.attempts[face_id] = result
    if settings.enforce_quality and (not chart_quality["accepted"] or not physical_quality["accepted"]):
        raise MeshError("deferred recombination rejected requested physical/chart quality before commit")
    if not size["accepted"]:
        raise MeshError("deferred recombination rejected requested physical size before commit: pre-pair non-regression")
    return physical, dict(enumerate(ids)), result


def finalize_components(geometry, mesh, registry, cancellation_check=None):
    from ._shared_triangle_split import repair_completed_cylindrical_neighbours
    from .hybrid import _neutral_shell_core, _publish_native_face_elements

    states = tuple(getattr(registry, "_deferred_cylindrical_components", {}).values())
    if not states:
        return {}
    for state in states:
        if state.phase != "collecting":
            raise MeshError("duplicate deferred cylindrical finalization")
        if set(state.staged) != set(state.faces) or any(not mesh.elements_of_face.get(f) for f in state.faces):
            raise MeshError("incomplete deferred cylindrical component")
        state.binding.validate()
    working = deepcopy(mesh)
    repair_registry = copy(registry)
    repair_registry._published_triangle_incidence = deepcopy(getattr(registry, "_published_triangle_incidence", {}))
    protected = {n: np.asarray(mesh.nodes[n]).tobytes() for sequence in mesh.nodes_of_edge.values() for n in sequence}
    all_results = {}
    try:
        for state in states:
            state.phase = "finalizing"
            _checkpoint(cancellation_check, "cylindrical deferred repair start")
            repairs = repair_completed_cylindrical_neighbours(
                working, geometry, state.binding, repair_registry, cancellation_check=cancellation_check
            )
            for face_id in state.faces:
                old = set(working.elements_of_face[face_id])
                if working.grid_of_face.get(face_id) is not None or any(
                    old.intersection(elements) for table in (working.elements_of_edge, working.elements_of_member)
                    for elements in table.values()
                ):
                    raise MeshError("deferred recombination has incompatible external element associations")
                core, ids, result = _recombine_face(geometry, working, face_id, state, cancellation_check)
                for element in old:
                    del working.tris[element]
                    working.activity.pop(element, None)
                _publish_native_face_elements(working, face_id, core, ids)
                for element in working.elements_of_face[face_id]:
                    working.activity[element] = 1.0
                result["repair"] = repairs.get(str(face_id))
                all_results[face_id] = result
        for sheet_id, sheet in geometry.sheets.items():
            working.elements_of_sheet[sheet_id] = sorted({
                element for use in sheet.face_use_ids
                for element in working.elements_of_face.get(geometry.face_uses[use].face_id, ())
            })
        assert_valid_mesh(_neutral_shell_core(working))
        if set(working.nodes) != set(mesh.nodes) or working.nodes_of_edge != mesh.nodes_of_edge:
            raise MeshError("deferred recombination changed source node/edge identity")
        if any(np.asarray(working.nodes[n]).tobytes() != data for n, data in protected.items()):
            raise MeshError("deferred recombination changed protected coordinate bytes")
        _checkpoint(cancellation_check, "cylindrical deferred before commit")
        # The last caller callback can mutate the owner or leave its journal open.
        # Authoritative checks follow that callback; no caller code runs afterward.
        for state in states:
            state.binding.validate()
    except BaseException:
        for state in states:
            state.phase = "rejected"
        raise
    # No callback, geometry evaluation, or fallible qualification within publication.
    mesh.__dict__.update({name: getattr(working, name) for name in (
        "nodes", "tris", "quads", "elements_of_face", "elements_of_sheet", "activity"
    )})
    for state in states:
        state.phase = "committed"
        for face_id in state.faces:
            state.staged[face_id]["diagnostics"]["deferred_cylindrical_recombination"] = all_results[face_id]
            getattr(registry, "_published_triangle_incidence", {}).pop(face_id, None)
    return all_results
