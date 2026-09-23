"""Local conformity repair for detached, already-generated T3 neighbours."""

import numpy as np

from .errors import MeshError


def _edges(nodes):
    return tuple(tuple(sorted((nodes[i], nodes[(i + 1) % 3]))) for i in range(3))


def propagate_triangle_split(mesh, face_ids, endpoints, node_id, *, cache):
    """Stage all incident-face replacements, then commit using source IDs.

    Each face incidence index is built once and updated locally. The caller
    owns the detached mesh, the registered straight-edge station and new node.
    Quadratic/recombined neighbours are deliberately not silently decomposed.
    """
    edge = tuple(sorted(map(int, endpoints)))
    plans = []
    next_id = max((*mesh.tris, *mesh.quads, *mesh.beams), default=0) + 1
    for face_id in sorted(set(face_ids)):
        elements = mesh.elements_of_face.get(face_id, ())
        if not elements:
            continue
        index = cache.get(face_id)
        if index is None:
            index = {}
            for element_id in elements:
                nodes = mesh.tris.get(element_id)
                if nodes is None or len(nodes) != 3:
                    raise MeshError("shared refinement requires staged linear T3 neighbours")
                for key in _edges(nodes):
                    index.setdefault(key, set()).add(element_id)
            cache[face_id] = index
        adjacent = index.get(edge, set())
        if len(adjacent) != 1:
            raise MeshError("shared-edge split requires exactly one incident boundary triangle per face")
        element_id = next(iter(adjacent))
        original = mesh.tris[element_id]
        start = next(
            i for i in range(3)
            if tuple(sorted((original[i], original[(i + 1) % 3]))) == edge
        )
        a, b, c = (original[(start + i) % 3] for i in range(3))
        replacements = ((a, node_id, c), (node_id, b, c))
        old = np.asarray([mesh.nodes[n] for n in (a, b, c)], dtype=np.float64)
        reference = np.cross(old[1] - old[0], old[2] - old[0])
        for nodes in replacements:
            points = np.asarray([mesh.nodes[n] for n in nodes], dtype=np.float64)
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            if not np.all(np.isfinite(normal)) or float(normal @ reference) <= 0.:
                raise MeshError("shared-edge split would invert or degenerate a neighbour")
        plans.append((face_id, index, element_id, next_id, original, replacements))
        next_id += 1
    for face_id, index, element_id, added_id, original, replacements in plans:
        for key in _edges(original):
            index[key].remove(element_id)
            if not index[key]:
                del index[key]
        mesh.tris[element_id], mesh.tris[added_id] = replacements
        mesh.elements_of_face[face_id].append(added_id)
        for identifier, nodes in zip((element_id, added_id), replacements):
            for key in _edges(nodes):
                index.setdefault(key, set()).add(identifier)
    return len(plans)


def repair_triangle_face(
    mesh, face_id, chart_points, *, protected_edges=(), cache,
    physical_evaluator=None, cancellation_check=None,
):
    """Repair a detached neighbour, keeping every protected point and edge exact."""
    from .optimization import constrained_smoothing, local_edge_flip
    from .quality_v2 import triangle_quality

    element_ids = tuple(sorted(mesh.elements_of_face[face_id]))
    if any(element not in mesh.tris or len(mesh.tris[element]) != 3 for element in element_ids):
        raise MeshError("shared refinement repair requires linear T3 neighbours")
    node_ids = tuple(sorted({node for element in element_ids for node in mesh.tris[element]}))
    local = {node: index for index, node in enumerate(node_ids)}
    points = np.asarray(chart_points, dtype=np.float64)
    if points.shape != (len(node_ids), 2) or not np.all(np.isfinite(points)):
        raise MeshError("shared refinement repair requires finite owner chart coordinates")
    before = np.asarray([[local[node] for node in mesh.tris[element]] for element in element_ids], dtype=np.int64)
    physical = np.asarray([mesh.nodes[node] for node in node_ids], dtype=np.float64)

    def signed_areas(triangles, coordinates=points):
        corners = coordinates[triangles]
        first, second = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
        return first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]

    def incidence(triangles):
        result = {}
        for row, triangle in enumerate(triangles):
            for edge in _edges(triangle):
                result.setdefault(edge, set()).add(row)
        return result

    areas = signed_areas(before)
    if not (np.all(areas > 0) or np.all(areas < 0)):
        raise MeshError("shared refinement neighbour has inconsistent chart orientation")
    original_incidence = incidence(before)
    if any(len(rows) > 2 for rows in original_incidence.values()):
        raise MeshError("shared refinement neighbour has invalid chart incidence")
    boundary = {edge for edge, rows in original_incidence.items() if len(rows) == 1}
    protected = boundary | {
        tuple(sorted((local[a], local[b])))
        for a, b in protected_edges
        if a in local and b in local and tuple(sorted((local[a], local[b]))) in original_incidence
    }
    result = local_edge_flip(points, before, protected_edges=sorted(protected), max_flips=max(16, 8 * len(before)))
    after = np.asarray(result.triangles, dtype=np.int64)
    if after.shape != before.shape or np.any(after < 0) or np.any(after >= len(points)):
        raise MeshError("shared refinement repair returned invalid connectivity")
    if np.all(areas < 0):
        after = after[:, (0, 2, 1)]
    changed_areas = signed_areas(after)
    changed_incidence = incidence(after)
    if (
        not np.all(changed_areas * np.sign(areas[0]) > 0)
        or set(map(int, after.ravel())) != set(range(len(node_ids)))
        or len({tuple(sorted(row)) for row in after}) != len(after)
        or any(len(rows) > 2 for rows in changed_incidence.values())
        or {edge for edge, rows in changed_incidence.items() if len(rows) == 1} != boundary
        or any(len(changed_incidence.get(edge, ())) != len(original_incidence[edge]) for edge in protected)
        or not np.isclose(np.sum(changed_areas), np.sum(areas), rtol=1.0e-12, atol=0.0)
    ):
        raise MeshError("shared refinement repair changed protected topology or chart coverage")

    def score(triangles, coordinates=physical):
        quality = triangle_quality(coordinates, triangles)
        if not all(np.all(np.isfinite(values)) for values in (quality.area, quality.aspect_ratio, quality.scaled_jacobian)):
            raise MeshError("shared refinement repair returned non-finite physical quality")
        return (
            int(np.count_nonzero((quality.area <= 0) | (quality.scaled_jacobian <= 0))),
            int(np.count_nonzero(quality.aspect_ratio > 5.0)),
            float(np.max(quality.aspect_ratio)),
            -float(np.min(quality.scaled_jacobian)),
        )

    initial_score, candidate_score = score(before), score(after)
    if candidate_score[0]:
        raise MeshError("shared refinement repair failed physical validity")
    committed_flips = int(result.flip_count) if candidate_score < initial_score else 0
    attempted_flips = int(result.flip_count)
    queue_visits = int(result.queue_visits)
    if not candidate_score < initial_score:
        after = before
        candidate_score = initial_score
    final_physical = physical
    moved_nodes = np.empty(0, dtype=np.int64)
    smoothing_iterations = 0
    attempted_smoothing_iterations = 0
    if candidate_score[1] and physical_evaluator is not None:
        if cancellation_check is not None:
            cancellation_check(f"cylindrical neighbour {face_id} smoothing start")
        smoothed = constrained_smoothing(
            points, after, constrained_edges=sorted(protected),
            iterations=4, relaxation=0.6,
        )
        attempted_smoothing_iterations = int(smoothed.iterations)
        smooth_points = np.asarray(smoothed.points, dtype=np.float64)
        fixed = np.asarray(sorted({node for edge in protected for node in edge}), dtype=np.int64)
        if (
            smooth_points.shape != points.shape
            or not np.all(np.isfinite(smooth_points))
            or smooth_points[fixed].tobytes() != points[fixed].tobytes()
        ):
            raise MeshError("shared refinement smoothing changed protected coordinates")
        moved = np.flatnonzero(np.any(smooth_points != points, axis=1))
        lifted = physical.copy()
        if len(moved):
            evaluated = np.asarray(physical_evaluator(smooth_points[moved]), dtype=np.float64)
            if evaluated.shape != (len(moved), 3) or not np.all(np.isfinite(evaluated)):
                raise MeshError("shared refinement smoothing returned invalid owner coordinates")
            lifted[moved] = evaluated
        final_flip = local_edge_flip(
            smooth_points, after, protected_edges=sorted(protected),
            max_flips=max(16, 8 * len(after)),
        )
        attempted_flips += int(final_flip.flip_count)
        queue_visits += int(final_flip.queue_visits)
        smoothed_rows = np.asarray(final_flip.triangles)
        if (
            smoothed_rows.dtype.kind not in "iu"
            or smoothed_rows.shape != before.shape
            or np.any(smoothed_rows < 0)
            or np.any(smoothed_rows >= len(points))
        ):
            raise MeshError("shared refinement smoothing returned invalid connectivity")
        if np.all(areas < 0):
            smoothed_rows = smoothed_rows[:, (0, 2, 1)]
        smooth_areas = signed_areas(smoothed_rows, smooth_points)
        smooth_incidence = incidence(smoothed_rows)
        if (
            not np.all(smooth_areas * np.sign(areas[0]) > 0)
            or set(map(int, smoothed_rows.ravel())) != set(range(len(node_ids)))
            or len({tuple(sorted(row)) for row in smoothed_rows}) != len(smoothed_rows)
            or any(len(rows) > 2 for rows in smooth_incidence.values())
            or {edge for edge, rows in smooth_incidence.items() if len(rows) == 1} != boundary
            or any(len(smooth_incidence.get(edge, ())) != len(original_incidence[edge]) for edge in protected)
            or not np.isclose(np.sum(smooth_areas), np.sum(areas), rtol=1.0e-12, atol=0.0)
        ):
            raise MeshError("shared refinement smoothing changed protected topology or coverage")
        smooth_score = score(smoothed_rows, lifted)
        if smooth_score[0]:
            raise MeshError("shared refinement smoothing failed physical validity")
        if smooth_score < candidate_score:
            after, candidate_score = smoothed_rows, smooth_score
            final_physical, moved_nodes = lifted, moved
            committed_flips += int(final_flip.flip_count)
            smoothing_iterations = int(smoothed.iterations)
    selected = candidate_score < initial_score
    if selected:
        replacements = {element: tuple(node_ids[index] for index in row) for element, row in zip(element_ids, after)}
        replacement_nodes = {node_ids[index]: tuple(map(float, final_physical[index])) for index in moved_nodes}
        replacement_index = {}
        for element, nodes in replacements.items():
            for edge in _edges(nodes):
                replacement_index.setdefault(edge, set()).add(element)
        if cancellation_check is not None:
            cancellation_check(f"cylindrical neighbour {face_id} repair before publish")
        mesh.nodes.update(replacement_nodes)
        mesh.tris.update(replacements)
        cache[face_id] = replacement_index
    return {
        "selected": selected,
        "flips": committed_flips if selected else 0,
        "attempted_flips": attempted_flips,
        "queue_visits": queue_visits,
        "initial_score": initial_score,
        "final_score": candidate_score if selected else initial_score,
        "coordinates_changed": bool(selected and len(moved_nodes)),
        "moved_nodes": int(len(moved_nodes)) if selected else 0,
        "smoothing_iterations": smoothing_iterations if selected else 0,
        "attempted_smoothing_iterations": attempted_smoothing_iterations,
        "protected_coordinates_changed": False,
    }


def repair_completed_cylindrical_neighbours(mesh, geometry, binding, registry, *, cancellation_check):
    """Run once all certified faces exist and no shared splits are pending."""
    cache = getattr(registry, "_published_triangle_incidence", {})
    sectors = binding.face_records
    if not cache or any(not mesh.elements_of_face.get(sector.face.id) for sector in sectors):
        return {}
    protected = tuple(
        tuple(sorted((a, b)))
        for sequence in mesh.nodes_of_edge.values()
        for a, b in zip(sequence, sequence[1:])
    )
    reports = {}
    for sector in sorted(sectors, key=lambda value: value.face.id):
        face_id = sector.face.id
        if face_id not in cache:
            continue
        if cancellation_check is not None:
            cancellation_check(f"cylindrical shared-boundary repair face {face_id}")
        chart = binding.chart_for(sector.face_use)
        nodes = sorted({node for element in mesh.elements_of_face[face_id] for node in mesh.tris[element]})
        physical = np.asarray([mesh.nodes[node] for node in nodes], dtype=np.float64)
        _, uv, distances = geometry.project_to_face_many(face_id, physical)
        scale = np.asarray((chart.circumferential_length, chart.axial_length))
        tolerance = geometry.tolerance.effective_length(float(np.max(scale)))
        if not np.all(np.isfinite(distances)) or np.any(distances > tolerance):
            raise MeshError("shared refinement neighbour left its owner cylinder")
        reports[str(face_id)] = repair_triangle_face(
            mesh, face_id, uv * scale, protected_edges=protected, cache=cache,
            physical_evaluator=chart.evaluate, cancellation_check=cancellation_check,
        )
    if cancellation_check is not None:
        cancellation_check("cylindrical shared-boundary repair complete")
    binding.validate()
    return reports
