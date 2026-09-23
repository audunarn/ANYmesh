"""Private whole-atlas quadratic refinement trial; public activation is separate."""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np

from ._cylindrical_atlas import CylindricalAtlasBinding
from ._cylindrical_patch import CylindricalPatchBinding
from ._cylindrical_quadratic import QuadraticComponentStage
from ._shared_triangle_split import propagate_triangle_split
from .core import MeshCore, corner_edges
from .errors import MeshError
from .native_v2 import ComponentSeedRegistry, frontal_delaunay_refine
from .quality_v2 import assert_valid_mesh, evaluate_quality
from .triangulation import PlanarTriangulation
from . import surface_mesh


@dataclass(frozen=True)
class QuadraticRefinementTrial:
    accepted: bool
    mesh: object
    diagnostics: dict
    boundary_stations: tuple = ()


def refine_quadratic_component(geometry, seed, binding, station_entries, settings, *,
                               eligible_edges=(), metric_model_uuid=None,
                               metric_geometry_revision=None, cancellation_check=None):
    """Refine a complete certified native sector component without publication.

    Seed creation remains the existing hybrid quadratic path. All faces stay T3
    until shared splits and the final repair sweep are complete. Failure of a
    requested quality gate returns the original whole component, never a mixed
    split/unsplit mesh. Operational failures and cancellation propagate.
    """
    if not isinstance(binding, (CylindricalAtlasBinding, CylindricalPatchBinding)):
        raise MeshError("quadratic refinement requires an owner-certified cylinder binding")
    binding.validate()
    faces = tuple(sorted(sector.face.id for sector in binding.face_records))
    if (seed.geometry_model_id, seed.geometry_revision) != (geometry.model_id, geometry.revision):
        raise MeshError("quadratic seed does not belong to the current working geometry")
    if set(seed.elements_of_face) != set(faces) or seed.beams or seed.couplings:
        raise MeshError("quadratic atlas trial requires a complete native-only component")
    if not settings.quadratic or settings.native_options.point_placement != "frontal_delaunay":
        raise MeshError("quadratic atlas trial requires quadratic frontal options")
    if settings.native_options.experimental_metric_provider is not None:
        raise MeshError("quadratic atlas trial does not admit unbound metric callbacks")
    from anygeometry.curves import Straight
    for edge in eligible_edges:
        if not isinstance(geometry.edges[edge].curve, Straight) or not set(geometry.faces_using_edge(edge)).issubset(faces):
            raise MeshError("quadratic shared splits require component-owned straight edges")
    stage = QuadraticComponentStage(seed, faces, station_entries, eligible_edges=eligible_edges)
    working = stage.mesh
    # Local max-ID allocators must not reuse an ID occupied by another component.
    highest_reserved_node = max(seed.nodes, default=0)
    if highest_reserved_node and highest_reserved_node not in working.nodes:
        working.nodes[highest_reserved_node] = np.asarray(seed.nodes[highest_reserved_node]).copy()

    def kernel_check(phase="quadratic cylindrical kernel"):
        if cancellation_check is not None:
            cancellation_check(phase)
    native_settings = replace(settings, order="linear", recombine=False)
    next_node = max(seed.nodes, default=0) + 1
    def allocate_node():
        nonlocal next_node
        node = next_node
        next_node += 1
        return node
    registry = ComponentSeedRegistry(next_node, node_id_allocator=allocate_node)
    next_element = max(seed.shells, default=0) + 1
    incidence_cache = {}
    diagnostics = {"faces": {}, "attempted_insertions": 0, "published_insertions": 0,
                   "reserved_node_reuses": 0, "accepted": False}
    charts = {sector.face.id: binding.chart_for(sector.face_use) for sector in binding.face_records}

    def checkpoint(phase):
        if cancellation_check is not None:
            cancellation_check(phase)
        binding.validate()

    def projection(face, xyz):
        _, uv, distances = geometry.project_to_face_many(face, np.asarray(xyz, dtype=np.float64))
        chart = charts[face]
        scale = np.array((chart.circumferential_length, chart.axial_length))
        tolerance = geometry.tolerance.effective_length(float(np.max(scale)))
        if not np.all(np.isfinite(distances)) or np.any(distances > tolerance):
            raise MeshError("quadratic candidate left its certified owner chart")
        result = np.asarray(uv) * scale
        if not np.all(np.isfinite(result)):
            raise MeshError("quadratic chart projection is non-finite")
        return result

    def face_data(face):
        elements = sorted(working.elements_of_face[face])
        ids = tuple(sorted({n for e in elements for n in working.tris[e]}))
        local = {n: i for i, n in enumerate(ids)}
        points = projection(face, [working.nodes[n] for n in ids])
        triangles = np.array([[local[n] for n in working.tris[e]] for e in elements], dtype=np.int64)
        p = points[triangles]
        delta, other = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
        signs = delta[:, 0] * other[:, 1] - delta[:, 1] * other[:, 0]
        sign = 1 if np.all(signs > 0) else -1 if np.all(signs < 0) else 0
        if not sign:
            raise MeshError("quadratic seed has inconsistent chart orientation")
        if sign < 0:
            triangles = triangles[:, (0, 2, 1)]
        # The native cavity contract compares canonical triangle identities.
        # Hybrid element cycles retain owner orientation but need not begin
        # with their smallest local node, so normalize before entering T3.
        triangles = np.asarray(sorted(
            tuple(int(n) for n in np.roll(row, -int(np.argmin(row))))
            for row in triangles
        ), dtype=np.int64)
        loops = []
        intervals = {}
        owner = geometry.faces[face]
        for loop in (owner.loop, *owner.holes):
            sequence = []
            for use in loop:
                nodes = tuple(working.nodes_of_edge[use.edge])
                if not use.forward:
                    nodes = nodes[::-1]
                sequence.extend(local[n] for n in nodes[:-1])
                if use.edge in stage._eligible:
                    for a, b in zip(nodes, nodes[1:]):
                        intervals[(local[a], local[b])] = (
                            use.edge, stage.station(use.edge, a), stage.station(use.edge, b))
            ring = np.array(sequence, dtype=np.int64)
            q = points[ring]
            area = np.sum(q[:, 0] * np.roll(q[:, 1], -1) - q[:, 1] * np.roll(q[:, 0], -1))
            if (not loops and area < 0) or (loops and area > 0):
                ring = ring[::-1]
            loops.append(ring)
        edges = np.array(sorted({tuple(sorted((int(a), int(b))))
                                 for ring in loops for a, b in zip(ring, np.roll(ring, -1))}), dtype=np.int64)
        return ids, points, triangles, loops, edges, intervals, sign

    def install_face(face, ids, points, triangles, sign, shared=()):
        nonlocal next_node, next_element
        mapping = dict(enumerate(ids))
        by_row = {row["local_node_id"]: row for row in shared}
        protected = {n for sequence in working.nodes_of_edge.values() for n in sequence}
        for row, point in enumerate(points):
            if row not in mapping:
                if row in by_row:
                    mapping[row] = int(by_row[row]["node_id"])
                else:
                    mapping[row] = next_node
                    next_node += 1
            node = mapping[row]
            if node in stage._protected or node in protected:
                continue
            working.nodes[node] = np.asarray(charts[face].evaluate(np.asarray((point,))))[0].copy()
        from ._element_identity import next_unoccupied_shell_id
        next_element = next_unoccupied_shell_id(working, next_element)
        for e in working.elements_of_face[face]:
            del working.tris[e]
        new_ids = []
        for tri in triangles:
            nodes = tuple(mapping[int(i)] for i in tri)
            if sign < 0:
                nodes = (nodes[0], nodes[2], nodes[1])
            working.tris[next_element] = nodes
            new_ids.append(next_element)
            next_element += 1
        working.elements_of_face[face] = new_ids
        incidence_cache.pop(face, None)

    def optimize(points, triangles, edges):
        initial = surface_mesh._make_candidate(points, triangles, settings=native_settings)
        result = surface_mesh._optimize_candidate(initial, edges, np.empty((0, 2)),
                                                  settings=native_settings, prefer_growth=True)
        protected = np.unique(edges)
        if result.points[protected].tobytes() != points[protected].tobytes():
            raise MeshError("quadratic repair moved protected chart nodes")
        return result

    from ._quadratic_boundary_prepare import synchronize_existing_midpoints
    preparation = synchronize_existing_midpoints(
        seed, working, stage, registry, faces, eligible_edges,
        geometry.faces_using_edge, projection, incidence_cache,
        settings.native_options.max_topology_operations,
        lambda phase="quadratic_boundary_preparation": (
            cancellation_check(phase) if cancellation_check is not None else None
        ),
    )
    checkpoint("quadratic_boundary_preparation_complete")
    diagnostics["boundary_preparation"] = preparation
    diagnostics["reserved_node_reuses"] += preparation["reused_node_count"]
    pending = list(faces)
    queued = set(faces)
    spent_points = dict.fromkeys(faces, 0)
    spent_operations = {face: preparation["face_operations"][str(face)] for face in faces}
    visits = dict.fromkeys(faces, 0)

    def schedule(face):
        if face not in queued and visits[face] < 3:
            pending.append(face)
            queued.add(face)

    while pending:
        face = pending.pop(0)
        queued.remove(face)
        remaining_points = settings.native_options.max_insertions - spent_points[face]
        remaining_operations = settings.native_options.max_topology_operations - spent_operations[face]
        if remaining_points <= 0 or remaining_operations <= 0:
            continue
        visits[face] += 1
        checkpoint("quadratic component face refinement start")
        ids, points, triangles, loops, edges, intervals, sign = face_data(face)
        from ._cdt_restore import restore_constrained_delaunay
        restored = restore_constrained_delaunay(points, triangles, edges,
                                               max_flips=remaining_operations,
                                               cancellation_check=kernel_check)
        spent_operations[face] += restored.flips
        remaining_operations -= restored.flips
        if not restored.converged or remaining_operations <= 0:
            diagnostics["reason"] = "constrained_delaunay_restoration_budget"
            return QuadraticRefinementTrial(False, deepcopy(seed), diagnostics)
        triangles = restored.triangles
        seed_tri = PlanarTriangulation(points=points, triangles=triangles, segments=edges,
                                      boundary_segments=edges, mandatory_segments=np.empty((0, 2), dtype=np.int64),
                                      outer_loop=loops[0], hole_loops=tuple(loops[1:]))
        chart = charts[face]
        proposal = stage.split_proposal(
            lambda edge, node, physical: projection(face, (physical,))[0],
            evaluate_edge=lambda edge, t: geometry.sample_edge(edge, (t,))[0])
        refined, report = frontal_delaunay_refine(
            seed_tri, replace(settings.native_options, max_insertions=remaining_points,
                              max_topology_operations=remaining_operations), target_size=settings.target_size,
            model_uuid=metric_model_uuid or str(geometry.model_id),
            geometry_revision=geometry.revision if metric_geometry_revision is None else metric_geometry_revision,
            metric_to_physical=chart.evaluate, metric_jacobian=chart.jacobians,
            automatically_seeded_shared_segments=intervals, component_seed_registry=registry,
            cancellation_check=cancellation_check, _split_proposal=proposal)
        for row in report["shared_nodes"]:
            edge, node = row["edge_id"], row["node_id"]
            sequence = working.nodes_of_edge[edge]
            if node in sequence:
                continue
            station = Fraction(*row["station"])
            found = [(i, a, b) for i, (a, b) in enumerate(zip(sequence, sequence[1:]))
                     if min(stage.station(edge, a), stage.station(edge, b)) < station <
                     max(stage.station(edge, a), stage.station(edge, b))]
            if len(found) != 1:
                raise MeshError("quadratic shared split lost its exact owner station")
            i, a, b = found[0]
            stage.record_split(edge, a, b, node, station)
            point = stage.proposed_point(edge, station)
            if node in working.nodes and working.nodes[node].tobytes() != point.tobytes():
                raise MeshError("quadratic shared split collided with an existing physical node")
            working.nodes[node] = point
            neighbours = sorted(set(geometry.faces_using_edge(edge)).difference((face,)))
            propagate_triangle_split(working, neighbours, (a, b), node, cache=incidence_cache)
            for neighbour in neighbours:
                schedule(neighbour)
            next_element = max(next_element, max(working.tris, default=0) + 1)
            sequence.insert(i + 1, node)
        best = optimize(refined.points, refined.triangles, refined.segments)
        install_face(face, ids, best.points, best.triangles, sign, report["shared_nodes"])
        spent_points[face] += report["insertions"] + report["reserved_node_reuses"]
        spent_operations[face] += report["topology_operations"]
        entry = diagnostics["faces"].setdefault(str(face), {"refinement_passes": []})
        entry["refinement_passes"].append({"refinement": report, "restoration_flips": restored.flips})
        entry["refinement"] = report
        entry["staged_quality"] = best.report
        entry["work_totals"] = {"staged_points": spent_points[face],
                                "topology_operations": spent_operations[face], "passes": visits[face]}
        diagnostics["attempted_insertions"] += report["insertions"]
        diagnostics["reserved_node_reuses"] += report["reserved_node_reuses"]
        if best.report["poor_element_ids"] and (report["insertions"] or best.moved_nodes):
            schedule(face)

    # Every face is repaired after the last shared-edge update. No early face
    # is promoted while a later face can still change its boundary incidence.
    for face in faces:
        checkpoint("quadratic component final repair")
        ids, points, triangles, loops, edges, intervals, sign = face_data(face)
        best = optimize(points, triangles, edges)
        from ._local_angle_repair import repair_triangle_angles
        remaining = max(0, settings.native_options.max_topology_operations -
                        spent_operations[face])
        repair = repair_triangle_angles(
            best.points, best.triangles, edges, native_settings,
            max_trials=min(2048, remaining), cancellation_check=kernel_check)
        best = surface_mesh._make_candidate(repair.points, best.triangles, settings=native_settings)
        install_face(face, ids, best.points, best.triangles, sign)
        diagnostics["faces"][str(face)]["final_chart_quality"] = best.report
        diagnostics["faces"][str(face)]["angle_repair"] = {
            "trials": repair.trials, "accepted_moves": repair.accepted_moves,
            "moved_nodes": [ids[row] for row in repair.moved_nodes],
            "budget_exhausted": repair.budget_exhausted}
        spent_operations[face] += repair.trials
        diagnostics["faces"][str(face)]["work_totals"]["topology_operations"] = spent_operations[face]

    from .hybrid import _neutral_shell_core
    from ._cylindrical_transition_repair import repair_component_transitions
    checkpoint("cylindrical_transition_repair_begin")
    repair_component_transitions(
        working, faces, charts, face_data, settings, spent_operations,
        diagnostics, lambda phase="cylindrical_transition_repair": (
            cancellation_check(phase) if cancellation_check is not None else None
        ),
    )
    checkpoint("cylindrical_transition_repair_end")
    core = _neutral_shell_core(working)
    assert_valid_mesh(core)
    quality = surface_mesh._quality_threshold_report(evaluate_quality(core), settings)
    diagnostics["physical_quality"] = quality
    from ._physical_chart_quality import physical_chart_quality
    policy_accepted = True
    for face in faces:
        ids, points, triangles, loops, edges, intervals, sign = face_data(face)
        face_policy = physical_chart_quality(
            points, triangles, settings, charts[face], points,
            np.asarray([working.nodes[node] for node in ids], dtype=np.float64),
        )
        diagnostics["faces"][str(face)]["final_policy_quality"] = face_policy
        policy_accepted = policy_accepted and not face_policy["poor_element_ids"]
    diagnostics["component_policy_accepted"] = bool(policy_accepted and quality["accepted"])
    if not quality["accepted"] or not policy_accepted:
        checkpoint("quadratic component rejected before promotion")
        diagnostics["reason"] = "component_quality_policy"
        return QuadraticRefinementTrial(False, deepcopy(seed), diagnostics)

    if settings.recombine:
        triangulated_working = deepcopy(working)
        for face in faces:
            checkpoint("quadratic component late recombination")
            next_element = max(next_element, max(working.shells, default=0) + 1)
            ids, points, triangles, loops, edges, intervals, sign = face_data(face)
            report, _ = surface_mesh._qualified_recombination(
                MeshCore(points, triangles), edges, replace(native_settings, recombine=True), cancellation_check)
            if report.mesh.node_coordinates.tobytes() != MeshCore(points, triangles).node_coordinates.tobytes():
                raise MeshError("quadratic late recombination changed node coordinates")
            for e in working.elements_of_face[face]:
                del working.tris[e]
            new_ids = []
            for table, rows, active in ((working.tris, report.mesh.triangle_connectivity, report.mesh.triangle_active),
                                        (working.quads, report.mesh.quad_connectivity, report.mesh.quad_active)):
                for row in rows[np.flatnonzero(active)]:
                    nodes = tuple(ids[int(i)] for i in row)
                    if sign < 0:
                        nodes = (nodes[0], *nodes[:0:-1])
                    table[next_element] = nodes
                    new_ids.append(next_element)
                    next_element += 1
            working.elements_of_face[face] = new_ids

    def interior(face, a, b):
        points = projection(face, (a, b))
        return charts[face].evaluate(np.asarray((0.5 * (points[0] + points[1]),)))[0]

    if settings.recombine:
        from ._physical_recombination_filter import retain_physical_recombination
        diagnostics["physical_recombination"] = retain_physical_recombination(
            working, triangulated_working, settings,
        )
    promotion = stage.promote(
        working, evaluate_edge=lambda edge, t: geometry.sample_edge(edge, (t,))[0],
        evaluate_interior=interior, cancellation_check=cancellation_check)
    promoted_core = _neutral_shell_core(promotion.mesh)
    assert_valid_mesh(promoted_core)
    final = surface_mesh._quality_threshold_report(evaluate_quality(promoted_core), settings)
    diagnostics["promoted_quality"] = final
    checkpoint("quadratic component trial final gate")
    if not final["accepted"]:
        diagnostics["reason"] = "promoted_quality_policy"
        return QuadraticRefinementTrial(False, deepcopy(seed), diagnostics)
    diagnostics["accepted"] = True
    diagnostics["published_insertions"] = diagnostics["attempted_insertions"]
    return QuadraticRefinementTrial(True, promotion.mesh, diagnostics, promotion.boundary_stations)
