"""Private transition repair on an already detached cylindrical T3 component."""
import numpy as np

from ._local_angle_repair import repair_triangle_angles
from ._cdt_restore import restore_constrained_delaunay
from ._transition_patch import repair_transition_patch
from ._physical_chart_quality import physical_chart_quality
from ._angle_star_projection import angle_star_target
from ._corner_cavity import repair_corner_cavity
from ._boundary_fan_cavity import repair_boundary_fan
from ._joint_triangle_repair import repair_joint_triangle_quality
from .surface_mesh import _make_candidate


def repair_component_transitions(mesh, faces, charts, face_data, settings,
                                 spent_operations, diagnostics, checkpoint):
    for face in faces:
        checkpoint()
        ids, points, triangles, _, protected, _, sign = face_data(face)
        initial_points = points.copy()
        initial_triangles = triangles.copy()
        initial = _make_candidate(points, triangles, settings=settings)
        reference_coordinates = np.asarray([mesh.nodes[node] for node in ids], dtype=np.float64)
        initial_quality = physical_chart_quality(
            points, triangles, settings, charts[face], initial_points, reference_coordinates,
        )
        if not initial_quality["poor_element_ids"]:
            continue
        available = max(0, settings.native_options.max_topology_operations - spent_operations[face])
        used = 0
        attempts = 0
        committed_flips = 0
        rounds = 0
        restoration_flips = 0
        projection_work = 0
        corner_attempts = 0
        corner_insertions = 0
        fan_insertions = 0
        joint_trials = 0
        locked_corner_trials = 0
        locked_before = 0
        fixed_nodes = {int(v) for edge in protected for v in edge}

        def physical_points(p):
            xyz = np.asarray(charts[face].evaluate(p), dtype=np.float64).copy()
            if xyz.shape != (len(p), 3) or not np.isfinite(xyz).all():
                raise ValueError("invalid owner coordinates in cylindrical transition")
            count = len(initial_points)
            unchanged = np.all(np.ascontiguousarray(p[:count]).view(np.uint64)
                               == initial_points.view(np.uint64), axis=1)
            xyz[:count][unchanged] = reference_coordinates[unchanged]
            return xyz

        def quality(p, t):
            # Existing coordinates retain their original raw owner values. New
            # interior rows have no protected reference and are owner-evaluated.
            extra = p[len(initial_points):]
            reference_points = initial_points
            coordinates = reference_coordinates
            if len(extra):
                reference_points = np.vstack((initial_points, extra))
                coordinates = np.vstack((reference_coordinates, charts[face].evaluate(extra)))
            return physical_chart_quality(p, t, settings, charts[face],
                                          reference_points, coordinates)

        def score(p, t):
            q = quality(p, t)
            return (q["invalid_element_count"], q["quality_violation_count"],
                    q["max_aspect_ratio"], -q["min_scaled_jacobian"], -q["min_angle"])

        def joint_relax(p, t, limit):
            nonlocal used, joint_trials
            current = quality(p, t)
            if (used >= available or not current["poor_element_ids"]
                    or not 0 < settings.min_angle < 60
                    or not np.isfinite(settings.max_element_growth)
                    or settings.max_element_growth <= 1):
                return p
            angle_only = (initial_quality["min_angle"] >= 20.
                          and current["max_element_growth"] <= settings.max_element_growth)
            joint = repair_joint_triangle_quality(
                p, t, protected, [int(i) - 1 for i in current["poor_element_ids"]],
                min_angle=settings.min_angle, max_growth=settings.max_element_growth,
                max_trials=min(limit, available - used), cancellation_check=checkpoint,
                evaluate_coordinates=physical_points,
                neighbourhood_rings=int(angle_only),
            )
            used += joint.trials
            joint_trials += joint.trials
            return joint.points if score(joint.points, t) < score(p, t) else p

        def relax(p, t):
            nonlocal used, projection_work, locked_corner_trials
            checkpoint()
            used += 1  # One candidate topology/quality evaluation.
            incoming_score = score(p, t)
            repaired = repair_triangle_angles(
                p, t, protected, settings,
                max_trials=min(512, max(0, available - used)),
                cancellation_check=checkpoint,
            )
            used += repaired.trials
            p = repaired.points
            current = quality(p, t)
            locked_after = sum(all(int(v) in fixed_nodes for v in t[int(i) - 1])
                               for i in current["poor_element_ids"])
            if locked_after < locked_before and locked_corner_trials < 4096:
                prior_trials = joint_trials
                p = joint_relax(p, t, min(2048, 4096 - locked_corner_trials))
                locked_corner_trials += joint_trials - prior_trials
            else:
                p = joint_relax(p, t, 256)
            # Do not displace the established successful relaxation trajectory.
            # Projection is a bounded fallback for a stalled candidate only.
            if score(p, t) < incoming_score or projection_work >= 512:
                return p
            bad = quality(p, t)["poor_element_ids"]
            movable = sorted({int(v) for i in bad for v in t[int(i) - 1]} - fixed_nodes)

            def consume_projection_work():
                nonlocal used, projection_work, target_work
                if used >= available or projection_work >= 512 or target_work >= 128:
                    return False
                used += 1
                projection_work += 1
                target_work += 1
                return True

            for node in movable:
                for angle in (settings.min_angle + .25, settings.min_angle + 1.):
                    if used >= available or not 0 < angle < 60:
                        break
                    target_work = 0
                    target = angle_star_target(
                        p, t, node, angle, consume_work=consume_projection_work,
                        cancellation_check=checkpoint,
                    )
                    if target is not None:
                        candidate = p.copy()
                        candidate[node] = target
                        if score(candidate, t) < score(p, t):
                            p = candidate
            return p

        # Restore the background before constructing a non-Delaunay transition.
        # Re-running Lawson restoration afterward can erase the required fan.
        if available:
            restored = restore_constrained_delaunay(
                points, triangles, protected, max_flips=available,
                cancellation_check=checkpoint,
            )
            used += restored.flips
            if restored.converged:
                triangles = restored.triangles
                restoration_flips = restored.flips
                repaired = repair_triangle_angles(
                    points, triangles, protected, settings,
                    max_trials=min(2048, max(0, available - used)),
                    cancellation_check=checkpoint,
                )
                used += repaired.trials
                points = repaired.points

        # A highly unequal protected corner can have no admissible motion in its
        # existing two-triangle cavity. Spend a bounded insertion there before
        # exhausting the face budget on fixed-connectivity relaxation.
        staged_points = diagnostics["faces"][str(face)]["work_totals"].get("staged_points")
        point_room = (max(0, settings.native_options.max_insertions - staged_points)
                      if staged_points is not None else 0)
        if available and point_room and initial_quality["min_angle"] < 20.:
            visited_corners = set()
            for _ in range(min(2, point_room)):
                if used >= available:
                    break
                current = quality(points, triangles)
                corner_work_end = used + min(4096, available - used)

                def consume_corner_work():
                    nonlocal used
                    if used >= corner_work_end:
                        return False
                    used += 1
                    return True

                corner = repair_corner_cavity(
                    points, triangles, protected,
                    [int(i) - 1 for i in current["poor_element_ids"]],
                    score=score, max_candidates=min(64, available - used), max_insertions=1,
                    cancellation_check=checkpoint, excluded_corners=visited_corners,
                    consume_work=consume_corner_work,
                    relax=lambda p, t: joint_relax(p, t, min(512, max(0, corner_work_end - used))),
                    target_met=lambda p, t: not quality(p, t)["poor_element_ids"],
                )
                corner_attempts += corner.attempts
                if corner.corner_node is None:
                    break
                visited_corners.add(corner.corner_node)
                if corner.improved:
                    points, triangles = corner.points, corner.triangles
                    corner_insertions += corner.added_points

        points = joint_relax(points, triangles, 2048)

        # Release a saturated boundary fan before exhausting the remaining
        # budget on fixed-connectivity smoothing. Protected edges stay intact.
        fan_work_end = min(available, used + 1024)

        def consume_fan_work():
            nonlocal used
            if used >= fan_work_end:
                return False
            used += 1
            return True

        fan = repair_boundary_fan(
            points, triangles, protected,
            [int(i) - 1 for i in quality(points, triangles)["poor_element_ids"]],
            score=score, max_candidates=6,
            max_insertions=int(point_room > corner_insertions),
            consume_work=consume_fan_work, cancellation_check=checkpoint,
            relax=lambda p, t: joint_relax(p, t, min(192, max(0, fan_work_end - used))),
            target_met=lambda p, t: not quality(p, t)["poor_element_ids"],
        )
        if fan.improved:
            points, triangles = fan.points, fan.triangles
            fan_insertions = fan.added_points

        for _ in range(6):
            before = quality(points, triangles)
            violations = before["quality_violation_count"]
            if not violations or used >= available:
                break
            locked_before = sum(all(int(v) in fixed_nodes for v in triangles[int(i) - 1])
                                for i in before["poor_element_ids"])
            patch = repair_transition_patch(
                points, triangles, protected,
                [int(i) - 1 for i in before["poor_element_ids"]],
                score=score, relax=relax, max_attempts=256,
                cancellation_check=checkpoint,
                target_met=lambda p, t: quality(p, t)["quality_violation_count"] < violations,
                work_available=lambda: used < available,
            )
            attempts += patch.attempts
            points, triangles = patch.points, patch.triangles
            committed_flips += patch.committed_flips
            points = joint_relax(points, triangles, 1024)
            repaired = repair_triangle_angles(
                points, triangles, protected, settings,
                max_trials=min(2048, max(0, available - used)),
                cancellation_check=checkpoint,
            )
            used += repaired.trials
            points = repaired.points
            rounds += 1
            if not patch.improved:
                break

        if score(points, triangles) > score(initial_points, initial_triangles):
            points, triangles = initial_points.copy(), initial_triangles.copy()
            committed_flips = 0
            restoration_flips = 0
            corner_insertions = 0
            fan_insertions = 0
        final = _make_candidate(points, triangles, settings=settings)
        final_quality = quality(points, triangles)
        fixed = sorted({int(v) for edge in protected for v in edge})
        if points[fixed].tobytes() != initial_points[fixed].tobytes():
            raise ValueError("cylindrical transition changed protected chart coordinates")
        if score(points, triangles) > score(initial_points, initial_triangles):
            raise ValueError("cylindrical transition worsened whole-face policy")
        changed = [i for i in range(len(points))
                   if i >= len(initial_points) or points[i].tobytes() != initial_points[i].tobytes()]
        checkpoint()
        old_element_ids = sorted(mesh.elements_of_face[face])
        added = len(points) - len(initial_points)
        if added != corner_insertions + fan_insertions or len(triangles) != len(old_element_ids) + 2 * added:
            raise ValueError("cylindrical transition changed unauthorized topology")
        group_updates = []
        if added:
            old_set = set(old_element_ids)
            for groups in (mesh.elements_of_sheet, mesh.elements_of_member):
                for key, members in groups.items():
                    if old_set.intersection(members):
                        if not old_set.issubset(members):
                            raise ValueError("ambiguous cylindrical transition group ownership")
                        group_updates.append((groups, key, list(members)))
        extra_ids = tuple(range(max(mesh.nodes, default=0) + 1,
                                max(mesh.nodes, default=0) + 1 + added))
        ids = tuple(ids) + extra_ids
        if changed:
            lifted = np.asarray(charts[face].evaluate(points[changed]), dtype=np.float64)
            if lifted.shape != (len(changed), 3) or not np.isfinite(lifted).all():
                raise ValueError("invalid owner lift during cylindrical transition repair")
            for local, coordinate in zip(changed, lifted, strict=True):
                mesh.nodes[ids[local]] = coordinate.copy()
        first_element = max(mesh.shells, default=0) + 1
        extra_elements = list(range(first_element, first_element + 2 * added))
        element_ids = old_element_ids + extra_elements
        for element_id, row in zip(element_ids, triangles, strict=True):
            if sign < 0:
                row = row[[0, 2, 1]]
            mesh.tris[element_id] = tuple(ids[int(v)] for v in row)
        mesh.elements_of_face[face] = element_ids
        for groups, key, members in group_updates:
            groups[key] = members + extra_elements
        spent_operations[face] += used
        entry = diagnostics["faces"][str(face)]
        entry["final_chart_quality"] = final.report
        entry["final_physical_quality"] = final_quality["physical_quality"]
        entry["work_totals"]["topology_operations"] = spent_operations[face]
        if added:
            entry["work_totals"]["staged_points"] += added
            diagnostics["attempted_insertions"] = diagnostics.get("attempted_insertions", 0) + added
        entry["transition_repair"] = {
            "attempts": attempts, "committed_flips": committed_flips,
            "committed_restoration_flips": restoration_flips,
            "moved_nodes": [ids[i] for i in changed], "rounds": rounds,
            "work_units": used, "available_work_units": available,
            "projection_work_units": projection_work,
            "corner_attempts": corner_attempts, "corner_insertions": added,
            "corner_cavity_insertions": corner_insertions,
            "boundary_fan_attempts": fan.attempts, "boundary_fan_insertions": fan_insertions,
            "joint_repair_trials": joint_trials,
            "unlocked_corner_joint_trials": locked_corner_trials,
            "budget_exhausted": used >= available,
            "target_met": not final_quality["poor_element_ids"],
            "initial_quality": initial.report, "final_quality": final.report,
            "initial_physical_quality": initial_quality["physical_quality"],
            "final_physical_quality": final_quality["physical_quality"],
        }
    checkpoint()
