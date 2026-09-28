"""Independent sampled checks for irregular planar and cylindrical owners."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from anygeometry import Cylinder
from anygeometry.curves import Arc, arc_frame
from anymesher.errors import MeshError
from anymesher.quad.domain import PlanarQuadDomain
from anymesher.quad.high_order import certify_mapping_validity, evaluate_mapping
from anymesher.refinement import SizeField
from benchmarks.sg1.measure import physical_edges, quadrature, samples


def _planar_owner_area(geometry, face, domain):
    """Exact Green-area reference, including circular source arcs."""
    def ring_area(uses):
        vertices = [
            geometry.edges[use.edge].start if use.forward else geometry.edges[use.edge].end
            for use in uses
        ]
        points = np.asarray([domain.project(geometry.vertex_position(v)) for v in vertices])
        area = .5 * float(np.dot(points[:, 0], np.roll(points[:, 1], -1)) -
                          np.dot(points[:, 1], np.roll(points[:, 0], -1)))
        for use in uses:
            edge = geometry.edges[use.edge]
            if not isinstance(edge.curve, Arc):
                continue
            frame = arc_frame(
                geometry.vertex_position(edge.start),
                geometry.vertex_position(edge.curve.via_vertex),
                geometry.vertex_position(edge.end),
            )
            orientation = float(np.dot(np.cross(frame.e1, frame.e2), domain.normal))
            area += (.5 * frame.radius**2 * (frame.sweep - np.sin(frame.sweep))
                     * orientation * (1. if use.forward else -1.))
        return abs(area)
    source = geometry.faces[face]
    return ring_area(source.loop) - sum(ring_area(loop) for loop in source.holes)


def audit_irregular(geometry, faces, mesh, h, refinements=(), center=None):
    field = SizeField(geometry, h, tuple(refinements))
    cells = {**mesh.quads, **mesh.tris}
    checks = []
    def check(name, passed, **detail):
        checks.append({"name": name, "status": "passed" if bool(passed) else "failed", **detail})

    all_edges = defaultdict(list)
    declared_boundary_uses = defaultdict(int)
    ratios, lengths = [], []
    core_lengths, remote_lengths = [], []
    corner_nodes = set()
    geometry_by_cell = {}
    physical_area = owner_area = 0.
    max_support = max_node_support = max_error_over_h = max_normal = max_aspect = 0.
    min_jacobian = 1.
    diameter = max(1., float(np.linalg.norm(np.ptp(np.array(list(mesh.nodes.values())), axis=0))))
    source_edges = {int(use.edge) for face in faces
                    for loop in (geometry.faces[face].loop, *geometry.faces[face].holes)
                    for use in loop}
    check("source-edge-inventory", set(mesh.nodes_of_edge) == source_edges)
    for edge_id, chain in mesh.nodes_of_edge.items():
        if edge_id not in source_edges or len(chain) < 2:
            check(f"edge/{edge_id}/stations", False)
            continue
        parameters = []
        residuals = []
        for node in chain:
            near, parameter, _ = geometry.closest_edge_point(edge_id, mesh.nodes[node])
            parameters.append(float(parameter))
            residuals.append(float(np.linalg.norm(np.asarray(mesh.nodes[node]) - near)))
        check(f"edge/{edge_id}/source-support",
              max(residuals) <= 1e-10*diameter, value=max(residuals))
        check(f"edge/{edge_id}/station-order",
              abs(parameters[0]) <= 1e-10 and
              abs(parameters[-1]-1.) <= 1e-10 and
              all(b-a > 1e-12 for a, b in zip(parameters, parameters[1:])))
    for face in faces:
        owner = geometry.faces[face].surface
        if isinstance(owner, Cylinder):
            uv = lambda point: np.asarray(owner.local_uv(tuple(point)), dtype=float)
            lift = lambda point: np.asarray(owner.evaluate(*point), dtype=float)
            normal = lambda point: (
                np.asarray(owner.evaluate(*point), dtype=float) - owner.origin
                - float(point[1]) * owner.height * owner.axis
            ) / owner.radius
            reference_area = None
        else:
            try:
                domain = PlanarQuadDomain.from_geometry(geometry, face)
            except MeshError:
                domain = None
            if domain is None:
                raise ValueError(f"irregular audit has no owner reference for {type(owner).__name__}")
            uv = lambda point: np.asarray(domain.project(point), dtype=float)
            lift = lambda point: np.asarray(domain.lift(point), dtype=float)
            base = np.asarray(geometry.vertex_position(domain.vertex_ids[0]), dtype=float)
            side_a = np.asarray(geometry.vertex_position(domain.vertex_ids[1]), dtype=float) - base
            side_b = np.asarray(geometry.vertex_position(domain.vertex_ids[-1]), dtype=float) - base
            direction = np.cross(side_a, side_b)
            direction /= np.linalg.norm(direction)
            normal = lambda point: direction
            reference_area = _planar_owner_area(geometry, face, domain)
        def ring(uses):
            points = []
            step = 2 if mesh.order == "quadratic" else 1
            for use in uses:
                chain = tuple(mesh.nodes_of_edge[use.edge])
                if not use.forward:
                    chain = tuple(reversed(chain))
                points.extend(uv(mesh.nodes[node]) for node in chain[:-1:step])
            return np.asarray(points)
        outer = ring(geometry.faces[face].loop)
        holes = [ring(loop) for loop in geometry.faces[face].holes]
        source_polygon = Polygon(outer, holes)
        check(f"face/{face}/valid-source-polygon", source_polygon.is_valid and source_polygon.area > 0)
        if reference_area is None:
            reference_area = source_polygon.area * abs(owner.radius * owner.sweep_angle * owner.height)
        owner_area += reference_area
        polygons = []
        face_edges = defaultdict(list)
        declared_face_boundary = set()
        for eid in mesh.elements_of_face.get(face, ()):
            body = cells[eid]
            corners = 4 if eid in mesh.quads else 3
            family = ("Q" if corners == 4 else "T") + str(len(body))
            corner_nodes.update(body[:corners])
            xyz = np.asarray([mesh.nodes[node] for node in body])
            for node_point in xyz:
                max_node_support = max(max_node_support, float(np.linalg.norm(
                    node_point - lift(uv(node_point))
                )))
            corners_uv = np.asarray([uv(point) for point in xyz[:corners]])
            polygons.append(Polygon(corners_uv))
            nvec = normal(np.mean(corners_uv, axis=0))
            certificate = certify_mapping_validity(xyz, family, reference_normal=nvec)
            if mesh.order == "quadratic":
                check(f"element/{eid}/strict-positive", certificate.status.value == "CERTIFIED_POSITIVE")
            evaluation = evaluate_mapping(xyz, family, samples(corners == 3), reference_normal=nvec)
            min_jacobian = min(min_jacobian, float(np.min(evaluation.normalized_quality)))
            cell_error = 0.
            for point, vector in zip(evaluation.points, evaluation.jacobian_vector):
                owner_uv = uv(point)
                projection = lift(owner_uv)
                error = float(np.linalg.norm(point - projection))
                cell_error = max(cell_error, error)
                max_support = max(max_support, error)
                max_error_over_h = max(max_error_over_h, error / float(field.size_at(
                    np.asarray(point, dtype=float).reshape((1, 3))
                )[0]))
                owner_normal = normal(owner_uv)
                dot = float(vector @ owner_normal / np.linalg.norm(vector) / np.linalg.norm(owner_normal))
                max_normal = max(max_normal, float(np.degrees(np.arccos(np.clip(dot, -1., 1.)))))
            geometry_by_cell[eid] = cell_error
            points, weights = quadrature(corners == 3)
            physical_area += float(weights @ evaluate_mapping(
                xyz, family, points, reference_normal=nvec
            ).jacobian_magnitude)
            edge_records = physical_edges(xyz, corners, field)
            max_aspect = max(max_aspect, max(item[0] for item in edge_records) /
                             min(item[0] for item in edge_records))
            for index, (length, ratio, midpoint) in enumerate(edge_records):
                a, b = body[index], body[(index+1) % corners]
                key = tuple(sorted((a, b)))
                mid = body[corners+index] if len(body) > corners else None
                face_edges[key].append((a, b, mid))
                if key not in all_edges:
                    lengths.append(length); ratios.append(ratio)
                    if center is not None:
                        distance = float(np.linalg.norm(np.asarray(midpoint) - np.asarray(center)))
                        if distance <= .35:
                            core_lengths.append(length)
                        if distance >= .7:
                            remote_lengths.append(length)
                all_edges[key].append((a, b, mid, face))
        check(f"face/{face}/nonempty", bool(polygons))
        if polygons:
            union = unary_union(polygons)
            overlap = sum(item.area for item in polygons) - union.area
            gap = union.symmetric_difference(source_polygon).area
            check(f"face/{face}/coverage", overlap <= 1e-8 and gap <= 1e-8,
                  overlap=overlap, gap=gap)
        for hole_index, uses in enumerate((geometry.faces[face].loop, *geometry.faces[face].holes)):
            for use in uses:
                chain = tuple(mesh.nodes_of_edge[use.edge])
                if not use.forward:
                    chain = tuple(reversed(chain))
                step = 2 if mesh.order == "quadratic" else 1
                for index in range(0, len(chain)-step, step):
                    a, b = chain[index], chain[index+step]
                    key = tuple(sorted((a, b)))
                    declared_face_boundary.add(key)
                    declared_boundary_uses[key] += 1
                    entries = face_edges.get(key, ())
                    directed = (b, a) if hole_index else (a, b)
                    check(f"face/{face}/edge/{use.edge}/{index}",
                          len(entries) == 1 and entries[0] ==
                          (*directed, chain[index+1] if step == 2 else None))
        observed_face_boundary = {
            key for key, entries in face_edges.items() if len(entries) == 1
        }
        check(f"face/{face}/boundary-complete",
              observed_face_boundary == declared_face_boundary,
              missing=len(declared_face_boundary-observed_face_boundary),
              extra=len(observed_face_boundary-declared_face_boundary))
    check("shell/ownership", sum(len(mesh.elements_of_face.get(face, ())) for face in faces) == len(cells))
    check("global/incidence", all(len(entries) <= 2 and
          (len(entries) < 2 or (entries[0][0:2] == entries[1][1::-1]
           and entries[0][2] == entries[1][2])) for entries in all_edges.values()))
    expected_free = {key for key, uses in declared_boundary_uses.items() if uses == 1}
    observed_free = {key for key, entries in all_edges.items() if len(entries) == 1}
    check("global/boundary-complete",
          all(uses <= 2 for uses in declared_boundary_uses.values())
          and observed_free == expected_free,
          missing=len(expected_free-observed_free),
          extra=len(observed_free-expected_free))
    if mesh.order == "quadratic":
        check("support", max_node_support <= 1e-10 * diameter, value=max_node_support)
        check("geometry-over-h", max_error_over_h <= .02, value=max_error_over_h)
        check("normal-degrees", max_normal <= 5., value=max_normal)
        check("area-relative", abs(physical_area-owner_area) / owner_area <= .005,
              value=abs(physical_area-owner_area) / owner_area)
        check("normalized-jacobian", min_jacobian >= .05, value=min_jacobian)
        check("aspect", max_aspect <= 20., value=max_aspect)
        check("sizing-median", .5 <= float(np.median(ratios)) <= 1.5,
              value=float(np.median(ratios)))
        check("sizing-fraction", float(np.mean((np.asarray(ratios) >= .25) &
              (np.asarray(ratios) <= 2.))) >= .95)
        check("sizing-max", max(ratios) <= 4., value=max(ratios))
    return {"checks": checks, "counts": {"nodes": len(mesh.nodes),
            "quads": len(mesh.quads), "triangles": len(mesh.tris),
            "shells": len(cells), "equivalent": len(mesh.quads)+.5*len(mesh.tris)},
            "max_geometry_error": max_support, "max_geometry_over_h": max_error_over_h,
            "max_node_support": max_node_support,
            "max_normal_degrees": max_normal, "min_normalized_jacobian": min_jacobian,
            "max_aspect": max_aspect, "owner_area": owner_area,
            "physical_area": physical_area, "physical_edge_lengths": lengths,
            "geometry_by_cell": geometry_by_cell,
            "core_corner_count": (sum(float(np.linalg.norm(np.asarray(mesh.nodes[node]) -
                np.asarray(center))) <= .35 for node in corner_nodes) if center is not None else None),
            "core_median": float(np.median(core_lengths)) if core_lengths else None,
            "remote_median": float(np.median(remote_lengths)) if remote_lengths else None}
