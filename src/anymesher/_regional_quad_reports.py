"""Translate public quad-first reports across a bijection of published mesh IDs.

Q4 seed matching, front edges and validation rows describe face-local states
before global labels are allocated. Q5 exposes both resident and global node
lists. The quadratic certificate contains global midpoint nodes.
"""
from copy import deepcopy

from .errors import MeshError
from .regional_update import (_as_int, _boolean, _count, _finite, _geometry_face_key,
                              _list, _mapping, _nullable, _report_fields, _string, _validate_report)

_KEYS = frozenset(("front", "geometry_family_by_face", "high_order_geometry", "layout_policy",
                   "line_search", "orientation", "q4", "q5", "quad_first_api", "quality_model",
                   "route", "seed_mode", "t3_angle_rule", "validation"))
def COUNT(fields):
    return _report_fields(fields, _count)


def NUMBER(fields):
    return _report_fields(fields, _finite)


def _face_map(item):
    def check(value, path):
        for face, row in _mapping(value, path).items():
            _geometry_face_key(face, path)
            _validate_report(row, f'{path}[{face}]', item)
    return check


def _pair(values, path):
    if len(_list(values, path)) != 2:
        raise MeshError(f'{path} requires two local indices')
    for item in values:
        _count(item, path)


_Q4 = {**COUNT("applied_pairs candidate_components eligible_components final_q4 final_t3 initial_q4 initial_t3 "
               "skipped_component_cap skipped_infeasible skipped_large skipped_nonbipartite skipped_unbalanced "
               "solved_components worker_calls"), "total_cost": _as_int, "status": _string,
       "arc_counts": [_count], "component_bounds": [_count], "component_sizes": [_count]}
_Q4_FACE = {**_Q4, **COUNT("generation_before generation_after"), "added_q4_ids": [_count], "selected_pairs": [_pair]}
_Q4_SUMMARY = {**_Q4, "faces": _face_map(_Q4_FACE), "added_q4_ids_by_face": _face_map([_count]),
               "generation_by_face": _face_map(COUNT("before after")), "selected_pairs_by_face": _face_map([_pair])}

_Q5 = {**COUNT("applied attempts budget budget_cap eligible_nodes worker_calls"),
       **NUMBER("max_displacement objective_final_sum objective_initial_sum"),
       "status": _string, "worker_statuses": [_string], "moved_node_ids": [_as_int]}
_TRANSITION = {**COUNT("evaluation_budget evaluations sweeps"), **NUMBER("final_excess initial_excess"),
               "status": _string, "moved_node_ids": [_count]}
_TRANSITION_OPTIONAL = {**NUMBER("max_displacement proposed_excess"), "global_moved_node_ids": [_as_int],
                        "attempted_moved_node_ids": [_count],
                        "assembled_shell": {**NUMBER("initial_excess proposed_excess"),
                                            "accepted": _boolean, "cross_domain_pairs_included": _boolean,
                                            "status": _string}}


def _transition(value, path):
    row = _mapping(value, path)
    _validate_report(row, path, {**_TRANSITION, **{k:v for k,v in _TRANSITION_OPTIONAL.items() if k in row}})


def _q5_face(value, path):
    schema = {**_Q5, 'resident_moved_node_ids': [_count]}
    if 'transition_quality' in value:
        schema['transition_quality'] = _transition
    _validate_report(value, path, schema)


def _map_q5(report, ids):
    _validate_report(report, 'q5', {**_Q5, 'faces': _face_map(_q5_face)})
    for row in (report, *report['faces'].values()):
        row['moved_node_ids'] = [ids.node(node, 'q5.moved_node_ids') for node in row['moved_node_ids']]
        transition = row.get('transition_quality', {})
        if 'global_moved_node_ids' in transition:
            transition['global_moved_node_ids'] = [
                ids.node(node, 'q5.transition') for node in transition['global_moved_node_ids']]


_FRONT = {**COUNT("attempts direct_accepts final_q4 final_t3 guided_accepts initial_t3 no_candidate queue_pushes "
                  "recovery_accepts recovery_attempts recovery_inserted_nodes stale_skips"),
          "exhausted_budget": _boolean, "field_diagnostics": [_string], "field_mode": _string,
          "front_remaining": [_pair], "seed_triangulation_backend": _string}
_VALIDATION = {**COUNT("boundary_edge_count face interior_edge_count node_count quad_count tri_count t3_angle_warning_count"),
               **NUMBER("area_ratio cell_area_sum min_cell_area n_eq q4_area_fraction q4_count_fraction reference_area"),
               **_report_fields("min_q4_scaled_jacobian max_q4_angle min_t3_angle", _nullable(_finite)),
               "t3_angle_warning_cells": [_count], "quality_gates": {**NUMBER("q4_maximum_angle q4_maximum_aspect_ratio "
               "q4_minimum_angle q4_minimum_scaled_jacobian t3_minimum_angle"), "t3_angle_action": _string}}


def _front_face(value, path):
    value = _mapping(value, path)
    # A front with no remaining triangles has no minimum triangle angle.
    angle = _nullable(_finite) if value.get('final_t3') == 0 else _finite
    schema = {**_FRONT, 'residual_flips': {**COUNT("flips passes"),
              **_report_fields("min_t3_angle_after min_t3_angle_before", angle)}}
    if 'layout_seed' in value:
        schema = {**schema, 'layout_seed': COUNT("boundary_stations interior_points targeted_reseed_points")}
    if 'edge_aligned_seed' in value:
        schema = {**schema, 'edge_aligned_seed': COUNT("boundary_stations interior_points tensor_cells")}
    _validate_report(value, path, schema)


def _midside(row, ids):
    from .quad.high_order import HighOrderBoundaryMidside
    path = 'high_order_geometry.boundary_midsides'
    _validate_report(row, path, {"canonical_edge": [_finite], "source_edge": _as_int, "station_interval": [_finite],
                     "midpoint_parameter": _finite, "midpoint_node": _as_int, "residual": _finite,
                     "curvature_class": _string})
    edge = row['source_edge']
    interval = tuple(row['station_interval'])
    if len(interval) != 2 or tuple(row['canonical_edge']) != (edge, *[round(t,15) for t in interval]):
        raise MeshError('quadratic midside canonical edge differs from its geometry station')
    return HighOrderBoundaryMidside(tuple(row['canonical_edge']), edge, interval, row['midpoint_parameter'],
                                    ids.node(row['midpoint_node'], path), row['residual'], row['curvature_class'])


def _high_order(value, ids, mesh):
    from .quad.high_order import HighOrderGeometryReport, HighOrderMeshCertificate
    if value == {'status': 'NOT_APPLICABLE', 'reports': []}:
        if mesh.is_quadratic:
            raise MeshError('quadratic mesh lacks a high-order geometry report')
        return deepcopy(value)
    record = deepcopy(dict(_mapping(value, 'high_order_geometry')))
    try:
        rows = record.pop('reports')
        boundary = record.pop('boundary_midsides')
        _validate_report(record, 'high_order_geometry',
            {**_report_fields('status model_id order route', _string),
             **COUNT('revision q8_count t6_count unique_midside_count unique_boundary_midside_count'),
             **NUMBER('max_geometry_residual target_size')})
        if (not mesh.is_quadratic or record['order'] != 'quadratic' or
                (record['q8_count'], record['t6_count']) != (len(mesh.quads), len(mesh.tris))):
            raise MeshError('high-order certificate shell family differs from published connectivity')
        face_schema = {**_report_fields('model_id geometry_family chart_kind boundary_projection '
                                    'interior_projection validity_status', _string),
                       **COUNT('revision face_id q8_count t6_count certified_elements total_elements '
                               'interior_midside_count invalid_elements'),
                       'chart_origin': [_finite], 'max_geometry_residual': _finite,
                       'edge_curvature_classes': [_string]}
        if {row['face_id'] for row in rows} != set(mesh.elements_of_face):
            raise MeshError('high-order certificate face coverage differs from published connectivity')
        midsides = {edge:set(nodes[1::2]) for edge,nodes in mesh.nodes_of_edge.items()}
        faces = []
        for row in rows:
            boundary_rows = row.pop('boundary_midsides')
            _validate_report(row, 'high_order_geometry.reports', face_schema)
            elements = mesh.elements_of_face[row['face_id']]
            if (row['q8_count'] != sum(e in mesh.quads for e in elements)
                    or row['t6_count'] != sum(e in mesh.tris for e in elements)):
                raise MeshError('high-order certificate face shell counts differ from published connectivity')
            if any(r['midpoint_node'] not in midsides.get(r['source_edge'], ()) for r in boundary_rows):
                raise MeshError('high-order midside does not occur on its registered source edge')
            row['boundary_midsides'] = tuple(_midside(item, ids) for item in boundary_rows)
            row['chart_origin'] = tuple(row['chart_origin'])
            row['edge_curvature_classes'] = tuple(row['edge_curvature_classes'])
            faces.append(HighOrderGeometryReport(**row))
        result = HighOrderMeshCertificate(reports=tuple(faces), **record).to_dict()
        if [_midside(item, ids).to_dict() for item in boundary] != result['boundary_midsides']:
            raise MeshError('quadratic boundary midsides differ from their face reports')
        return result
    except (KeyError, TypeError, ValueError) as error:
        raise MeshError(f'invalid high-order geometry report: {error}') from error


def quad_first_fields(record, ids, mesh):
    if 'quad_first_api' not in record:
        return {}
    if record['quad_first_api'] != 'public/1' or not _KEYS.issubset(record):
        raise MeshError('unrecognized or incomplete public quad-first diagnostic')
    fields = _KEYS | ({'quadratic_promotion'} if 'quadratic_promotion' in record else set())
    out = {k:deepcopy(record[k]) for k in fields}
    for key in ('layout_policy','line_search','orientation','quad_first_api','quality_model','route','seed_mode'):
        _string(out[key], key)
    _validate_report(out['geometry_family_by_face'], 'geometry_family_by_face', _face_map(_string))
    _validate_report(out['front'], 'front', {'faces': _face_map(_front_face)})
    _validate_report(out['validation'], 'validation', {'faces': _face_map(_VALIDATION)})
    _validate_report(out['t3_angle_rule'], 't3_angle_rule', {'minimum_angle': _finite, 'action': _string,
                                                           'warning_count': _count})
    _validate_report(out['q4'], 'q4', _Q4_SUMMARY)
    _map_q5(out['q5'], ids)
    out['high_order_geometry'] = _high_order(out['high_order_geometry'], ids, mesh)
    if 'quad_quality_repair' in record:
        from ._regional_quad_repair import translate_quad_quality_repair
        out['quad_quality_repair'] = translate_quad_quality_repair(record['quad_quality_repair'], ids)
    if 'quadratic_promotion' in out:
        row = out['quadratic_promotion']
        _validate_report(row, 'quadratic_promotion', {**COUNT('exact_boundary_midsides repair_count unique_midsides'),
                          'max_corner_displacement': _finite, 'repaired_corner_nodes': [_as_int], 'status': _string})
        row['repaired_corner_nodes'] = [ids.node(n, 'quadratic_promotion') for n in row['repaired_corner_nodes']]
    return out
