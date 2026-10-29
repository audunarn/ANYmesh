"""Saved-state diagnostic only: legal two-flip corner sequences, full mixed gates.

Run explicitly through the authorised root harness. Saved arrays confer no
geometry, source-cell or publication authority. No source geometry is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from anymesher.core import MeshCore
from anymesher.native_v2 import MutableT3Topology, orient2d
from anymesher._physical_t3_refinement import _alternative_progress
from anymesher.quality_v2 import evaluate_quality
from anymesher.surface_mesh import SurfaceMeshOptions, _quality_threshold_report, _published_quality_report


class ReplayRefused(Exception):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with np.load(path,allow_pickle=False) as data:
        return {key:data[key].copy() for key in data.files}


def replacement_ids(old_cells,old_ids,new_cells):
    """Preserve unchanged cell IDs; deterministically reuse only removed IDs."""
    old = {tuple(sorted(map(int,cell))):int(identifier) for cell,identifier in zip(old_cells,old_ids)}
    keys = [tuple(sorted(map(int,cell))) for cell in new_cells]
    if len(old) != len(old_cells) or len(set(keys)) != len(keys):
        raise ValueError('ambiguous saved triangle identities')
    removed = sorted(identifier for key,identifier in old.items() if key not in set(keys))
    added = [key for key in keys if key not in old]
    if len(removed) != len(added):
        raise ValueError('flip changed the saved triangle count')
    identifiers = {**old,**dict(zip(sorted(added),removed))}
    return np.asarray([identifiers[key] for key in keys],dtype=np.int64)


def edge_context(topology,edge):
    attached = topology._topology_index.attached(edge)
    result = {'edge':list(edge),'attached_T3_count':len(attached),'protected':edge in topology.protected_edges}
    if len(attached) == 2:
        opposite = [next(int(node) for node in topology.triangles[row] if int(node) not in edge) for row in attached]
        signs = []
        for a,b,c,d in ((*edge,*opposite),(*opposite,*edge)):
            signs.append([float(orient2d(topology.points[a],topology.points[b],topology.points[c])),
                          float(orient2d(topology.points[a],topology.points[b],topology.points[d]))])
        result.update(opposites=opposite,diagonal_signs=signs,
            strictly_convex=all((a > 0 and b < 0) or (a < 0 and b > 0) for a,b in signs))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uv',required=True,type=Path)
    parser.add_argument('--physical',required=True,type=Path)
    parser.add_argument('--cells-json',required=True,type=Path)
    parser.add_argument('--output-directory',required=True,type=Path)
    parser.add_argument('--max-growth',required=True,type=float)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True,exist_ok=False)
    started = perf_counter()
    inputs = (args.uv,args.physical,args.cells_json)
    input_hashes = {str(path.resolve()):sha(path) for path in inputs}
    uv,physical = read(args.uv),read(args.physical)
    metadata = json.loads(args.cells_json.read_text(encoding='utf-8-sig'))
    for key in ('triangles','triangle_active','triangle_ids','quads','quad_active','quad_ids'):
        if not np.array_equal(uv[key],physical[key]):
            raise ValueError('saved UV/physical connectivity correspondence differs: '+key)
    points,xyz = uv['points'][:,:2].copy(),physical['points'].copy()
    cells = physical['triangles'][physical['triangle_active']].copy()
    ids = physical['triangle_ids'][physical['triangle_active']].copy()
    quads = physical['quads'][physical['quad_active']].copy()
    quad_ids = physical['quad_ids'][physical['quad_active']].copy()
    input_arrays = (points,xyz,cells,ids,quads,quad_ids)
    input_bytes = tuple(array.tobytes() for array in input_arrays)
    if len(points) != len(xyz):
        raise ValueError('saved point correspondence differs')
    for record in metadata['poor_cells']:
        if record['kind'] == 'triangle':
            selected = np.flatnonzero(ids == record['id'])
            if len(selected) != 1 or not np.array_equal(cells[selected[0]],record['nodes']):
                raise ValueError('saved poor-cell identity differs')
            if not np.array_equal(xyz[cells[selected[0]]],record['xyz']):
                raise ValueError('saved poor-cell XYZ differs')
    thresholds = metadata['physical_report']['thresholds']
    settings = SurfaceMeshOptions(min_scaled_jacobian=thresholds['minimum_scaled_jacobian'],
        max_aspect_ratio=thresholds['maximum_aspect_ratio'],min_angle=thresholds['minimum_angle'],
        max_angle=thresholds['maximum_angle'],max_warpage=thresholds['maximum_warpage'],
        max_element_growth=args.max_growth)

    def score(triangles,triangle_ids):
        core = MeshCore(xyz,triangles,quads,triangle_ids=triangle_ids,quad_ids=quad_ids)
        quality = evaluate_quality(core)
        threshold = _quality_threshold_report(quality,settings)
        report = _published_quality_report(core,settings,quality,threshold)
        report['violation_counts'] = dict(threshold['violation_counts'])
        report['invalid_element_count'] = len(quality.validity.errors)
        return report

    baseline = score(cells,ids)
    incidence = {}
    for cell in cells:
        for a,b in zip(cell,np.roll(cell,-1)):
            edge = tuple(sorted((int(a),int(b))))
            incidence[edge] = incidence.get(edge,0)+1
    quad_edges = {tuple(sorted((int(a),int(b)))) for quad in quads for a,b in zip(quad,np.roll(quad,-1))}
    protected = tuple(edge for edge,count in incidence.items() if count == 1 or edge in quad_edges)
    outcomes = []
    sequences = ((24,((24,169),(24,193))),(24,((24,193),(24,169))),
                 (72,((72,650),(72,626))),(72,((72,626),(72,650))))
    for ordinal,(corner,sequence) in enumerate(sequences):
        topology = MutableT3Topology(points,cells,protected,
            triangle_owners=np.full(len(cells),3,dtype=np.int64))
        current_ids = ids.copy()
        outcome = {'corner':corner,'sequence':[list(edge) for edge in sequence],'steps':[]}
        for step,edge in enumerate(sequence):
            entry_cells = topology.triangles
            record = {'ordinal':step,**edge_context(topology,edge)}
            proposal = []
            def validate(trial_points,trial_cells):
                if not np.array_equal(trial_points,points):
                    raise ValueError('pure flip altered saved UV points')
                mapped = replacement_ids(entry_cells,current_ids,trial_cells)
                report = score(trial_cells,mapped)
                proposal.append((mapped,report))
                record['full_mixed_report'] = report
                record['complete_progress'] = _alternative_progress(baseline,report)
                if report['invalid_element_count']:
                    raise ReplayRefused('physical_validity_refusal')
                if step == 1 and not record['complete_progress']:
                    raise ReplayRefused('complete_quality_or_growth_refusal')
            at = perf_counter()
            try:
                changed = topology.flip_edge(edge,_validate_candidate=validate)
                record['changed'] = bool(changed)
                record['status'] = ('committed_detached' if changed else
                    'primitive_nonconvex_refusal' if record.get('strictly_convex') is False else 'primitive_refusal_before_scoring')
            except ReplayRefused as error:
                changed = False
                record.update(changed=False,status=str(error))
            record['seconds'] = perf_counter()-at
            outcome['steps'].append(record)
            if changed:
                current_ids = proposal[-1][0]
            else:
                break
        final_cells = topology.triangles
        outcome['final_report'] = score(final_cells,current_ids)
        outcome['complete_pair_committed'] = len(outcome['steps']) == 2 and all(step['changed'] for step in outcome['steps'])
        outcome['points_unchanged'] = np.array_equal(topology.points,points)
        np.savez_compressed(args.output_directory / f'corner-{corner}-order-{ordinal}.npz',
            uv_points=topology.points,physical_points=xyz,triangles=final_cells,triangle_ids=current_ids,
            quads=quads,quad_ids=quad_ids,protected_edges=np.asarray(protected,dtype=np.int64))
        outcomes.append(outcome)
        if tuple(array.tobytes() for array in input_arrays) != input_bytes:
            raise ValueError('replay changed its detached input arrays')
    if {str(path.resolve()):sha(path) for path in inputs} != input_hashes:
        raise ValueError('saved input files changed during replay')
    source = Path(__file__).resolve().parents[1] / 'src' / 'anymesher'
    result = {'classification':'saved_state_diagnostic_only','input_hashes':input_hashes,
        'source_hashes':{str(path.resolve()):sha(path) for path in (
            Path(__file__),source/'native_v2.py',source/'surface_mesh.py',source/'quality_v2.py',source/'_physical_t3_cavity.py')},
        'settings':{'thresholds':thresholds,'max_element_growth':args.max_growth},
        'counts':{'points':len(points),'active_T3':len(cells),'active_Q4':len(quads)},
        'baseline':baseline,'outcomes':outcomes,'seconds':perf_counter()-started}
    (args.output_directory/'replay.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'outcome_count':len(outcomes),'evidence':str((args.output_directory/'replay.json').resolve()),
        'complete_pairs':sum(row['complete_pair_committed'] for row in outcomes)},allow_nan=False))


if __name__ == '__main__':
    main()
