"""Replay a root-captured real pre-transition stage with freshly issued owner data."""
import argparse,json,pickle,time,hashlib
from pathlib import Path
from fractions import Fraction as Q
from types import SimpleNamespace
import numpy as np
import anygeometry as owner
from anygeometry.nurbs_intersection_curve import intersection_curve_operation
from anymesher._material_region_binding import MaterialRegionBinding
from anymesher._analytic_metric_chart import AnalyticMetricChart
from anymesher.surface_mesh import _curvilinear_mixed_preview,_mixed_preview_key


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',required=True,type=Path)
    parser.add_argument('--output-directory',required=True,type=Path)
    parser.add_argument('--require-compound',action='store_true')
    parser.add_argument('--check-compound-exception',action='store_true')
    args=parser.parse_args(); args.output_directory.mkdir(parents=True,exist_ok=False)
    def event(phase,**data):
        with (args.output_directory/'events.jsonl').open('a',encoding='utf8') as stream:
            stream.write(json.dumps(dict(phase=phase,clock=time.perf_counter(),**data))+'\n')
    event('load.begin')
    source=args.capture
    context=json.loads((source/'owner-context.json').read_text())
    with np.load(source/'candidate.npz',allow_pickle=False) as archive:
        saved={name:archive[name].copy() for name in archive.files}
    # Explicit root-owned local capture only; never a public geometry proof.
    with (source/'settings-report.pickle').open('rb') as stream:
        settings,report,candidate=pickle.load(stream)
    with intersection_curve_operation():
        model=owner.from_dict(json.loads((source/'geometry.json').read_text()))
        collection=owner.query_material_surface_regions(model,tuple(context['collection_faces']))
        region=next(item for item in collection.regions if any(face.id==context['face'] for face in item.faces))
        binding=MaterialRegionBinding(model,collection,region,context['face'],
            geometry_approximation_error=context['approximation_error'])
        event('chords.begin',intervals=len(context['intervals']))
        chords=owner.query_prepared_material_boundary_chords(model,collection,
            tuple((model.handle('edge',int(edge)),Q(a),Q(b)) for edge,a,b in context['intervals']),
            chord_error=context['chord_error'])
        owner.validate_prepared_material_boundary_chords_binding(model,collection,chords)
        binding._curvilinear_state['chords']=chords
        chart=AnalyticMetricChart(model,context['face'],region_binding=binding)
        np.testing.assert_array_equal(chart.transform,saved['transform'])
        np.testing.assert_allclose(chart.evaluate(saved['points']),saved['xyz'],rtol=2e-14,atol=2e-14)
        assert np.array_equal(candidate.points,saved['points']) and np.array_equal(candidate.triangles,saved['triangles'])
        original=candidate.points.tobytes(),candidate.triangles.tobytes()
        original_model=owner.to_dict(model)
        if args.check_compound_exception:
            from importlib import import_module
            mixed=import_module('anymesher._physical_mixed_repair')
            original_repair=mixed.repair_mixed_coordinates
            failure=RuntimeError('injected genuine compound polish failure')
            def fault(*positional,**keywords):
                if keywords.get('_trial_guard') is not None:
                    raise failure
                return original_repair(*positional,**keywords)
            mixed.repair_mixed_coordinates=fault
            try:
                try:
                    _curvilinear_mixed_preview(candidate,SimpleNamespace(segments=saved['protected']),
                        settings,chart.evaluate,candidate.points[saved['pinned']],None,
                        topology_remaining=max(0,settings.native_options.max_topology_operations-report['topology_operations']))
                except RuntimeError as caught:
                    assert caught is failure
                else:
                    raise AssertionError('genuine compound branch was not reached')
            finally:
                mixed.repair_mixed_coordinates=original_repair
            assert original==(candidate.points.tobytes(),candidate.triangles.tobytes())
            assert original_model==owner.to_dict(model)
            (args.output_directory/'failure-check.json').write_text(
                json.dumps(dict(exception_identity=True,input_unchanged=True,source_unchanged=True)),encoding='utf8')
            print('genuine compound exception preserved inputs and source')
            return
        event('preview.begin',points=len(candidate.points),triangles=len(candidate.triangles))
        preview=_curvilinear_mixed_preview(candidate,SimpleNamespace(segments=saved['protected']),
            settings,chart.evaluate,candidate.points[saved['pinned']],None,
            topology_remaining=max(0,settings.native_options.max_topology_operations-report['topology_operations']))
        event('preview.end',accepted=preview['accepted'],exchange_work=preview['report'].exchange_work,
            topology_work=preview['corner_work'])
        assert original==(candidate.points.tobytes(),candidate.triangles.tobytes())
        assert original_model==owner.to_dict(model)
        facts=preview['corner_facts']
        if args.require_compound:
            from anymesher._physical_corner_cavity import _score
            from anymesher._physical_t3_refinement import _alternative_progress
            assert facts.get('compound_polish_attempts',0)>0
            seeds=facts.get('compound_seeds',())
            assert facts['compound_polish_attempts'] >= 2 and len(seeds) == 2
            assert all(seed['status'] in ('progress', 'refused') and seed['work'] > 0 for seed in seeds)
            assert sum(seed['work'] for seed in seeds)<=facts['polish_work']
            assert preview['report'].exchange_work<=settings.max_recombination_work
            assert _score(preview['report'].mesh,settings)==facts['final_quality']
            if facts['commits']:
                assert _alternative_progress(facts['initial_quality'],facts['final_quality'])
        output=dict(accepted=preview['accepted'],quality=preview['quality'],corner=preview['corner_facts'],
            exchange_work=preview['report'].exchange_work,topology_work=preview['corner_work'],
            inputs={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in
                ('owner-context.json','geometry.json','candidate.npz','settings-report.pickle')})
        (args.output_directory/'replay.json').write_text(json.dumps(output,indent=2),encoding='utf8')
        core=preview['report'].mesh
        np.savez(args.output_directory/'candidate.npz',points=preview['points'],xyz=core.node_coordinates,
            triangles=core.triangle_connectivity,quads=core.quad_connectivity,
            triangle_active=core.triangle_active,quad_active=core.quad_active)
        print(json.dumps(dict(accepted=preview['accepted'],corner=preview['corner_facts']['commits'])))


if __name__=='__main__': main()
