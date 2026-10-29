"""Explicit saved-state replay through a freshly issued genuine owner chart.

No saved payload is geometry authority. Literal source supports are reconstructed
using the existing public pair fixture and owner-qualified outer material chart.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from test_physical_mixed_repair_batch import chart_fixture
from corner_fan_saved_replay import read,sha
from anymesher._physical_corner_cavity import repair_corner_cavities,_score,_edges
from anymesher._analytic_metric_chart import AnalyticMetricChart
from anymesher._material_region_binding import MaterialRegionBinding
from anymesher.core import MeshCore
from anymesher.recombine import RecombinationReport
from anymesher.surface_mesh import SurfaceMeshOptions
import anygeometry as owner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uv',required=True,type=Path)
    parser.add_argument('--physical',required=True,type=Path)
    parser.add_argument('--output-directory',required=True,type=Path)
    args=parser.parse_args()
    args.output_directory.mkdir(parents=True,exist_ok=False)
    uv,physical=read(args.uv),read(args.physical)
    for key in ('triangles','triangle_active','triangle_ids','quads','quad_active','quad_ids'):
        if not np.array_equal(uv[key],physical[key]):
            raise ValueError('saved state correspondence differs: '+key)
    points=uv['points'][:,:2].copy()
    generator=chart_fixture.__wrapped__()
    try:
        model,inner=next(generator)
        face=next(face.id for face in model.faces.values()
                  if face.holes and face.surface == model.faces[inner.face_id].surface)
        regions=owner.query_material_surface_regions(model,(face,))
        loops=(model.faces[face].loop,*model.faces[face].holes)
        intervals=tuple((model.handle('edge',use.edge),i/4,(i+1)/4)
                        for loop in loops for use in loop for i in range(4))
        chords=owner.query_prepared_material_boundary_chords(model,regions,intervals,chord_error=.1)
        binding=MaterialRegionBinding(model,regions,regions.regions[0],face,
                                      geometry_approximation_error=.1)
        binding._curvilinear_state['chords']=chords
        chart=AnalyticMetricChart(model,face,region_binding=binding)
        # Derive fixed rows solely from complete active mixed boundary incidence.
        counts={}
        for cells,active in ((uv['triangles'],uv['triangle_active']),(uv['quads'],uv['quad_active'])):
            for cell in cells[active]:
                for edge in _edges(cell): counts[edge]=counts.get(edge,0)+1
        edges=tuple(edge for edge,count in counts.items() if count==1)
        fixed=sorted({node for edge in edges for node in edge})
        interior=np.asarray(sorted(set(range(len(points)))-set(fixed)),dtype=int)
        xyz=chart.evaluate(points)
        # Refuse a different metric map/support. Boundary chord witnesses are
        # intentionally kept literal, as in the public final registry overwrite.
        if not np.allclose(xyz[interior],physical['points'][interior],rtol=2e-14,atol=2e-14):
            raise ValueError('fresh owner chart does not reproduce saved interior coordinates')
        core=MeshCore(physical['points'],physical['triangles'],physical['quads'],
            triangle_ids=physical['triangle_ids'],quad_ids=physical['quad_ids'],
            triangle_active=physical['triangle_active'],quad_active=physical['quad_active'])
        report=RecombinationReport(core,np.empty((0,2),dtype=np.int64),
            np.empty(0,dtype=np.int64),np.empty(0),0,0)
        settings=SurfaceMeshOptions(min_angle=15.)
        old=(points.tobytes(),core.node_coordinates.tobytes(),core.triangle_connectivity.tobytes())
        result,updated,work,facts=repair_corner_cavities(points,report,settings,chart.evaluate,
            fixed,edges,topology_remaining=256)
        assert old==(points.tobytes(),core.node_coordinates.tobytes(),core.triangle_connectivity.tobytes())
        data=dict(inputs={str(path.resolve()):sha(path) for path in (args.uv,args.physical)},
            sources={str(path.resolve()):sha(path) for path in
                (Path(__file__),Path(__file__).parents[1]/'src/anymesher/_physical_corner_cavity.py',
                 Path(__file__).parents[1]/'src/anymesher/_physical_mixed_repair.py',
                 Path(__file__).parents[1]/'src/anymesher/surface_mesh.py')},
            transform=chart.transform.tolist(),before=_score(core,settings),after=_score(updated.mesh,settings),
            work=work,facts=facts)
        np.savez(args.output_directory/'candidate.npz',points=result,xyz=updated.mesh.node_coordinates,
                 triangles=updated.mesh.triangle_connectivity,quads=updated.mesh.quad_connectivity,
                 triangle_active=updated.mesh.triangle_active,quad_active=updated.mesh.quad_active)
        (args.output_directory/'replay.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
        print(json.dumps(dict(commits=facts['commits'],proposals=facts['complete_proposals'],work=work)))
    finally:
        generator.close()


if __name__=='__main__': main()
