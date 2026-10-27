"""One raw provider feasibility case; criteria are loaded before generation."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path)
    a = p.parse_args()
    manifest_path = a.manifest or Path(__file__).resolve().parents[1] / 'docs/volume_feasibility_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    case = next(c for c in manifest['cases'] if c['id'] == a.case)
    from anysolid import SolidMesh
    import pytetwild
    assert importlib.metadata.version('pytetwild') == manifest['backend'].split('==')[1]
    dims = np.array(case['dimensions'])
    x = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                  [0,0,1],[1,0,1],[1,1,1],[0,1,1]], float) * dims
    loops = ((0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))
    triangles = np.array([tri for q in loops for tri in ((q[0],q[1],q[2]),(q[0],q[2],q[3]))], dtype=np.int32)
    started = time.perf_counter()
    vertices, cells = pytetwild.tetrahedralize(
        x, triangles, edge_length_abs=case['target_edge_length'],
        epsilon=case['boundary_tolerance']/float(np.linalg.norm(dims)),
        **manifest['options'])
    backend_wall = time.perf_counter()-started
    vertices = np.asarray(vertices, float)
    cells = np.array(cells, dtype=np.int64, copy=True)
    determinants = np.linalg.det(vertices[cells[:,1:]]-vertices[cells[:,:1]])
    reverse = determinants < 0
    cells[reverse,1:3] = cells[reverse,2:0:-1]
    mesh = SolidMesh(vertices, cells, provenance={'provider': 'pytetwild0.4.2', 'case':case})
    f = mesh.boundary_facets
    points = vertices[f]
    cross = np.cross(points[:,1]-points[:,0], points[:,2]-points[:,0])
    lengths = np.linalg.norm(cross, axis=1)
    normals = cross / lengths[:,None]
    sources = [(2,0,-1),(2,dims[2],1),(1,0,-1),(0,dims[0],1),(1,dims[1],1),(0,0,-1)]
    matches = []
    tol = case['boundary_tolerance']
    cosine = np.cos(np.deg2rad(manifest['limits']['maximum_face_normal_degrees']))
    face_areas = []
    worst_plane = 0.0
    for axis, coordinate, sense in sources:
        distances = np.max(np.abs(points[:,:,axis]-coordinate), axis=1)
        match = (distances <= tol) & (sense*normals[:,axis] >= cosine)
        matches.append(match)
        face_areas.append(float(np.sum(lengths[match])/2))
        if np.any(match):
            worst_plane = max(worst_plane, float(np.max(distances[match])))
    ownership = np.stack(matches).sum(axis=0)
    expected_areas = np.array([dims[0]*dims[1],dims[0]*dims[1],dims[0]*dims[2],dims[1]*dims[2],dims[0]*dims[2],dims[1]*dims[2]])
    area_error = float(np.max(np.abs(np.array(face_areas)/expected_areas-1)))
    volume = float(np.sum(np.abs(determinants))/6)
    volume_error = abs(volume/float(np.prod(dims))-1)
    predicates = {
        'cell_cap':len(cells)<=manifest['limits']['maximum_cells'],
        'complete_unique_face_ownership':bool(np.all(ownership==1)),
        'face_areas':area_error<=manifest['limits']['maximum_relative_face_area_error'],
        'volume':volume_error<=manifest['limits']['maximum_relative_volume_error'],
        'bounds':bool(np.all(vertices>=-tol) and np.all(vertices<=dims+tol)),
        'anysolid_constructor':True,
    }
    a.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(a.output/'mesh.npz',coordinates=vertices,cells=cells,boundary_facets=f,face_matches=np.stack(matches))
    result = dict(case=case, options=manifest['options'], backend_version=importlib.metadata.version('pytetwild'),
                  backend_wall_seconds=backend_wall,nodes=len(vertices),cells=len(cells),facets=len(f),
                  canonicalized_negative_cells=int(np.sum(reverse)),unowned_facets=int(np.sum(ownership==0)),
                  ambiguous_facets=int(np.sum(ownership>1)),maximum_plane_distance=worst_plane,
                  maximum_face_area_relative_error=area_error,relative_volume_error=volume_error,
                  predicates=predicates,status='FEASIBLE' if all(predicates.values()) else 'FAILED_FEASIBILITY')
    (a.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    return 0 if all(predicates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
