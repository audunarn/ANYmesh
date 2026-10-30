"""Headless owner -> admitted Tet4 -> optional ANYsolid P1/P2 consumer example.

The six explicit tetrahedra are a controlled fixture, not a generated mesh or
provider qualification. Run against installed candidate wheels; see README.md.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from anygeometry import GeometryModel
from anymesher import VolumeMesh, validate_volume_mesh


def controlled_fixture():
    model = GeometryModel()
    region = model.add_solid_prism(
        [[0, 0, 0], [1, 0, 0], [1, .5, 0], [0, .5, 0]],
        [0, 0, .25], offset=.125)
    source = model.solid_region_snapshot(region)
    cells = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
                      [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]])
    # Enumerate the explicit fixture's exterior independently of mesher helpers.
    incidence = {}
    for cell in cells:
        for triple in itertools.combinations(cell, 3):
            key = tuple(sorted(map(int, triple)))
            incidence.setdefault(key, []).append(cell)
    facets, owners = [], []
    for key, adjacent in sorted(incidence.items()):
        if len(adjacent) != 1:
            continue
        face = list(key)
        points = source.coordinates[face]
        opposite = next(row for row in adjacent[0] if row not in face)
        if np.dot(np.cross(points[1]-points[0], points[2]-points[0]),
                  source.coordinates[opposite]-points[0]) > 0:
            face[1], face[2] = face[2], face[1]
        matches = [ref for ref, loop in zip(source.face_refs, source.face_loops)
                   if set(face) <= set(loop)]
        if len(matches) != 1:
            raise ValueError('fixture triangle must have one source face owner')
        facets.append(face)
        owners.append(matches[0])
    boundary_ids = np.arange(300, 300+len(facets))
    mesh = VolumeMesh(
        source.coordinates, cells, node_ids=np.arange(100, 108),
        cell_ids=np.arange(200, 206), boundary_facets=facets,
        boundary_ids=boundary_ids,
        boundary_groups={name: boundary_ids[[owner in refs for owner in owners]]
                         for name, refs in source.face_groups.items()},
        cell_groups={source.region_id: np.arange(200, 206)},
        face_owners=dict(zip(map(int, boundary_ids), owners)),
        provenance=dict(fixture='explicit six-Tet4 box; no native provider',
                        source_model_id=source.model_id,
                        source_revision=source.revision,
                        source_region_id=source.region_id,
                        source_signature=source.source_signature))
    validate_volume_mesh(mesh, source, boundary_tolerance=1e-7)
    return source, mesh


def run():
    # Explicit optional consumer: ANYmesher does not depend on ANYsolid.
    import anysolid

    source, mesh = controlled_fixture()
    solid = anysolid.SolidMesh.snapshot(mesh)
    E, nu = 1000., .3
    gradient = np.array([[.03, .08, -.02], [.04, -.01, .05], [.02, -.03, .06]])
    strain = (gradient+gradient.T)/2
    stress = E/(1+nu)*strain + E*nu/((1+nu)*(1-2*nu))*np.trace(strain)*np.eye(3)
    results = []
    for order in (1, 2):
        prepared = anysolid.prepare(solid, {'elastic_modulus': E, 'poisson_ratio': nu}, order=order)
        k = prepared.assemble_linear()
        values = prepared.coefficient_coordinates @ gradient.T + [.1, -.2, .3]
        q = values[np.arange(prepared.ndof), prepared.coefficient_components]
        loads = np.zeros(prepared.ndof)
        for name, refs in source.face_groups.items():
            assert len(refs) == 1
            loop = source.face_loops[source.face_refs.index(refs[0])]
            points = source.coordinates[list(loop)]
            normal = np.cross(points[1]-points[0], points[2]-points[0])
            normal /= np.linalg.norm(normal)
            loads += prepared.traction(name, stress@normal)
        recovery = prepared.recover(q)
        np.testing.assert_allclose(recovery.strain, np.broadcast_to(strain, recovery.strain.shape), rtol=1e-10, atol=1e-13)
        np.testing.assert_allclose(recovery.stress, np.broadcast_to(stress, recovery.stress.shape), rtol=1e-10, atol=1e-10)
        residual = float(np.linalg.norm(k@q-loads)/max(np.linalg.norm(loads), 1.))
        symmetry = float(np.linalg.norm((k-k.T).data)/max(np.linalg.norm(k.data), 1.))
        assert residual <= 1e-9 and symmetry <= 1e-10
        energy = float(q@k@q)
        assert energy > 0
        np.testing.assert_allclose(energy, source.volume*np.sum(strain*stress), rtol=1e-10, atol=1e-12)
        body = np.array([2., -3., 5.])
        resultant = np.bincount(prepared.coefficient_components, weights=prepared.body_load(body), minlength=3)
        np.testing.assert_allclose(resultant, source.volume*body, rtol=1e-10, atol=1e-12)
        results.append(dict(order=order, coefficients=int(prepared.ndof),
                            equilibrium_residual=residual, stiffness_symmetry=symmetry,
                            energy=energy, body_resultant=resultant.tolist(),
                            strain=strain.tolist(), stress=stress.tolist()))
    return dict(status='PASS', scope='controlled fixture; delivery only',
                native_meshing_calls=0, nodes=len(mesh.coordinates), cells=len(mesh.cells),
                exterior_facets=len(mesh.boundary_facets), orders=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(run(), indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(result+'\n', encoding='utf-8')
    print(result)
