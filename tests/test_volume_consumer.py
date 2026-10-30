"""Installed consumer contract on controlled meshes, not provider qualification."""
import numpy as np
import pytest

anysolid = pytest.importorskip('anysolid', reason='read-only ANYsolid consumer lane')
from anygeometry import GeometryModel
from anymesher import generate_volume_mesh


@pytest.mark.parametrize('order', [1, 2])
def test_native_prism_controlled_provider_affine_elasticity(monkeypatch, order):
    model = GeometryModel()
    region = model.add_solid_prism(
        [[0,0,0],[1,0,0],[1,.5,0],[0,.5,0]], [0,0,.25], offset=.125)
    source = model.solid_region_snapshot(region)
    # Independent six-tet subdivision is a controlled fixture, never the public
    # meshing algorithm or a fallback when the optional provider fails.
    cells = np.array([[0,1,2,6],[0,2,3,6],[0,3,7,6],
                      [0,7,4,6],[0,4,5,6],[0,5,1,6]])
    def controlled(source, edge, timeout, cancel):
        return source.coordinates.copy(), cells.copy(), {'backend':'controlled_consumer_fixture'}
    monkeypatch.setattr('anymesher._volume_process.run_provider', controlled)
    mesh = generate_volume_mesh(source, target_edge_length=.25, boundary_tolerance=1e-7)
    solid = anysolid.SolidMesh.snapshot(mesh)
    assert len(solid.boundary_facets) == 12
    assert set(solid.boundary_groups) == set(source.face_groups)
    np.testing.assert_array_equal(solid.node_ids, mesh.node_ids)
    np.testing.assert_array_equal(solid.boundary_ids, mesh.boundary_ids)
    E, nu = 1000., .3
    prepared = anysolid.prepare(solid, {'elastic_modulus':E,'poisson_ratio':nu}, order=order)
    k = prepared.assemble_linear()
    gradient = np.array([[.03,.08,-.02],[.04,-.01,.05],[.02,-.03,.06]])
    values = prepared.coefficient_coordinates @ gradient.T + [.1,-.2,.3]
    q = values[np.arange(prepared.ndof), prepared.coefficient_components]
    strain = (gradient+gradient.T)/2
    mu, lam = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
    stress = 2*mu*strain + lam*np.trace(strain)*np.eye(3)
    loads = np.zeros(prepared.ndof)
    for name in source.face_groups:
        assert len(source.face_groups[name]) == 1
        index = source.face_refs.index(source.face_groups[name][0])
        points = source.coordinates[list(source.face_loops[index])]
        normal = np.cross(points[1]-points[0],points[2]-points[0])
        normal /= np.linalg.norm(normal)
        loads += prepared.traction(name, stress@normal)
    assert np.linalg.norm(k@q-loads)/max(np.linalg.norm(loads),1.) <= 1e-9
    assert np.linalg.norm((k-k.T).data)/max(np.linalg.norm(k.data),1.) <= 1e-10
    recovery = prepared.recover(q)
    np.testing.assert_allclose(recovery.strain,np.broadcast_to(strain,recovery.strain.shape),rtol=1e-10,atol=1e-13)
    np.testing.assert_allclose(recovery.stress,np.broadcast_to(stress,recovery.stress.shape),rtol=1e-10,atol=1e-10)
    energy = float(q@k@q)
    assert energy > 0
    np.testing.assert_allclose(energy,source.volume*np.sum(strain*stress),rtol=1e-10,atol=1e-12)
    rigid = prepared.rigid_modes()
    assert rigid.shape[1] == 6
    assert np.linalg.norm(k@rigid)/max(np.linalg.norm(k.data),1.) <= 1e-9
    body = np.array([2.,-3.,5.])
    force = prepared.body_load(body)
    resultant = np.bincount(prepared.coefficient_components,weights=force,minlength=3)
    np.testing.assert_allclose(resultant,source.volume*body,rtol=1e-10,atol=1e-12)
