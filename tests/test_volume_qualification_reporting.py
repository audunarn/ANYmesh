"""A nonfinite numerical result must never become accepted qualification JSON."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

anysolid = pytest.importorskip('anysolid', reason='read-only qualification consumer lane')


def qualification_tool():
    path = Path(__file__).resolve().parents[1]/'tools/qualify_retained_volume.py'
    spec = importlib.util.spec_from_file_location('volume_qualification_reporting',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nonfinite_stress_is_failed_by_actual_qualification_entrypoint(monkeypatch,tmp_path):
    coordinates = np.array([[0,0,0],[1,0,0],[1,.5,0],[0,.5,0],
        [0,0,.25],[1,0,.25],[1,.5,.25],[0,.5,.25]],dtype=float)
    cells = np.array([[0,1,2,6],[0,2,3,6],[0,3,7,6],
        [0,7,4,6],[0,4,5,6],[0,5,1,6]])
    raw = tmp_path/'raw.npz'
    np.savez(raw,coordinates=coordinates,cells=cells)
    manifest = {'status':'READY_RETAINED_MESH_QUALIFICATION','backend':'controlled fixture',
        'limits':{'maximum_coefficients':1000,'residual':1e-9,'work_symmetry':1e-10,
                  'resultant':1e-8,'patch_tensors':1e-10,'solved_displacement':1e-7},
        'cases':[{'id':'box','dimensions':[1,.5,.25],'target_edge_length':.125,
            'boundary_tolerance':2.5e-5,'retained_mesh':'raw.npz',
            'retained_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()}]}
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest),encoding='utf-8')
    original = anysolid.prepare

    class NonfiniteStress:
        def __init__(self,prepared): self.prepared=prepared
        def __getattr__(self,name): return getattr(self.prepared,name)
        def recover(self,q):
            recovered = self.prepared.recover(q)
            return replace(recovered,stress=np.full_like(recovered.stress,np.nan))

    def prepare(mesh,material,*,order=2):
        return NonfiniteStress(original(mesh,material,order=order))

    monkeypatch.setattr(anysolid,'prepare',prepare)
    output = tmp_path/'result'
    monkeypatch.setattr(sys,'argv',['qualify_retained_volume.py','--manifest',str(path),
        '--case','box','--workspace',str(tmp_path),'--output',str(output)])
    assert qualification_tool().main()==1
    def reject(value): raise AssertionError('nonfinite JSON constant '+value)
    result = json.loads((output/'result.json').read_text(),parse_constant=reject)
    assert result['status']=='FAILED'
    assert result['native_calls']==0
    assert result['orders']==[]
    assert 'nonfinite recovered' in result['error']


def test_nonfinite_resultant_input_is_refused():
    from types import SimpleNamespace
    prepared = SimpleNamespace(coefficient_components=np.arange(3),
        coefficient_coordinates=np.zeros((3,3)),ndof=3)
    with pytest.raises(ValueError,match='nonfinite load'):
        qualification_tool().vector_resultants(prepared,np.array([1.,np.nan,0.]))


@pytest.mark.parametrize('bad_modes',['zero','dependent','nonphysical'])
def test_rigid_gate_rejects_false_modes_even_with_zero_stiffness(bad_modes):
    from types import SimpleNamespace
    from scipy.sparse import csr_matrix
    corners = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=float)
    coords = np.repeat(corners,3,axis=0)
    if bad_modes=='zero': modes=np.zeros((12,6))
    elif bad_modes=='dependent': modes=np.ones((12,6))
    else: modes=np.eye(12)[:,:6]
    prepared=SimpleNamespace(ndof=12,coefficient_coordinates=coords,
        coefficient_components=np.tile(np.arange(3),4),rigid_modes=lambda:modes)
    limits={'work_symmetry':1e-10,'residual':1e-9}
    if bad_modes in ('zero','dependent'):
        with pytest.raises(ValueError,match='zero|rank'):
            qualification_tool().rigid_checks(prepared,csr_matrix((12,12)),limits)
    else:
        assert not qualification_tool().rigid_checks(prepared,csr_matrix((12,12)),limits)['checks']['six_rigid_modes']
