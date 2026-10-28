"""Changed registration and immutable reuse controls only; no native calls."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import pytest

PATH=Path(__file__).resolve().parents[1]/'tools/qualify_thin_volume_v2.py'
spec=importlib.util.spec_from_file_location('thin_v2',PATH)
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


def manifests():
    return json.loads(study.MANIFEST.read_text()),json.loads(study.MANIFEST.with_name('thin_resolution_manifest_20261028.json').read_text())


def test_only_sizing_case_delta_and_unchanged_governing_limits():
    revised,original=manifests()
    study.registration_check(revised,original)
    for case in revised['cases']:
        assert case['target_edge_length']==.002
        assert case['boundary_tolerance']==1e-6
    assert revised['execution']==dict(maximum_native_calls=2,native_calls_per_case=1,maximum_native_process_seconds=120,complete_case_seconds=480,complete_process_budget_seconds=1200,retries=0)


@pytest.mark.parametrize('field',['backend','backend_wheel','backend_identity','options','execution','limits','metric_definitions'])
def test_registration_refuses_changed_governing_contract(field):
    revised,original=manifests();revised[field]=None
    with pytest.raises(ValueError):study.registration_check(revised,original)


@pytest.mark.parametrize('mutation',['offset','identity','duplicate','sizing'])
def test_registration_refuses_any_other_case_delta(mutation):
    revised,original=manifests()
    if mutation=='offset':revised['cases'][1]['input_offset']=5
    elif mutation=='identity':revised['cases'][0]['id']='other'
    elif mutation=='duplicate':revised['cases'][1]=copy.deepcopy(revised['cases'][0])
    else:revised['cases'][0]['target_edge_length']=.0025
    with pytest.raises(ValueError):study.registration_check(revised,original)


def test_reuse_preserves_original_bytes_snapshot_ids_and_analytics(tmp_path,monkeypatch):
    q=study.helpers();revised,_=manifests()
    original=json.loads((study.ORIGINAL/'frozen/frozen.json').read_text())
    monkeypatch.setattr(q,'backend_identity',lambda *args:original['backend'])
    output=tmp_path/'frozen'
    frozen=study.reuse_exact_inputs(q,revised,original,output)
    proof=json.loads((output/'reuse-proof.json').read_text())
    for case in revised['cases']:
        case_id=case['id'];expected=copy.deepcopy(original['cases'][case_id]);expected['parameters']=case
        assert frozen['cases'][case_id]==expected
        for name in ('geometry.json','snapshot.json','input.npz'):
            assert (output/case_id/name).read_bytes()==(study.ORIGINAL/'frozen'/case_id/name).read_bytes()
            assert proof[case_id][name]['byte_identical']
    with pytest.raises(FileExistsError):study.reuse_exact_inputs(q,revised,original,output)


def test_unchanged_metric_implementations_reuse_original_controlled_evidence():
    original=PATH.with_name('qualify_thin_volume.py')
    old=ast.parse(original.read_text());new=ast.parse(PATH.read_text())
    for name in ('regular_tet_condition','sampling_fraction'):
        extract=lambda tree:ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        assert extract(old)==extract(new)
    identity=json.loads((study.ORIGINAL/'frozen/study-identity.json').read_text())
    for name in ('qualify_thin_volume.py','workflow2_process_supervisor.py','qualify_retained_volume.py','volume_peak_worker.py'):
        assert study.stream_sha(PATH.with_name(name))==identity['files']['ANYmesh/tools/'+name]
    import xml.etree.ElementTree as ET
    suite=ET.parse(study.ORIGINAL/'linux-unit.xml').getroot().find('testsuite')
    assert suite.attrib['tests']=='14' and suite.attrib['failures']==suite.attrib['errors']=='0'
