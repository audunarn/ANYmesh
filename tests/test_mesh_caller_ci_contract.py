"""CI-only routing and exact positive caller JUnit contracts; no fake coverage."""
import argparse
from pathlib import Path
import re
import tomllib
import xml.etree.ElementTree as ET
import pytest

ROOT=Path(__file__).resolve().parents[1]
CASE='test_actual_parallel_warm_closure_matches_identical_serial_owner'
CONTRACT='mesh-warm-closure-qualified-v1'
FULL="github.event_name == 'workflow_dispatch' || github.event_name == 'schedule' || startsWith(github.ref, 'refs/tags/v')"
ORDINARY=('smoke','native-v2-contract','pytest','native-absent','gmsh','wheel','wheel-full-linux','qualified-caller-planning')


def jobs(text):
    pieces=re.split(r'^  ([a-z][a-z0-9-]*):\s*$',text.split('\njobs:\n',1)[1],flags=re.M)
    return dict(zip(pieces[1::2],pieces[2::2]))


def condition(block):
    value=re.search(r'^    if: (.*?)(?=^    [a-z]|\Z)',block,flags=re.M|re.S).group(1)
    return ' '.join(value.removeprefix('>-').split())


def verify_positive(path):
    selected=[case for case in ET.parse(path).iter('testcase')
        if case.get('classname','').rsplit('.',1)[-1]=='test_preparation_callback_context' and case.get('name')==CASE]
    if len(selected)!=1:raise ValueError('missing or duplicate required caller positive')
    case=selected[0];properties=[(p.get('name'),p.get('value')) for p in case.findall('./properties/property')]
    required={('caller_planning_contract',CONTRACT),('execution_scope','original-profile-assertions'),
              ('original_test_body_executed','True'),('numerical_worker_execution','True')}
    if any(case.find(kind) is not None for kind in ('failure','error','skipped')) or not required.issubset(properties):
        raise ValueError('required caller did not pass original positive assertions')
    if any(name=='execution_scope' and value!='original-profile-assertions' for name,value in properties):
        raise ValueError('initial fallback cannot count as caller qualification')
    return 1


def test_build_backend_is_available_only_in_dev_profile():
    config=tomllib.loads((ROOT/'pyproject.toml').read_text())
    assert 'setuptools>=77.0.3' in config['project']['optional-dependencies']['dev']
    assert config['project']['dependencies']==['numpy>=1.26','ANYgeometry[planar]>=0.4.7,<0.5']
    assert config['build-system']==dict(requires=['setuptools>=77','wheel'],**{'build-backend':'setuptools.build_meta'})


def test_diagnostic_boolean_routes_only_diagnostic_and_retains_normal_full_conditions():
    text=(ROOT/'.github/workflows/ci.yml').read_text();blocks=jobs(text)
    assert re.search(r'workflow_dispatch:\s+inputs:\s+s0_diagnostic_only:\s+type: boolean\s+default: false',text)
    assert set(blocks)==set(ORDINARY)|{'s0-diagnostic'}
    assert condition(blocks['s0-diagnostic'])=='inputs.s0_diagnostic_only == true'
    assert 'uses: ./.github/workflows/s0-mac-diagnostic.yml' in blocks['s0-diagnostic']
    assert "run-name: ${{ inputs.s0_diagnostic_only == true && 'DIAGNOSTIC ONLY - six S0 Mac faces' || 'Tests' }}" in text
    assert 'DIAGNOSTIC ONLY' in blocks['s0-diagnostic']
    assert condition(blocks['smoke'])=='inputs.s0_diagnostic_only != true'
    for name in ORDINARY[1:]:assert condition(blocks[name])=='inputs.s0_diagnostic_only != true && ('+FULL+')'
    diagnostic=(ROOT/'.github/workflows/s0-mac-diagnostic.yml').read_text()
    assert '  workflow_call:' in diagnostic and 'timeout-minutes: 15' in diagnostic
    assert '2d3b0f876976b4d2e970c531c4b71941f2bb78a6' in diagnostic


def test_full_matrix_and_immutable_owner_pin_are_preserved():
    text=(ROOT/'.github/workflows/ci.yml').read_text();blocks=jobs(text)
    assert 'os: [windows-latest, ubuntu-latest, macos-latest]' in blocks['pytest']
    assert 'python-version: ["3.11", "3.12", "3.13", "3.14"]' in blocks['pytest']
    assert 'python -m pytest --durations=25' in blocks['pytest']
    assert '--junitxml=mesh-results.xml' in blocks['pytest']
    pins=re.findall(r'\bref: ([0-9a-f]{40})',text)
    assert pins and set(pins)=={'68a577ed37d07d9991018f973c2349c295463e4f'}


def test_required_caller_lane_builds_only_geometry_planning_helper_and_pins_profile():
    block=jobs((ROOT/'.github/workflows/ci.yml').read_text())['qualified-caller-planning']
    for value in ('runs-on: windows-latest','python-version: "3.14.2"','numpy==2.4.6','scipy==1.17.1',
                  'ANYGEOMETRY_REQUIRE_PLANNING_NATIVE: "1"','ANYGEOMETRY_REQUIRE_QUALIFIED_PLANNING_TESTS: "1"',
                  'OMP_NUM_THREADS: "1"','OPENBLAS_NUM_THREADS: "1"','MKL_NUM_THREADS: "1"',
                  'tools/ci_planning_profile.py --preflight','test_preparation_callback_context.py::'+CASE,
                  'test_mesh_caller_ci_contract.py --verify-positive caller-planning-positive.xml'):
        assert value in block
    assert 'ANYGEOMETRY_ACCELERATE' not in block


@pytest.mark.parametrize('qualified,required',[(False,False),(False,True),(True,False),(True,True)])
def test_required_positive_flag_cannot_classify_initial_refusal(qualified,required):
    from test_preparation_callback_context import _caller_execution_scope
    if required and not qualified:
        with pytest.raises(AssertionError,match='qualified caller lane requires'):_caller_execution_scope(qualified,required)
    else:
        assert _caller_execution_scope(qualified,required)==('original-profile-assertions' if qualified else 'initial-admission-serial-fallback')


@pytest.mark.parametrize('variant',['initial','unknown','staged','attempted','abandoned','manifest','model','infrastructure','unretired','parallel','legacy'])
def test_simulated_initial_receipt_validation_never_claims_postcapture(variant):
    from test_preparation_callback_context import _assert_initial_serial_receipt
    # Pure receipt validation only. This is not actual mocked-platform coverage.
    receipt=dict(schema='R5-production-v1',ci_verified=False,disposition='serial',
        reason='planning resource helper requires Windows CPython with GIL',stages=[],attempted_stages=[],retired=True,costs={'abandoned':{}})
    if variant=='unknown':receipt['reason']='source-less module is not qualified'
    elif variant=='staged':receipt['stages']=['pair']
    elif variant=='attempted':receipt['attempted_stages']=['pair']
    elif variant=='abandoned':receipt['costs']['abandoned']={'pair':1}
    elif variant=='manifest':receipt['manifest_sha256']='captured'
    elif variant=='model':receipt['model_sha256']='accepted'
    elif variant=='infrastructure':receipt['infrastructure']={'host':'accepted'}
    elif variant=='unretired':receipt['retired']=False
    elif variant=='parallel':receipt['disposition']='parallel'
    elif variant=='legacy':receipt['reason']='infrastructure function global owner'
    import sys
    if variant=='initial' or variant=='legacy' and sys.version_info[:2]<(3,14):_assert_initial_serial_receipt(receipt)
    else:
        with pytest.raises(AssertionError):_assert_initial_serial_receipt(receipt)


@pytest.mark.parametrize('variant',['positive','missing','duplicate','failure','error','skipped','initial','false-body','missing-worker'])
def test_junit_validator_requires_actual_exact_positive_case(tmp_path,variant):
    root=ET.Element('testsuites');suite=ET.SubElement(root,'testsuite')
    for index in range(0 if variant=='missing' else 2 if variant=='duplicate' else 1):
        case=ET.SubElement(suite,'testcase',classname='test_preparation_callback_context',name=CASE)
        props=ET.SubElement(case,'properties')
        for name,value in [('caller_planning_contract',CONTRACT),('execution_scope','original-profile-assertions'),
                           ('original_test_body_executed','True'),('numerical_worker_execution','True')]:
            if variant=='missing-worker' and name=='numerical_worker_execution':continue
            if variant=='initial' and name=='execution_scope':value='initial-admission-serial-fallback'
            if variant=='false-body' and name=='original_test_body_executed':value='False'
            ET.SubElement(props,'property',name=name,value=value)
        if variant in ('failure','error','skipped'):ET.SubElement(case,variant)
    path=tmp_path/'actual.xml';ET.ElementTree(root).write(path)
    if variant=='positive':assert verify_positive(path)==1
    else:
        with pytest.raises(ValueError):verify_positive(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify-positive',type=Path,required=True)
    print(verify_positive(parser.parse_args().verify_positive))
