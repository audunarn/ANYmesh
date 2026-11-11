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
    assert 'wheel' in config['project']['optional-dependencies']['dev']
    assert config['project']['dependencies']==['numpy>=1.26','ANYgeometry[planar]>=0.4.7,<0.5']
    assert config['build-system']==dict(requires=['setuptools>=77','wheel'],**{'build-backend':'setuptools.build_meta'})


def test_qualified_failure_receipt_is_exact_and_unelided(tmp_path, capsys):
    import json
    from test_preparation_callback_context import _persist_caller_parallel_receipt
    receipt = dict(disposition='serial', reason='exact initial boundary ' + 'x' * 500,
                   stages=[], attempted_stages=[], retired=True,
                   costs={'abandoned': {}, 'callback_calls': 0})
    _persist_caller_parallel_receipt(tmp_path, receipt)
    assert json.loads((tmp_path / 'caller-parallel-receipt.json').read_text()) == receipt
    assert receipt['reason'] in capsys.readouterr().out


def test_trial_artifact_patterns_match_actual_hosted_pytest_directory_names():
    from fnmatch import fnmatchcase
    blocks = jobs((ROOT / '.github/workflows/ci.yml').read_text())
    # Actual completed Mac313/Mac314/Linux313 JUnit names, not inferred from
    # the longer Python function name. Pytest truncates the temporary basename.
    for job, base in [('pytest', 'ci-pytest-temp'),
                      ('wheel-full-linux', 'installed-ci-pytest-temp')]:
        patterns = [line.strip() for line in blocks[job].splitlines()
                    if line.strip().startswith(base + '/')]
        for directory in ('test_work_bearing_quadratic_tr2', 'test_work_bearing_quadratic_tr3'):
            for name in ('trial.json', 'oracle.json'):
                actual = f'{base}/{directory}/{name}'
                assert any(fnmatchcase(actual, pattern) for pattern in patterns), actual


def test_initial_refusal_receipt_precedes_unchanged_known_reason_gate():
    import ast
    from fnmatch import fnmatchcase
    source = ast.parse((ROOT / 'tests/test_preparation_callback_context.py').read_text())
    body = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                and node.name == '_initial_serial_warm_closure_matches_identical_serial_owner').body
    def call_index(name):
        return next(i for i, node in enumerate(body) if isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == name)
    assert call_index('_persist_caller_parallel_receipt') < call_index('_assert_initial_serial_receipt')
    blocks = jobs((ROOT / '.github/workflows/ci.yml').read_text())
    for job, base in [('pytest', 'ci-pytest-temp'),
                      ('wheel-full-linux', 'installed-ci-pytest-temp')]:
        patterns = [line.strip() for line in blocks[job].splitlines()
                    if line.strip().startswith(base + '/')]
        actual = f'{base}/test_actual_parallel_warm_clos0/caller-parallel-receipt.json'
        assert any(fnmatchcase(actual, pattern) for pattern in patterns)


def test_qualified_lane_retains_only_own_receipt_and_status_artifacts():
    import ast
    source = ast.parse((ROOT / 'tests/test_preparation_callback_context.py').read_text())
    body = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                and node.name == '_qualified_parallel_warm_closure_matches_identical_serial_owner').body
    persist = next(i for i, node in enumerate(body) if isinstance(node, ast.Expr)
                   and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                   and node.value.func.id == '_persist_caller_parallel_receipt')
    assert persist < next(i for i, node in enumerate(body) if isinstance(node, ast.Assert))
    block = jobs((ROOT / '.github/workflows/ci.yml').read_text())['qualified-caller-planning']
    assert '--basetemp=caller-planning-temp' in block
    assert 'if: always()' in block
    prefix = 'caller-planning-temp/test_actual_parallel_warm_clos*/'
    assert prefix + 'caller-parallel-receipt.json' in block
    for name in ('receipt.json', 'initialization-phases.json', 'admitted-*.json',
                 'cleanup.json', 'workers-cleanup.json'):
        assert prefix + 'production/**/' + name in block
    assert 'production/**/*.json' not in block
    assert 'production/**/session.json' not in block


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


CURRENT_GEOMETRY_REF='33dfbaad89a76dd9e925fd6c628602213ce8c2b8'
OLD_GEOMETRY_REF='f645523c8006efe61e94678a5056fba3c9a6a39d'
OLD_MESHER_REF='aa19c0fa6b796a0d7cb1793a87840f52899dfa24'


def owner_checkout_pins(text):
    """Bind every explicit owner checkout to its exact ref AND role path."""
    rows=[]
    for block in text.split('\n      - '):
        repos=re.findall(r'^\s*repository: audunarn/(ANYgeometry|ANYmesh)\s*$',block,re.M)
        if not repos:continue
        refs=re.findall(r'^\s*ref: ([0-9a-f]{40})\s*$',block,re.M)
        paths=re.findall(r'^\s*path: (\S+)\s*$',block,re.M)
        assert len(repos)==len(refs)==len(paths)==1,'owner checkout must have exactly one immutable ref and role path'
        rows.append((repos[0],refs[0],paths[0]))
    return rows


def test_owner_checkout_parser_binds_exact_ref_and_role_path():
    text='\n      - uses: actions/checkout@v4\n        with:\n          repository: audunarn/ANYgeometry\n          ref: '+CURRENT_GEOMETRY_REF+'\n          path: .ecosystem/ANYgeometry\n'
    assert owner_checkout_pins(text)==[('ANYgeometry',CURRENT_GEOMETRY_REF,'.ecosystem/ANYgeometry')]
    altered=text.replace('.ecosystem/ANYgeometry','.s0-old-geometry')
    assert owner_checkout_pins(altered)!=owner_checkout_pins(text)


@pytest.mark.parametrize('mutation',['branch','short_sha','missing_path','duplicate_ref'])
def test_owner_checkout_parser_rejects_missing_or_ambiguous_metadata(mutation):
    text='\n      - uses: actions/checkout@v4\n        with:\n          repository: audunarn/ANYgeometry\n          ref: '+CURRENT_GEOMETRY_REF+'\n          path: .ecosystem/ANYgeometry\n'
    if mutation=='branch':text=text.replace(CURRENT_GEOMETRY_REF,'main')
    elif mutation=='short_sha':text=text.replace(CURRENT_GEOMETRY_REF,CURRENT_GEOMETRY_REF[:12])
    elif mutation=='missing_path':text=text.replace('          path: .ecosystem/ANYgeometry\n','')
    else:text+='          ref: '+OLD_GEOMETRY_REF+'\n'
    with pytest.raises(AssertionError,match='exactly one'):owner_checkout_pins(text)


def test_full_matrix_and_immutable_owner_pin_are_preserved():
    text=(ROOT/'.github/workflows/ci.yml').read_text();blocks=jobs(text)
    assert 'os: [windows-latest, ubuntu-latest, macos-latest]' in blocks['pytest']
    assert 'python-version: ["3.11", "3.12", "3.13", "3.14"]' in blocks['pytest']
    assert 'python -m pytest --durations=25' in blocks['pytest']
    assert '--junitxml=mesh-results.xml' in blocks['pytest']
    assert '--basetemp=ci-pytest-temp' in blocks['pytest']
    assert '--basetemp=ci-pytest-temp --maxfail=1' in blocks['pytest']
    assert 'ci-pytest-temp/test_work_bearing_quadratic_tr*/trial.json' in blocks['pytest']
    assert '--junitxml=installed-mesh-results.xml --basetemp=installed-ci-pytest-temp' in blocks['wheel-full-linux']
    assert '--basetemp=installed-ci-pytest-temp --maxfail=1' in blocks['wheel-full-linux']
    assert 'installed-ci-pytest-temp/test_work_bearing_quadratic_tr*/trial.json' in blocks['wheel-full-linux']
    from collections import Counter
    assert Counter(owner_checkout_pins(text))==Counter({
        ('ANYgeometry',CURRENT_GEOMETRY_REF,'.ecosystem/ANYgeometry'):8,
        ('ANYgeometry',OLD_GEOMETRY_REF,'.s0-old-geometry'):2,
        ('ANYmesh',OLD_MESHER_REF,'.s0-old-mesher'):2})
    for name in ('pytest','wheel-full-linux'):
        assert set(owner_checkout_pins(blocks[name]))=={
            ('ANYgeometry',CURRENT_GEOMETRY_REF,'.ecosystem/ANYgeometry'),
            ('ANYgeometry',OLD_GEOMETRY_REF,'.s0-old-geometry'),
            ('ANYmesh',OLD_MESHER_REF,'.s0-old-mesher')}


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
@pytest.mark.parametrize('reason',['planning resource helper requires Windows CPython with GIL',
                                 'Windows private-commit accounting unavailable'])
def test_simulated_initial_receipt_validation_never_claims_postcapture(variant,reason):
    from test_preparation_callback_context import _assert_initial_serial_receipt
    # Pure receipt validation only. This is not actual mocked-platform coverage.
    receipt=dict(schema='R5-production-v1',ci_verified=False,disposition='serial',
        reason=reason,stages=[],attempted_stages=[],retired=True,costs={'abandoned':{}})
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
    if variant=='initial':_assert_initial_serial_receipt(receipt)
    else:
        with pytest.raises(AssertionError):_assert_initial_serial_receipt(receipt)


def test_unattributed_infrastructure_owner_failure_is_not_a_runtime_profile():
    from test_preparation_callback_context import _assert_initial_serial_receipt
    receipt=dict(schema='R5-production-v1',ci_verified=False,disposition='serial',
        reason='infrastructure function global owner',stages=[],attempted_stages=[],
        retired=True,costs={'abandoned':{}})
    with pytest.raises(AssertionError):
        _assert_initial_serial_receipt(receipt)


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
