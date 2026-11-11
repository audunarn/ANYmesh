"""Synthetic acceptance refusal controls; do not run a native operation."""
import copy
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

root = Path(__file__).resolve().parents[1]
path = root/'runner.py' if (root/'runner.py').exists() else root/'tools/workflow2_windows_native.py'
spec = importlib.util.spec_from_file_location('native_guard_driver',path)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


@pytest.fixture
def evidence(tmp_path):
    expected = [f'tests/test_fixture.py::test_{n}' for n in range(116)] + [
        'tests/test_workflow2_native_operations.py::test_required_geometry_native_plane_kernel',
        'tests/test_workflow2_native_operations.py::test_required_mesher_native_orientation_kernel']
    actual = dict(completed=True,exitstatus=0,selected=list(expected),deselected=[],reports=[
        dict(nodeid=node,when=phase,outcome='passed') for node in expected
        for phase in ('setup','call','teardown')])
    xml = tmp_path/'pytest.xml'
    suite = ET.Element('testsuite',tests='118',failures='0',errors='0',skipped='0')
    for node in expected:
        ET.SubElement(suite,'testcase',name=node)
    ET.ElementTree(suite).write(xml)
    return expected,actual,xml


def test_complete_exact_inventory_is_accepted(evidence):
    assert driver.validate_outcomes(*evidence)['passed']==118


@pytest.mark.parametrize('mutation',['deselected','skipped','missing_witness','missing_phase','incomplete','bad_exit'])
def test_incomplete_or_narrowed_execution_is_refused(evidence,mutation):
    expected,actual,xml = evidence
    actual = copy.deepcopy(actual)
    if mutation=='deselected': actual['deselected']=[expected[-1]]
    if mutation=='skipped': actual['reports'][-2]['outcome']='skipped'
    if mutation=='missing_witness':
        actual['selected'].pop()
        actual['reports']=[r for r in actual['reports'] if r['nodeid']!=expected[-1]]
    if mutation=='missing_phase': actual['reports'].pop()
    if mutation=='incomplete': actual['completed']=False
    if mutation=='bad_exit': actual['exitstatus']=1
    with pytest.raises(ValueError): driver.validate_outcomes(expected,actual,xml)


@pytest.mark.parametrize('mutation',['missing','truncated','skipped','missing_case'])
def test_missing_or_bad_junit_is_refused(evidence,mutation):
    expected,actual,xml = evidence
    if mutation=='missing': xml.unlink()
    if mutation=='truncated': xml.write_text('<testsuites>')
    if mutation in ('skipped','missing_case'):
        tree=ET.parse(xml)
        if mutation=='skipped': ET.SubElement(tree.getroot()[0],'skipped')
        else: tree.getroot().remove(tree.getroot()[0])
        tree.write(xml)
    with pytest.raises((ValueError,FileNotFoundError,ET.ParseError)):
        driver.validate_outcomes(expected,actual,xml)


def test_environment_drops_external_pytest_and_python_controls(monkeypatch,tmp_path):
    keys=('PYTEST_ADDOPTS','PYTEST_PLUGINS','PYTHONOPTIMIZE','PYTHONHOME','PYTHONSTARTUP','PYTHONINSPECT')
    for key in keys: monkeypatch.setenv(key,'hostile-control')
    env=driver.environment(tmp_path)
    assert all(key not in env for key in keys)
    assert env['PYTEST_DISABLE_PLUGIN_AUTOLOAD']=='1'
