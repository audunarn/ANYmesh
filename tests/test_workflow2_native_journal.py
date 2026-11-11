"""Live journal preserves phase and failure evidence without claiming completion."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def tool():
    path=Path(__file__).resolve().parents[1]/'tools/workflow2_native_gate.py'
    spec=importlib.util.spec_from_file_location('native_journal_under_test',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_failure_report_is_flushed_with_complete_longrepr(tmp_path):
    m=tool();j=m.Journal(tmp_path/'events.jsonl')
    j.pytest_runtest_logstart('sample::failure',None)
    j.pytest_runtest_call(SimpleNamespace(nodeid='sample::failure'))
    j.pytest_runtest_logreport(SimpleNamespace(nodeid='sample::failure',when='call',outcome='failed',duration=.25,longrepr='complete failure detail',failed=True,skipped=False))
    result=m.journal_summary(j.path)
    assert result['counts']['failed']==1
    assert result['reports'][0]['longrepr']=='complete failure detail'
    assert result['active_node']=='sample::failure'
    assert not result['session_finished']
    j.handle.close()


def test_partial_run_retains_active_phase_and_no_full_pass(tmp_path):
    m=tool();j=m.Journal(tmp_path/'events.jsonl')
    j.pytest_runtest_logstart('sample::hang',None)
    j.pytest_runtest_setup(SimpleNamespace(nodeid='sample::hang'))
    j.pytest_runtest_logreport(SimpleNamespace(nodeid='sample::hang',when='setup',outcome='passed',duration=.1,longrepr=None,failed=False,skipped=False))
    j.pytest_runtest_call(SimpleNamespace(nodeid='sample::hang'))
    result=m.journal_summary(j.path)
    assert (result['active_node'],result['active_phase'])==('sample::hang','call')
    assert result['counts']['passed']==0 and result['finished_nodes']==0
    assert not result['session_finished']
    j.handle.close()
