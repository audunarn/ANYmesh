"""Controlled descendant-session timeout/cancellation checks; no real mesher."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import pytest

PATH=Path(__file__).resolve().parents[1]/'tools/workflow2_process_supervisor.py'
spec=importlib.util.spec_from_file_location('owned_descendant_supervisor',PATH)
supervisor=importlib.util.module_from_spec(spec);spec.loader.exec_module(supervisor)
LINUX_PIDFD=(sys.platform.startswith('linux') and hasattr(os,'pidfd_open')
             and hasattr(signal,'pidfd_send_signal'))


@pytest.mark.skipif(not LINUX_PIDFD,reason='Linux controller requires pidfd APIs; child ABI support is separate')
@pytest.mark.parametrize('mode',['timeout','cancel','interrupt'])
def test_nested_new_sessions_are_killed_and_reaped(mode,tmp_path):
    ready=tmp_path/'ready.json'
    grandchild=("import os,json,time;from pathlib import Path;"
        f"Path({str(ready)!r}).write_text(json.dumps(dict(child=os.getppid(),grandchild=os.getpid())));time.sleep(30)")
    child=f"import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',{grandchild!r}],start_new_session=True);time.sleep(30)"
    case=f"import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',{child!r}],start_new_session=True);time.sleep(30)"
    thread=None
    if mode=='interrupt':
        def interrupt_after_ready():
            deadline=time.monotonic()+5
            while not ready.exists() and time.monotonic()<deadline:time.sleep(.01)
            if ready.exists():os.kill(os.getpid(),signal.SIGTERM)
        thread=threading.Thread(target=interrupt_after_ready,daemon=True);thread.start()
    started=time.monotonic()
    cancel=(lambda:ready.exists() and time.monotonic()-started>.2) if mode=='cancel' else None
    result=supervisor.run_linux([sys.executable,'-c',case],tmp_path/'process',1.5,
        dict(os.environ),tmp_path,cancel=cancel)
    if thread is not None:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert result['status']=={'timeout':'timeout','cancel':'cancelled','interrupt':'interrupted'}[mode]
    assert result['cleanup']['verified'] and result['cleanup']['remaining_members']==[]
    pids=json.loads(ready.read_text())
    assert len({row['session'] for row in result['cleanup']['owned_identities']})>=3
    for pid in (result['pid'],pids['child'],pids['grandchild']):
        assert not Path('/proc',str(pid)).exists()
    assert json.loads((tmp_path/'process/result.json').read_text())['cleanup']['verified']


@pytest.mark.skipif(not LINUX_PIDFD,reason='Linux controller requires pidfd APIs; child ABI support is separate')
@pytest.mark.parametrize('caller',['generic-cli','source-gate'])
def test_reusable_check_entrypoints_clean_a_child_in_another_session(caller,tmp_path,monkeypatch):
    from types import SimpleNamespace
    ready=tmp_path/'pid.txt'
    child=f"import os,time;from pathlib import Path;Path({str(ready)!r}).write_text(str(os.getpid()));time.sleep(30)"
    case=f"import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',{child!r}],start_new_session=True);time.sleep(30)"
    repo=PATH.parents[1]
    if caller=='generic-cli':
        command=[sys.executable,str(repo/'tools/workflow2_check.py'),'--name','caller',
            '--cwd',str(repo),'--evidence-root',str(tmp_path/'evidence'),'--timeout','1',
            '--',sys.executable,'-c',case]
        result=supervisor.run_linux(command,tmp_path/'outer',15,dict(os.environ),repo)
        assert result['status']=='process_error' and result['exit_code']==1
        folders=list((tmp_path/'evidence').glob('caller-*'))
        assert len(folders)==1
        receipt=json.loads((folders[0]/'process/result.json').read_text())
        identity=json.loads((folders[0]/'identity.json').read_text())
    else:
        source=repo/'tools/workflow2_source_gate.py'
        caller_spec=importlib.util.spec_from_file_location('source_check_caller',source)
        module=importlib.util.module_from_spec(caller_spec);caller_spec.loader.exec_module(module)
        monkeypatch.setattr(module,'ROOT',tmp_path/'source-campaign')
        assert module.run(SimpleNamespace(name='caller',command=[sys.executable,'-c',case],
            timeout=1,pythonpath=[]))==1
        receipt=json.loads((module.ROOT/'caller/process/result.json').read_text())
        identity=json.loads((module.ROOT/'caller/identity.json').read_text())
    assert receipt['status']=='timeout' and receipt['cleanup']['verified']
    assert 'workflow2_process_supervisor.py' in identity['supervisor_path']
    assert not Path('/proc',ready.read_text()).exists()


@pytest.mark.skipif(not sys.platform.startswith('linux'),reason='Linux missing-capability refusal')
def test_missing_pidfd_is_refused_before_any_process_launch(monkeypatch,tmp_path):
    monkeypatch.delattr(supervisor.os,'pidfd_open',raising=False)
    def forbidden(*args,**kwargs):pytest.fail('unsupported controller launched a process')
    monkeypatch.setattr(supervisor.subprocess,'Popen',forbidden)
    with pytest.raises(RuntimeError,match='pidfd'):
        supervisor.run_linux([sys.executable,'-c','pass'],tmp_path/'unstarted',1,dict(os.environ),tmp_path)
    assert not (tmp_path/'unstarted').exists()
