"""Dedicated Linux check supervision across descendant-created sessions.

The calling process must have no other children and runs one check at a time.
It becomes a subreaper, inventories descendant identities, and uses pidfds to
terminate only its owned descendants on timeout, SIGINT/SIGTERM, or completion.
Uncatchable controller death/host failure is outside this userspace contract.
"""
from __future__ import annotations
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


class _Interrupted(BaseException):
    pass


def _table():
    records={}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            values=(entry/'stat').read_text().rsplit(')',1)[1].split()
            records[int(entry.name)]=dict(state=values[0],ppid=int(values[1]),
                group=int(values[2]),session=int(values[3]),start=int(values[19]))
        except (OSError,IndexError,ValueError):continue
    return records


def _descendants(records,parent):
    ids={parent};result={}
    while True:
        added={pid:row for pid,row in records.items() if pid not in ids and row['ppid'] in ids}
        if not added:return result
        result.update(added);ids.update(added)


def _kill_identity(pid,row):
    try:
        descriptor=os.pidfd_open(pid,0)
    except ProcessLookupError:return
    try:
        current=_table().get(pid)
        if current is not None and current['start']==row['start']:
            signal.pidfd_send_signal(descriptor,signal.SIGKILL,None,0)
    except ProcessLookupError:
        pass
    finally:
        os.close(descriptor)


def run_linux(command,folder,timeout,environment,cwd,cancel=None):
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Linux descendant supervisor required')
    if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
        raise RuntimeError('pidfd identity-safe cleanup unavailable')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    controller=os.getpid();process=None;owned={};receipt={};interrupted=False
    started=time.perf_counter();status='controller_error';error=None
    old_handlers={}
    def handle_signal(signum,frame):
        raise _Interrupted(f'controller signal {signum}')
    def remember():
        records=_table();owned.update(_descendants(records,controller));return records
    try:
        if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:
            raise OSError('cannot enable child subreaper')
        if _descendants(_table(),controller):
            raise RuntimeError('dedicated supervisor must have no pre-existing child processes')
        for signum in (signal.SIGINT,signal.SIGTERM):
            old_handlers[signum]=signal.signal(signum,handle_signal)
        with (folder/'stdout.log').open('wb') as stdout,(folder/'stderr.log').open('wb') as stderr:
            process=subprocess.Popen(command,cwd=cwd,env=environment,stdout=stdout,
                stderr=stderr,stdin=subprocess.DEVNULL,start_new_session=True)
            receipt['pid']=process.pid
            while True:
                remember()
                code=process.poll()
                if code is not None:
                    status='completed' if code==0 else 'process_error';break
                if cancel is not None and cancel():status='cancelled';break
                if time.perf_counter()-started>=timeout:status='timeout';break
                time.sleep(.025)
    except (KeyboardInterrupt,_Interrupted) as exc:
        status='interrupted';interrupted=True;error=str(exc)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    finally:
        # A second graceful interrupt must not bypass the owned cleanup block.
        for signum in old_handlers:signal.signal(signum,signal.SIG_IGN)
        before=[];remaining=[];cleanup_error=None
        if process is not None:
            try:
                records=remember()
                before=[dict(pid=pid,**row) for pid,row in records.items()
                    if pid in owned and owned[pid]['start']==row['start']]
                deadline=time.perf_counter()+5
                while True:
                    records=remember()
                    current={pid:row for pid,row in records.items()
                        if pid in owned and row['start']==owned[pid]['start']}
                    for pid,row in current.items():
                        if row['state']!='Z':_kill_identity(pid,row)
                    if process.poll() is not None:process.wait(timeout=.1)
                    for pid in tuple(current):
                        if pid==process.pid:continue
                        try:os.waitpid(pid,os.WNOHANG)
                        except ChildProcessError:pass
                    records=remember()
                    remaining=[dict(pid=pid,**row) for pid,row in records.items()
                        if pid in owned and row['start']==owned[pid]['start']]
                    if not remaining:break
                    if time.perf_counter()>=deadline:
                        raise OSError('owned descendants remain after cleanup')
                    time.sleep(.01)
                receipt['exit_code']=process.wait(timeout=.1)
            except Exception as exc:
                cleanup_error=f'{type(exc).__name__}: {exc}';status='cleanup_failed'
        for signum,handler in old_handlers.items():signal.signal(signum,handler)
        receipt.update(status=status,process_wall_seconds=time.perf_counter()-started,
            timeout_seconds=timeout,controller_pid=controller,command=list(map(str,command)),
            interrupted=interrupted,cleanup=dict(verified=cleanup_error is None,
                members_before_cleanup=before,remaining_members=remaining,
                owned_identities=[dict(pid=pid,**row) for pid,row in owned.items()],error=cleanup_error))
        if error is not None:receipt['error']=error
        (folder/'result.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))
    return receipt
