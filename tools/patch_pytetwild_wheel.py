"""Apply the explicitly approved optional-accessor fix to an official 0.4.2 wheel.

Does not download, install, publish, or change native binaries. The output is a
separately identified local compatibility artifact with a regenerated RECORD.
"""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

VERSION = '0.4.2+anyworkflow2.1'
OLD = b'from . import _accessor as _accessor'
NEW = (b'from importlib.util import find_spec as _find_spec\n'
       b'if _find_spec("pyvista") is not None:\n'
       b'    from . import _accessor as _accessor\n'
       b'del _find_spec')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('wheel', type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if not a.wheel.name.startswith('pytetwild-0.4.2-'):
        raise ValueError('only an official 0.4.2 wheel is accepted')
    with zipfile.ZipFile(a.wheel) as z:
        contents = {name:z.read(name) for name in z.namelist() if not name.endswith('/')}
    init = contents['pytetwild/__init__.py']
    if init.count(OLD) != 1:
        raise ValueError('unexpected optional accessor import; refuse an unknown patch target')
    original_hashes = {name:sha(data) for name,data in contents.items()}
    contents['pytetwild/__init__.py'] = init.replace(OLD, NEW)
    old_dist = 'pytetwild-0.4.2.dist-info/'
    new_dist = 'pytetwild-' + VERSION + '.dist-info/'
    made = {}
    for name,data in contents.items():
        name = new_dist+name[len(old_dist):] if name.startswith(old_dist) else name
        if name == new_dist+'RECORD':
            continue
        if name == new_dist+'METADATA':
            if b'Version: 0.4.2\n' not in data:
                raise ValueError('unexpected metadata version')
            data = data.replace(b'Version: 0.4.2\n', b'Version: '+VERSION.encode()+b'\n',1)
        made[name] = data
    record = io.StringIO(newline='')
    writer = csv.writer(record, lineterminator='\n')
    for name,data in sorted(made.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        writer.writerow([name,'sha256='+digest,str(len(data))])
    writer.writerow([new_dist+'RECORD','',''])
    made[new_dist+'RECORD'] = record.getvalue().encode()
    a.output.mkdir(parents=True,exist_ok=True)
    target = a.output/a.wheel.name.replace('pytetwild-0.4.2-','pytetwild-'+VERSION+'-',1)
    if target.exists():
        raise FileExistsError(target)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(made.items()):
            info = zipfile.ZipInfo(name, date_time=(2026,10,27,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info,data)
    native = {name:sha(data) for name,data in made.items()
              if name.endswith(('.pyd','.dll','.so','.dylib')) or '.so.' in name}
    if not native or any(original_hashes.get(name)!=digest for name,digest in native.items()):
        raise ValueError('native binary identity not preserved')
    manifest = dict(original_wheel=str(a.wheel),original_sha256=sha(a.wheel.read_bytes()),
                    patched_wheel=str(target),patched_sha256=sha(target.read_bytes()),version=VERSION,
                    patch={'old':OLD.decode(),'new':NEW.decode()},native_hashes=native,
                    changed_runtime_files=['pytetwild/__init__.py'],native_bytes_unchanged=True)
    target.with_suffix('.patch.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
