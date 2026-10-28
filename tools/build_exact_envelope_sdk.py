"""Provision upstream-CI MPIR/zlib into an owned Windows SDK prefix only."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
from build_exact_envelope import save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args = p.parse_args();root=args.root.resolve()
    if not root.name.startswith('exact-envelope-'):
        raise ValueError('owned exact-envelope root required')
    prefix=root/'windows-sdk'
    if prefix.exists():
        raise FileExistsError(prefix)
    packages=root/'windows-conda-packages'
    config=root/'windows-sdk.condarc'
    config.write_text('register_envs: false\nshortcuts: false\ncreate_default_packages: []\n')
    env=dict(os.environ)
    env.update(CONDARC=str(config),CONDA_REGISTER_ENVS='false',CONDA_SHORTCUTS='false',
               CONDA_PKGS_DIRS=str(packages),CONDA_ENVS_PATH=str(root/'windows-conda-envs'),
               CONDA_NO_PLUGINS='true',CONDA_SOLVER='classic',CONDA_AUTO_UPDATE_CONDA='false')
    conda=Path('C:/Users/AudunArnesenNyhus/miniconda3/Scripts/conda.exe')
    command=[str(conda),'create','--prefix',str(prefix),'--override-channels','-c',
             'https://conda.anaconda.org/conda-forge','--solver','classic',
             '--no-default-packages','--yes','--json','mpir=3.0.0','zlib=1.3.1']
    save(root/'windows-sdk-command.json',dict(command=command,explicit_environment={key:env[key] for key in (
        'CONDARC','CONDA_REGISTER_ENVS','CONDA_SHORTCUTS','CONDA_PKGS_DIRS','CONDA_ENVS_PATH','CONDA_NO_PLUGINS','CONDA_SOLVER','CONDA_AUTO_UPDATE_CONDA')}))
    with (root/'windows-sdk-create.json').open('wb') as out,(root/'windows-sdk-create.stderr.txt').open('wb') as err:
        result=subprocess.run(command,env=env,stdout=out,stderr=err)
    if result.returncode:
        raise RuntimeError(f'isolated conda SDK provisioning failed ({result.returncode}); retained JSON/stderr')
    explicit=subprocess.check_output([str(conda),'list','--prefix',str(prefix),'--explicit','--md5'],env=env,text=True)
    (root/'windows-sdk-explicit.txt').write_text(explicit)
    inventory=[]
    for path in sorted((prefix/'conda-meta').glob('*.json')):
        record=json.loads(path.read_text())
        inventory.append({key:record.get(key) for key in ('name','version','build','channel','url','sha256','md5','license','license_family')})
    save(root/'windows-sdk-packages.json',inventory)
    archives={path.name:sha(path) for path in packages.glob('*') if path.is_file() and path.suffix in ('.conda','.bz2')}
    save(root/'windows-sdk-package-archives.json',archives)
    licenses=[]
    for package in packages.iterdir():
        if not package.is_dir():continue
        for path in (package/'info').rglob('*'):
            if path.is_file() and any(key in path.name.lower() for key in ('license','copying','copyright')):
                target=root/'license-evidence/windows-sdk'/package.name/path.relative_to(package/'info')
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
                licenses.append(dict(path=str(path),sha256=sha(path),retained=str(target)))
    save(root/'windows-sdk-license-inventory.json',licenses)
    files={str(path.relative_to(prefix)):sha(path) for path in (prefix/'Library').rglob('*') if path.is_file() and path.suffix in ('.h','.lib','.dll')}
    save(root/'windows-sdk-input-files.json',files)
    print(json.dumps(dict(prefix=str(prefix),packages=len(inventory),archives=len(archives),licenses=len(licenses),files=len(files))))


if __name__=='__main__':
    main()
