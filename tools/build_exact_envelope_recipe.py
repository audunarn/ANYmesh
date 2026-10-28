"""Reproducible candidate preparation, dependency inventory and import-only smoke."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from build_exact_envelope import VERSION, FTETWILD_COMMIT, FAST_ENVELOPE_COMMIT, save, sha
from patch_pytetwild_wheel import OLD, NEW

WRAPPER_COMMIT = '85c28acd6e1546b7c19b96fd6a5ee5bc2184b176'


def flag_proof(args):
    cache = (args.build / 'CMakeCache.txt').read_text()
    flag = next(line for line in cache.splitlines() if line.startswith('FLOAT_TETWILD_WITH_EXACT_ENVELOPE:'))
    target = 'CMakeFiles/PyfTetWildWrapper.dir/src/FTetWildWrapper.cpp.o'
    command = subprocess.check_output(['ninja','-C',str(args.build),'-t','commands',target],text=True,timeout=20).strip()
    if flag != 'FLOAT_TETWILD_WITH_EXACT_ENVELOPE:BOOL=ON' or '-DNEW_ENVELOPE' not in command:
        raise ValueError('effective exact-envelope flag/macro missing')
    save(args.root / 'native-flag-proof.json', dict(cmake_cache_flag=flag, wrapper_compile_command=command,
          build_ninja_sha256=sha(args.build / 'build.ninja'),
          ftetwild_source_unchanged=not git(args.source / 'src/fTetWild','status','--short')))
    for name in ('CMakeCache.txt','compile_commands.json','build.ninja'):
        target = args.root / 'native-build-configuration' / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes((args.build / name).read_bytes())


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True, timeout=20).strip()


def prepare(args):
    source = args.source.resolve()
    ftet = source / 'src/fTetWild'
    if git(source, 'rev-parse', 'HEAD') != WRAPPER_COMMIT or git(ftet, 'rev-parse', 'HEAD') != FTETWILD_COMMIT:
        raise ValueError('unexpected upstream source identity')
    init = source / 'src/pytetwild/__init__.py'
    before = init.read_bytes()
    if before.count(OLD) != 1:
        raise ValueError('unknown accessor patch target')
    init.write_bytes(before.replace(OLD, NEW))
    result = dict(candidate_version=VERSION, wrapper_commit=WRAPPER_COMMIT,
                  ftetwild_commit=FTETWILD_COMMIT, fast_envelope_commit=FAST_ENVELOPE_COMMIT,
                  cmake_option={'FLOAT_TETWILD_WITH_EXACT_ENVELOPE':'ON'},
                  source_patch=dict(path='src/pytetwild/__init__.py', before_sha256=hashlib.sha256(before).hexdigest(),
                                    after_sha256=sha(init), old=OLD.decode(), new=NEW.decode()),
                  native_algorithm_source_edits=False,
                  native_bytes_unchanged=False,
                  source_status=git(source, 'status', '--short'))
    save(args.root / 'source-preparation.json', result)
    print(json.dumps(result))


def glue(args):
    source = args.source.resolve()
    cpp = source / 'src/FTetWildWrapper.cpp'
    before = cpp.read_bytes()
    old = b'''    if (!params.init(tree.get_sf_diag())) {
        throw std::runtime_error("FTetWildWrapper.cpp: Parameters initialization failed");
    }
'''
    new = old + b'''
#ifdef NEW_ENVELOPE
    // Match the standalone fTetWild initialization after absolute epsilon is set.
    tree.init_sf_tree(input_points, input_faces, params.eps);
#endif
'''
    module = b'NB_MODULE(PyfTetWildWrapper, m) {\n'
    marker = module + b'''#ifdef NEW_ENVELOPE
    m.attr("_workflow2_exact_envelope") = true;
#else
    m.attr("_workflow2_exact_envelope") = false;
#endif
'''
    if before.count(old) != 1 or before.count(module) != 1:
        raise ValueError('unexpected raw wrapper initialization/module target')
    cpp.write_bytes(before.replace(old,new).replace(module,marker))
    if git(source / 'src/fTetWild','status','--short'):
        raise ValueError('upstream fTetWild algorithm source is dirty')
    result = dict(candidate_version=VERSION, path='src/FTetWildWrapper.cpp',
                  before_sha256=hashlib.sha256(before).hexdigest(), after_sha256=sha(cpp),
                  raw_initialization=dict(old=old.decode(), new=new.decode()),
                  marker=dict(old=module.decode(), new=marker.decode()),
                  native_algorithm_source_edits=False,
                  csg_route='not initialized by this scoped patch; unqualified and out of scope')
    save(args.root / 'wrapper-glue-patch.json', result)
    (args.root / 'candidate-source.patch').write_text(git(source,'diff','--','src/pytetwild/__init__.py','src/FTetWildWrapper.cpp'),encoding='utf-8')
    preparation = json.loads((args.root / 'source-preparation.json').read_text())
    preparation.update(candidate_version=VERSION, additional_wrapper_glue_patch='wrapper-glue-patch.json')
    save(args.root / 'source-preparation.json', preparation)
    print(json.dumps(result))


def inventory(args):
    roots = [args.source.resolve(), args.source.resolve() / 'src/fTetWild']
    if args.build is not None:
        roots.extend(sorted(p for p in (args.build / '_deps').glob('*-src') if (p / '.git').exists()))
        for parent in tuple(roots):
            roots.extend(sorted(p for p in (parent / '3rdparty').glob('*') if (p / '.git').exists()))
    roots = list(dict.fromkeys(roots))
    records = []
    license_root = args.root / 'license-evidence'
    license_root.mkdir(exist_ok=True)
    for i, repo in enumerate(roots):
        licenses = []
        # Include root notices plus common nested license folders, while avoiding
        # broad source dumps or generated build outputs.
        candidates = set()
        for pattern in ('LICENSE*','LICENCE*','COPYING*','NOTICE*','LICENSES/**/*','licenses/**/*'):
            candidates.update(p for p in repo.glob(pattern) if p.is_file())
        for path in sorted(candidates):
            relative = path.relative_to(repo)
            target = license_root / f'{i:02d}-{repo.name}' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            licenses.append(dict(path=str(relative), sha256=sha(path), retained=str(target)))
        records.append(dict(path=str(repo), commit=git(repo,'rev-parse','HEAD'),
                            remote=git(repo,'remote','get-url','origin'),
                            dirty=git(repo,'status','--short'), licenses=licenses))
    if args.build is not None:
        exact = [r for r in records if r['commit'] == FAST_ENVELOPE_COMMIT]
        if len(exact) != 1:
            raise ValueError('expected exactly the pinned FastEnvelope dependency')
    save(args.root / 'dependency-inventory.json', records)
    embedded = []
    for repo in roots:
        for directory in ('3rdparty/i_pred','indirectPredicates'):
            for path in sorted((repo / directory).glob('*')):
                if path.is_file() and path.suffix in ('.h','.hpp','.c','.cpp'):
                    data = path.read_bytes()
                    text = data[:8192].decode('utf-8', errors='replace')
                    markers = [term for term in ('GNU Lesser General Public License','GNU General Public License','SPDX-License-Identifier') if term in text]
                    if markers:
                        target = license_root / 'embedded-notices' / repo.name / path.relative_to(repo)
                        target.parent.mkdir(parents=True,exist_ok=True)
                        target.write_bytes(data)
                        embedded.append(dict(path=str(path),sha256=sha(path),markers=markers,retained=str(target)))
    save(args.root / 'embedded-license-notices.json', embedded)
    if sys.platform.startswith('linux'):
        system = []
        for package in ('libgmp10','libgmp-dev','libmpfr6','libc6','libstdc++6','gcc-13-base'):
            path = Path('/usr/share/doc') / package / 'copyright'
            if path.exists():
                target = license_root / 'system' / (package + '.copyright')
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(path.read_bytes())
                system.append(dict(package=package,sha256=sha(path),retained=str(target)))
        save(args.root / 'system-license-inventory.json', system)
    # Python build-tool distributions may include statically incorporated native
    # headers (nanobind/robin_map); retain their shipped notices as well.
    wheel_licenses = []
    for wheel in sorted((args.root / 'build-tool-wheels').glob('*.whl')):
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if ('license' in name.lower() or 'copying' in name.lower() or name.endswith('/METADATA')) and not name.endswith('/'):
                    target = license_root / 'build-tools' / wheel.stem / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(name))
                    wheel_licenses.append(dict(wheel=wheel.name, path=name, sha256=sha(target)))
    save(args.root / 'build-tool-license-inventory.json', wheel_licenses)
    # Source file hashes make locally generated/versioned files explicit too.
    identities = {}
    for repo in roots:
        files = git(repo,'ls-files','-z').split('\0')
        for name in files:
            path = repo / name
            if name and path.is_file():
                identities[str(path)] = sha(path)
    save(args.root / 'source-files-sha256.json', identities)
    if args.build is not None:
        flag_proof(args)
    print(json.dumps(dict(repositories=len(records), source_files=len(identities),
                          license_files=sum(len(r['licenses']) for r in records))))


def complete(args):
    """Complete retained inventory without repeating already captured hashes."""
    records = json.loads((args.root / 'dependency-inventory.json').read_text())
    known = {str(Path(record['path']).resolve()) for record in records}
    licenses = []
    git_closure = []
    scanned = set()
    for original in tuple(known):
        for current, directories, files in os.walk(original):
            if current in scanned:
                directories[:] = []
                continue
            scanned.add(current)
            path = Path(current)
            if '.git' in directories or '.git' in files:
                git_closure.append(dict(path=current,commit=git(path,'rev-parse','HEAD'),
                                        remote=git(path,'remote','get-url','origin')))
            directories[:] = [name for name in directories if name not in ('.git','.cache')]
            for name in files:
                lower = name.lower()
                if not any(term in lower for term in ('license','licence','copying','copyright','notice','lgpl.txt','gpl.txt')):
                    continue
                source = path / name
                relative = source.relative_to(Path(original))
                target = args.root / 'license-evidence' / 'recursive' / Path(original).name / relative
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(source.read_bytes())
                licenses.append(dict(source=str(source),sha256=sha(source),retained=str(target)))
    save(args.root / 'recursive-git-closure.json',git_closure)
    save(args.root / 'recursive-license-inventory.json',licenses)
    # CLI11 is an upstream-pinned tarball dependency rather than a git checkout.
    third_party = args.build / '_deps/exact_envelope-src/3rdparty'
    cli = third_party / 'cli11'
    archive = next((third_party / '.cache/cli11').rglob('v1.6.1.tar.gz'))
    cli_hashes = {str(path.relative_to(cli)):sha(path) for path in cli.rglob('*') if path.is_file()}
    save(args.root / 'cli11-source-inventory.json',dict(version='1.6.1',
         url='https://github.com/CLIUtils/CLI11/archive/v1.6.1.tar.gz',
         upstream_md5='48ef97262adb0b47a2f0a7edbda6e2aa',archive=str(archive),archive_sha256=sha(archive),
         source_files_sha256=cli_hashes,license='BSD-3-Clause',license_sha256=sha(cli/'LICENSE')))
    target = args.root / 'license-evidence/cli11-1.6.1/LICENSE'
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((cli/'LICENSE').read_bytes())
    flag_proof(args)
    link = subprocess.check_output(['ninja','-C',str(args.build),'-t','commands','PyfTetWildWrapper.abi3.so'],text=True,timeout=20)
    (args.root / 'native-build-configuration/relink-commands.txt').write_text(link)
    archives = {str(path):sha(path) for path in args.build.rglob('*.a')}
    native = args.build / 'PyfTetWildWrapper.abi3.so'
    dependencies = subprocess.check_output(['ldd',str(native)],text=True,timeout=20)
    (args.root / 'native-build-configuration/ldd.txt').write_text(dependencies)
    save(args.root / 'relink-inputs.json',dict(source_root=str(args.source),build_root=str(args.build),
         build_venv=str(args.root/'linux-build-venv'),static_libraries_sha256=archives,
         native_unstripped_sha256=sha(native),source_file_identity='source-files-sha256.json',
         additional_tarball_identity='cli11-source-inventory.json',
         compiled_wrapper_patch='candidate-source.patch',build_tools_identity='build-environment.json',
         retained_configuration='native-build-configuration',release_license_gate='pending; includes LGPL-3.0-or-later i_pred'))
    print(json.dumps(dict(git_repositories=len(git_closure),recursive_license_files=len(licenses),
                          cli11_files=len(cli_hashes),static_libraries=len(archives))))


def smoke(args):
    wheels = sorted((args.root / 'wheels').glob('*.whl'))
    if len(wheels) != 1:
        raise ValueError('one candidate wheel required')
    wheel = wheels[0]
    directory = args.root / 'import-only-smoke'
    directory.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(directory)
    sys.path.insert(0, str(directory))
    provider = importlib.import_module('pytetwild')
    if provider.__version__ != VERSION:
        raise ValueError('candidate import version mismatch')
    if 'pyvista' in sys.modules:
        raise ValueError('unexpected PyVista import')
    native_module = importlib.import_module('pytetwild.PyfTetWildWrapper')
    if native_module._workflow2_exact_envelope is not True:
        raise ValueError('compiled exact-envelope macro capability marker is not true')
    native = {str(p.relative_to(directory)):sha(p) for p in directory.rglob('*')
              if p.is_file() and p.suffix in ('.so','.pyd','.dll')}
    if not native:
        raise ValueError('candidate contains no native module')
    result = dict(status='import_only_passed', meshing_invoked=False, version=provider.__version__,
                  imported_from=str(provider.__file__), wheel=str(wheel), wheel_sha256=sha(wheel),
                  native_sha256=native, compiled_exact_envelope_marker=True,
                  python=sys.version, platform=sys.platform)
    save(args.root / 'import-only-smoke.json', result)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare','glue','inventory','complete','smoke'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--build', type=Path)
    args = parser.parse_args()
    args.source = args.source or args.root / 'source'
    args.build = args.build or args.root / 'linux-build'
    globals()[args.phase](args)


if __name__ == '__main__':
    main()
