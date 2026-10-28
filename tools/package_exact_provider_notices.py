"""Additive exact1 notice/source/relink handoff; never compiles or meshes.

Run mutable work through ``run ACTION`` for fresh bounded process accounting.
The historical evidence root is read-only. All outputs have exclusive names.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import time
import traceback
import urllib.request
import zipfile

REPO = Path(__file__).resolve().parents[1]
WORK = REPO.parent
OLD = WORK / 'evidence/exact-envelope-20261028-a1'
OUT = WORK / 'evidence/provider-notices-20261028-a1'
REVISION = ''
CHILD_PYTHON = None
MAX_SOURCE = 1024**3
MPIR_SHA = '52f63459cf3f9478859de29e00357f004050ead70b45913f2c2269d9708675bb'
SKIP_DIRS = {'.git', '.github', '__pycache__', 'build', '_build', 'tests', 'test',
             'examples', 'example', 'tutorial', 'docs', 'doc', 'data', 'datasets',
             'screenshots', 'assets', 'node_modules'}
SKIP_SUFFIXES = {'.o', '.obj', '.a', '.lib', '.dll', '.so', '.pyd', '.pyc',
                 '.exe', '.pdb', '.exp', '.dylib', '.png', '.jpg', '.jpeg',
                 '.gif', '.mp4', '.pdf', '.zip', '.whl'}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def digest_stream(handle):
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(1024*1024), b''):
        digest.update(chunk)
        size += len(chunk)
    return digest.digest(), size


def local_path(value):
    value = str(value).replace('\\', '/')
    if value.startswith('/mnt/c/'):
        value = 'C:/' + value[7:]
    elif value.startswith('/mnt/d/'):
        value = 'D:/' + value[7:]
    if os.name != 'nt' and re.match(r'^[CD]:/', value, re.I):
        value = '/mnt/' + value[0].lower() + value[2:]
    return Path(value)


def safe_name(name):
    name = name.replace('\\', '/')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or ':' in name:
        raise ValueError(f'unsafe archive member: {name}')
    return str(path)


def permitted_metadata(name):
    parts = PurePosixPath(name).parts
    if not parts or not parts[0].endswith('.dist-info'):
        return False
    return len(parts) > 1 and (parts[1] in {'METADATA', 'RECORD', 'NOTICE', 'licenses'}
                              or parts[1].startswith('NOTICE'))


def repack(original, destination, additions, metadata_note):
    """Keep original runtime bytes, append metadata, regenerate/verify RECORD."""
    original, destination = Path(original), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with zipfile.ZipFile(original) as source:
        names = source.namelist()
        if len(names) != len(set(names)):
            raise ValueError('duplicate original wheel members')
        for name in names:
            safe_name(name)
        metadata = [name for name in names if name.endswith('.dist-info/METADATA')]
        if len(metadata) != 1:
            raise ValueError('wheel must have one metadata directory')
        prefix = metadata[0].rsplit('/', 1)[0]
        record = prefix + '/RECORD'
        extra = {prefix + '/' + safe_name(name): value for name, value in additions.items()}
        if any(not permitted_metadata(name) for name in extra):
            raise ValueError('additions may only change permitted dist-info members')
        if any(name in names for name in extra):
            raise ValueError('additions would overwrite historical metadata')
        metadata_bytes = source.read(metadata[0])
        updated = metadata_bytes + b'\n\n' + metadata_note.encode('utf-8') + b'\n'
        rows = []
        with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as target:
            for entry in source.infolist():
                if entry.filename == record:
                    continue
                if entry.filename == metadata[0]:
                    content = updated
                    target.writestr(entry, content)
                    raw, size = hashlib.sha256(content).digest(), len(content)
                else:
                    with source.open(entry) as reader, target.open(entry, 'w') as writer:
                        digest = hashlib.sha256(); size = 0
                        for chunk in iter(lambda: reader.read(1024*1024), b''):
                            writer.write(chunk); digest.update(chunk); size += len(chunk)
                        raw = digest.digest()
                rows.append((entry.filename, 'sha256=' + base64.urlsafe_b64encode(raw).decode().rstrip('='), str(size)))
            for name, path_or_bytes in sorted(extra.items()):
                content = path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_bytes()
                target.writestr(name, content)
                rows.append((name, 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip('='), str(len(content))))
            buffer = io.StringIO(newline='')
            writer = csv.writer(buffer, lineterminator='\n')
            writer.writerows(rows + [(record, '', '')])
            target.writestr(record, buffer.getvalue().encode('utf-8'))
    return verify_wheel(original, destination)


def verify_wheel(original, refreshed):
    preserved, changed = {}, []
    with zipfile.ZipFile(original) as before, zipfile.ZipFile(refreshed) as after:
        if len(after.namelist()) != len(set(after.namelist())):
            raise ValueError('duplicate refreshed members')
        if not set(before.namelist()).issubset(after.namelist()):
            raise ValueError('historical member removed')
        for name in after.namelist():
            safe_name(name)
            with after.open(name) as stream:
                raw, size = digest_stream(stream)
            if name in before.namelist():
                with before.open(name) as stream:
                    old_raw, old_size = digest_stream(stream)
                if (raw, size) != (old_raw, old_size):
                    if not permitted_metadata(name):
                        raise ValueError(f'runtime member changed: {name}')
                    changed.append(name)
                elif not permitted_metadata(name):
                    preserved[name] = dict(sha256=raw.hex(), bytes=size)
            elif not permitted_metadata(name):
                raise ValueError(f'new runtime member: {name}')
        records = [name for name in after.namelist() if name.endswith('.dist-info/RECORD')]
        if len(records) != 1:
            raise ValueError('missing/duplicate RECORD')
        record = records[0]
        rows = list(csv.reader(io.StringIO(after.read(record).decode('utf-8'))))
        if len(rows) != len({row[0] for row in rows}) or {row[0] for row in rows} != set(after.namelist()):
            raise ValueError('RECORD membership differs')
        for name, claimed, length in rows:
            if name == record:
                if claimed or length:
                    raise ValueError('RECORD must not hash itself')
                continue
            with after.open(name) as stream:
                raw, size = digest_stream(stream)
            expected = 'sha256=' + base64.urlsafe_b64encode(raw).decode().rstrip('=')
            if claimed != expected or length != str(size):
                raise ValueError(f'RECORD mismatch: {name}')
    return dict(original_sha256=sha(original), refreshed_sha256=sha(refreshed),
                preserved_runtime_members=preserved, changed_metadata_members=changed,
                runtime_bytes_identical=True, record_verified=True)


def selected_sources():
    roots = [('wrapper', OLD/'source')]
    roots += [('dependencies/' + p.name.removesuffix('-src'), p)
              for p in sorted((OLD/'linux-build/_deps').glob('*-src')) if p.is_dir()]
    nanobind = OLD/'linux-build-venv/lib/python3.12/site-packages/nanobind'
    if not nanobind.is_dir():
        raise ValueError('retained nanobind source is missing')
    roots.append(('dependencies/nanobind', nanobind))
    records = []
    for alias, root in roots:
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d.lower() not in SKIP_DIRS)
            for name in sorted(files):
                path = Path(current)/name
                if path.is_symlink() or path.name == '.git' or path.suffix.lower() in SKIP_SUFFIXES:
                    continue
                relative = Path(alias)/path.relative_to(root)
                records.append((path, 'source/' + relative.as_posix(), path.stat().st_size))
    return roots, records


def collect_notices(folder):
    folder.mkdir()
    records = []
    inventories = ['recursive-license-inventory.json', 'embedded-license-notices.json',
                   'system-license-inventory.json', 'windows-sdk-license-inventory.json']
    candidates = []
    for name in inventories:
        for item in json.loads((OLD/name).read_text()):
            value = item.get('retained') or item.get('path')
            if value:
                candidates.append((local_path(value), name))
    for item in json.loads((OLD/'dependency-inventory.json').read_text()):
        for license_item in item.get('licenses', []):
            candidates.append((local_path(license_item['retained']), 'dependency-inventory.json'))
    # LGPL3 incorporates GPL3; this complete retained text is explicitly selected.
    candidates += [(OLD/'linux-build/_deps/eigen-src/COPYING.GPL', 'GNU GPL3'),
                   (OLD/'linux-build/_deps/exact_envelope-src/3rdparty/i_pred/lgpl.txt', 'GNU LGPL3')]
    seen = set()
    for path, origin in candidates:
        if not path.is_file():
            raise ValueError(f'missing retained license: {path}')
        digest = sha(path)
        if digest in seen:
            continue
        seen.add(digest)
        name = digest[:12] + '-' + path.name
        shutil.copyfile(path, folder/name)
        records.append(dict(file='licenses/'+name, sha256=digest, provenance=origin,
                            historical_source=str(path.relative_to(OLD))))
    # nanobind is linked native code, although distributed to the builder as a
    # Python build-tool wheel. Its license lives outside the imported package.
    wheels=list((OLD/'build-tool-wheels').glob('nanobind-*.whl'))
    if len(wheels)!=1:raise ValueError('expected one retained nanobind build wheel')
    with zipfile.ZipFile(wheels[0]) as archive:
        for entry in archive.infolist():
            if '/licenses/' not in entry.filename or entry.is_dir():continue
            content=archive.read(entry);digest=hashlib.sha256(content).hexdigest()
            if digest in seen:continue
            seen.add(digest);name=digest[:12]+'-'+PurePosixPath(entry.filename).name
            (folder/name).write_bytes(content)
            records.append(dict(file='licenses/'+name,sha256=digest,
                provenance='retained linked nanobind build wheel',
                historical_source=str(wheels[0].relative_to(OLD))+'!'+entry.filename))
    return records


def fetch_mpir(cache):
    cache.mkdir()
    for path in sorted(OUT.glob('cache*/*.tar.bz2')):
        if path.parent!=cache and sha(path)==MPIR_SHA:
            attempts=[dict(verified_cached_source=str(path.relative_to(OUT)),sha256=MPIR_SHA,matches_recipe=True)]
            write_json(cache/'acquisition.json',dict(status='FIXED_RECIPE_SOURCE_VERIFIED',attempts=attempts))
            return path,attempts
    urls = ['https://sources.buildroot.net/mpir/mpir-3.0.0.tar.bz2',
            'https://www.mpir.org/mpir-3.0.0.tar.bz2']
    attempts = []
    for index, url in enumerate(urls):
        path = cache/f'mpir-attempt{index}.tar.bz2'
        try:
            with urllib.request.urlopen(url, timeout=20) as response, path.open('xb') as target:
                count = 0
                while chunk := response.read(1024*1024):
                    count += len(chunk)
                    if count > 32*1024**2:
                        raise ValueError('source download exceeds 32MiB bound')
                    target.write(chunk)
            actual = sha(path)
            attempts.append(dict(url=url, sha256=actual, matches_recipe=actual==MPIR_SHA))
            if actual == MPIR_SHA:
                write_json(cache/'acquisition.json', dict(status='FIXED_RECIPE_SOURCE_VERIFIED', attempts=attempts))
                return path, attempts
        except Exception as error:
            attempts.append(dict(url=url, error=f'{type(error).__name__}: {error}'))
    write_json(cache/'acquisition.json', dict(status='UNAVAILABLE', attempts=attempts))
    return None, attempts


LIBRARY_NAMES = {'libnanobind-static.a', 'libnanobind-static-abi3.a', 'libFloatTetwild.a',
                 'libtbb.a', 'libgeogram.a', 'libFastEnvelope.a', 'libIndirectPredicates.a',
                 'libfmt.a', 'libpredicates.a', 'libspdlog.a', 'nanobind-static.lib',
                 'nanobind-static-abi3.lib', 'FloatTetwild.lib', 'tbb12.lib', 'geogram.lib',
                 'FastEnvelope.lib', 'IndirectPredicates.lib', 'fmt.lib', 'predicates.lib', 'spdlog.lib'}


def relink_member(name):
    name = safe_name(name)
    return (PurePosixPath(name).name in LIBRARY_NAMES or
            bool(re.search(r'CMakeFiles/PyfTetWildWrapper\.dir/src/FTetWildWrapper\.cpp\.(o|obj)$', name)))


def write_posix_relink_recipe(path, command):
    """Keep recipient shell bytes portable even when exported on Windows."""
    text = '#!/bin/sh\nset -eu\n# Run from this directory with compatible GCC13/system GMP/OpenMP.\n'+command
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def export_relink(folder):
    folder.mkdir()
    results = {}
    configs = {
        'linux312abi3': (OLD/'linux311/previous-abi-relink-inputs.tar.gz', None, OLD/'native-build-configuration'),
        'windows311': (OLD/'windows312/previous-abi-relink-inputs.tar.gz', None, OLD/'windows-native-build-configuration'),
        'linux311': (None, OLD/'linux-build', OLD/'linux311/build-configuration'),
        'windows312abi3': (None, OLD/'windows-build', OLD/'windows312/build-configuration'),
    }
    for abi, (archive, build, configuration) in configs.items():
        destination = folder/abi; destination.mkdir()
        rows = []
        ninja=(configuration/'build.ninja').read_text()
        link_line=re.search(r'^  LINK_LIBRARIES = (.+)$',ninja,re.M)
        if link_line is None:raise ValueError(f'missing effective linker inputs for {abi}')
        direct=[part.replace('\\','/') for part in re.split(r'\s{2,}',link_line.group(1))]
        wanted=[part for part in direct if PurePosixPath(part).name in LIBRARY_NAMES]
        if len(wanted)!=9:raise ValueError(f'unexpected direct static-library closure for {abi}: {wanted}')
        def selected(name):
            name=safe_name(name)
            return name in wanted or bool(re.search(r'CMakeFiles/PyfTetWildWrapper\.dir/src/FTetWildWrapper\.cpp\.(o|obj)$',name))
        if archive:
            with tarfile.open(archive, 'r:gz') as source:
                for member in source:
                    if member.isfile() and selected(member.name):
                        name = safe_name(member.name)
                        target = destination/name; target.parent.mkdir(parents=True, exist_ok=True)
                        with source.extractfile(member) as reader, target.open('xb') as writer:
                            shutil.copyfileobj(reader, writer, 1024*1024)
                        rows.append(dict(path=name, bytes=target.stat().st_size, sha256=sha(target)))
        else:
            for current, dirs, files in os.walk(build):
                dirs[:] = [d for d in dirs if not d.endswith('-src') and d not in {'.git', 'CMakeScratch'}]
                for filename in files:
                    path = Path(current)/filename; name = path.relative_to(build).as_posix()
                    if selected(name):
                        target=destination/name;target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path,target)
                        rows.append(dict(path=name, bytes=target.stat().st_size, sha256=sha(target)))
        if not any('FTetWildWrapper.cpp.' in row['path'] for row in rows):
            raise ValueError(f'no application object for {abi}')
        for path in configuration.iterdir():
            if path.is_file():
                target=destination/'historical-configuration'/path.name
                target.parent.mkdir(exist_ok=True);shutil.copyfile(path,target)
        command_files = list((destination/'historical-configuration').glob('*relink*'))
        if not command_files:
            command_files = list((destination/'historical-configuration').glob('build.ninja'))
        # Recipient commands use paths relative to this ABI directory. SDK/runtime
        # import libraries remain explicit prerequisites, not silently copied.
        objects=[row['path'] for row in rows if 'FTetWildWrapper.cpp.' in row['path']]
        libraries=wanted
        if set(libraries)-{row['path'] for row in rows}:raise ValueError(f'missing direct archive for {abi}')
        if abi.startswith('linux'):
            output='PyfTetWildWrapper.abi3.so' if abi=='linux312abi3' else 'PyfTetWildWrapper.cpython-311-x86_64-linux-gnu.so'
            command='c++ -fPIC -O3 -fsigned-char -ffp-contract=off -shared -Wl,--gc-sections -o '+output+' '+ ' '.join(objects+libraries)+' -fopenmp -lgmp -ldl\n'
            write_posix_relink_recipe(destination/'relink.sh', command)
        else:
            python_lib='python311.lib' if abi=='windows311' else 'python3.lib'
            command='link.exe /nologo /dll /machine:x64 /INCREMENTAL:NO /out:PyfTetWildWrapper.pyd '+ ' '.join('"'+p.replace('/','\\')+'"' for p in objects+libraries)+' "%PYTHON_LIB_DIR%\\'+python_lib+'" "%MPIR_LIB_DIR%\\gmp.lib" psapi.lib kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib\n'
            (destination/'relink.cmd').write_text('@echo off\nrem Run from this directory in an MSVC14.50 x64 developer prompt.\nrem Set PYTHON_LIB_DIR and MPIR_LIB_DIR to compatible development import libraries.\n'+command+'if errorlevel 1 exit /b %errorlevel%\n')
        results[abi]=dict(files=rows, historical_configuration='historical-configuration',
                          status='INPUTS_EXPORTED_RECIPIENT_RELINK_NOT_EXECUTED',
                          prerequisites='Compatible compiler/SDK, Python import library on Windows, system GMP/OpenMP on Linux; Windows MPIR development import library and runtime DLLs.',
                          command='relink.sh' if abi.startswith('linux') else 'relink.cmd')
    return results


NOTICE = '''Exact1 provider component notice and engineering status

The wrapper/fTetWild are MPL-2.0; this binary is not MPL-only. The exact
envelope route statically links LGPL-3.0-or-later i_pred (IndirectPredicates).
Windows wheels additionally bundle LGPL3 MPIR and Microsoft runtime DLLs.
The LGPL library and its use are covered by the accompanying LGPLv3/GPLv3
texts. Other linked inputs include FastEnvelope (MIT), TBB (Apache-2.0),
Geogram (BSD plus embedded third-party notices), Eigen/libigl (file-specific
MPL/permissive terms), nanobind (BSD), fmt/spdlog/json (MIT), CLI11 (BSD), and
Shewchuk predicates (public domain). See the accompanying license inventory.

The external provider-notices handoff contains selected corresponding source,
source identities, application/static relink inputs for each ABI, and recipient
directions. This wheel alone is not the complete source/relink handoff.
Retained notices also cover fetched build-source components; their presence
does not claim every vendored or archive object is retained in the binary.
No compilation, relink, numerical qualification, publication or distribution
acceptance is established by this metadata-only repack.
'''


def build():
    stage=OUT/('payload'+REVISION);stage.mkdir(parents=True, exist_ok=False)
    roots, sources=selected_sources()
    size=sum(row[2] for row in sources)
    write_json(OUT/('source-size-estimate'+REVISION+'.json'),dict(selected_files=len(sources),uncompressed_bytes=size,
        limit_bytes=MAX_SOURCE,excluded_directories=sorted(SKIP_DIRS),excluded_suffixes=sorted(SKIP_SUFFIXES)))
    if size>MAX_SOURCE:
        raise ValueError('selected source exceeds 1GiB; narrow source scope before archiving')
    license_records=collect_notices(stage/'licenses')
    mpir, acquisition=fetch_mpir(OUT/('cache'+REVISION))
    if mpir and size+mpir.stat().st_size>MAX_SOURCE:
        raise ValueError('selected source plus verified MPIR exceeds 1GiB')
    historic={}
    for inventory in ('source-files-sha256.json','nested-source-files-sha256.json'):
        historic.update({str(local_path(p).resolve()):digest for p,digest in json.loads((OLD/inventory).read_text()).items()})
    source_rows=[]
    with tarfile.open(stage/'corresponding-source.tar.gz','x:gz',compresslevel=1) as archive:
        for path,name,length in sources:
            archive.add(path,arcname=name,recursive=False)
            actual=sha(path);expected=historic.get(str(path.resolve()))
            if expected is not None and actual!=expected:raise ValueError(f'protected source changed: {path}')
            source_rows.append(dict(path=name,bytes=length,sha256=actual,
                historical_origin=str(path.relative_to(OLD)),historic_identity_verified=expected is not None))
        if mpir:
            archive.add(mpir,arcname='source/mpir-3.0.0.tar.bz2',recursive=False)
    write_json(stage/'source-files.json',source_rows)
    write_json(stage/'source-root-map.json',[dict(historical_root=str(root.relative_to(OLD)),
        archive_root='source/'+alias) for alias,root in roots])
    relink=export_relink(stage/'relink')
    write_json(stage/'relink-manifest.json',relink)
    pins=[]
    for row in json.loads((OLD/'dependency-inventory.json').read_text()):
        pins.append(dict(historical_path=str(local_path(row['path']).relative_to(OLD)),
                         commit=row['commit'],upstream=row.get('remote'),
                         selection='build source retained; no final-symbol retention claim'))
    write_json(stage/'source-pins.json',pins)
    write_json(stage/'license-manifest.json',license_records)
    for name in ('candidate-source.patch','wrapper-glue-patch.json','recursive-git-closure.json',
                 'windows-recursive-git-closure.json','windows-bundled-dll-origin-proven.json',
                 'windows-sdk-package-archives.json','native-flag-proof.json'):
        shutil.copyfile(OLD/name,stage/name)
    microsoft=[row for row in license_records if 'vc14_runtime-' in row['historical_source'] or 'ucrt-' in row['historical_source']]
    blockers=['Recipient rebuild/relink commands are supplied but not executed under this notice-only authority.',
              'Final linker-retained archive-member/header closure has not been measured; compilation/link-input evidence is distinguished from final retention.',
              'Microsoft documents and exact bundled-DLL origin are retained; qualification of applicable distribution conditions is not claimed.']
    if not mpir:blockers.append('MPIR corresponding-source archive unavailable at the fixed recipe SHA256; binary conda package/recipe/license remain retained.')
    write_json(stage/'component-manifest.json',dict(status='NOTICE_PACKAGING_ONLY',
        direct_link_targets=['nanobind','FloatTetwild','TBB','FastEnvelope','predicates','spdlog','fmt','geogram','IndirectPredicates'],
        lgpl_static_component=dict(name='i_pred',revision='e144a34c85cf25ee64ae9983745b548b27f9d996',license='LGPL-3.0-or-later',compiled_source='3rdparty/i_pred/numerics.cpp'),
        source_archive=dict(path='corresponding-source.tar.gz',sha256=sha(stage/'corresponding-source.tar.gz'),selected_uncompressed_bytes=size),
        mpir_source=dict(expected_sha256=MPIR_SHA,verified=mpir is not None,attempts=acquisition),
        microsoft_documents=microsoft,blocked_distribution_items=blockers,
        linked_vs_compiled='Direct library targets are proven by original Ninja linker rules. Exported library members and compiled Geogram third-party objects are not all asserted retained in the final native binary.',
        source_selection='Patched wrapper/fTetWild and fetched dependency build sources; excludes Git, build outputs, tests/examples/docs and datasets. Complete retained license notices are supplied separately.'))
    (stage/'README.md').write_text('# Exact1 source and relink handoff\n\n'+NOTICE+'\nExtract corresponding-source.tar.gz here. Source paths are relative under source/.\nEach relink/<ABI>/ directory has application/static inputs and a relative-path\nrelink.sh or relink.cmd; run from that directory after meeting its compiler/SDK\nprerequisites. Historical configuration is evidence and contains original paths.\nRebuild a modified i_pred library from source/dependencies/exact_envelope using\nits original compile flags in historical compile_commands.json, replace the\nIndirectPredicates archive in the matching ABI directory, then run its relink\ncommand. Generated native bytes would be a new unqualified candidate.\nWindows repaired wheels also require their unchanged packaged MPIR/MS runtime\nDLLs/bootstrap; a new modified native binary must be packaged separately.\nSee component-manifest.json for the exact unresolved distribution constraints.\n')
    summary=json.loads((OLD/'four-abi-candidate-summary.json').read_text())
    wheel_results={}
    additions={'NOTICE.exact1.txt':NOTICE.encode(),
               'NOTICE.exact1-components.json':(stage/'component-manifest.json').read_bytes(),
               'NOTICE.exact1-licenses.json':(stage/'license-manifest.json').read_bytes()}
    additions.update({'licenses/exact1/'+Path(row['file']).name:stage/row['file'] for row in license_records})
    for abi,record in summary['artifacts'].items():
        original=local_path(record['wheel'])
        if sha(original)!=record['wheel_sha256']:raise ValueError('historical wheel identity mismatch')
        refreshed=OUT/('wheels'+REVISION)/abi/original.name
        proof=repack(original,refreshed,additions,NOTICE)
        proof.update(original_wheel=original.relative_to(OLD).as_posix(),refreshed_wheel=refreshed.relative_to(OUT).as_posix(),
                     version='0.4.2+anyworkflow2.exact1',source_payload_sha256=sha(stage/'corresponding-source.tar.gz'))
        write_json(OUT/('proofs'+REVISION)/f'{abi}.json',proof);wheel_results[abi]=proof
    write_json(OUT/('summary'+REVISION+'.json'),dict(status='FOUR_NOTICE_REFRESHES_RUNTIME_BYTES_IDENTICAL',
        artifacts=wheel_results,source_files=len(source_rows),source_uncompressed_bytes=size,
        source_archive_sha256=sha(stage/'corresponding-source.tar.gz'),blockers=blockers,
        native_calls=0,compile_or_relink_calls=0,publication=False,distribution_ready=False))
    print(json.dumps(dict(status='NOTICE_REFRESH_COMPLETE',source_files=len(source_rows),source_bytes=size,wheels=len(wheel_results),blockers=len(blockers))),flush=True)


def smoke(abi):
    proof=json.loads((OUT/('proofs'+REVISION)/f'{abi}.json').read_text())
    wheel=OUT/safe_name(proof['refreshed_wheel']);folder=OUT/('smokes'+REVISION+'-portable')/abi
    folder.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(wheel) as archive:
        for entry in archive.infolist():safe_name(entry.filename)
        archive.extractall(folder/'extracted')
    sys.path.insert(0,str(folder/'extracted'))
    import importlib.metadata
    import pytetwild
    from pytetwild import PyfTetWildWrapper as native
    if importlib.metadata.version('pytetwild')!='0.4.2+anyworkflow2.exact1':raise ValueError('version changed')
    if getattr(native,'_workflow2_exact_envelope',False) is not True:raise ValueError('compiled marker missing')
    if Path(pytetwild.__file__).resolve().parent!=(folder/'extracted/pytetwild').resolve():raise ValueError('wrong imported origin')
    relative=Path(native.__file__).relative_to(folder/'extracted').as_posix()
    if sha(native.__file__)!=proof['preserved_runtime_members'][relative]['sha256']:raise ValueError('native imported bytes differ')
    if 'pyvista' in sys.modules:raise ValueError('unexpected PyVista')
    write_json(folder/'result.json',dict(status='PASS',python=sys.version,executable=sys.executable,
        wheel_sha256=sha(wheel),native_sha256=sha(native.__file__),native_calls=0,
        imported_from=str(pytetwild.__file__),version=importlib.metadata.version('pytetwild')))
    print(json.dumps(dict(abi=abi,status='PASS',native_calls=0)),flush=True)


def audit():
    """Check portable handoff membership without importing any provider."""
    stage=OUT/'payload-final'
    estimate=json.loads((OUT/'source-size-estimate-final.json').read_text())
    mpir=next(path for path in (OUT/'cache-network').glob('*.tar.bz2') if sha(path)==MPIR_SHA)
    with tarfile.open(mpir,'r:bz2') as archive:
        mpir_bytes=sum(member.size for member in archive if member.isfile())
    if estimate['uncompressed_bytes']+mpir_bytes>MAX_SOURCE:raise ValueError('expanded corresponding source exceeds 1GiB')
    with tarfile.open(stage/'corresponding-source.tar.gz','r:gz') as archive:
        names=[]
        for member in archive:
            name=safe_name(member.name)
            if '.git' in PurePosixPath(name).parts:raise ValueError('Git metadata in source archive')
            if PurePosixPath(name).suffix.lower() in {'.o','.obj','.lib','.a','.dll','.pyd','.so'}:raise ValueError('build output in source archive')
            names.append(name)
    if 'source/mpir-3.0.0.tar.bz2' not in names:raise ValueError('verified MPIR source missing')
    licenses=json.loads((stage/'license-manifest.json').read_text())
    if not any(row['provenance']=='retained linked nanobind build wheel' for row in licenses):raise ValueError('nanobind notice missing')
    if not any(row['sha256']==sha(OLD/'linux-build/_deps/eigen-src/COPYING.GPL') for row in licenses):raise ValueError('complete GPLv3 text missing')
    portable={}
    for abi in ('windows311','windows312abi3','linux311','linux312abi3'):
        proof=json.loads((OUT/'proofs-final'/f'{abi}.json').read_text())
        proof['original_wheel']=safe_name(proof['original_wheel'])
        proof['refreshed_wheel']=safe_name(proof['refreshed_wheel'])
        if sha(OLD/proof['original_wheel'])!=proof['original_sha256']:raise ValueError('historical wheel changed')
        if sha(OUT/proof['refreshed_wheel'])!=proof['refreshed_sha256']:raise ValueError('refreshed wheel changed')
        portable[abi]=proof
    write_json(OUT/'portable-artifact-manifest.json',dict(artifacts=portable,
        source_selected_bytes=estimate['uncompressed_bytes'],mpir_expanded_bytes=mpir_bytes,
        source_expanded_total_bytes=estimate['uncompressed_bytes']+mpir_bytes,
        source_limit_bytes=MAX_SOURCE,source_archive_members=len(names),
        source_archive_sha256=sha(stage/'corresponding-source.tar.gz'),native_calls=0,
        note='POSIX relative-path index; original per-wheel proof files are retained unchanged.'))
    print(json.dumps(dict(status='PORTABLE_HANDOFF_AUDITED',licenses=len(licenses),source_members=len(names))),flush=True)


def run(action,abi=None):
    OUT.mkdir(parents=True,exist_ok=True)
    ledger_path=OUT/'budget.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(limit_seconds=900.,used_seconds=0.,attempts=[])
    remaining=900-ledger['used_seconds']
    if remaining<=0:raise RuntimeError('notice process budget exhausted')
    name=action+('-'+abi if abi else '')+REVISION+('-controller312' if CHILD_PYTHON else '')
    folder=OUT/'processes'/name
    if folder.exists():raise FileExistsError(folder)
    command=([sys.executable,'-I','-m','pytest','-q','tests/test_exact_provider_notices.py',
              '--junitxml='+str(OUT/('focused'+REVISION+'.xml'))] if action=='checks' else
             [CHILD_PYTHON or sys.executable,'-I',str(Path(__file__).resolve()),action])
    if abi:command += ['--abi',abi]
    if REVISION and action!='checks':command += ['--revision',REVISION.lstrip('-')]
    timeout=min(300.,remaining);env=dict(os.environ)
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONOPTIMIZE'):env.pop(key,None)
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    helper=Path.home()/'bin/claude-review.py' if os.name=='nt' else REPO/'tools/workflow2_process_supervisor.py'
    spec=importlib.util.spec_from_file_location('notice_process_supervisor',helper)
    supervisor=importlib.util.module_from_spec(spec);spec.loader.exec_module(supervisor)
    receipt=None;started=time.perf_counter()
    try:
        if os.name=='nt':
            folder.mkdir(parents=True)
            receipt,_=supervisor.run_process(command,folder,'stdout',timeout,env,REPO)
        else:
            receipt=supervisor.run_linux(command,folder,timeout,env,REPO)
    finally:
        if receipt is None:
            receipt=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else dict(status='controller_error',process_wall_seconds=time.perf_counter()-started)
        ledger['used_seconds']+=receipt['process_wall_seconds'];ledger['remaining_seconds']=900-ledger['used_seconds']
        entry=dict(receipt);entry.update(name=name,command=command,runner_sha256=sha(__file__),supervisor_sha256=sha(helper),
            controller_python=sys.version,controller_executable=sys.executable,child_executable=command[0])
        ledger['attempts'].append(entry)
        temporary=ledger_path.with_suffix('.tmp');temporary.write_text(json.dumps(ledger,indent=2,allow_nan=False));temporary.replace(ledger_path)
    print(json.dumps(dict(name=name,status=receipt['status'],used=ledger['used_seconds'],remaining=ledger['remaining_seconds'])),flush=True)
    return 0 if receipt['status']=='completed' and receipt.get('exit_code')==0 and receipt.get('cleanup',{}).get('verified') else 1


def main():
    global REVISION,CHILD_PYTHON
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('run','build','smoke','acquire','audit'))
    parser.add_argument('action',nargs='?',choices=('build','smoke','checks','acquire','audit'))
    parser.add_argument('--abi',choices=('windows311','windows312abi3','linux311','linux312abi3'))
    parser.add_argument('--revision',choices=('final','checks-repair'))
    parser.add_argument('--child-python')
    args=parser.parse_args()
    REVISION='-'+args.revision if args.revision else ''
    CHILD_PYTHON=args.child_python
    if args.phase=='run':return run(args.action,args.abi)
    if args.phase=='build':build();return 0
    if args.phase=='acquire':fetch_mpir(OUT/'cache-network');return 0
    if args.phase=='audit':audit();return 0
    if args.abi is None:parser.error('--abi is required for smoke')
    smoke(args.abi);return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        traceback.print_exc();raise
