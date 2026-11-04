"""Produce one cold six-case frozen-source S0 baseline for this exact CI image.

No build, installation, numerical override, cache reset or golden update occurs.
The reviewed manual Mac diagnostic remains a separate diagnostic-only entrypoint.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import traceback
import zipfile

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent / 'tests'))
import _s0_baseline_contract as contract
from diagnostic_s0_mac import BUILD_FILES, ENV, inventory, supervise, write

BOOTSTRAP = '''\n\n# start delvewheel patch
def _delvewheel_patch_1_13_0():
    import os
    if os.path.isdir(libs_dir := os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'anymesher.libs'))):
        os.add_dll_directory(libs_dir)


_delvewheel_patch_1_13_0()
del _delvewheel_patch_1_13_0
# end delvewheel patch
'''


def git_bytes(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], timeout=15)


def lf(body):
    return body.replace(b'\r\n', b'\n')


def compiler_proof(old_repo, current_repo):
    """Exact Git blobs plus current working inputs; only two adopted metadata deltas."""
    rows = {}
    for name in BUILD_FILES:
        old = git_bytes(old_repo, 'show', contract.REFS['mesher'] + ':' + name)
        current = git_bytes(current_repo, 'show', 'HEAD:' + name)
        assert old == current, 'native compiler/adapter Git blob changed: ' + name
        assert lf((current_repo / name).read_bytes()) == lf(current), 'working compiler input changed: ' + name
        rows[name] = contract.hash_bytes(old)
    old_py = git_bytes(old_repo, 'show', contract.REFS['mesher'] + ':pyproject.toml')
    current_py = (current_repo / 'pyproject.toml').read_bytes()
    configs = [tomllib.loads(body.decode()) for body in (old_py, current_py)]
    deps = [config['project'].pop('dependencies') for config in configs]
    assert deps == [['numpy>=1.26', 'ANYgeometry[planar]>=0.4.5,<0.5'],
                    ['numpy>=1.26', 'ANYgeometry[planar]>=0.4.7,<0.5']], 'unexpected runtime metadata change'
    dev = configs[1]['project']['optional-dependencies']['dev']
    assert dev.count('setuptools>=77.0.3') == 1
    dev.remove('setuptools>=77.0.3')
    assert configs[0] == configs[1], 'build-system/version/config differs beyond two adopted metadata exceptions'
    tree = ast.parse(git_bytes(old_repo, 'show', contract.REFS['mesher'] + ':setup.py').decode())
    imports = sorted({n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} |
                     {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names})
    assert all(name and name.split('.')[0] in ('__future__', 'os', 'shutil', 'sys', 'pathlib', 'setuptools')
               for name in imports), 'setup imports geometry/build-generated owner constants'
    return dict(exact_git_blobs=rows, setup_imports=imports,
                pyproject_sha256=[contract.hash_bytes(old_py), contract.hash_bytes(current_py)],
                exceptions=dict(runtime_dependencies=deps, dev_addition='setuptools>=77.0.3'),
                remaining_parsed_build_version_config_equal=True,
                newline_policy='working checkout CRLF is recorded; Git compiler blobs compare byte-for-byte')


def archive_source(repo, ref, output):
    assert git_bytes(repo, 'rev-parse', ref + '^{commit}').decode().strip() == ref
    body = git_bytes(repo, 'archive', '--format=zip', ref, 'src')
    output.mkdir()
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        for item in archive.infolist():
            path = Path(item.filename)
            assert not path.is_absolute() and '..' not in path.parts and path.parts[0] == 'src'
        archive.extractall(output)
    return dict(commit=ref, archive_sha256=contract.hash_bytes(body), source=inventory(output / 'src'))


def loader_payload(package_root, current_repo):
    """Verify actual wheel Python against source and declare exactly one known loader patch."""
    patch = ''
    rows = {}
    actual_paths = {p.relative_to(package_root).as_posix(): p for p in package_root.rglob('*.py')}
    source_root = current_repo / 'src' / 'anymesher'
    expected_paths = {p.relative_to(source_root).as_posix(): p for p in source_root.rglob('*.py')}
    assert actual_paths.keys() == expected_paths.keys(), 'actual package Python inventory differs from source'
    for name, path in actual_paths.items():
        actual = lf(path.read_bytes())
        expected = lf(expected_paths[name].read_bytes())
        if name == '__init__.py' and actual != expected:
            assert actual.count(BOOTSTRAP.encode()) == 1, 'unknown wheel loader/bootstrap'
            assert actual.replace(BOOTSTRAP.encode(), b'', 1) == expected, 'wheel differs beyond exact loader patch'
            patch = BOOTSTRAP
        else:
            assert actual == expected, 'wheel Python/source differs: ' + name
        rows[name] = dict(actual_sha256=contract.file_hash(path), source_sha256=contract.file_hash(expected_paths[name]))
    return dict(python_files=rows, bootstrap=patch,
                bootstrap_sha256=contract.hash_bytes(patch.encode()),
                scope='actual native artifact, with explicitly declared loader adaptation; not pure Git source')


def add_loader(old_init, patch):
    original = old_init.read_bytes()
    if not patch:
        return dict(original_sha256=contract.hash_bytes(original), overlay_sha256=contract.hash_bytes(original), patch='none')
    source = original.decode()
    tree = ast.parse(source)
    prefix = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == '__future__']
    assert prefix, 'old loader has no declared future-import insertion anchor'
    line = max(node.end_lineno for node in prefix)
    lines = source.splitlines(keepends=True)
    lines.insert(line, patch)
    old_init.write_text(''.join(lines), encoding='utf8', newline='')
    return dict(original_sha256=contract.hash_bytes(original), overlay_sha256=contract.file_hash(old_init),
                patch_sha256=contract.hash_bytes(patch.encode()), insertion_after_line=line,
                scope='explicit non-Git loader-only overlay; old numerical source retained')


def child(config_path):
    config = json.loads(Path(config_path).read_text())
    output = Path(config['output'])
    output.mkdir()
    roots = [Path(config[k]) for k in ('geometry', 'mesher', 'fixtures')]
    before = [inventory(p) for p in roots]
    write(output / 'source-before.json', before)
    assert not any(n.split('.')[0] in ('anygeometry', 'anymesher') for n in sys.modules), 'not a fresh ANY namespace'
    assert all(os.environ.get(k) == v for k, v in ENV.items()), 'effective old-child controls differ'
    sys.path[:0] = list(map(str, roots))
    cases = []
    failure = None
    try:
        import anygeometry
        import anymesher
        fixture = importlib.import_module('test_qf_nurbs_s0_characterisation')
        for module, root in ((anygeometry, roots[0]), (anymesher, roots[1]), (fixture, roots[2])):
            assert Path(module.__file__).resolve().is_relative_to(root.resolve()), 'unexpected child source origin'
        assert contract.runtime_identity() == config['runtime'], 'old child runtime differs'
        geometry_backend = contract.checked_geometry_backend(anygeometry.backend_info())
        assert geometry_backend['effective'] == config['geometry_backend']['effective'], 'old/current geometry backend differs'
        native = contract.native_identity(anymesher)
        assert native['extension'] == config['native']['extension'] and native['libraries'] == config['native']['libraries'], 'old/current native payload differs'
        assert fixture.GOLDEN == contract.HISTORICAL
        from anymesher.native import _run_reference_native_seed
        target = _run_reference_native_seed.__code__
        extension = importlib.import_module('anymesher._native') if native['extension'] else None
        native_function = getattr(extension, 'reference_delaunay', None)
        observations = []
        def observe(frame, event, arg):
            if event == 'return' and frame.f_code is target:
                observations.append(dict(event='reference_seed_result', accepted=arg is not None and arg[0] is not None))
            if event == 'c_call' and native_function is not None and arg is native_function:
                observations.append(dict(event='actual_compiled_reference_delaunay_call'))
        for name in contract.CASES:
            write(output / 'active.json', dict(case=name, completed=len(cases)))
            previous = sys.getprofile()
            observations.clear()
            try:
                sys.setprofile(observe)
                mesh = fixture.build(name)
            finally:
                sys.setprofile(previous)
            body, counts = contract.encode_mesh(mesh)
            assert contract.hash_bytes(body) == fixture.digest(mesh)
            (output / (name + '.bin')).write_bytes(body)
            cases.append(dict(case=name, **counts, bytes=len(body), sha256=contract.hash_bytes(body),
                              native_observations=list(observations)))
            write(output / 'partial-cases.json', cases)
        assert geometry_backend == contract.checked_geometry_backend(anygeometry.backend_info()), 'old geometry backend drift'
        write(output / 'origins.json', dict(geometry=anygeometry.__file__, geometry_backend=geometry_backend,
              mesher=anymesher.__file__,
              fixture=fixture.__file__, native=native, runtime=config['runtime']))
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        after = [inventory(p) for p in roots]
        write(output / 'source-after.json', after)
        write(output / 'cold-result.json', dict(cases=cases, source_unchanged=before == after, failure=failure))
    return 0 if failure is None and before == after and len(cases) == 6 else 1


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    current_repo = args.mesher_source.resolve()
    geometry_repo = args.geometry_source.resolve()
    old_m_repo = args.old_mesher.resolve()
    old_g_repo = args.old_geometry.resolve()
    proof = compiler_proof(old_m_repo, current_repo)
    write(output / 'compiler-proof.json', proof)
    old_g = archive_source(old_g_repo, contract.REFS['geometry'], output / 'old-geometry')
    old_m = archive_source(old_m_repo, contract.REFS['mesher'], output / 'old-mesher')
    import anygeometry
    import anymesher
    package_root = Path(anymesher.__file__).resolve().parent
    assert Path(anygeometry.__file__).resolve().is_relative_to(geometry_repo / 'src'), 'current geometry origin differs'
    # Wheel mode deliberately binds installed/artifact package source to this checkout, rather than shadowing it.
    payload = loader_payload(package_root, current_repo)
    identity = contract.current_identity()
    fixtures = output / 'fixtures'
    fixtures.mkdir()
    for name in contract.FIXTURES:
        shutil.copyfile(current_repo / 'tests' / name, fixtures / name)
    fixture_hashes = contract.fixture_identity(fixtures)
    contract.validate_fixture_contract(fixtures)
    overlay = output / 'old-mesher' / 'src' / 'anymesher'
    loader = add_loader(overlay / '__init__.py', payload['bootstrap'])
    for name in identity['native']['extension']:
        shutil.copyfile(package_root / name, overlay / name)
    for name in identity['native']['libraries']:
        destination = overlay.parent / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package_root.parent / name, destination)
    write(output / 'provenance.json', dict(old_geometry=old_g, old_mesher=old_m, compiler=proof,
          actual_package_loader=payload, old_overlay_loader=loader,
          current_geometry_commit=git_bytes(geometry_repo, 'rev-parse', 'HEAD').decode().strip(),
          current_mesher_commit=git_bytes(current_repo, 'rev-parse', 'HEAD').decode().strip(),
          actual_current_origins=dict(geometry=anygeometry.__file__, mesher=anymesher.__file__)))
    config = dict(geometry=str(output / 'old-geometry' / 'src'), mesher=str(overlay.parent),
                  fixtures=str(fixtures), output=str(output / 'cold'),
                  runtime=identity['runtime'], native=identity['native'], geometry_backend=identity['geometry_backend'])
    write(output / 'child-config.json', config)
    command = [sys.executable] + (['-I'] if sys.flags.isolated else []) + ['-B', str(Path(__file__).resolve()), '--child', str(output / 'child-config.json')]
    env = dict(os.environ, **ENV)
    # No inherited baseline may be mistaken for the one being produced.
    for key in (contract.PATH_ENV, contract.SHA_ENV, contract.REQUIRE_ENV):
        env.pop(key, None)
    terminal = supervise(command, output, env, output / 'process')
    assert contract.completed(terminal), 'cold child did not complete with verified cleanup'
    cold = json.loads((output / 'cold' / 'cold-result.json').read_text())
    assert cold['failure'] is None and cold['source_unchanged'] is True
    assert identity == contract.current_identity(), 'current runtime/native/source drifted during old child'
    assert fixture_hashes == contract.fixture_identity(current_repo / 'tests'), 'fixture drift'
    assert proof == compiler_proof(old_m_repo, current_repo), 'compiler source drift'
    for name in contract.CASES:
        shutil.copyfile(output / 'cold' / (name + '.bin'), output / (name + '.bin'))
    manifest = dict(schema=1, refs=contract.REFS, order=list(contract.CASES), historical_golden=contract.HISTORICAL,
                    identity=identity, fixtures=fixture_hashes, completed=True, source_unchanged=True,
                    terminal=terminal, cases=cold['cases'],
                    scope='same-runtime frozen-old cold versus current warm represented bytes; not geometric accuracy/native qualification')
    path = output / 'manifest.json'
    write(path, manifest)
    seal = contract.file_hash(path)
    contract.load_packet(path, seal, identity=identity, fixtures=fixture_hashes)
    write(output / 'binding.json', dict(path=str(path), sha256=seal))
    if args.github_env:
        with args.github_env.open('a', encoding='utf8') as stream:
            stream.write(f'{contract.PATH_ENV}={path}\n{contract.SHA_ENV}={seal}\n')
    print(json.dumps(dict(baseline=str(path), sha256=seal, cases=6, cleanup_verified=True)))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', type=Path)
    parser.add_argument('--geometry-source', type=Path)
    parser.add_argument('--mesher-source', type=Path, default=TOOLS.parent)
    parser.add_argument('--old-geometry', type=Path)
    parser.add_argument('--old-mesher', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--github-env', type=Path)
    args = parser.parse_args()
    if args.child:
        return child(args.child)
    if not all((args.geometry_source, args.old_geometry, args.old_mesher, args.output)):
        parser.error('all source roots and a fresh output directory are required')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
