"""Exact represented-byte S0 identity, not an accuracy or native qualification claim.

CI's cold frozen-source producer seals the manifest outside pytest. Current
full-suite cases compare against it after building, so warm-state drift fails.
An ordinary local invocation without this explicit contract keeps its historical
Windows GOLDEN check. No geometry/mesher function is wrapped or replaced.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import struct
import sys

CASES = ('planar', 'cylinder', 'cone', 'ruled', 'coons', 'native_nurbs')
FIXTURES = ('test_qf_nurbs_s0_characterisation.py', 'test_curved_native_qualification.py',
            'test_cylindrical_atlas_binding.py')
REFS = {'geometry': 'f645523c8006efe61e94678a5056fba3c9a6a39d',
        'mesher': 'aa19c0fa6b796a0d7cb1793a87840f52899dfa24'}
HISTORICAL = {
    'planar': '357065ccbca528c3d08905e9f38473141781069f9fd8db668cada0731565af67',
    'cylinder': '1b7a9cdfdfb3563f4ac95817ff63f77d510050c436f60476e108ea3c211d8949',
    'cone': '843a32d901a5aa5ebb7108e0b2870058afa648f6124050b297da3de2fd0f278f',
    'ruled': '0a3a26e9df46528c148a5eebfe5ec21f4610cc4442a0d69494fd334b9e8c6e31',
    'coons': '0a3a26e9df46528c148a5eebfe5ec21f4610cc4442a0d69494fd334b9e8c6e31',
    'native_nurbs': 'eb02e4af2fc2683b4e1864667dc4e1011bbe10978826ccc10c22abad16dfc84b',
}
MAX_BYTES = 4_000_000
PATH_ENV = 'ANYMESHER_S0_BASELINE'
SHA_ENV = 'ANYMESHER_S0_BASELINE_SHA256'
REQUIRE_ENV = 'ANYMESHER_REQUIRE_S0_BASELINE'
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
# Test-input identities, not production/native trust pins. These retain the
# original builder bodies and their two unchanged helper modules across CI.
BUILDERS = {
    'digest': 'bc9971c5188db23c9d1d9e42df5eae6c64af7933e0e2405d11a49af68c3eb2d0',
    'planar_model': '91b8b82f7e472099e957d7df771371410131058a2e906ba8fd7e90d95d17a9c2',
    'quad_first': 'd5eca3ea1defc74df83793fc3aa8472fcb2784db51a9ea10ed85048b2fe442ef',
    'nurbs_patch': '4c9c0f6ee09d157d0c39d66d0fcf9ef5ecc8d8a320599356ab084f53437dcf43',
    'build': 'e0b3d5bdee0fa0c960e7cc45e3d034103e23bde268ff98522ef782ca8d4bd896',
}
HELPERS = {
    'test_curved_native_qualification.py': 'fb48d37ca05afe90e762d8812d261fc274bcabb22ba2bb3c2b9887d6731765c9',
    'test_cylindrical_atlas_binding.py': 'c524b97635823c8cef22bb068155956658d3b47c16750bcfc4f7951c59a05d74',
}


def hash_bytes(body):
    return hashlib.sha256(body).hexdigest()


def semantic_ast(node):
    # Python 3.12 added an empty FunctionDef.type_params field and 3.13 changed
    # ast.dump empty-field formatting. Neither changes these original builders.
    if isinstance(node, ast.AST):
        return [type(node).__name__, {key: semantic_ast(value) for key, value in ast.iter_fields(node)
                                     if key != 'type_params' or value}]
    if isinstance(node, list):
        return [semantic_ast(value) for value in node]
    return node


def file_hash(path):
    return hash_bytes(Path(path).read_bytes())


def source_inventory(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob('*.py'))}


def native_identity(package):
    """Actual selected extension and adjacent loader payload; absence is explicit."""
    root = Path(package.__file__).resolve().parent
    spec = importlib.util.find_spec(package.__name__ + '._native')
    binaries = {}
    if spec is not None:
        assert spec.origin and Path(spec.origin).resolve().parent == root, 'native image escaped package'
        assert any(spec.origin.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES), 'not a native extension'
        binaries[Path(spec.origin).name] = file_hash(spec.origin)
    libraries = {}
    for directory in (root / '.dylibs', root.parent / (root.name + '.libs')):
        if directory.exists():
            for path in sorted(directory.rglob('*')):
                if path.is_file():
                    libraries[path.relative_to(root.parent).as_posix()] = file_hash(path)
    return dict(extension=binaries, libraries=libraries,
                loader_sha256=file_hash(root / '__init__.py'))


def runtime_identity():
    import numpy
    import shapely
    def image(module):
        return dict(version=module.__version__, origin=str(Path(module.__file__).resolve()),
                    source_sha256=file_hash(module.__file__))
    # Record actual dependency native images; version alone cannot bind an ABI.
    dependency_native = {}
    for package in (numpy, shapely):
        root = Path(package.__file__).resolve().parent
        paths = [p for p in root.rglob('*') if p.is_file() and
                 any(str(p).endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES)]
        for directory in (root / '.dylibs', root.parent / (root.name + '.libs')):
            if directory.exists():
                paths.extend(p for p in directory.rglob('*') if p.is_file())
        dependency_native[package.__name__] = {str(p.resolve()): file_hash(p) for p in sorted(set(paths))}
    return dict(executable=str(Path(sys.executable).resolve()), python=sys.version,
                cache_tag=sys.implementation.cache_tag, platform=sys.platform,
                machine=platform.machine(), byteorder=sys.byteorder, isolated=sys.flags.isolated,
                numpy=image(numpy), shapely=image(shapely), dependency_native=dependency_native)


def checked_geometry_backend(report):
    """This portable profile attests Python G, never an unexamined G core image."""
    effective = {key: report[key] for key in ('backend', 'module', 'build')}
    assert effective == dict(backend='python', module=None, build=None), 'S0 requires the actual Python geometry backend'
    assert type(report['reason']) is str, 'missing geometry backend reason'
    return dict(effective=effective, reason=report['reason'])


def current_identity():
    import anygeometry
    import anymesher
    return dict(runtime=runtime_identity(), geometry_backend=checked_geometry_backend(anygeometry.backend_info()),
                geometry_origin=str(Path(anygeometry.__file__).resolve()),
                mesher_origin=str(Path(anymesher.__file__).resolve()),
                geometry_source=source_inventory(Path(anygeometry.__file__).parent),
                mesher_source=source_inventory(Path(anymesher.__file__).parent), native=native_identity(anymesher))


def fixture_identity(directory):
    return {name: file_hash(Path(directory) / name) for name in FIXTURES}


def validate_fixture_contract(directory):
    directory = Path(directory)
    tree = ast.parse((directory / FIXTURES[0]).read_text(encoding='utf8'))
    actual = {n.name: hash_bytes(json.dumps(semantic_ast(n), sort_keys=True).encode())
              for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in BUILDERS}
    assert actual == BUILDERS, 'original S0 builder/serialization changed'
    constants = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'GOLDEN' for t in n.targets))
    assert constants == HISTORICAL, 'historical GOLDEN literals changed'
    assert {name: hash_bytes((directory / name).read_bytes().replace(b'\r\n', b'\n'))
            for name in HELPERS} == HELPERS, 'original S0 helper fixture changed'


def encode_mesh(mesh):
    """Verbatim byte ordering/dtypes of the unchanged historical digest."""
    import numpy as np
    body = bytearray()
    for key in sorted(mesh.nodes):
        body.extend(np.int64(key).tobytes() + np.asarray(mesh.nodes[key], dtype=np.float64).tobytes())
    for name in ('quads', 'tris'):
        body.extend(name.encode())
        for key in sorted(getattr(mesh, name)):
            body.extend(np.int64(key).tobytes() + np.asarray(getattr(mesh, name)[key], dtype=np.int64).tobytes())
    counts = dict(nodes=len(mesh.nodes), quads=len(mesh.quads), tris=len(mesh.tris))
    validate_raw(bytes(body), counts)
    return bytes(body), counts


def validate_raw(body, row):
    counts = [row[k] for k in ('nodes', 'quads', 'tris')]
    assert all(type(n) is int and n >= 0 for n in counts), 'invalid counts'
    n, q, t = counts
    assert n > 0 and q + t > 0, 'empty S0 fixture'
    assert len(body) <= MAX_BYTES and len(body) == 32*n + 40*q + 32*t + 9, 'raw size/count mismatch'
    nodes = []
    for offset in range(0, 32*n, 32):
        key, *xyz = struct.unpack_from('=qddd', body, offset)
        assert all(math.isfinite(x) for x in xyz), 'nonfinite coordinate'
        nodes.append(key)
    assert nodes == sorted(set(nodes)), 'unsorted/duplicate node IDs'
    node_set = set(nodes)
    offset = 32*n
    for marker, count, width in ((b'quads', q, 4), (b'tris', t, 3)):
        assert body[offset:offset+len(marker)] == marker, 'raw marker mismatch'
        offset += len(marker)
        ids = []
        for _ in range(count):
            key, *connectivity = struct.unpack_from('=' + 'q'*(width+1), body, offset)
            offset += 8*(width+1)
            assert all(node in node_set for node in connectivity), 'unknown connectivity node'
            ids.append(key)
        assert ids == sorted(set(ids)), 'unsorted/duplicate element IDs'
    assert offset == len(body)


def completed(terminal):
    return (terminal['status'] == 'completed' and type(terminal['exit_code']) is int and
            terminal['exit_code'] == 0 and terminal['cleanup_verified'] is True and
            terminal['remaining_owned'] == [] and terminal['errors'] == [])


def load_packet(path, seal, *, identity, fixtures):
    path = Path(path)
    assert re.fullmatch('[0-9a-f]{64}', seal or ''), 'invalid manifest SHA'
    assert path.stat().st_size <= MAX_BYTES, 'manifest size bound'
    encoded = path.read_bytes()
    assert hash_bytes(encoded) == seal, 'manifest SHA mismatch'
    row = json.loads(encoded)
    assert row['schema'] == 1 and row['refs'] == REFS and row['order'] == list(CASES), 'reference/order mismatch'
    assert row['historical_golden'] == HISTORICAL, 'historical constants changed'
    assert row['identity'] == identity and row['fixtures'] == fixtures, 'runtime/native/source/fixture binding mismatch'
    assert row['source_unchanged'] is True and row['completed'] is True and completed(row['terminal']), 'incomplete baseline'
    assert [c['case'] for c in row['cases']] == list(CASES), 'case inventory mismatch'
    result = {}
    for case in row['cases']:
        raw_path = path.parent / (case['case'] + '.bin')
        assert raw_path.resolve().parent == path.resolve().parent, 'raw path escaped packet'
        assert raw_path.stat().st_size <= MAX_BYTES
        body = raw_path.read_bytes()
        assert len(body) == case['bytes'] and hash_bytes(body) == case['sha256'], 'raw integrity mismatch'
        validate_raw(body, case)
        result[case['case']] = body
    return result


def baseline_environment():
    path, seal = os.environ.get(PATH_ENV), os.environ.get(SHA_ENV)
    if not path and not seal:
        assert os.environ.get(REQUIRE_ENV) != '1', 'required S0 cold baseline is missing'
        return None
    assert path and seal, 'partial S0 baseline environment'
    return path, seal


def assert_output(name, mesh, golden, *, digest):
    validate_fixture_contract(Path(__file__).parent)
    assert golden == HISTORICAL, 'historical GOLDEN literals changed'
    body, counts = encode_mesh(mesh)
    assert hash_bytes(body) == digest(mesh), 'raw encoding differs from original digest'
    environment = baseline_environment()
    if environment is None:
        assert digest(mesh) == golden[name]
        return
    assert all(os.environ.get(key) == '1' for key in THREADS), 'S0 thread controls changed'
    assert all(os.environ.get(key) == '1' for key in ('ANYMESH_Q4_DISABLE_WORKER', 'ANYMESH_Q5_DISABLE_WORKER'))
    rows = load_packet(*environment, identity=current_identity(), fixtures=fixture_identity(Path(__file__).parent))
    assert body == rows[name], 'current warm S0 output differs from frozen-source cold raw baseline: ' + name
