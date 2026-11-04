"""Pure compiler-proof contracts; no S0 fixture, mesh or native execution."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('s0_compiler_metadata_tool', ROOT / 'tools/ci_s0_baseline.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


OLD = '''[build-system]
requires = ["setuptools>=77", "wheel"]
build-backend = "setuptools.build_meta"
[project]
name = "ANYmesher"
version = "0.5.1"
dependencies = ["numpy>=1.26", "ANYgeometry[planar]>=0.4.5,<0.5"]
[project.optional-dependencies]
dev = ["build>=1.2", "pytest>=8", "twine>=5"]
'''
CURRENT = OLD.replace('>=0.4.5,<0.5', '>=0.4.7,<0.5').replace(
    '"build>=1.2",', '"build>=1.2", "setuptools>=77.0.3", "wheel",')


def inputs(monkeypatch, tmp_path, *, current=CURRENT, historical=OLD,
           git_change=False, working_change=False):
    old = tmp_path / 'old'
    current_repo = tmp_path / 'current'
    blobs = {name: b'compiler bytes\n' for name in tool.BUILD_FILES}
    blobs['setup.py'] = b'from pathlib import Path\nfrom setuptools import setup\n'
    for name, body in blobs.items():
        path = current_repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body.replace(b'\n', b'\r\n'))
    (current_repo / 'pyproject.toml').write_text(current, encoding='utf-8')
    changed = tool.BUILD_FILES[-1]
    if working_change:
        with (current_repo / changed).open('ab') as stream:
            stream.write(b'working mutation')

    def git_bytes(repo, operation, key):
        assert operation == 'show'
        ref, name = key.split(':', 1)
        assert ref == (tool.contract.REFS['mesher'] if repo == old else 'HEAD')
        if name == 'pyproject.toml':
            assert repo == old
            return historical.encode()
        body = blobs[name]
        return body + b'git mutation' if git_change and repo == current_repo and name == changed else body

    monkeypatch.setattr(tool, 'git_bytes', git_bytes)
    return old, current_repo


def test_exact_adopted_metadata_and_crlf_working_inputs_are_allowed(monkeypatch, tmp_path):
    proof = tool.compiler_proof(*inputs(monkeypatch, tmp_path))
    assert proof['exceptions']['dev_addition'] == 'setuptools>=77.0.3'
    assert proof['exceptions']['dev_wheel_addition'] == 'wheel'
    assert proof['exceptions']['runtime_dependencies'] == [
        ['numpy>=1.26', 'ANYgeometry[planar]>=0.4.5,<0.5'],
        ['numpy>=1.26', 'ANYgeometry[planar]>=0.4.7,<0.5']]
    assert set(proof['exact_git_blobs']) == set(tool.BUILD_FILES)
    assert proof['remaining_parsed_build_version_config_equal'] is True


@pytest.mark.parametrize('replacement', ['', '"wheel", "wheel",', '"wheel>=0.47",', '"Wheel",'])
def test_missing_duplicate_versioned_or_case_changed_wheel_is_refused(monkeypatch, tmp_path, replacement):
    current = CURRENT.replace('"setuptools>=77.0.3", "wheel",',
                              '"setuptools>=77.0.3", ' + replacement)
    with pytest.raises(AssertionError):
        tool.compiler_proof(*inputs(monkeypatch, tmp_path, current=current))


@pytest.mark.parametrize('replacement', ['', '"setuptools>=77.0.3", "setuptools>=77.0.3",',
                                          '"setuptools>=78",'])
def test_original_exact_setuptools_exception_is_preserved(monkeypatch, tmp_path, replacement):
    with pytest.raises(AssertionError):
        tool.compiler_proof(*inputs(monkeypatch, tmp_path,
            current=CURRENT.replace('"setuptools>=77.0.3",', replacement)))


@pytest.mark.parametrize('current', [
    CURRENT + '\n[tool.unknown]\nvalue = true\n',
    CURRENT.replace('version = "0.5.1"', 'version = "0.5.2"'),
    CURRENT.replace('requires = ["setuptools>=77", "wheel"]',
                    'requires = ["setuptools>=78", "wheel"]'),
    CURRENT.replace('"numpy>=1.26", "ANYgeometry', '"numpy>=1.26", "scipy>=1.13", "ANYgeometry'),
    CURRENT.replace('ANYgeometry[planar]>=0.4.7,<0.5', 'ANYgeometry[planar]>=0.4.6,<0.5'),
    CURRENT.replace('"build>=1.2",', '"build>=1.2", "ruff",'),
])
def test_unknown_config_version_build_runtime_or_dev_changes_are_refused(monkeypatch, tmp_path, current):
    with pytest.raises(AssertionError):
        tool.compiler_proof(*inputs(monkeypatch, tmp_path, current=current))


def test_wheel_must_be_a_current_addition_not_a_changed_old_baseline(monkeypatch, tmp_path):
    old = OLD.replace('"build>=1.2",', '"build>=1.2", "wheel",')
    with pytest.raises(AssertionError):
        tool.compiler_proof(*inputs(monkeypatch, tmp_path, historical=old))


@pytest.mark.parametrize('git_change,working_change,message', [
    (True, False, 'native compiler/adapter Git blob changed'),
    (False, True, 'working compiler input changed'),
])
def test_compiler_git_and_working_mutations_still_refuse(monkeypatch, tmp_path, git_change, working_change, message):
    with pytest.raises(AssertionError, match=message):
        tool.compiler_proof(*inputs(monkeypatch, tmp_path,
            git_change=git_change, working_change=working_change))
