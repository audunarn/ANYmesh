"""The CI reporter lists failed test ids exactly as pytest prints them."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location('ci_failed_ids', ROOT / 'tools' / 'ci_failed_ids.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ids_match_pytests_own_report_for_a_real_run(tmp_path):
    project = tmp_path / 'project'
    (project / 'tests' / 'sub').mkdir(parents=True)
    (project / 'tests' / 'sub' / 'test_a.py').write_text(textwrap.dedent('''
        import pytest
        def test_ok(): pass
        @pytest.mark.parametrize('case', ['ruled', 'a.b', 'x[1]'])
        def test_bad(case): assert False
        class TestGroup:
            def test_err(self, missing_fixture): pass
            def test_fail(self): assert False
    '''), encoding='utf8')
    # Its own rootdir, as in CI, where the run starts at the repository root.
    (project / 'pytest.ini').write_text('[pytest]\n', encoding='utf8')
    (project / 'tests' / 'test_broken.py').write_text('import not_a_module_anywhere\n', encoding='utf8')
    (project / 'tests' / 'test_skip.py').write_text('import pytest\ndef test_s(): pytest.skip("z")\n',
                                                    encoding='utf8')
    junit = tmp_path / 'r.xml'
    done = subprocess.run([sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider', '-rfE', '-q',
                           '--continue-on-collection-errors', '--junitxml', str(junit), 'tests'],
                          cwd=project, capture_output=True, text=True)
    assert done.returncode == 1, done.stdout
    # pytest prints backslashes on Windows; the reporter always uses '/'.
    printed = sorted({line.split(' ', 1)[1].split(' - ', 1)[0].replace(chr(92), '/')
                      for line in done.stdout.splitlines() if line.startswith(('FAILED ', 'ERROR '))})
    # Collection continued: the other module's tests ran.
    assert 'tests/sub/test_a.py::test_bad[a.b]' in printed and 'tests/test_broken.py' in printed
    assert _tool().failed_ids(junit) == printed


def _suite(cases, tests=None, failures=0, errors=0):
    tests = cases.count('<testcase') if tests is None else tests
    return (f'<testsuites><testsuite name="pytest" tests="{tests}" failures="{failures}" errors="{errors}">'
            + cases + '</testsuite></testsuites>')


def test_status_line_and_exit_tell_missing_evidence_from_a_clean_report(tmp_path, monkeypatch):
    tool = _tool()
    summary = tmp_path / 'summary.md'
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(summary))
    reports = {
        'clean': _suite('<testcase classname="tests.t" name="test_x" />'),
        'internal': _suite('<testcase classname="pytest" name="internal"><error /></testcase>', errors=1),
        'bad': '<testsuites',
        'unrelated': '<project><testcase classname="tests.t" name="test_x"><failure /></testcase></project>',
        'declared_without_cases': _suite('', tests=1, failures=1),
        'no_cases': _suite(''),
        'undercounted': _suite('<testcase classname="tests.t" name="test_x"><failure /></testcase>', failures=0),
        'no_suite': '<testsuites />',
        'missing_counts': '<testsuites><testsuite><testcase classname="tests.t" name="test_x" /></testsuite></testsuites>',
    }
    for name, text in reports.items():
        (tmp_path / f'{name}.xml').write_text(text, encoding='utf8')
    out = tmp_path / 'ids.txt'
    for name, expected in [('clean', ['# report: ok, 0 failed']),
                           ('internal', ['# report: ok, 1 failed', 'INTERNALERROR'])]:
        assert tool.main([str(tmp_path / f'{name}.xml'), '--output', str(out)]) == 0
        assert out.read_text(encoding='utf8').splitlines() == expected
    # Fail closed: without complete evidence the step, and so the leg, fails.
    for name, status in [('missing', 'unreadable'), ('bad', 'unreadable'), ('unrelated', 'incomplete'),
                         ('declared_without_cases', 'incomplete'), ('no_cases', 'incomplete'),
                         ('undercounted', 'incomplete'), ('no_suite', 'incomplete'), ('missing_counts', 'incomplete')]:
        assert tool.main([str(tmp_path / f'{name}.xml'), '--output', str(out)]) == 1, name
        assert out.read_text(encoding='utf8').startswith(f'# report: {status} ('), name
    assert summary.read_text(encoding='utf8').count('### Failed test ids (') == 10


def test_write_errors_never_fail_the_step(tmp_path, monkeypatch):
    clean = tmp_path / 'clean.xml'
    clean.write_text(_suite('<testcase classname="tests.t" name="test_x" />'), encoding='utf8')
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(tmp_path / 'no-such-dir' / 'summary.md'))
    assert _tool().main([str(clean), '--output', str(tmp_path / 'no-such-dir' / 'ids.txt')]) == 0
