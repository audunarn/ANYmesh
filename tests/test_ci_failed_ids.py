"""The CI reporter lists failed test ids exactly as pytest prints them."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location('ci_failed_ids', ROOT / 'tools' / 'ci_failed_ids.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPORT = '''<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">
<testcase classname="tests.sub.test_a" name="test_ok" />
<testcase classname="tests.sub.test_a" name="test_bad[ruled]"><failure message="x" /></testcase>
<testcase classname="tests.test_b.TestGroup" name="test_err"><error message="y" /></testcase>
<testcase classname="tests.test_b" name="test_skip"><skipped message="z" /></testcase>
<testcase classname="tests.test_c" name="test_with_file" file="tests/test_c.py"><failure /></testcase>
</testsuite></testsuites>'''


def test_failed_and_errored_ids_in_pytest_form(tmp_path, monkeypatch):
    report = tmp_path / 'r.xml'
    report.write_text(REPORT, encoding='utf8')
    out = tmp_path / 'ids.txt'
    summary = tmp_path / 'summary.md'
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(summary))
    assert _tool().main([str(report), '--output', str(out)]) == 0
    assert out.read_text(encoding='utf8').splitlines() == [
        'tests/sub/test_a.py::test_bad[ruled]',
        'tests/test_b.py::TestGroup::test_err',
        'tests/test_c.py::test_with_file',
    ]
    assert summary.read_text(encoding='utf8').startswith('### Failed test ids: 3\n')


def test_unreadable_report_is_reported_and_never_fails_the_step(tmp_path, monkeypatch):
    monkeypatch.delenv('GITHUB_STEP_SUMMARY', raising=False)
    out = tmp_path / 'ids.txt'
    assert _tool().main([str(tmp_path / 'missing.xml'), '--output', str(out)]) == 0
    assert out.read_text(encoding='utf8') == ''
    bad = tmp_path / 'bad.xml'
    bad.write_text('<testsuites', encoding='utf8')
    assert _tool().main([str(bad), '--output', str(out)]) == 0
