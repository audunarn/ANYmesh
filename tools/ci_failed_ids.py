"""List the failed and errored test ids of a pytest JUnit XML report.

Used after the full-profile pytest legs in CI. The whole suite runs (no
``--maxfail``, and ``--continue-on-collection-errors``), so a red leg is
compared by test id with a ``main`` run that used the same commands.

The output file always starts with one status line, so that missing evidence
cannot pass for a clean leg:

- ``# report: ok, <n> failed`` followed by the ids, one per line and sorted, in
  the form pytest prints them (``file::name``; a collection error is the module
  path; an internal error is ``INTERNALERROR``);
- ``# report: unreadable (<reason>)`` when the JUnit file is missing or malformed;
- ``# report: incomplete (<reason>)`` when it is not a complete pytest report:
  another root element, no test case, or a ``<testsuite>`` whose declared
  ``tests``/``failures``/``errors`` differ from its test cases.

The same goes to the job summary. Contract (owner standing approval,
2026-11-26): best-effort parsing of our own pytest JUnit output. An unreadable
or incomplete report fails the step, and so the leg, because that leg cannot
be compared by test id. Otherwise the step exits 0 and pytest's own exit
status decides the leg. Write errors here are warnings.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def _module_path(dotted: str) -> str:
    return dotted.replace('.', '/') + '.py'


def case_id(case) -> str:
    name = case.get('name', '')
    classname = case.get('classname', '')
    file = case.get('file')
    if classname == 'pytest' and name == 'internal':
        return 'INTERNALERROR'
    if not classname:
        # A collection error: pytest names the module by its dotted path.
        return _module_path(name)
    if file:
        return f'{file}::{name}'
    parts = classname.split('.')
    classes = []
    while parts and parts[-1][:1].isupper():
        classes.insert(0, parts.pop())
    return '::'.join(['/'.join(parts) + '.py'] + classes + [name])


class Incomplete(ValueError):
    """The report is not a complete pytest JUnit report."""


def failed_ids(report: Path) -> list[str]:
    root = ET.parse(report).getroot()
    if root.tag not in ('testsuites', 'testsuite'):
        raise Incomplete(f'root element <{root.tag}> is not a pytest JUnit report')
    suites = [root] if root.tag == 'testsuite' else root.findall('testsuite')
    if not suites:
        raise Incomplete('no <testsuite>')
    total = 0
    for suite in suites:
        cases = suite.findall('testcase')
        counted = {'tests': len(cases),
                   'failures': sum(len(case.findall('failure')) for case in cases),
                   'errors': sum(len(case.findall('error')) for case in cases)}
        for key, value in counted.items():
            declared = suite.get(key)
            if declared is None or not declared.isdigit() or int(declared) != value:
                raise Incomplete(f'<testsuite {key}="{declared}"> but {value} found')
        total += len(cases)
    if total == 0:
        raise Incomplete('no test cases')
    return sorted({case_id(case) for suite in suites for case in suite.findall('testcase')
                   if case.find('failure') is not None or case.find('error') is not None})


def render(report: Path) -> tuple[str, list[str] | None]:
    try:
        ids = failed_ids(report)
    except (OSError, ET.ParseError) as error:
        return f'# report: unreadable ({type(error).__name__}: {error})', None
    except Incomplete as error:
        return f'# report: incomplete ({error})', None
    return f'# report: ok, {len(ids)} failed', ids


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    status, ids = render(args.report)
    body = ''.join(i + '\n' for i in ids or [])
    print(status)
    print(body, end='')
    try:
        args.output.write_text(status + '\n' + body, encoding='utf8')
    except OSError as error:
        print(f'::warning title=failed ids::could not write {args.output}: {error}')
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        try:
            with open(summary, 'a', encoding='utf8') as stream:
                stream.write(f'### Failed test ids ({status[2:]})\n')
                if ids:
                    stream.write('```\n' + body + '```\n')
        except OSError as error:
            print(f'::warning title=failed ids::could not write the job summary: {error}')
    if ids is None:
        # Fail closed: without complete evidence the leg cannot be compared by test id.
        print(f'::error title=failed ids::{status[2:]}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
