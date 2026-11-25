"""List the failed and errored test ids of a pytest JUnit XML report.

Used after the full-profile pytest legs in CI. The whole suite runs (no
``--maxfail``, and ``--continue-on-collection-errors``), so a red leg is
compared by test id with a ``main`` run that used the same commands.

The output file always starts with one status line, so that missing evidence
cannot pass for a clean leg:

- ``# report: ok, <n> failed`` followed by the ids, one per line and sorted, in
  the form pytest prints them (``file::name``; a collection error is the module
  path; an internal error is ``INTERNALERROR``);
- ``# report: unreadable (<reason>)`` when the JUnit file is missing or malformed.

The same goes to the job summary. This step never changes the leg's result:
pytest's own exit status does that, and every write error here is a warning.
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


def failed_ids(report: Path) -> list[str]:
    root = ET.parse(report).getroot()
    return sorted({case_id(case) for case in root.iter('testcase')
                   if case.find('failure') is not None or case.find('error') is not None})


def render(report: Path) -> tuple[str, list[str] | None]:
    try:
        ids = failed_ids(report)
    except (OSError, ET.ParseError) as error:
        return f'# report: unreadable ({type(error).__name__}: {error})', None
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
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
