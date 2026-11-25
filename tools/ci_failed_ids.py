"""List the failed and errored test ids of a pytest JUnit XML report.

Used after the full-profile pytest legs in CI. The whole suite runs (no
``--maxfail``), so a red leg is compared with ``main`` by test id. The ids
(``file::name`` as pytest prints them, one per line, sorted) are written to
``--output`` (uploaded with the leg's results) and to the job summary. This
step never changes the leg's result: pytest's own exit status does that.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def failed_ids(report: Path) -> list[str]:
    root = ET.parse(report).getroot()
    ids = set()
    for case in root.iter('testcase'):
        if case.find('failure') is None and case.find('error') is None:
            continue
        name = case.get('name', '')
        file = case.get('file')
        if not file:
            # pytest's junit gives the dotted classname; rebuild the path part.
            parts = case.get('classname', '').split('.')
            cls = []
            while parts and parts[-1][:1].isupper():
                cls.insert(0, parts.pop())
            file = '/'.join(parts) + '.py' if parts else ''
            name = '::'.join(cls + [name])
        ids.add(f'{file}::{name}' if file else name)
    return sorted(ids)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        ids = failed_ids(args.report)
        text = ''.join(i + '\n' for i in ids)
        heading = f'### Failed test ids: {len(ids)}\n'
    except (OSError, ET.ParseError) as error:
        ids, text = None, ''
        heading = f'### Failed test ids: no readable report ({error})\n'
    args.output.write_text(text, encoding='utf8')
    print(heading.strip())
    print(text, end='')
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        try:
            with open(summary, 'a', encoding='utf8') as stream:
                stream.write(heading)
                if ids:
                    stream.write('```\n' + text + '```\n')
        except OSError as error:
            print(f'::warning title=failed ids summary::could not write the job summary: {error}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
