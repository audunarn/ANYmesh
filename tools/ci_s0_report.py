"""Report the CI S0 baseline outcome and isolate it from the test legs.

Runs after the (non-blocking) S0 step in both full CI profiles:

1. Containment: any process still running from the S0 producer (its script or
   its output directory on the command line, and their descendants) is
   killed, and the step fails if one survives.  A stray S0 process must not
   share the runner with pytest.
2. Binding: only after a successful S0 outcome, and only when the producer
   wrote exactly one path/SHA pair to its binding file, are that pair and
   ``ANYMESHER_REQUIRE_S0_BASELINE=1`` exported.  Otherwise
   ``ANYMESHER_S0_BASELINE_UNAVAILABLE=<reason>`` is exported, no binding is
   visible to pytest, and the tests that need the baseline skip with that
   reason.
3. The outcome is written to the job summary, and a non-success as a warning.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PATH_ENV = 'ANYMESHER_S0_BASELINE'
SHA_ENV = 'ANYMESHER_S0_BASELINE_SHA256'
REQUIRE_ENV = 'ANYMESHER_REQUIRE_S0_BASELINE'
UNAVAILABLE_ENV = 'ANYMESHER_S0_BASELINE_UNAVAILABLE'
MARKERS = ('ci_s0_baseline.py',)


def stray_processes(output: Path):
    """Live processes started for S0 (by command line), with their descendants."""
    try:
        import psutil
    except ImportError:  # the S0 step failed before it could start a child
        return None, []
    markers = MARKERS + (str(output.resolve()), str(output))
    me = os.getpid()
    found = {}
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            if process.pid == me:
                continue
            line = ' '.join(process.info['cmdline'] or [])
            if any(marker in line for marker in markers):
                found[process.pid] = process
                for child in process.children(recursive=True):
                    found[child.pid] = child
        except psutil.Error:
            continue
    return psutil, list(found.values())


def contain(output: Path) -> list[int]:
    psutil, processes = stray_processes(output)
    if not processes:
        return []
    for process in processes:
        try:
            process.kill()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(processes, timeout=30)
    return [process.pid for process in alive]


def read_binding(path: Path):
    """The producer's single path/SHA pair, or None."""
    if not path.is_file():
        return None
    rows = [line for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
    pairs = dict(line.split('=', 1) for line in rows if '=' in line)
    if len(rows) != 2 or set(pairs) != {PATH_ENV, SHA_ENV} or not all(pairs.values()):
        return None
    return pairs


def report(outcome: str, binding: Path, output: Path, github_env: Path, summary: Path | None) -> int:
    alive = contain(output)
    pairs = read_binding(binding) if outcome == 'success' else None
    if outcome == 'success' and pairs is None:
        outcome = 'success-without-binding'
    lines = []
    if pairs is not None:
        lines = [f'{PATH_ENV}={pairs[PATH_ENV]}', f'{SHA_ENV}={pairs[SHA_ENV]}', f'{REQUIRE_ENV}=1']
    else:
        lines = [f'{UNAVAILABLE_ENV}={outcome}']
        print(f'::warning title=S0 baseline {outcome}::No frozen-source S0 baseline in this leg; '
              'tests that need it are skipped with this reason, all other tests run. '
              'See the s0-cold artifact and docs/ci-s0-baseline.md.')
    with github_env.open('a', encoding='utf8') as stream:
        stream.write(''.join(line + '\n' for line in lines))
    if summary is not None:
        with summary.open('a', encoding='utf8') as stream:
            stream.write(f'### S0 frozen-source baseline: {outcome}\n')
    if alive:
        print(f'::error title=S0 process containment::S0 processes survived cleanup: {alive}')
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outcome', required=True)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--github-env', type=Path, required=True)
    args = parser.parse_args(argv)
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    return report(args.outcome or 'skipped', args.binding, args.output, args.github_env,
                  Path(summary) if summary else None)


if __name__ == '__main__':
    raise SystemExit(main())
