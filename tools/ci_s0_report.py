"""Report the CI S0 baseline outcome and isolate it from the test legs.

Runs after the (non-blocking) S0 step in both full CI profiles:

1. Containment: any process still running from the S0 producer is killed,
   and the step fails if one survives.  Ownership is established at launch:
   the S0 step's environment carries ``ANYMESHER_S0_OWNER=<token>``, which
   every descendant inherits (also one reparented after its parent exited);
   the script name and the output directory on a command line are matched
   as well.  A stray S0 process must not share the runner with pytest.
2. Binding: only after a successful S0 outcome, and only when the producer
   wrote exactly one path/SHA pair to its binding file whose path is a file
   inside the output directory and whose seal is that file's lowercase
   hexadecimal SHA-256, are that pair and
   ``ANYMESHER_REQUIRE_S0_BASELINE=1`` exported.  Otherwise
   ``ANYMESHER_S0_BASELINE_UNAVAILABLE=<reason>`` is exported, no binding is
   visible to pytest, and the tests that need the baseline skip with that
   reason.
3. The outcome is written to the job summary, and a non-success as a warning.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path
import sys

PATH_ENV = 'ANYMESHER_S0_BASELINE'
SHA_ENV = 'ANYMESHER_S0_BASELINE_SHA256'
REQUIRE_ENV = 'ANYMESHER_REQUIRE_S0_BASELINE'
UNAVAILABLE_ENV = 'ANYMESHER_S0_BASELINE_UNAVAILABLE'
OWNER_ENV = 'ANYMESHER_S0_OWNER'
MARKERS = ('ci_s0_baseline.py',)


def stray_processes(output: Path, owner: str | None = None):
    """Live S0 processes: owner token in the environment, or S0 markers on the
    command line, with their descendants."""
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
            owned = False
            if owner:
                try:
                    owned = process.environ().get(OWNER_ENV) == owner
                except psutil.Error:
                    owned = False
            if owned or any(marker in line for marker in markers):
                found[process.pid] = process
                for child in process.children(recursive=True):
                    found[child.pid] = child
        except psutil.Error:
            continue
    return psutil, list(found.values())


def contain(output: Path, owner: str | None = None, rounds: int = 20) -> list[int]:
    """Kill S0 processes until a fresh scan finds none; return the survivors.

    An owned process can start a child between a scan and its kill, so the
    scan repeats after every kill round; only an empty fresh scan counts.
    """
    import time

    alive = []
    quiet = 0
    for _ in range(rounds):
        psutil, processes = stray_processes(output, owner)
        if not processes:
            # Two consecutive empty scans a second apart: a child started just
            # before a scan by a parent that then exited is seen by the next one.
            quiet += 1
            if quiet == 2 or psutil is None:
                return []
            time.sleep(1.)
            continue
        quiet = 0
        for process in processes:
            try:
                process.kill()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(processes, timeout=30)
    psutil, processes = stray_processes(output, owner)
    return sorted({process.pid for process in alive} | {process.pid for process in processes})


def read_binding(path: Path, output: Path):
    """The producer's single validated path/SHA pair, or None."""
    try:
        rows = [line for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError):
        return None
    pairs = dict(line.split('=', 1) for line in rows if '=' in line)
    if len(rows) != 2 or set(pairs) != {PATH_ENV, SHA_ENV}:
        return None
    seal = pairs[SHA_ENV]
    if not re.fullmatch(r'[0-9a-f]{64}', seal):
        return None
    try:
        # The producer binds exactly <output>/manifest.json: the exported path
        # must be that path itself (no alias), a regular file and no symlink,
        # inside the real output directory.
        given = Path(pairs[PATH_ENV])
        expected = output / 'manifest.json'
        if given.is_symlink() or expected.is_symlink() or output.is_symlink():
            return None
        if os.path.normcase(str(given.absolute())) != os.path.normcase(str(expected.absolute())):
            return None
        manifest = given.resolve(strict=True)
        if not manifest.is_file() or not manifest.is_relative_to(output.resolve(strict=True)):
            return None
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != seal:
            return None
    except Exception:  # any malformed path (OSError, RuntimeError on loops, ...) is unavailable
        return None
    return pairs


def report(outcome: str, binding: Path, output: Path, github_env: Path, summary: Path | None,
           owner: str | None = None) -> int:
    alive = contain(output, owner)
    pairs = read_binding(binding, output) if outcome == 'success' else None
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
        try:
            with summary.open('a', encoding='utf8') as stream:
                stream.write(f'### S0 frozen-source baseline: {outcome}\n')
        except OSError as error:  # reporting must never block the leg
            print(f'::warning title=S0 summary::could not write the job summary: {error}')
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
    parser.add_argument('--owner', help='the ANYMESHER_S0_OWNER token given to the S0 step')
    args = parser.parse_args(argv)
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    return report(args.outcome or 'skipped', args.binding, args.output, args.github_env,
                  Path(summary) if summary else None, args.owner)


if __name__ == '__main__':
    raise SystemExit(main())
