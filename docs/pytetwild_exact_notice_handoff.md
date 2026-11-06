# Exact1 provider notice handoff

State: **notice/source/relink handoff implemented; root independent byte/RECORD audit accepted notice packaging only**.
No publication or distribution-readiness claim. Root audit is retained in
`evidence/provider-notices-20261028-a1/root-review.json` (all four wheels,
zero provider imports/native calls). Its exact supervised command/cleanup receipt
is `evidence/provider-notice-root-review-14959fd85f`, 0.515000000014 seconds.
The four previously qualified wheels and all historical evidence remain immutable.

Plan and acceptance:

1. Estimate the selected corresponding-source payload before archiving; refuse
   more than 1 GiB. Export patched wrapper and pinned build sources without Git,
   build outputs or unrelated datasets. Record every selected file and hash.
2. Export per-ABI application/static relink inputs and effective configuration,
   with portable relative paths and recipient commands. Distinguish linker
   inputs from archive members whose final retention was not measured.
3. Add licenses, component notices and source/relink directions to new wheels.
   Only `.dist-info` notice/license/METADATA/RECORD members may change. Verify
   every original runtime member, package version, tags and regenerated RECORD.
4. Run focused refusal/byte-preservation checks and, within the fresh 900-second
   process budget, at most one isolated import/metadata smoke per ABI.

Open constraints: source/relink recipient usability, fixed-SHA MPIR source
availability, and the redistribution-document basis for the actual Microsoft
DLLs. Notice packaging alone cannot establish these or provider promotion.
Root reviews the implementation and payload and owns final adjudication/Git.

## Verified outputs

Evidence root: `../evidence/provider-notices-20261028-a1`. Final artifacts are
`wheels-final/`, `payload-final/`, `proofs-final/` and the canonical POSIX index
`portable-artifact-manifest.json`. Earlier packaging and failed attempts remain
unchanged. `payload-final/README.md` gives recipient source/relink directions;
each `payload-final/relink/<ABI>/` contains application/static inputs, effective
historical configuration, input hashes and a relative `relink.sh`/`relink.cmd`.
Commands are supplied for recipient use and were **not** executed here.

Selected source: 5,839 files / 89,125,560 bytes. Expanded MPIR source adds
26,525,283 bytes: total 115,650,843 bytes, below 1 GiB. The source archive has
5,840 outer members and SHA256
`394bec49e0167059ecd3c88574640c989829bf2976413f304b98b77b63ab8272`.
Git metadata, build outputs and unrelated datasets are excluded. Source files
are mapped to archive-relative paths; available historical source hashes are
checked before export. Retained source snapshots, patches and dependency pins
are included. MPIR 3.0.0 was fetched from Buildroot's source mirror and matches
the retained recipe SHA256
`52f63459cf3f9478859de29e00357f004050ead70b45913f2c2269d9708675bb`.
Restricted-network failures remain under `cache/`; verified acquisition is
recorded under `cache-network/` and reused by `cache-final/`.

All four refreshed wheels preserve every original runtime member byte, DLL
name/bootstrap, version and WHEEL tag. RECORDs are regenerated and verified.
GPLv3/LGPLv3, linked nanobind's license and retained component/MS documents are
present; fetched-source notices are distinguished from actual direct linker
targets and unmeasured archive-member retention.

## Verification and failures

Exact commands, interpreter identities, runner/supervisor hashes, wall times
and cleanup receipts are in `budget.json` and `processes/`.

- `runtime/Scripts/python.exe -B tools/package_exact_provider_notices.py run checks
  --revision checks-repair`: **11 passed**, zero failures/errors/skips.
- `runtime/Scripts/python.exe -B tools/package_exact_provider_notices.py run audit
  --revision final`: portable manifest, source limit, Git/build-output exclusion,
  MPIR inclusion, complete GPLv3/nanobind notices and artifact hashes pass.
- Four isolated import/metadata smokes pass, one actual provider import per ABI:
  Windows311, Windows312 ABI3, Linux311 and Linux312 ABI3. Linux uses the existing
  Python3.12 controller/pidfd supervisor with the appropriate child interpreter.
  Origins, native hashes, package version and compiled exact-envelope marker
  match. No PyVista import or native meshing occurred.

An earlier 10-test run also passed. A later pytest invocation rejected an
accidentally forwarded revision argument before tests ran; its receipt remains.
Two restricted WSL launches failed before Python started, the Linux311
Python3.11 controller lacked pidfd support, and the first Linux312 child failed
on Windows separators before provider import. Root expressly authorized those
harness repairs and the first actual Linux imports. Windows imports were not
repeated. See `launcher-failures.json`; outer failed-launch timing is identified
as tool-reported wall, not reconstructed child timing.

Supervised child-process usage: **76.543884135/900 seconds**. The two denied WSL
launches add a separately identified 0.6800592-second budget charge. Native
meshing, C++ compilation and relink executions: **0**.

## Remaining constraints

Recipient rebuild/relink commands have not been executed. Original linker
inputs and compiled objects are distinguished from exact final retained
symbols/headers, which were not measured. The actual Microsoft documents and
bundled-DLL origins are retained, but applicable redistribution conditions have
not been adjudicated. These are explicit technical/release limits, not an
invented additional approval ceremony. No distribution-ready/provider promotion
or broader numerical acceptance is inferred from this notice-only refresh.
