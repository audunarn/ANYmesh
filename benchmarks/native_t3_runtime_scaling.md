# Incremental T3 runtime milestone

Status (2026-09-06): **cache/queue submilestone qualified; full foundation
milestone remains open**. Independent review accepted the bounded measurements
and recorded resource request `ab55ebcb455b460c9176bdda7160a642` as
`COMPLETED_PASS`. This is not release or ecosystem closeout.

## Implemented

- Per-cell immutable coordinates and exact location tolerance, retaining the original ordered predicate scan.
- Cached triangle angles and metric-dependent quality qualification.
- Persistent deterministic work heap with non-reused tokens, stale-entry pruning and consumed-survivor reactivation.
- Geometry cache reuse through retained incidence snapshots; rejected edits retain the prior published topology.
- Writable detached candidate coordinates for experimental metric callbacks.
- Original positive queue-admission predicate, including non-admission of NaN severity.

Defaults, versions, output diagnostics and topology contracts are unchanged.
This is not a spatial-index implementation: location and metric/gradation scans remain global.
Incidence snapshot update and cache retention also still scan existing records.

## Focused qualification

On Windows, Python 3.13, with the separately built optional native extension:

`build/native-t3-owned-connectivity-20260906/lib/anymesher/_native.cp313-win_amd64.pyd`

The combined runtime/storage/incidence/frontal, selected foundation,
planar/curved corpus and benchmark-authorization suite passed:
**83 tests in 4.12 seconds**. No test was skipped in that run.

The insertion-bearing frontal fixture compares actual points, triangles, protected
segments and diagnostics with a fresh geometry/queue rebuild. It requires
positive insertion count; a seed-only early exit cannot satisfy this regression.

These are focused correctness results, not a performance claim or full CI result.

## Performance request

Original request `123717860b46472bb527a057dd5f138e` is preserved as rejected
pre-execution evidence. Independent review found an approval-ledger column
mismatch. The corrected parser has five passing behavioral format fixtures.

Successor request: `ab55ebcb455b460c9176bdda7160a642`.
Status: **COMPLETED_PASS, independently accepted**. The approved command ran
exactly once; all six workers exited zero without timeout and the lock was
released. No benchmark retry or cleanup occurred.

The exact command is registered in:

`C:/Github/.resource-manager/requests/ab55ebcb455b460c9176bdda7160a642.json`

It requires administrator approval and the workspace-wide exclusive lock.
No request or benchmark script grants that approval.

The new `native_t3_runtime_scaling.py` harness compares reference rebuild and
incremental modes in separate processes at 10k, 100k and 500k triangle records.
It performs initialization, one warmup and seven measured updates, changing
metric inputs and exercising queue consumption/reactivation. It preserves
decision signatures, raw worker output, exit information, source identities,
timings and peak RSS in a fresh directory. It has a shared 20-minute deadline.

This is explicitly a **cache/queue microbenchmark**, not actual-front scaling.
The benchmark source passed AST/compile and the approved workload completed.

## Accepted measurements

Windows 11, Python 3.13.9, NumPy 2.4.3; one warmup and seven measured updates
per mode, separate serial processes. Two percent of triangle records change
metric inputs between rounds; geometry is fixed.

| Triangle records | Rebuild median | Incremental median | Time ratio | Peak-RSS ratio |
| --- | --- | --- | --- | --- |
| 10,000 | 0.355764 s | 0.011985 s | 0.033688 | 0.995437 |
| 100,000 | 3.640997 s | 0.147710 s | 0.040569 | 1.029771 |
| 500,000 | 18.814486 s | 0.887005 s | 0.047145 | 1.031631 |

All refreshed priority signatures and the first eight pops per round match.
Incremental quality evaluation touches 2% of records, with 98% cache hits.
The 500k incremental process peak RSS was 665,948,160 bytes, compared with
645,529,600 bytes for the rebuild reference. Timing includes signature hashing
and consumption; RSS includes initialization. These are not full-mesh speedups.

Preserved evidence directory:

`benchmarks/results-t3-runtime-ab55ebcb455b460c9176bdda7160a642/`

Terminal `result.json`: 11,542 bytes, SHA-256
`147F217D72081B8AC589E44FC781A0EA3BEE0A0DD0A12F6360C31C18CE0A5CFC`.

The result binds four measured source identities and preserves all raw worker
outputs and exit receipts. Independent review confirmed source stability,
raw/result equality, measurement counts, ratios and released resources.

## Remaining acceptance gates

- CLOSED: administrator-approved cache/queue scaling measurements at all three scales.
- Actual insertion-bearing end-to-end Frontal-Delaunay scaling, including the
  registered runtime and memory ceilings.
- Mapped zero-use runtime/RSS comparison.
- Source and installed-wheel cross-platform corpus and normal CI.
- Persistent spatial location and removal of remaining full-mesh per-edit work,
  where profiling demonstrates it is required.

The open gates cannot be inferred from the microbenchmark. Actual-front
qualification must perform nonzero insertion/refinement work and compare against
a frozen corrected legacy baseline at matched resolution. At 100k, its median
must meet the registered 1.25x legacy bound and the native-v2 2x peak-RSS ceiling.
Mapped zero-use acceptance requires the registered 3% runtime and 5% RSS limits.

The existing mapped comparator is not suitable for immediate closeout: its
wrapper pins an older baseline, requires clean source worktrees and performs
worktree cleanup. It was inspected, not executed. No baseline was fabricated,
no current source was represented as a clean commit, and no normal CI or
installed-wheel/platform result is claimed here.

Next implementation/qualification work is the insertion-bearing end-to-end
profile and remaining global per-edit phases, followed by the frozen mapped
comparison and source/wheel platform matrix. These remain task-owned work,
not waived or silently converted into completed gates.

Do not infer completion of the quad-front milestone or release qualification
from this incremental runtime milestone.
