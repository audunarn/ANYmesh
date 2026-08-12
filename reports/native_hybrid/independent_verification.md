# Independent Verification Ledger

Status: partial, not a closeout

Date: 2026-08-12

Independent ecosystem review is coordinated by task
`019ff655-abd9-7eb1-b94e-d80252ff9215`. The entries below record only results
that task explicitly reproduced or inspected and accepted in a bounded scope.

## Independently reproduced functional evidence

| Slice | Independent result | Accepted scope |
|---|---:|---|
| ANYmesh native public path | 21 passed | Shared mapped/native edge identity, authoritative Plane handling, fail-closed missing charts. |
| ANYfem native project path | 3 passed | Declared focused project integration. |
| ANYsolver activity | 17 passed | Activity/deletion public seam and focused integrations. |
| Incremental runtime publication | 11 passed | Declared focused publication behavior. |
| Compiled boundary and charts | 19 passed, 1 warning | Windows CPython 3.13 C++17 boundary, PEP3118 arrays, predicates, adjacency, charts, metric operations, smoothing/flips, parity. The warning was subsequently removed. |
| Intersection and persistence | 31 passed | Public kernel intersection path, fail-closed behavior, and mesh persistence cases. |
| Focused solver qualification | 61 passed | Declared solver qualification cases. |
| Full functional solver-mode matrix | 109 passed in 67.58 s | Contact, nonlinear static/limit/DNV, nonlinear diagnostics/state, dynamics, modal, and buckling modules in the exact leased command. |
| Curved native qualification | 13 passed | Cylinder, Cone, RuledSurface, CoonsSurface, reprojection, mixed elements, refinement, and quadratic midsides. |
| Runtime cache reuse | 7 passed in 1.01 s | Dirty-only generation, exact clean-publication retention, empty-resolution zero generation, and related focused runtime behavior. |
| Serial/concurrent component determinism | 2 passed in 0.87 s | Two model-bound faces under one-worker serial versus reversed two-worker scheduling. |
| Safe-phase cancellation | ANYmesh 1 passed; ANYfem 3 passed | Exception propagation after real triangulation and last-valid publication retention. |

## Independently inspected performance evidence

The coordinator inspected
`incremental_performance_results.json` and accepted its schema and internal
coherence for the recorded Windows CPython 3.13.9 run:

- 20 persistent components and 8,000 initial elements.
- One dirty component, 400 dirty elements, dirty fraction `0.05`.
- Full generation `37.870554 s`; dirty response `2.045820 s`.
- Full-over-dirty ratio `18.5112x`.
- Clean publication identity and canonical hashes retained.
- Dirty publication replaced.
- Governing `<=0.20` dirty and `>=5x` gates passed.

This is not broad scaling, peak-memory, or cross-platform evidence.

## Upstream kernel evidence recorded by independent coordination

- ANYgeometry persistence: 116 focused tests for schema v1-v4 reads,
  canonical v4 writes, deterministic migration, checksums, and fail-closed
  unverified legacy relationship evidence.
- Structural/local model contracts: 30 focused passes.
- Intersection/predicate/workflow contracts: 52 focused passes.
- Straight-extrusion Plane support and shell/sheet CONNECT: 3 focused passes
  before a later hostile review hold.
- Batch evaluator/projection hostile fixes: 26 focused passes.
- Structural attachment/radial/curved slice: 45 focused passes.

The ANYgeometry public tree remains withheld pending its final intersection and
full/package qualification. These upstream counts therefore do not yet
authorize the downstream shell/sheet rerun.

## Committed solver handoff

- Activity implementation delivery: ANYsolver commit `1fd1c19`.
- SHA ledger update: ANYsolver commit `7daa6e8`.
- Public seam and merge order: `C:\Github\ANYsolver\ACTIVITY_S4_HANDOFF.md`.

## Evidence run locally but awaiting independent acceptance

- ANYfem app-facing cancellation bridge: 4 passed in 0.98 s.
- Downstream migration matrix: ANYfem 60 passed; ANYmesh 18 passed.
- Dedicated ANYfem legacy migration module: 30 passed in 9.13 s.
- Planar native corpus and benchmark fixture: 5 passed in 2.21 s.
- Incremental structural correctness: 9 passed in 0.09 s.

## Open independent gates

- Renewed ANYgeometry `PUBLIC TREE READY`, final SHA, and contract ledger.
- ANYfem Sheet/FaceUse ownership plus shell/sheet CONNECT and conformal mesh
  qualification against that frozen tree.
- 10k/100k/500k generation, memory, serialization, and scaling evidence.
- Like-for-like frozen-baseline/current mapped regression budgets.
- Isolated installed-wheel import and platform artifact matrix.
- Combined activity/S4 merge and regression matrix.
- Broad migration, application/UI, edit/restart, and ecosystem regressions.
- Final report/commit inventory and independent completion review.

No `ECOSYSTEM CLOSEOUT: OK` verdict has been issued.
