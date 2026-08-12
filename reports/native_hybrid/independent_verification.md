# Independent Verification Ledger

Status: partial, not a closeout

Date: 2026-08-13

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
| ANYfem schema-6 backend migration | 40 passed | Recursive schema 1-5 migration, schema-6 canonical selector, session snapshot, and save/reopen behavior. |
| ANYfem selector UI | 5 passed | Project selector identity, snapshot, provenance, and fail-hard UI behavior in the focused gate. |
| ANYsolver consumer surface | 3 passed | Exact `>=0.1,<0.3` source range, neutral runtime API, and order-independent installed METADATA requirement semantics. |

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

ANYgeometry's final public branch tip is
`37234b7bc6b6c3f2e02cf1c53acb875245d9c3aa`; qualified kernel/package/test code
is at parent `8828019e0f940b0d6f240b98f8be17d6f306155b`. The final handoff records:

- Full kernel suite: 389 passed in 11.94 seconds.
- Evaluator contract: 26 passed.
- Intersection/policy/radial contract: 67 passed in 7.66 seconds.
- Structural/source/radial/curved split: 45 passed in 0.25 seconds.
- Offline build and Twine checks passed.
- Isolated external-TEMP installed-wheel origin/version/typing/CLI smoke passed.
- Schema 1 through 4 reads and canonical checksummed schema 4 writes.
- Qualified face/face `CONNECT` with shared Edge and both Sheets' FaceUse/Coedge
  incidence; unsupported cases remain typed and fail closed.

## Committed solver handoff

- Activity implementation delivery: ANYsolver commit `1fd1c19`.
- SHA ledger update: ANYsolver commit `7daa6e8`.
- Consumer compatibility commits: ANYsolver `a49fee8` and `925639e`.
- Public seam and merge order: `C:\Github\ANYsolver\ACTIVITY_S4_HANDOFF.md`.

## Evidence run locally but awaiting independent acceptance

- ANYfem app-facing cancellation bridge: 4 passed in 0.98 s.
- Downstream migration matrix: ANYfem 60 passed; ANYmesh 18 passed.
- Dedicated ANYfem legacy migration module: 30 passed in 9.13 s.
- Planar native corpus and benchmark fixture: 5 passed in 2.21 s.
- Incremental structural correctness: 9 passed in 0.09 s.

## Open independent gates

- Broad ANYfem Sheet/FaceUse shell/sheet CONNECT and conformal-mesh application
  qualification against the frozen public geometry tree.
- 10k/100k/500k generation, memory, serialization, and scaling evidence.
- Like-for-like frozen-baseline/current mapped regression budgets.
- Dependency-complete installed-wheel import and platform artifact matrix in
  CI/release infrastructure. The local path was retired without qualification.
- ANYfileIO range correction and a fully hashed combined resolver/lock selecting
  ANYmesher 0.2.1; no normal resolved ecosystem claim exists yet.
- Combined activity/S4 merge and regression matrix.
- Broad migration, application/UI, edit/restart, and ecosystem regressions.
- Final report/commit inventory and independent completion review.

No `ECOSYSTEM CLOSEOUT: OK` verdict has been issued.
