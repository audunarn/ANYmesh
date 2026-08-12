# Native Hybrid Mesher Core Delivery Summary

Date: 2026-08-13

Status: core source delivery ready for bounded review; ecosystem closeout not
granted.

## A. Ready core and native implementation

### Authoritative branch deliveries

- ANYgeometry final tip: `37234b7bc6b6c3f2e02cf1c53acb875245d9c3aa`.
- ANYgeometry qualified code parent: `8828019e0f940b0d6f240b98f8be17d6f306155b`.
- ANYmesh core pipeline: `97058e0a1213ba7f0da506ff1a00d4ef10093d20`.
- ANYmesh compiled triangulation/defaults: `3fb32f825416314df8ca8464bfae82b1533b3283`.
- ANYmesh authorization clarification: `574fac99db064cc447bdb3e91ff029047a3c2248`.
- ANYmesh attempt history: `955028ff6e506556a28f7be70b6259b3f5523312`.
- ANYfem incremental runtime: `b17f1d47ba79e5c04692301e96d23dd5ac5627cb`.
- ANYfem schema-6/UI handoff: `7a41baca4bd4d1a5cb538ec6148c6ca51c79d1f2`.
- ANYsolver activity implementation: `1fd1c196518ac92b9dee920676f54c2d0cf58d26`.
- ANYsolver activity SHA handoff: `7daa6e8c61954cfc1bc4469457fef0db154d3375`.
- ANYsolver consumer range: `a49fee8a9bdd9ff1f700d88e4eea5a1da457b96f`.
- ANYsolver METADATA semantic fix: `925639e35efc7883f2c4a57a51cdc407d9b6ca67`.

### Delivered capabilities

- Public ANYgeometry closure, batched evaluation/derivatives/normals/projection,
  changed-region audit, indexed overlaps, and typed intersection/imprint APIs.
- Canonical schema-4 geometry persistence with model identity, structural
  ownership, lineage, qualified relationship evidence, and fail-closed legacy
  semantics.
- Mapped/native component strategy, compact MeshCore, stable IDs and handle
  associations, shared boundary registry, native constrained triangulation,
  quad recombination, curved charts, Q8/T6 insertion, quality metrics, mesh BVH,
  structural pipeline, and damage-ready adjacency.
- Optional C++17 predicate/adjacency/triangulation boundary with strict native,
  explicit Python, and absence-only auto fallback semantics.
- ANYfem revisioned incremental sessions, cache generations, safe-phase
  cancellation, deterministic component scheduling, atomic publication,
  schema-6 backend migration, and selector UI handoff.
- ANYsolver dense activity/deletion lifecycle plus source/runtime compatibility
  with the neutral ANYmesher 0.1 and additive 0.2 API surface.

### Accepted bounded evidence

- ANYgeometry final suite: 389 passed, plus focused evaluator,
  intersection/policy/radial, structural, package, and installed-wheel gates.
- ANYmesh public/native and persistence slices: independently reproduced
  21, 31, 13, and 19-test focused results in their declared scopes.
- ANYfem runtime publication, cache reuse, two-component scheduling,
  schema-6 migration, and selector UI focused gates were independently accepted.
- ANYsolver full functional solver-mode matrix: 109 passed in the leased module
  scope; focused restart matrix: 10 passed.
- Incremental 20-component run: 8,000 elements, 5 percent dirty, 18.5112x
  full-over-dirty speedup, with clean reuse and dirty replacement gates passed.

These results qualify only their recorded scopes. They do not imply all final
acceptance criteria in the governing plan have passed.

## B. Deferred resolver, publication, and broad qualification

### Dependency and publication gates

- ANYfileIO still requires `ANYmesher<0.2`. Its owner must widen the cap before
  a normal environment can resolve ANYfem, ANYsolver, and ANYfileIO together on
  ANYmesher 0.2.1.
- The fully hashed combined resolver lock, pip report, and clean installed import
  smoke remain blocked on that owner change.
- Local Windows wheel Attempts 1 through 4 produced no accepted installed ANYmesh
  behavior bundle. The path is retired for this core slice and preserved only as
  truthful attempt history.
- CI/release must provide exact wheel tags, hashes, origins, backend behavior,
  oracle/predicate parity, non-Windows results, and publication-readiness evidence.

### Performance and regression gates

- Mapped 3 percent time, 5 percent memory, and 5 percent serialization comparison
  gates have not been executed under the approved frozen method.
- 10k, 100k, 500k, and workstation-scale generation/time/RSS/quality evidence is
  incomplete. Only the focused incremental result is qualified.
- Broad adversarial quality, nonlinear curved convergence, application/UI,
  long-running edit/restart, and full cross-repository suites remain open.
- Broad downstream Sheet/FaceUse shell/sheet CONNECT and conformal-mesh
  application qualification against the frozen geometry tree remains open.
- ANYsolver activity must be reconciled with the separately owned S4 branch and
  pass the combined regression matrix.

### Delivery and closeout gates

- ANYgeometry has published its coordinated branch; the other local delivery
  commits have not been claimed as pushed or released by this report.
- Unrelated ANYfem UI scene/viewport work and ANYgeometry untracked workflow,
  IDE, and distribution artifacts are excluded from this delivery.
- Independent completion review must verify the final commit/report inventory.
- No `ECOSYSTEM CLOSEOUT: OK` verdict has been issued.

## Completion interpretation

The repository-resident native-hybrid core is implemented and logically
committed. The ecosystem is not release-qualified. Deferred gates must remain
visible and fail closed; they may not be converted into claims by configuration,
partial local builds, or source-tree imports.
