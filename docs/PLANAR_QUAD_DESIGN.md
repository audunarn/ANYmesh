# ANYmesher Genuine Planar Quad Production Design (PQ0 freeze)

## Baseline

- Commit: `68341621392b5f3a558c2f2735652ded0aca26e9`
- Branch: `opencode/planar-quad-v1`

## Current production reality

The explicit quad-first path is a fixed demonstration, not a general production driver.

- Each selected explicit quad-first face must currently be an untrimmed planar four-corner face, and each is independently reduced to one Q4; multi-face selection/publication exists, but resolution remains one Q4 per selected quad-first face.
- It seeds only the four corner nodes and two T3 seeds.
- It performs a single `front_step` and publishes one Q4.
- `target_size` is validated but does not control production topology or resolution.
- The Q4 production call is a fixed `CountInstance` demonstration.
- The Q5 production call is a fixed synthetic `PatchSpec`.
- The front logic relies on total-incidence assumptions that do not represent a Q4/T3 residual-region interface.
- Protected endpoint rejection conflates a fixed identity with forbidden element participation.

None of the above is a claim that PQ-M1 is implemented. It documents the starting reality.

## Frozen invariants

These invariants are binding for the production design and must not be violated.

- `ANYgeometry` is authoritative for geometry, topology, and ownership.
- No coordinate welding is performed.
- Source geometry is immutable and revision-checked.
- Canonical boundary station identity is reused across reverse shared-edge uses.
- A resident state contains active accepted Q4 plus finalized T3 that exactly partition the domain, with no helper T3 under quads.
- Fixed nodes may participate but cannot move or be deleted.
- Protected segments cannot be removed or crossed.
- All operations are deterministic, bounded, and cancellable.
- Explicit quad-first never silently succeeds as legacy-only.
- `quad_options=None` preserves legacy behavior.

## Frozen ID semantics

Correct identity handling is required to prevent accidental conflation.

- Source geometry vertex/edge/face IDs are owner identities.
- Mesh node IDs are generated topology IDs.
- Canonical boundary stations are owner-edge plus exact station identities; they are neither source vertex IDs nor mesh node IDs.
- Shared conformity must be by owner/station identity, never by coordinate matching.

## Capability table

| Area | Status | PQ0 assessment |
| --- | --- | --- |
| Resident mixed T3/Q4 state | Existing | Usable substrate for planar quad work |
| Atomic local transaction machinery | Existing | Reuse for local updates; do not rebuild |
| Front pairing | Partial | Needs corrected residual-region interface |
| Recovery / insertion | Existing | Requires corrected front/protection semantics |
| Cross-field guidance | Existing | Present but not wired end to end |
| Transitions | Existing | Present but not a general production driver |
| MCF worker | Existing | Kernel exists; missing geometry-derived use |
| TinyAD | Existing | Kernel exists; missing actual-mesh use |
| Public planar route | Needs correction | Current path is a fixed demonstration |
| Target-size-driven seed | Missing | Not present in production today |
| Holes / concavity | Missing for public route | Out of scope for PQ-M1 |
| Curved / higher-order | Out of scope | Explicitly out of PQ-M1 |

## Architecture direction (PQ0 to PQ3)

The production design proceeds in ordered phases.

### PQ0 â€” Freeze contract

- Freeze invariants and ID semantics as stated above.
- Define the capability table and the scope boundary.
- Establish the contract that later phases must satisfy.

### PQ1 â€” Correct resident semantics and locality

- Correct the front/residual-region interface and pairing.
- Fix protected endpoint semantics so identity and participation are distinct.
- Ground recovery and insertion in the corrected local transaction machinery.

### PQ2 â€” Planar domain, stations, size-aware constrained seed

- Introduce a size-aware constrained seed for the planar domain.
- Use canonical boundary stations as the conformity reference.
- Preserve the active accepted Q4 plus finalized T3 partition without helper T3.

### PQ3 â€” Finite size-driven front driver, independent validation, public dispatch

- Replace the fixed `front_step` demonstration with a finite size-driven front driver.
- Add independent validation of the produced resident state.
- Wire the public dispatch path so quad-first becomes a real production route.

### Orchestration glue

`hybrid.py` remains the orchestration glue. Production work must evolve
`QuadMeshState`; it must not duplicate the topology engine.

## PQ-M1 boundary

PQ-M1 scopes the first production milestone precisely.

- In scope: single planar face first.
- Deferred unless naturally enabled: holes, concavity, graded refinement, multiface production.
- Out of scope: curved elements, higher-order elements, anisotropy.

## Non-claims

## PQ-M1 implementation status (PQ3)

PQ-M1 is implemented on the supported planar scope. The PQ0 baseline statements
above remain historical starting-point evidence; they no longer describe the
explicit public quad-first route.

- The public route captures one canonical `BoundaryStationRegistry` for all
  selected planar domains, builds a fresh target-size PQ2 seed per face, runs a
  bounded deterministic front driver, independently validates the resident
  result, then publishes every final Q4 and residual T3.
- `target_size` causally controls topology. On P01, `h=1.0/0.5/0.25` produces
  exactly `60/240/960` Q4 with no residual T3 and exact area closure.
- Boundary station identity, not coordinate welding, controls shared-edge
  conformity. Complete intrinsic source-edge station chains are published in
  `nodes_of_edge`; exact endpoint station provenance supplies `node_of_vertex`.
- Recovery on a Q4/T3 active-front interface is conforming: a Steiner split is
  represented on both sides by a bounded local re-tile in one transaction, so
  no hanging node is published. Failed candidate re-tiles roll back without
  consuming resident IDs.
- At PQ-M1 the historical Q4 MCF and Q5 TinyAD adapters remained direct-only.
  PQ5 activates the existing TinyAD worker on bounded real Q4 patches after the
  front driver. PQ6 now activates Q4 MCF earlier, on the real constrained T3 seed
  before the front driver; no fixed/demo `CountInstance` is part of the public
  path. Final validation/publication remains after the front and Q5 stages.
- `quad_options=None` remains the legacy dispatch sentinel. Curved surfaces and
  higher-order elements remain outside the qualified planar scope.
- PQ4a separately qualifies simple planar exterior loops beyond four corners,
  single and multiple interior holes, and concavity. Canonical source-edge station
  identity is retained on every outer and hole loop; analytic planar boundary
  curves are sampled at the exact geometry stations, while area closure is
  validated against the resulting authoritative station/chord domain.
- PQ4b qualifies the remaining staged planar scope. The explicit route constructs
  one existing `SizeField` from the public `Refinement` set and shares it between
  canonical boundary seeding and each face seed. The uniform branch remains
  byte-compatible with PQ4a; nonuniform interiors retain the coarse target-size
  lattice and add bounded deterministic fine candidates only where the local size
  field is smaller. Shared-edge identity still comes exclusively from one
  `BoundaryStationRegistry`, never coordinate welding. Qualified-S3 preparation is
  applied after quad-first publication (or after the mixed beam merge); all-Q4
  meshes report the established no-triangle status, while residual T3 retain the
  existing Sheet/FaceUse owner-authority requirement. Compiled triangulation is
  qualification-equivalent to Python on the staged seed corpus but is not required
  to replace the deterministic Python seed backend.


### PQ5 — bounded actual-mesh TinyAD optimization

PQ5 is a coordinate-only quality stage on the resident mesh produced by the
front driver. It does not change topology, ownership, station identity, or the
source geometry.

- Eligible centers are unprotected interior nodes incident only to Q4 cells.
  The optimization patch contains every Q4 incident to that one free center;
  protected/boundary nodes are fixed and residual-T3-touching centers are
  excluded in this tranche.
- Candidate ordering is deterministic. Pure-Python patch energy ranks the
  candidates; at most `min(max_local_optimizations, 8)` TinyAD worker calls are
  allowed over one public execution.
- A worker result is committed only when the independently validated TinyAD
  response is `CONVERGED` with a strict objective decrease and the staged
  resident patch remains positively oriented. The transaction then moves only
  the center coordinate. `NOIMPROVE` leaves the resident state unchanged.
- Public grading shares the existing `SizeField`; the local desired size is used
  for the patch where a domain/field is available. Uniform routes preserve the
  previous target-size behavior.
- `max_local_optimizations=0` is an explicit `DISABLED` state. An unavailable
  worker is `UNAVAILABLE_SKIPPED` and leaves coordinates untouched; worker
  crashes, malformed replies, and invalid solutions are not hidden.
- Final planar validation runs after TinyAD, so any accepted move must still
  satisfy the existing incidence, area-closure, positivity, and station gates.

The exact acceptance evidence and final gate results are recorded in
`reports/quad_first/planar-production/WORK_STATUS.md`.

### PQ6 - geometry-derived seed-stage MCF count planning

PQ6 integrates the qualified integer MCF worker as a bounded topology stage on
the actual constrained planar T3 seed, before the front driver.

- Candidate arcs are geometry-derived pairs of resident T3 cells sharing one
  removable diagonal whose union is a strict canonical Q4. A protected shared
  diagonal is never consumed; protected endpoints may still participate.
- Connected candidate components are classified deterministically. Production
  caps are four solved components per face, 12 cells per component, and 12
  admissible pair arcs. Large/bounded, component-cap, unbalanced, non-bipartite
  and infeasible skips are reported explicitly together with component sizes and
  arc counts.
- Each eligible balanced bipartite component produces a real integer
  `CountInstance` with unit supplies/demands and deterministic nonnegative costs
  derived from the generated geometry/local desired size. Non-candidate cross
  arcs are blocked. `CountRejected` is an internal modelling error and
  propagates; only `CountInfeasible` is a nonfatal component skip.
- All component solves complete before topology mutation. Selected disjoint pairs
  are applied in one transaction, replacing two T3 by one Q4 per selected pair.
  The report records allocated Q4 IDs and resident generation before/after; a
  successful nonempty application advances generation exactly once. Worker
  unavailability, cancellation and infeasibility consume no resident IDs.
- The public diagnostics aggregate the real per-face MCF status and provenance.
  The existing front driver then operates on the residual T3 region, Q5 follows
  unchanged, and the independent final validator remains authoritative.

The qualified skew witness starts from 78 T3. Its candidate graph contains a
71-cell/91-arc component that is bounded out and one eligible 6-cell/5-arc
component. One worker call selects three pairs, producing 3 Q4 + 72 T3 in one
generation before the front driver. The final mesh remains 38 Q4 / 2 T3 with
exact area ratio 1.0, while front attempts fall from 44 with Q4 MCF disabled to
41 with it enabled and the final Q4 body set changes. Uniform P01 at `h=0.5` is
a bounded no-op witness: its 480-cell/688-arc candidate component is skipped,
zero Q4 worker calls occur, and the established 240 Q4 / 0 T3 final mesh is
unchanged.
