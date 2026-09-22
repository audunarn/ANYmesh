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

### PQ0 — Freeze contract

- Freeze invariants and ID semantics as stated above.
- Define the capability table and the scope boundary.
- Establish the contract that later phases must satisfy.

### PQ1 — Correct resident semantics and locality

- Correct the front/residual-region interface and pairing.
- Fix protected endpoint semantics so identity and participation are distinct.
- Ground recovery and insertion in the corrected local transaction machinery.

### PQ2 — Planar domain, stations, size-aware constrained seed

- Introduce a size-aware constrained seed for the planar domain.
- Use canonical boundary stations as the conformity reference.
- Preserve the active accepted Q4 plus finalized T3 partition without helper T3.

### PQ3 — Finite size-driven front driver, independent validation, public dispatch

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
- The historical Q4 MCF and Q5 TinyAD adapters remain available for direct
  qualification, but they are not in the PQ-M1 public dataflow and are reported
  as `NOT_INTEGRATED` without being required at runtime.
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

The exact acceptance evidence and final gate results are recorded in
`reports/quad_first/planar-production/WORK_STATUS.md`.
