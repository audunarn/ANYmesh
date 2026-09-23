# Curved / higher-order quad programme â€” work status

## CH0 â€” contract freeze

**Status: COMPLETE through CH0 qualification; this changeset is the CH0 milestone commit.**

Baseline: `a6c191466525b68b77f0a6cd5ad8bb4080b69243` on
`opencode/curved-quad-v1` in
`C:\Github\ANYmesh\.worktrees\curved-quad-v1`.

CH0 changes documentation and baseline-contract tests only; no
`src/anymesher` production file is changed.

### Frozen capability matrix

| Capability | CH0 status |
| --- | --- |
| Planar linear explicit quad-first | QUALIFIED through PQ6 |
| Planar explicit Q8/T6 quad-first | MISSING; CH2 target |
| Existing native cylinder route | EXISTING, route-specific |
| Existing native quadratic cylinder route | EXISTING, route-specific |
| Explicit quad-first cylinder | MISSING; CH3 target |
| Strict high-order global validity certificate | MISSING; CH1 target |
| Cone / ruled / Coons explicit quad-first | MISSING; later curved tranche |
| Q9 and orders above quadratic serendipity | DEFERRED / NO CONTRACT |
### Isolated environment evidence

- Python 3.14.2; numpy 2.4.6; ANYmesher distribution 0.5.0.
- Worktree import is the curved-quad worktree `src/anymesher`.
- ANYgeometry distribution 0.4.3; its interpreter-reported import location was
  recorded as metadata only. No sibling repository source was inspected for
  accepted CH0 evidence.
- No importable `anymesher._native` compiled extension is present in this new
  worktree, so the three compiled-parity tests are expected capability skips.
- Local Q4 worker:
  `third_party/quad/worker/out/lemon/quad_mcf_worker.exe`,
  SHA-256 `13fbb4c4961c0993bffba94584cdac687fc2cfc2d52a5415ea719588fce41d48`.
- Local Q5 worker:
  `third_party/quad/worker/out/tinyad/quad_tinyad_optimizer.exe`,
  SHA-256 `02e25212cd0b0cd53c96f0f4231eacbec5b57579704bc8255e2f76cae13b9329`.
- Both workers were built inside this worktree using the committed
  checkout-independent scripts and remain ignored outputs.

### CH0 gate evidence

```text
python -m pytest tests/quad_first_curved/test_ch0_baseline_contract.py -q
  -> 9 passed in 0.24 s

python -m pytest tests/quad_first_planar -q
  -> 63 passed, 3 skipped in 39.29 s
     skips: compiled triangulation parity because _native is not rebuilt here

python -m pytest tests/quad_first -q
  -> 275 passed in 3.77 s
python -m pytest \
  tests/test_charts.py \
  tests/test_cylindrical_metric_chart.py \
  tests/test_cylindrical_atlas_binding.py \
  tests/test_cylindrical_patch_binding.py \
  tests/test_cylindrical_public_quadratic.py \
  tests/test_cylindrical_quadratic_staging.py \
  tests/test_quadratic_boundary_prepare.py \
  tests/test_coupling.py \
  tests/test_quality_and_serialize.py -q
  -> 93 passed in 308.63 s

git diff --check
  -> clean after closeout corrections
```

### Isolation incident and correction

The initial CH0 worker briefly probed the primary checkout while diagnosing
missing ignored worker binaries, and draft documentation temporarily recorded
primary worker paths. The administrator aborted that worker. No primary file was
modified. Q4/Q5 workers were then built locally in this curved worktree and all
accepted CH0 evidence was corrected to use isolated-worktree facts only.

### Next after CH0 commit

- CH1: shared Q4/Q8/T3/T6 interpolation geometry and strict higher-order
  validity kernel.
- CH2: planar explicit quad-first Q8/T6 promotion with unique shared midsides.
- CH3, separately after CH-M1: cylindrical explicit quad-first work.

## CH1 — shared interpolation and strict high-order validity

**Status: QUALIFIED; ready for the CH1 milestone commit.**

CH1 adds `src/anymesher/quad/high_order.py` only on the production side.  It
keeps the CH0 public-route guard and the existing `quality_v2` skeleton metrics
unchanged.  Qualified families are Q4/Q8 on the reference square and T3/T6 on
the reference triangle.

The strict certificate uses conservative Bernstein/subdivision bounds for the
signed Jacobian.  Positive samples never certify; `INVALID` requires an actual
evaluated non-positive witness; budget exhaustion is `UNRESOLVED`.  Reports
carry a conservative whole-element envelope, scale-aware tolerance, witness,
subdivision/depth counters, deterministic JSON-safe serialization and bounded
cancellation checkpoints.

### CH1 gate evidence

```text
python -m pytest tests/quad_first_curved/test_ch1_high_order_validity.py -q
  -> 29 passed in 0.46 s

python -m pytest tests/quad_first_curved/test_ch0_baseline_contract.py \
  tests/test_coupling.py tests/test_quality_and_serialize.py -q
  -> 33 passed in 0.27 s

python -m pytest tests/test_cylindrical_quadratic_staging.py -q
  -> 11 passed in 0.19 s
python -m pytest tests/test_quadratic_boundary_prepare.py -q
  -> 3 passed in 0.18 s
python -m pytest tests/test_cylindrical_public_quadratic.py -q
  -> 4 passed in 277.22 s

python -m pytest tests/quad_first_curved -q
  -> 38 passed in 0.46 s
python -m pytest tests/quad_first_planar -q
  -> 63 passed, 3 skipped in 38.93 s
     skips: compiled triangulation parity because _native is not rebuilt here
python -m pytest tests/quad_first -q
  -> 275 passed in 3.72 s

git diff --check
  -> clean (line-ending warning only before documentation normalization)
```

CH1 adversarial evidence includes hidden Q8 and T6 midside inversions that the
frozen corner-only quality metrics do not see, a near-zero fail-closed case,
genuinely curved positive Q8/T6 mappings, dense independent bound checks, and a
case that progresses from `UNRESOLVED` to `CERTIFIED_POSITIVE` with a larger
bounded subdivision budget.  Input coordinates remain unchanged on normal,
budget-exhausted and cancelled certification paths.

### Next after CH1

- CH2: promote the already-qualified planar explicit topology to Q8/T6 using
  one shared midside per canonical edge; hard P01 h=0.5 target is 240 Q8,
  0 T6 and 785 nodes (273 retained corner nodes + 512 unique midsides).
- CH3 remains separate and is not opened until CH2 is complete.

## CH2 — planar quad-first Q8/T6 promotion

**Status: QUALIFIED; ready for the CH2 milestone commit.**

CH2 opens `order="quadratic"` only for the already-qualified explicit planar quad-first route. The linear topology remains authoritative, then promotion stages one midside per unique final shell edge and publishes Q8/T6 atomically only after the CH1 strict mapping-validity gate. Curved-surface quad-first remains typed unsupported; Q9 is deferred; quadratic beam/coupling output fails closed because B3 ownership is not qualified in CH2.

### Product evidence

- P01 h=0.5: linear `273 nodes / 240 Q4 / 0 T3`; quadratic `785 nodes / 240 Q8 / 0 T6`; `512` unique midsides, `64` exact source-boundary midsides, `0` repairs. All original node IDs/coordinates, shell IDs, corner connectivity, face ownership and source geometry are retained.
- Residual trapezoid h=0.75: linear `40 nodes / 26 Q4 / 6 T3`; quadratic `111 nodes / 26 Q8 / 6 T6`; `71` unique midsides, `20` exact boundary midsides, no repair. Q8/Q8 and Q8/T6 interfaces reuse the same midside IDs and all final mappings certify positive.
- P03 circular hole h=0.5: `795 nodes / 233 Q8 / 10 T6`; `519` unique midsides and `76` exact source-boundary midsides. Added hole-boundary midsides are exact source-arc samples and stay at radius `0.9` about `(3.2, 2.4)` within `1e-10`. Three unprotected interior opposite corners require bounded T6 repair; maximum displacement is `0.12266702535935825`, below the `0.5*h` cap. Protected boundary/station nodes and source geometry do not move.
- P07 local grading h=1.0: `275 nodes / 82 Q8 / 2 T6`; `179` unique midsides and `24` exact boundary midsides, with the qualified linear graded corner topology preserved.
- Promotion is idempotent and staged; a cancellation at the final ready checkpoint leaves a supplied linear mesh unchanged.

### CH2 gate evidence

```text
focused CH2
  -> 15 passed in 16.39 s
CH0 + CH1
  -> 38 passed in 0.47 s
PQ3/PQ4a/PQ4b/PQ5/PQ6 + Q6/Q7 regression bundle
  -> 84 passed, 3 skipped in 32.00 s
     skips are the known compiled-triangulation parity capability skips
full tests/quad_first_curved
  -> 53 passed in 16.39 s
full tests/quad_first_planar
  -> 63 passed, 3 skipped in 39.21 s
full tests/quad_first
  -> 275 passed in 3.69 s
serialization / quality / coupling / quadratic-staging consumers
  -> 35 passed in 0.26 s
```

Fresh review confirmed exact source-edge midpoint ownership, one canonical midside per unique final shell edge, strict Q8/T6 validity, stable topology IDs, atomic cancellation, fail-closed mixed-order beam handling, unchanged `quality_v2`, and bounded edge-linear promotion work. No CH3 cylinder tranche is opened by CH2.
## CH3 — owner-qualified cylindrical linear quad-first

**Status: QUALIFIED; ready for the CH3 milestone commit.**

CH3 activates the public explicit quad-first route for owner-qualified `Cylinder` faces with `order="linear"`. It reuses `_cylindrical_public.prepare_bindings`, the qualified physical cylindrical chart, and the existing boundary/seed/Q4/Q5/front/validator/publication dataflow. No cylindrical topology fork or coordinate welding was added. Cylindrical quadratic remains fail-closed for CH4.

### Product evidence

- pi/4 sector, h=0.5: `17 nodes / 8 Q4 / 2 T3`, `N_eq=9`; 14 front attempts, 8 guided accepts.
- same sector, h=0.25: `43 nodes / 30 Q4 / 2 T3`, `N_eq=31`; 36 front attempts, 30 guided accepts.
- full eight-sector ring, h=0.5: `88 nodes / 64 Q4 / 16 T3`, `N_eq=72`; the sector-0/sector-7 physical seam reuses one exact source-station node chain and repeat counts are deterministic.
- all published shell nodes in the unit-cylinder fixtures satisfy radial residual <= `1e-10`; source geometry is unchanged. Cancellation at `quad-first:face-seed` publishes no partial result.

### CH3 gate evidence

- focused CH3 product: `5 passed`.
- CH0 + CH1 + CH2 focused regression: `53 passed`.
- cylindrical metric chart + atlas binding + patch binding + frontal integration: `15 passed`.
- PQ3/PQ4a/PQ4b/PQ5/PQ6 + Q6/Q7 public regression bundle: accepted without CH3 regression.
- full `tests/quad_first_curved`: `58 passed`.
- full `tests/quad_first_planar`: `63 passed, 3 skipped` (known compiled-triangulation capability skips in this worktree).
- full `tests/quad_first`: `275 passed`.
- `git diff --check`: clean after documentation update.

Fresh review confirmed owner binding is qualified once before per-face capture, source-edge/station identity survives periodic publication, the source model is read-only, cancellation is atomic, the existing front driver remains the only quad topology engine, target size changes real cylinder topology, and no cylindrical Q8/T6 or other CH4 scope is accidentally opened.

## CH4 — owner-qualified cylindrical quadratic promotion

**Status: QUALIFIED; ready for the CH4 milestone commit.**

CH4 promotes the accepted CH3 cylindrical linear Q4/T3 topology to Q8/T6 using the CH2 staged promotion and CH1 strict validity kernel. Source-boundary midsides remain exact owner-edge midpoint samples; cylindrical interior midsides are physical-chart midpoints lifted through the already-qualified `CylindricalQuadDomain`. One canonical shell edge owns one midside and periodic source seams retain exact station identity.

### Product evidence

- pi/4 sector h=0.5: `17 nodes / 8 Q4 / 2 T3` -> `43 nodes / 8 Q8 / 2 T6`, `26` added unique midsides.
- pi/4 sector h=0.25: `43 nodes / 30 Q4 / 2 T3` -> `117 nodes / 30 Q8 / 2 T6`, `74` added unique midsides.
- full eight-sector ring h=0.5: `88 nodes / 64 Q4 / 16 T3` -> `256 nodes / 64 Q8 / 16 T6`, `168` added unique midsides; the periodic physical seam reuses one expanded source-edge chain and its midsides remain on radius 1 within `1e-10`.
- Every final cylindrical Q8/T6 is `CERTIFIED_POSITIVE`; linear corner IDs/coordinates, shell IDs, face ownership and source geometry are retained. Cancellation before quadratic publication is atomic.

### CH4 gate evidence

- CH0+CH1+CH2+CH3+CH4 focused: `62 passed in 29.83s`.
- cylindrical chart/atlas/patch/frontal/surface-metric/quadratic consumers and boundary preparation: `76 passed in 595.62s`.
- accepted PQ public regression bundle: `84 passed, 3 skipped in 31.80s`; skips are the existing unreconstructed compiled-triangulation capability checks.
- full curved quad-first: `62 passed in 30.07s`.
- full planar quad-first: `63 passed, 3 skipped in 42.98s`.
- full quad-first: `275 passed in 3.70s`.

Fresh review confirmed the CH3 topology remains authoritative, boundary ownership precedes cylindrical interior projection, periodic station identity is not welded by coordinates, promotion is staged and strictly certified, and CH4 does not open general curved surfaces or Q9+.
- serialization / quality / coupling / quadratic-staging consumers: `35 passed in 0.26s`.

## CH5 — high-order geometry provenance certificate

**Status: QUALIFIED; ready for the CH5 milestone commit.**

CH5 closes the CH0 reporting seam without changing accepted topology. Successful explicit quadratic quad-first promotion now publishes an immutable, JSON-safe global `HighOrderMeshCertificate` containing one `HighOrderGeometryReport` per source face plus canonical boundary/interface midside provenance. Linear quad-first reports the capability truthfully as `NOT_APPLICABLE`.

The certificate is built from the exact staged Q8/T6 coordinates/connectivity immediately after the CH1 strict final validity pass and before atomic publication. It checks model/revision and face uniqueness, per-face/global Q8/T6 counts, canonical source-edge/station midside identity, unique boundary counts and the global residual envelope. Conflicting canonical provenance or incomplete positive certification fails closed.
### Product evidence

- P01 h=0.5: `785 nodes / 240 Q8 / 0 T6`, 512 unique midsides, 64 unique source-boundary midsides, residual `0.0`, chart origin `(0,0,0)`, curvature class `straight`.
- P03 h=0.5: `795 nodes / 233 Q8 / 10 T6`, 519 unique midsides, 76 unique source-boundary midsides, residual `5.140558480280521e-13`; straight and analytic-curved source intervals are distinguished.
- Cylindrical pi/4 sector h=0.5: `43 nodes / 8 Q8 / 2 T6`, 26 unique midsides, 14 unique source-boundary midsides, residual `2.854117595420714e-13`; physical chart origin lifts to `(1,0,0)`.
- Full eight-sector ring h=0.5: `256 nodes / 64 Q8 / 16 T6`, 168 unique midsides, 72 globally unique canonical source-boundary/interface midsides, eight face reports, residual `2.8563285926006375e-13`.
- Final-checkpoint cancellation remains atomic and source geometry remains unchanged.

### CH5 gate evidence

```text
focused CH5
  -> 8 passed in 9.00 s
post-review full tests/quad_first_curved
  -> 70 passed in 38.83 s
post-review full tests/quad_first_planar
  -> 63 passed, 3 skipped in 42.11 s
     skips: known compiled-triangulation parity capability; native extension not rebuilt
post-review full tests/quad_first
  -> 275 passed in 3.67 s
pre-review cylindrical consumer bundle
  -> 69 passed in 339.84 s
```

Fresh review added the originally frozen chart-origin and edge-curvature-class semantics, strengthened global count/provenance consistency, and confirmed CH5 remains report-only: no new surface family, topology engine, option schema, Q9 path, coordinate welding or `quality_v2` change.


## CH6 — quadratic quad-first B3 ownership

**Status: QUALIFIED; ready for the CH6 milestone commit.**

CH6 removes the obsolete public blanket rejection of planar quadratic quad-first requests with straight beam/member content. The existing mapped quadratic beam generator remains the sole B3 implementation; arbitrary pre-existing B2 content still fails closed in direct shell promotion, and curved quadratic beam lines remain unsupported.

### Product evidence

- straight member-through-face, target size 1.0: `13 nodes / 1 Q8 / 0 T6 / 2 B3 / 1 coupling`;
- member edge chain: five stations, with two B3 spans and exact start-mid-end chord midpoint geometry;
- through-face coupling references the centre member station at `(0.5, 0.5, 0.0)` and retains the established shell interpolation weights;
- quadratic output contains no B2 beam body and no linear shell body;
- repeated execution is deterministic; cancellation at `quad-first:quadratic-promotion-ready` is atomic; source geometry remains unchanged;
- a curved beam edge remains a typed straight-sided-B3 rejection; cylindrical quadratic shell/beam ownership remains typed unsupported.

### CH6 gate evidence

- focused CH6 + evolved CH2 quadratic contract: `19 passed`;
- curved programme plus beam/coupling/serialization consumers: `107 passed`;
- full `tests/quad_first_curved`: `74 passed`;
- full `tests/quad_first_planar`: `63 passed, 3 skipped` (known compiled-triangulation parity capability skips);
- full `tests/quad_first`: `275 passed`.


## CH7 — cylindrical quadratic B3 ownership

**Status: QUALIFIED; ready for the CH7 milestone commit.**

CH7 qualifies coherent straight B3/member content on the CH4 owner-qualified cylindrical Q8/T6 public route. The existing mapped B3 generator remains authoritative and the structural pipeline keeps target-face ownership exact. A quadratic-only second BVH lookup is permitted when the exact owner-surface attachment lies just outside the polynomial shell map; its tolerance is bounded by the allowed target elements' midside-to-chord deviation.

### Product evidence

- full eight-sector unit-cylinder ring, h=0.5, one radial member: `261 nodes / 64 Q8 / 16 T6 / 2 B3 / 1 coupling`;
- B3 midsides equal endpoint chord midpoints within `1e-12`;
- centre member station is the exact owner-cylinder point and the coupling records a small nonzero physical projection eccentricity below `1e-3`;
- all shell nodes remain on radius 1 within `1e-10`;
- serialization preserves B3 connectivity and member station chains;
- repeat execution is deterministic, cancellation is atomic, and source geometry is unchanged.

### CH7 gate evidence

- focused CH6+CH7 plus structural preparation/pipeline, coupling and serialization: `63 passed in 13.15s`;
- full `tests/quad_first_curved`: `75 passed in 51.84s`;
- full `tests/quad_first_planar`: `63 passed, 3 skipped in 39.22s` (known compiled-triangulation capability skips);
- full `tests/quad_first`: `275 passed in 5.82s`.

Fresh review confirmed the tolerance expansion is quadratic-only, restricted to the explicitly owned target-face element set, derived from actual midside/chord curvature, and does not change linear attachment behavior. CH7 adds no topology engine, coordinate welding, new option schema, or general curved-surface family.

## CH8 — analytic conical linear quad-first

**Status: QUALIFIED; ready for the CH8 milestone commit.**

CH8 adds an exact developable `ConicalMetricChart` and `ConicalQuadDomain` for analytic ANYgeometry Cone faces, then reuses the existing target-size boundary registry, seed, MCF/Q5, front driver, validator and publication path. No second topology engine or coordinate welding is introduced. The chart and domain are model/revision bound through `FaceChart`; source geometry remains read-only.

### Product evidence

- h=0.6: `42 nodes / 28 Q4 / 6 T3`, `N_eq=31`, edge-chain lengths `[6,6,6,6]`, 52 front attempts, 24 guided accepts, 4 recovery accepts, owner residual `2.48e-16`;
- h=0.3: `118 nodes / 89 Q4 / 14 T3`, `N_eq=96`, edge-chain lengths `[11,12,11,12]`, 120 front attempts, 86 guided accepts, owner residual `3.14e-16`;
- target-size halving changes real topology (`N_eq` 31 -> 96) while the owner-developed reference area remains `7.148291545723455` and final cells close the sampled boundary polygon;
- repeat generation is deterministic; cancellation is atomic; `quad-first-conical` diagnostics and per-face `conical` family are public;
- conical quadratic remains typed unsupported for CH9, while Plane/Cylinder routes remain unchanged.

### CH8 gate evidence

- focused CH8: `7 passed`;
- CH0-CH7 focused curved contracts: `75 passed in 52.04s`;
- curved native/chart/physical-quality/serialization consumers: `43 passed in 8.58s`;
- accepted PQ public regression bundle: `84 passed, 3 skipped in 31.76s` (known native triangulation extension capability skips);
- full curved: `82 passed in 53.83s`;
- full planar: `63 passed, 3 skipped in 38.44s`;
- full quad-first: `275 passed in 3.67s`.

Fresh review confirmed the analytic unroll is locally isometric, inverse angle selection is unique within the owning sector, source/edge identity is unchanged, and CH8 opens no quadratic cone, ruled/Coons or Q9 path.

## CH9 — analytic conical Q8/T6 promotion

**Status: QUALIFIED; ready for the CH9 milestone commit.**

CH9 activates staged Q8/T6 promotion on the CH8 analytic conical linear topology. It reuses the CH1 validity kernel, CH2/CH4 canonical midside ownership and CH5 geometry-provenance certificate. No conical topology engine, coordinate welding or new option schema is introduced.

### Product evidence

- h=0.6: `42 linear nodes / 28 Q4 / 6 T3` -> `117 nodes / 28 Q8 / 6 T6`, `75` unique midsides, high-order geometry residual `2.283103467203869e-12`, owner support residual `3.1401849173675503e-16`;
- h=0.3: `118 linear nodes / 89 Q4 / 14 T3` -> `338 nodes / 89 Q8 / 14 T6`, `220` unique midsides, high-order geometry residual `2.8219887411506448e-12`, owner support residual `3.510833468576701e-16`;
- all original linear nodes, shell IDs and corner connectivity are retained exactly;
- all Q8/T6 mappings are `CERTIFIED_POSITIVE`; shared edges reuse one midside; source-edge chains preserve the CH8 stations at even positions;
- conical interior midsides use exact developed-chart midpoint lifts; boundary midsides retain source-edge parameter ownership;
- repeat generation is deterministic, cancellation is atomic and source geometry is unchanged;
- conical quadratic beam/coupling content remains typed unsupported; ruled/Coons and Q9+ remain closed.

### CH9 gate evidence

- focused CH9: `5 passed in 6.38s`;
- full curved: `87 passed in 78.19s`;
- full planar: `63 passed, 3 skipped in 40.75s` for the known native triangulation capability;
- full quad-first: `275 passed in 5.60s`;
- curved-native / quality / coupling / quadratic-staging consumers: `41 passed in 8.19s`.


## CH10 — metric parametric curved linear quad-first

**Status: QUALIFIED; implementation commit `9c97c55` plus this closeout evidence.**

CH10 activates explicit linear quad-first meshing for non-planar ANYgeometry RuledSurface and CoonsSurface faces. `ParametricQuadDomain` remains source model/revision/face bound through `FaceChart`, uses owner UV for topology, and applies a centre `J^T J` Cholesky normalization only to make chart distances approximately physical. Boundary divisions remain exact geometry/SizeField seeding. The established boundary registry, constrained seed, MCF/Q5, front driver, validator and publication path remain authoritative.

### Product evidence

- ruled h=0.6 -> `15 nodes / 8 Q4 / 0 T3`, `N_eq=8`, max physical corner edge `0.6908830462963139`, edge chains `[5,3,5,3]`, 8 attempts;
- ruled h=0.3 -> `40 nodes / 28 Q4 / 0 T3`, `N_eq=28`, max physical corner edge `0.3247592599913309`, edge chains `[8,5,8,5]`, 28 attempts;
- coons h=0.6 -> `15 nodes / 8 Q4 / 0 T3`, `N_eq=8`, max physical corner edge `0.6908830462963139`, edge chains `[5,3,5,3]`, 8 attempts;
- coons h=0.3 -> `40 nodes / 28 Q4 / 0 T3`, `N_eq=28`, max physical corner edge `0.3247592599913309`, edge chains `[8,5,8,5]`, 28 attempts;
- owner support residuals are `0.0` to `2.24e-16`; repeat generation is deterministic; cancellation is atomic; source geometry is unchanged; diagnostics are `quad-first-parametric-curved` with exact ruled/coons family labels.

Ruled/Coons quadratic output remains typed unsupported for CH11. Plane, Cylinder and Cone routes are unchanged.

### CH10 closeout gate evidence

- focused CH10: `8 passed in 0.74s`;
- chart/physical-quality/serialization/coupling consumers: `47 passed in 0.73s`;
- accepted PQ public regressions: `84 passed, 3 skipped in 31.60s`;
- full curved: `95 passed in 60.37s`;
- full planar: `63 passed, 3 skipped in 38.56s`;
- full quad-first: `275 passed in 5.51s`.

Fresh review confirmed metric normalization is reported as approximate rather than globally isometric, source edge/station ownership remains geometry-authoritative, no coordinate welding or topology fork was introduced, cancellation/source immutability remain intact, and CH10 does not activate quadratic ruled/Coons meshing.


## CH11 — metric curved Q8/T6 promotion

**Status: QUALIFIED; ready for the CH11 milestone commit.**

CH11 activates staged Q8/T6 promotion on the CH10 non-planar ruled and Coons linear routes. The existing higher-order promotion path now treats `ParametricQuadDomain` as owner-chart geometry for interior midsides; boundary midsides remain exact source-edge parameter samples. CH1 strict validity, CH5 provenance, stable shell/corner IDs, atomic publication and the single accepted quad-first topology pipeline remain authoritative.

### Product evidence

- ruled h=0.6: `15 nodes / 8 Q4 / 0 T3` -> `37 nodes / 8 Q8 / 0 T6`, 22 unique midsides, certificate residual `2.5438405243138006e-16`;
- ruled h=0.3: `40 / 28 Q4 / 0 T3` -> `107 / 28 Q8 / 0 T6`, 67 midsides, residual `2.2887833992611187e-16`;
- coons h=0.6: `15 / 8 / 0` -> `37 / 8 Q8 / 0 T6`, 22 midsides, residual `5.580527502014626e-16`;
- coons h=0.3: `40 / 28 / 0` -> `107 / 28 Q8 / 0 T6`, 67 midsides, residual `3.3766115072321297e-16`;
- locally graded h=0.6 for each family: `35 nodes / 25 Q4 / 4 T3` -> `98 nodes / 25 Q8 / 4 T6`, 63 midsides, 14 exact boundary midsides, zero repairs; all final Q8/T6 certify positive.

Quadratic ruled/Coons beam/coupling content remains typed unsupported. Plane, cylindrical and conical quadratic behavior is unchanged; Q9+ remains deferred.

### CH11 gate evidence

- focused CH11: `11 passed in 5.52s`;
- full curved: `106 passed in 65.21s`;
- full planar: `63 passed, 3 skipped in 39.04s`;
- full quad-first: `275 passed in 5.67s`;
- chart / physical-quality / serialization / coupling / quadratic-staging consumers: `58 passed in 0.73s`.

Fresh review confirmed exact owner-chart interior midsides, exact source-boundary ownership, stable linear topology IDs, strict validity and provenance truth, atomic cancellation, no topology fork, and no activation of curved B3 or Q9.

## CH12 — conical quadratic B3 ownership

**Status: QUALIFIED; ready for the CH12 milestone commit.**

CH12 narrows the CH9 blanket beam prohibition. Independent straight member content now reuses the existing mapped B3 generator and structural coupling pipeline alongside the accepted conical Q8/T6 shell. A beam that is itself a conical source-boundary edge remains typed unsupported; curved quadratic beam edges remain rejected; ruled/Coons quadratic B3 remains deferred to CH13.

### Product evidence

- cone h=0.6 + one straight through-face member: `126 nodes / 28 Q8 / 6 T6 / 4 B3 / 1 coupling`;
- member edge: 9 station nodes / 4 B3 elements; all B3 midsides exact chord midpoints;
- coupling eccentricity magnitude: `1.9027882047564422e-05`;
- maximum shell owner-support residual: `3.1401849173675503e-16`;
- every final Q8/T6 is `CERTIFIED_POSITIVE`; serialization/determinism/cancellation/source immutability are green.

### CH12 gate evidence

- focused CH12: `5 passed in 2.89s`;
- CH6+CH7+CH9+CH11+CH12: `26 passed in 26.61s`;
- structural/coupling/serialization consumers: `58 passed in 0.62s`;
- full curved: `111 passed in 68.89s`;
- full planar: `63 passed, 3 skipped in 38.56s`;
- full quad-first: `275 passed in 3.67s`.

## CH13 — metric-curved quadratic B3 ownership

**Status: QUALIFIED; ready for the CH13 milestone commit.**

CH13 narrows the CH11 blanket beam prohibition for non-planar `RuledSurface` and `CoonsSurface` faces. Independent straight members now reuse the accepted mapped B3 generator, CH7 quadratic attachment tolerance and structural coupling pipeline alongside CH11 Q8/T6 shells. Source-boundary beam co-ownership remains typed unsupported; curved quadratic beams remain rejected; no topology engine or option schema is added.

### Product evidence

- ruled h=0.6 + straight through-face member: `42 nodes / 8 Q8 / 0 T6 / 2 B3 / 1 coupling`, 5 member stations, eccentricity `4.163336342344337e-16`, shell owner residual `2.2247786310271853e-16`;
- coons h=0.6 + straight through-face member: `42 nodes / 8 Q8 / 0 T6 / 2 B3 / 1 coupling`, 5 member stations, eccentricity `4.163336342344337e-16`, shell owner residual `3.3335590258932494e-16`;
- all B3 midsides are exact endpoint chord midpoints; all Q8/T6 are `CERTIFIED_POSITIVE`;
- serialization, deterministic repeat, cancellation atomicity and source immutability are green.

### CH13 gate evidence

- focused CH13: `6 passed in 1.40s`;
- ownership/structural/coupling/serialization gate: `77 passed in 28.54s`;
- full curved: `117 passed in 69.89s`;
- full planar: `63 passed, 3 skipped in 38.49s`;
- full quad-first: `275 passed in 3.66s`.
