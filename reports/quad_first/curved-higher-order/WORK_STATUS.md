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