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
