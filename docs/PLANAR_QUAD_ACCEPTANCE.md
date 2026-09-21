# ANYmesher Genuine Planar Quad Acceptance (PQ0 freeze)

## Baseline

- Commit: `68341621392b5f3a558c2f2735652ded0aca26e9`
- Branch: `opencode/planar-quad-v1`
- Companion to `PLANAR_QUAD_DESIGN.md` (PQ0 freeze).

## Frozen fixture registry

### P01 — One planar rectangle (PQ-M1 mandatory)

- ONE unpartitioned planar 10 m x 6 m rectangle.
- Resolutions: `h = 1.0`, `h = 0.5`, `h = 0.25`.
- This is the PQ-M1 fixture set; no other fixture may replace it.

### P02 — Rigidly transformed variant (PQ2 qualification)

- P02 is ONLY a rigid rotation plus translation of P01.
- Same topology/count/quality intent as P01 at the same `h`; identity and
  stations, not coordinates, must drive conformity.

### Staged fixtures (beyond PQ-M1 unless naturally enabled)

- P03: off-centre hole plate.
- P05: concave planar domain.
- P06: narrow ligament / transition-heavy domain.
- P07: graded isotropic size field.
- P09: two planar faces sharing one refined edge used in reverse
  orientation; one canonical station chain across both uses.

## PQ-M1 gates

### Sizing

- Equivalent count: `N_eq = N_Q4 + N_T3/2`.
- P01 area is `A = 60 m^2`; nominal `A/h^2` expectations are `60`, `240`,
  and `960` for `h = 1.0`, `h = 0.5`, and `h = 0.25` respectively.
- For P01 at `h = 1.0`, `h = 0.5`, `h = 0.25`:
  `0.70 <= N_eq/(A/h^2) <= 1.40`.
- Halving `h` on nontrivial P01 resolutions: `3.2 <= N_eq(h/2)/N_eq(h) <= 5.0`.

### Interior structure

- At `h = 0.5`: at least 3 internal element rows and genuine interior mesh
  nodes. A single boundary-hugging layer is not acceptable.

### Quad dominance

- Hard minimum, nominal regular cases: >= 85% Q4 by active cell count AND
  >= 85% Q4 by active area.
- Target (aspirational above the hard minimum): >= 90% Q4 on the nominal
  regular case.
- Transition-heavy proposed target: >= 75% Q4 by count and by area. This is
  a proposed target only and is NOT a PQ-M1 hard gate unless activated later;
  validity still dominates.
- A Q4 fraction is a quality statistic, not a substitute for validity: every
  validity and quality gate always wins over the percentage. No gate may be
  met by degrading validity, and no validity failure may be waived to reach
  a percentage.

### Distribution honesty

- Normalized edge-size distribution must be meaningful and broadly sized.
- Count or area gates must not be gamed by tiny-cell concentration or
  micro-cell dumping.

### Validity and ownership (independent validator)

- Every active shell has finite positive orientation/Jacobian.
- Complete domain coverage: no overlaps, no pockets, no hole fill.
- No seed/helper triangles under active Q4; active Q4 plus finalized T3
  exactly partition the domain.
- Exact geometry face/edge/vertex ownership; exact boundary station chain.
- Source geometry face count, topology, and revision unchanged by meshing.

### Causality and route hygiene

- At least one public-path case whose accepted mesh causally requires
  recovery/insertion; with that mechanism disabled the same case must fail
  or change materially.
- Fixed demonstration MCF and synthetic TinyAD are forbidden on the PQ-M1
  route; their stages must report `NOT_APPLICABLE` / `NOT_INTEGRATED` until
  geometry-derived and actual-mesh integration is landed.

## Baseline non-target statement

The current PQ0 baseline one-Q4 behavior (four corner nodes, two T3 seeds,
one `front_step`, one published Q4, fixed demonstration CountInstance and
synthetic PatchSpec) is historical evidence of the starting point only. It
is NOT an acceptance target and must not be cited as PQ-M1 evidence.

## Acceptance matrix

| Phase | Acceptable at this stage |
| --- | --- |
| PQ0 | Freeze contract; gates defined; no meshing claims |
| PQ1 | Correct resident semantics, locality, protection; P01 still not required |
| PQ2 | Size-responsive constrained seed/domain/stations proven on P01, plus transform invariance checks on P02; the complete final P01/P02 public PQ-M1 gates are not yet required |
| PQ3 | Size-driven front driver, independent validation, public dispatch; the full public P01 PQ-M1 gates and causal recovery are required here |
| PQ4-PQ10 | Staged fixtures P03/P05/P06/P07/P09 as enabled; curved and higher-order remain later |

## Legacy compatibility and exclusions

- Explicit quad-first must not silently succeed as legacy-only: any explicit
  quad-first selection that lands on the legacy path is a failure, not an
  acceptable fallback.
- `quad_options=None` behavior is unchanged from legacy; legacy behavior is
  preserved, not improved, by this work.
- Curved surfaces, higher-order elements, and unsupported anisotropy remain
  unsupported until later phases; they are out of PQ-M1 explicitly.

## Resource bounds

- Routine development cases: 100 to 1000 active elements.
- P01 `h = 0.25` is the documented ceiling at about 960 equivalent.
- 500k / workstation-scale (or equivalent) cases are excluded from PQ-M1
  and must not appear as acceptance evidence.

## Evidence rules

- Focused tests activated under `tests/quad_first_planar/`.
- Relevant `tests/quad_first/` regressions activated and green.
- `git diff --check` clean at acceptance.
- Local milestone commits only; no push, no merge, no release or
  promotion action in PQ-M1.
- Reports must state exact counts, exact Q4 fractions (count and area),
  counters, and normalized edge-size statistics; rounded or approximate
  evidence is not acceptable against the gates above.

## Non-claims

This document freezes the acceptance contract. It does not claim PQ-M1 is
implemented, that any fixture passes, or that the recovery/insertion causal
case exists today.
