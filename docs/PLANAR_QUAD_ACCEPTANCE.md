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

## PQ4a activated planar-domain gates

PQ4a activates the previously staged P03 and P05 scope without changing the
frozen PQ-M1 thresholds:

- P03 is a true off-centre analytic circular hole in the 10 m x 6 m planar
  plate; P03b extends the same contract to two disjoint holes.
- P05 is a simple concave planar exterior loop. Explicit quad-first must accept
  these qualified planar domains without legacy fallback and preserve source
  geometry.
- Hole and outer source edges retain complete canonical station chains. Every
  analytic circular-hole station lies on the source circle within `1e-10` and
  the maximum chord is bounded by `1.5 h`.
- Halving P03 from `h=0.5` to `h=0.25` must retain `3.2 <= N_eq(h/2)/N_eq(h)
  <= 5.0`, at least 75% Q4 by active count and area, exact discrete-domain area
  closure, deterministic repeat topology, and no non-boundary node intrusion
  beyond the conservative chord sagitta.
- Curved *surfaces* and higher-order elements remain unqualified; this gate is
  for planar faces whose boundary edges may be analytic curves.
## PQ4b activated staged gates

PQ4b activates the remaining staged planar fixtures without changing the
PQ-M1 or PQ4a validity rules:

- P06 is the narrow-ligament / transition-heavy planar domain at `h=0.25`.
  It must remain valid with exact area closure, deterministic repeat topology,
  actual node/cell occupancy in the narrow connector, and at least 75% Q4 by
  active count and area.
- P07 activates public isotropic `Refinement` plumbing. A local
  `size=0.25`, `radius=0.75` zone about `(2,2,0)` on an 8 m x 4 m face with
  base `target_size=1.0` must increase total and near-zone node counts without
  globally collapsing the mesh to the finest size. Source geometry remains
  immutable.
- P09 activates two selected planar faces sharing one edge in reverse
  orientation. Both faces reuse the exact same global station/node IDs; the
  face-oriented station chains are exact reversals.
- `qualified_s3=True` may be combined with the explicit quad-first route. An
  all-Q4 result truthfully reports `NOT_APPLICABLE_NO_TRIANGLES` with legacy
  fallback forbidden; if residual T3 are present, the existing qualified-S3
  Sheet/FaceUse owner-authority contract remains mandatory.
- When the compiled triangulation extension is available, the canonical seed
  rows for P06, graded P07, and both P09 faces must reproduce Python
  preparation exactly: point bytes, segments, boundary/mandatory segments,
  and triangle connectivity. No native fallback or coordinate reordering is
  accepted as parity.

## PQ5 activated actual-mesh TinyAD gates

PQ5 activates Q5 only on the resident mesh produced by the genuine public
quad-first route. At the PQ5 checkpoint it did not activate Q4 MCF; PQ6 below
adds that separate seed-stage integration without changing the PQ5 contract.

- `max_local_optimizations=0` must make zero Q5 worker calls and report
  `DISABLED`; it must not fall back to the historical `NOT_INTEGRATED` tag.
- A causal real-mesh case must retain identical Q4/T3 connectivity with Q5 on
  and off, preserve every canonical boundary-station coordinate exactly, move
  at least one unprotected interior node, and strictly reduce the validated Q5
  objective while final planar validation remains green.
- The qualified causal case uses the real locally graded P07 mesh. With base
  `target_size=1.0` and the existing `size=0.25`, `radius=0.75`, `growth=1.5`
  refinement, Q5 must use the real generated Q4 patch and not a synthetic test
  patch.
- The uniform P01 `h=0.5` mesh is a no-regression/no-op witness: topology and
  coordinates must remain unchanged unless an accepted displacement is below
  the explicit negligible tolerance.
- Protected/boundary nodes are never free. First-tranche eligible centers touch
  Q4 cells only; residual-T3-touching centers are excluded.
- Worker calls are bounded globally by `min(max_local_optimizations, 8)`.
  Missing worker capability is truthful `UNAVAILABLE_SKIPPED` with no mutation;
  crash/malformed/invalid-solution failures remain typed failures rather than
  silent fallbacks.
- The independent planar validator runs after accepted coordinate moves. Area
  closure, positive cells, incidence, owner/station identity, legacy sentinel,
  and source-geometry immutability remain mandatory.
- At the PQ5 checkpoint Q4 MCF remains `NOT_INTEGRATED`; the separately
  qualified PQ6 gates below supersede that current-status tag.

## PQ6 activated geometry-derived Q4 MCF gates

PQ6 qualifies the count worker only on bounded components of the real planar T3
seed and before the genuine front driver. The following are mandatory:

- Candidate pairing must be derived from resident seed geometry. A protected
  shared diagonal is ineligible while protected endpoints remain legal. No fixed
  or demonstration `CountInstance` may drive the public route.
- Component processing is deterministic and bounded to four solved components
  per face, 12 cells per component and 12 admissible pair arcs. Diagnostics must
  expose component sizes/arc counts and explicit large, component-cap,
  unbalanced, non-bipartite and infeasible skip counts.
- Internally generated `CountRejected` must propagate as a modelling defect. A
  worker-reported `CountInfeasible` is the only nonfatal solver skip. Missing
  worker capability is truthful `UNAVAILABLE_SKIPPED` with no mutation.
- All selected pairs are solved before mutation and applied atomically. A
  successful nonempty application advances the resident generation exactly once
  and reports the added Q4 IDs; cancellation/unavailability/infeasibility must
  leave digest, generation and resident allocator IDs unchanged.
- The causal skew fixture at `h=0.5` must apply three seed pairs with one worker
  call, retain a strict-valid final 38 Q4 / 2 T3 mesh, change the final Q4 body
  set relative to worker-disabled execution, and reduce front attempts from 44
  to 41. Source geometry must remain immutable in both executions.
- Uniform P01 at `h=0.5` is a mandatory bounded no-op: the oversized seed
  component is reported as skipped, zero count-worker calls occur, and the final
  240 Q4 / 0 T3 result remains unchanged.
- A real geometry perturbation must change the derived component/cost signature;
  hard-coded production cost matrices are forbidden.

Curved surfaces, higher-order elements and unsupported anisotropy remain outside
the qualified planar scope.

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

This document remains the frozen acceptance contract. PQ-M1 implementation
and evidence are now recorded separately in `WORK_STATUS.md`: P01 passes all
three mandatory resolutions with 100% Q4 count/area fraction and exact area
closure; P02 is rigid-transform invariant at the qualified resolution; and a
small trapezoid public-path case proves causal, conforming recovery by improving
from 24 Q4 / 8 T3 with recovery disabled to 26 Q4 / 6 T3 with recovery enabled.
The frozen thresholds above are unchanged by that implementation evidence.
