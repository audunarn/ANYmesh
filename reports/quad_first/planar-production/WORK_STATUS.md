# ANYmesher Planar Quad Production Work Status

## Programme baseline

- Baseline: `68341621392b5f3a558c2f2735652ded0aca26e9`
- Branch: `opencode/planar-quad-v1`
- Worktree: `C:\Github\ANYmesh\.worktrees\planar-quad-v1`
- Worker model: `ollama/qwen3.8:27b`
- Administrator: ChatGPT
- Q4/Q5 worker executables: built locally from committed checkout-independent scripts

## PQ0 — contract freeze

**Status: COMPLETE.**

Current baseline behavior is intentionally recorded, not accepted as the product target:
- one untrimmed planar four-corner face -> four mesh nodes -> one Q4;
- changing `target_size` from 1.0 to 0.5 does not change public topology;
- Q4 MCF and Q5 TinyAD calls are fixed demonstrations rather than geometry-derived work;
- source geometry remains unchanged by the current public call.

PQ-M1 target remains genuine target-size-driven one-face planar meshing.

## PQ0 deliverables

- `docs/PLANAR_QUAD_DESIGN.md`
- `docs/PLANAR_QUAD_ACCEPTANCE.md`
- `tests/quad_first_planar/__init__.py`
- `tests/quad_first_planar/fixtures.py`
- `tests/quad_first_planar/test_pq0_baseline.py`
- this work-status record

## PQ0 evidence

```text
python -m pytest tests/quad_first_planar/test_pq0_baseline.py -q
  -> 1 passed in 0.22 s
python -m pytest tests/quad_first/test_q6_public_integration.py -q
  -> 32 passed in 0.47 s
python -m pytest tests/quad_first/test_q7_qualification.py -q
  -> 7 passed in 0.25 s
git diff --check
  -> clean
```

No production code changed in PQ0. No push, merge, release, publication, or default-route promotion occurred.

## PQ1 — residual front semantics + allocator/locality gates

**Status: COMPLETE.**

- Q4/T3 interface edges are active front; the source is the unique residual T3;
  a Q4 on the accepted side is permitted and ignored.
- Q3b transitions reconcile the front by residual-T3 incidence
  (_is_residual_front) instead of total single-incident count: a Q4-only
  boundary is not active front, so collision/closure consume the front fully
  and stage no residual additions.
- _apply_delta validates front/bits/protected only on locally changed/touched
  keys and applies them as in-place set patches (difference_update / update),
  with no full iteration or copy of resident front/protected sets.
- Transaction-owned node/cell allocators: ids allocated in the delta are
  reused (never consumed) on rollback and advance exactly once on commit;
  stale transactions cannot publish after another commit; cancellation and
  move-only commits preserve digest/generation without whole-map or
  front/protected-set iteration (locality gate tests).

### PQ1 evidence

```text
python -m pytest tests/quad_first_planar/test_pq1_front_semantics.py \
             tests/quad_first/test_q3b_transitions.py -q
  -> 35 passed in 0.27 s
python -m pytest tests/quad_first/test_q1_state.py \
             tests/quad_first/test_q1_front.py \
             tests/quad_first/test_q2_recovery.py \
             tests/quad_first/test_q3b_transitions.py \
             tests/quad_first/test_q3_guidance.py -q
  -> 138 passed in 0.27 s
python -m pytest tests/quad_first -q
  -> 272 passed in 6.07 s
python -m pytest tests/quad_first_planar -q
  -> 11 passed in 0.23 s
git diff --check
  -> clean
```

No push, merge, release, publication, or default-route promotion occurred.
## PQ2 — target-size planar seed + canonical boundary stations

**Status: COMPLETE.**

- `PlanarQuadDomain` freezes source model/revision/face identity and builds a deterministic orthonormal 2D chart without mutating source geometry.
- `BoundaryStationRegistry` solves target-size edge divisions once and owns canonical stations by exact source vertex/edge identity; reversed edge use reuses the same station identities in reverse order.
- `build_planar_quad_seed` creates a deterministic chart-space interior lattice, performs one Python constrained planar triangulation, and publishes an all-positive-T3 `QuadMeshState` with the exact outer segments as active front and protected boundary features.
- No public `_quad_first_execute` / hybrid route changed in PQ2; public-driver integration remains PQ3.

### PQ2 P01 size evidence

| target `h` | boundary divisions | nodes | boundary nodes | T3 | `N_eq=T3/2` | `A/h^2` |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 10, 6, 10, 6 | 77 | 32 | 120 | 60 | 60 |
| 0.5 | 20, 12, 20, 12 | 273 | 64 | 480 | 240 | 240 |
| 0.25 | 40, 24, 40, 24 | 1025 | 128 | 1920 | 960 | 960 |

Halving `h` therefore gives the exact 4.0 equivalent-count ratio. Boundary segment medians equal the target spacing on P01 and the maximum remains within the PQ2 1.5h gate.

P02 (rigidly rotated/translated P01) produces the same chart geometry within tolerance, the same division tuple, and at `h=0.5` the same 273 nodes / 480 T3. A two-face registry fixture with one exact shared geometry edge proves that the second face receives precisely the first face's station identities reversed; stations are not regenerated or coordinate-welded. Product tests also preserve source `model_id`, revision, vertices, edges, and face topology.

### PQ2 evidence

```text
python -m pytest tests/quad_first_planar/test_pq2_size_seed.py -q
  -> 8 passed in 7.09 s
python -m pytest tests/quad_first_planar/test_pq0_baseline.py \
             tests/quad_first_planar/test_pq1_front_semantics.py -q
  -> 11 passed in 0.22 s
python -m pytest tests/test_seeding.py \
             tests/test_compiled_triangulation_contract.py \
             tests/quad_first/test_q1_state.py -q
  -> 43 passed, 4 skipped in 0.24 s
     (compiled triangulation extension not rebuilt; Python contract exercised)
python -m pytest tests/quad_first -q
  -> 272 passed in 5.31 s
python -m pytest tests/quad_first_planar -q
  -> 19 passed in 7.10 s
git diff --check
  -> clean
```

No push, merge, release, publication, default-route promotion, or primary-worktree edit occurred.

## PQ3 / PQ-M1 - genuine target-size planar quad driver

**Status: COMPLETE, pending the local milestone commit.**

The committed PQ2 seed is now consumed by a finite deterministic public driver. A shared canonical station registry is captured before face execution; each selected planar face receives a fresh seed, one frozen cross-field report, guided/direct local pairing, bounded conforming recovery, independent validation, and exact neutral-mesh publication. The old fixed `CountInstance` and synthetic `PatchSpec` calls are absent from the PQ-M1 public dataflow. Q4 MCF and Q5 TinyAD remain direct qualified adapters, while public PQ-M1 diagnostics report both as `NOT_INTEGRATED` and do not require their binaries.

### P01 production evidence

| target `h` | nodes | Q4 | residual T3 | `N_eq` | Q4 count fraction | Q4 area fraction | final area | edge-chain lengths |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1.0 | 77 | 60 | 0 | 60 | 1.0 | 1.0 | 60.0 | [7,7,11,11] |
| 0.5 | 273 | 240 | 0 | 240 | 1.0 | 1.0 | 60.0 | [13,13,21,21] |
| 0.25 | 1025 | 960 | 0 | 960 | 1.0 | 1.0 | 60.0 | [25,25,41,41] |

Equivalent count is exactly `A/h^2` at every mandatory resolution and halving `h` gives the exact 4.0 count ratio. Driver counters are deterministic:

| `h` | initial T3 | attempts | stale skips | guided accepts | direct accepts | recovery accepts | queue pushes | final Q4/T3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1.0 | 120 | 60 | 55 | 60 | 0 | 0 | 136 | 60 / 0 |
| 0.5 | 480 | 240 | 239 | 240 | 0 | 0 | 512 | 240 / 0 |
| 0.25 | 1920 | 960 | 967 | 960 | 0 | 0 | 1984 | 960 / 0 |

P02 at `h=0.5` has the same 273 nodes, 240 Q4, zero T3, `N_eq=240`, and `[13,13,21,21]` station-chain lengths after the prescribed rigid transform. Source geometry identity/revision/topology remains unchanged by public meshing.

### Causal conforming recovery evidence

The public recovery fixture is the planar trapezoid `(0,0)-(4,0)-(3,4)-(1,4)` at `h=0.75`.

- recovery enabled: 26 Q4 / 6 T3, `N_eq=29`, Q4 count fraction `0.8125`, Q4 area fraction `0.946875`, one accepted recovery and one inserted node;
- same fresh seed with `allow_recovery=False`: 24 Q4 / 8 T3, `N_eq=28`, Q4 count fraction `0.75`, Q4 area fraction `0.9`;
- both close area at 12.0 within floating tolerance;
- strict canonical-boundary/incidence validation passes after the Q4/T3-interface split is made conforming by a same-transaction accepted-side local re-tile;
- a forced accepted-side re-tile rejection proves rollback preserves digest, generation, and next node/cell IDs.

The mixed quad-first/legacy merge preserves residual quad-first T3 IDs in one shared shell-element namespace and remaps legacy Q4/T3/beam IDs above the occupied quad-first shell IDs.

### PQ-M1 qualification evidence

```text
python -m pytest tests/quad_first/test_q2_recovery.py tests/quad_first_planar/test_pq3_public_driver.py -q
  -> 31 passed
python -m pytest tests/quad_first/test_q6_public_integration.py tests/quad_first/test_q7_qualification.py -q
  -> 40 passed
python -m pytest tests/quad_first/test_q1_front.py tests/quad_first/test_q2_recovery.py tests/quad_first/test_q3_guidance.py -q
  -> 92 passed
python -m pytest tests/quad_first_planar -q
  -> 27 passed
python -m pytest tests/quad_first -q
  -> 275 passed
git diff --check
  -> clean
```

Fresh bounded PQ3 review against target-size causality, worker isolation, exact station ownership, residual publication, conforming recovery/rollback, cancellation, validator incidence/area closure, legacy sentinel, source immutability, and structural mixed/S3/beam contracts found no material blocker after the final recovery correction.

No push, main merge, tag, release, PyPI operation, PQ4 work, or primary-worktree edit is part of PQ-M1.

## PQ4a - holed and concave planar domains

**Status: COMPLETE through fresh review; ready for the local milestone commit.**

PQ4a extends the genuine public quad-first route from the PQ-M1 four-corner
nominal scope to qualified simple planar exterior loops with concavity and true
interior holes. `PlanarQuadDomain` now records outer/hole loop provenance,
validates closed planar non-overlapping topology, and reports net source-chart
area. `BoundaryStationRegistry` includes every outer and hole source edge once,
and the seed passes one outer loop plus all hole station loops to the constrained
Python triangulator. Boundary station/protection provenance therefore covers all
loops without coordinate welding.

For analytic circular hole edges, the mesh boundary uses the exact geometry
station coordinates. The strict final validator continues to enforce resident
incidence/nonmanifold/boundary rules; its area reference is the authoritative
station/chord seed domain, avoiding a false comparison against the coarse source
corner polygon while keeping exact source-curve ownership.

### PQ4a P03 refinement evidence

| target `h` | nodes | Q4 | residual T3 | `N_eq` | Q4 count fraction | Q4 area fraction | discrete area | hole edge-chain lengths | max hole chord |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 0.5 | 276 | 233 | 10 | 238 | 0.9588477 | 0.9890414 | 57.57 | [4,4,4,4] | 0.4658743 |
| 0.25 | 1009 | 927 | 12 | 933 | 0.9872204 | 0.99737636 | 57.4842788816 | [7,7,7,7] | 0.23494715 |

The equivalent-count refinement ratio is `933/238 = 3.920168...`. Every P03
hole boundary station is on the exact radius-0.9 source circle about
`(3.2, 2.4)` within `1e-10`; maximum chord remains below `1.5 h`; non-hole
nodes remain outside the exact circle up to the conservative chord-sagitta
allowance. A fresh repeated `h=0.25` run reproduces node/Q4/T3 counts, edge-chain
lengths and validation fractions. Source model/revision/topology is unchanged.

P05 (concave exterior loop) and P03b (two disjoint holes) also execute on the
public `quad_first` route with exact discrete-area closure, at least 75% Q4 by
count/area where gated, complete source-edge chains, no hole fill, and
deterministic repeated topology. The obsolete Q7 hole/concave rejection cases
are intentionally converted to positive qualified-route assertions; unrelated
unsupported curved-surface and higher-order contracts remain unchanged.

### PQ4a gate evidence

```text
python -m pytest tests/quad_first_planar/test_pq4_general_domains.py -q
  -> 10 passed in 19.29 s
python -m pytest tests/quad_first_planar/test_pq3_public_driver.py \
             tests/quad_first/test_q2_recovery.py \
             tests/quad_first/test_q6_public_integration.py \
             tests/quad_first/test_q7_qualification.py \
             tests/test_seeding.py tests/test_compiled_triangulation_contract.py -q
  -> 91 passed, 4 skipped in 12.43 s
     (compiled triangulation extension not rebuilt)
python -m pytest tests/quad_first_planar -q
  -> 37 passed in 32.29 s
python -m pytest tests/quad_first -q
  -> 275 passed in 3.41 s
git diff --check
  -> clean
```

Primary `C:\Github\ANYmesh` remains on `main` at `6834162` with only its known
pre-existing unrelated local changes. PQ4a has not pushed, merged, tagged,
released or published anything.

The bounded fresh PQ4a review found one performance issue but no validity or
ownership defect: `_interior_lattice` performed a redundant O(N^2) duplicate
scan even though Cartesian lattice coordinates are unique by construction. The
scan was removed without changing topology or acceptance results. The affected
PQ4a product gate then improved from 48.17 s to 19.29 s and the full planar
suite from 92.23 s to 32.29 s on the same workstation. A proposed duplicate
self-intersection check was intentionally not added because authoritative
`ANYgeometry` already rejects self-intersecting face loops at geometry commit;
PQ4a does not duplicate that kernel responsibility.

## PQ4b - grading, narrow domains, shared-edge publication, S3 and native parity

**Status: COMPLETE through staged qualification and compiled-parity review.**

PQ4b was delivered in bounded local milestones: `1880f9e` activates graded
quad-first refinement and `58ec9aa` combines the explicit quad-first route with
qualified-S3 preparation. The final parity gate replaces the former pending
marker with exact Python/native checks on the actual staged seed corpus.

### PQ4b staged evidence

- P06 narrow domain (`h=0.25`): 569 nodes, 416 Q4, 92 residual T3, Q4 count
  fraction `0.8188976377952756`, Q4 area fraction `0.988951032540676`, exact
  area ratio `1.0`, deterministic repeat, and real node/cell occupancy in the
  narrow connector.
- P07 uniform (`h=1.0`): 45 nodes, 32 Q4, 0 T3, 5 nodes within radius 1.25 of
  the refinement centre. With the local `size=0.25`, `radius=0.75`,
  `growth=1.5` refinement: 96 nodes, 82 Q4, 2 T3, 53 near-centre nodes, Q4
  count fraction `0.9761904761904762`, Q4 area fraction `0.99755859375`. The
  mesh therefore refines locally rather than globally using the finest size.
- P09 two-face shared edge (`h=0.5`): 27 global nodes, 16 Q4, 0 T3; both faces
  publish 8 Q4 and reuse one 3-node shared-edge chain with exact reversed
  face orientation.
- Explicit all-Q4 `qualified_s3=True`: 32 Q4 / 0 T3 and the established
  `NOT_APPLICABLE_NO_TRIANGLES` record with legacy fallback `FORBIDDEN`.
  Residual-T3 cases continue to require authoritative Sheet/FaceUse normals;
  PQ4b does not weaken that existing S3 contract.
- Compiled parity: the rebuilt `anymesher._native` CPython 3.14 extension is
  present. P06 (569 prepared points / 924 seed T3), graded P07 (88 / 150), and
  both P09 face seeds match Python exactly in point bytes, segments, boundary
  and mandatory segments, and triangle connectivity; selected native backend is
  `anymesher-cpp17`.

### PQ4b gate evidence

```text
python -m pytest tests/quad_first_planar/test_pq4b_staged_domains.py -q
  -> 7 passed in 4.45 s
python -m pytest tests/test_compiled_triangulation_contract.py -q
  -> 13 passed, 1 skipped in 0.21 s
     (the one skip is the inverse capability-absent contract because native is present)
python -m pytest tests/quad_first_planar/test_pq2_size_seed.py \
             tests/quad_first_planar/test_pq3_public_driver.py \
             tests/quad_first_planar/test_pq4_general_domains.py \
             tests/quad_first/test_q6_public_integration.py -q
  -> 59 passed in 31.69 s
python -m pytest tests/test_seeding.py -q
  -> 10 passed in 0.20 s
python -m pytest tests/quad_first_planar -q
  -> 44 passed in 36.66 s
python -m pytest tests/quad_first -q
  -> 275 passed in 5.84 s
git diff --check
  -> clean
```

Primary `C:\Github\ANYmesh` remains on `main` at `6834162` with only its known
pre-existing unrelated changes. No push, merge, tag, release, PyPI operation,
reset or clean is part of PQ4b.

The bounded PQ4b review confirmed that uniform seed generation remains the PQ4a
path, graded refinement is bounded to at most 250000 fine candidate sites, exact
shared station identity is unchanged, qualified-S3 owner authority is not
bypassed, and native parity is qualification-only rather than a silent backend
switch in production. No material blocker remains in the activated PQ4b scope.

## PQ5 - actual-mesh TinyAD optimization

**Status: COMPLETE and committed at `cc81fb7`, with diagnostic closeout correction `367da9c`.**

PQ5 activates the already-qualified Q5 TinyAD worker on bounded patches of the
real resident Q4 mesh. The stage runs after the front driver and before the
independent final validator/publication. It is coordinate-only: no topology,
source ownership, canonical boundary station, or element ID is changed. Q4 MCF
remains truthfully `NOT_INTEGRATED`.

### PQ5 causal evidence

The causal public fixture is the real graded P07 8 m x 4 m face at base
`target_size=1.0`, using the already-qualified local refinement
`size=0.25`, `radius=0.75`, `growth=1.5` about `(2,2,0)`.

- Q5 disabled: 96 nodes, 82 Q4, 2 residual T3; status `DISABLED`, zero worker
  calls and zero coordinate moves.
- Q5 enabled with `max_local_optimizations=4`: the same 96 nodes / 82 Q4 / 2
  T3 and identical connectivity; 37 eligible real interior nodes, 4 attempts,
  4 worker calls, 4 accepted moves.
- Aggregate validated Q5 objective decreases from `85.42386140538405` to
  `72.14196687331342`; maximum accepted displacement is
  `0.2560234505295457` m. Moved resident node IDs are `[84, 77, 33, 78]`.
- Final strict planar validation remains exact: area ratio `1.0`, Q4 count
  fraction `0.9761904761904762`, Q4 area fraction `0.9975585937500001`,
  `N_eq=83`, and final area `32.0`.
- Every published canonical boundary-station node is byte-for-byte unchanged
  between disabled and enabled runs; only unprotected interior coordinates
  move.

Uniform P01 at `h=0.5` is the no-op witness: 273 nodes / 240 Q4 / 0 T3,
153 eligible interior candidates, 8 bounded worker calls, all returning
`NOIMPROVE`; objective remains `32.0 -> 32.0`, no node moves, and topology and
coordinates remain unchanged.

Missing worker capability is explicitly `UNAVAILABLE_SKIPPED` with zero worker
calls and no mutation. `max_local_optimizations=0` is explicitly `DISABLED`.
Protected nodes are never offered as free variables; residual-T3-touching
centres are excluded in this first causal Q5 tranche. Crash, malformed-response,
and invalid-solution failures remain typed and are not hidden by the public
optimizer.

### PQ5 gate evidence

```text
python -m pytest tests/quad_first_planar/test_pq5_real_optimization.py -q
  -> 7 passed in 2.18 s
python -m pytest tests/quad_first/test_q5_tinyad.py -q
  -> 28 passed in 2.38 s
python -m pytest tests/quad_first_planar/test_pq3_public_driver.py \
             tests/quad_first_planar/test_pq4_general_domains.py \
             tests/quad_first_planar/test_pq4b_staged_domains.py \
             tests/quad_first/test_q6_public_integration.py \
             tests/quad_first/test_q7_qualification.py -q
  -> 65 passed in 31.94 s
python -m pytest tests/quad_first_planar -q
  -> 51 passed in 40.34 s
python -m pytest tests/quad_first -q
  -> 275 passed in 5.44 s
git diff --check
  -> clean before documentation closeout
```

Primary `C:\Github\ANYmesh` remains on `main` at `6834162`; its unrelated
pre-existing `pyproject.toml` / documentation changes were not touched.

## PQ6 - geometry-derived real-seed MCF count planning

**Status: COMPLETE through closeout gates; this changeset is the PQ6 milestone commit.**

PQ6 runs the existing qualified Q4 integer MCF worker on bounded connected
components of the actual constrained T3 seed before the front driver. The public
route no longer uses a fixed/demo count instance. Q5 remains downstream and is
otherwise unchanged.

### PQ6 causal and contract evidence

- Skew fixture `((0,0),(4,0),(3.2,2.5),(0.4,2.5))`, `h=0.5`: seed has 78 T3.
  Candidate components are `(71 cells, 91 arcs)` and `(6 cells, 5 arcs)`. The
  first is reported `skipped_large`; the balanced 6-cell component is solved in
  one worker call and selects cell pairs `(13,14)`, `(16,18)`, `(17,20)`.
- The three pairs are applied in one transaction as Q4 IDs `[78,79,80]`; resident
  generation advances `0 -> 1`, yielding 3 Q4 / 72 T3 before the front driver.
- Enabled public execution finishes with 38 Q4 / 2 T3, 53 nodes, area ratio
  `1.0`, Q4 count fraction `0.95`, Q4 area fraction
  `0.9891176470588237`, and 41 front attempts. With
  `ANYMESH_Q4_DISABLE_WORKER=1`, the same source geometry remains strict-valid
  with the same 38 Q4 / 2 T3 counts but 44 front attempts and a different final
  Q4 body set. Source model/revision/topology is unchanged in both runs.
- Uniform P01 at `h=0.5`: the real seed candidate graph is one 480-cell /
  688-arc component, reported `skipped_large`; zero Q4 worker calls and zero
  seed mutation occur. The established final result remains 273 nodes / 240 Q4 /
  0 T3, area ratio `1.0`.
- Protected shared diagonals are excluded while protected endpoints remain
  admissible. Missing workers, explicit infeasibility and cancellation do not
  mutate resident generation or allocator IDs. `CountRejected` propagates as an
  internal modelling error; only `CountInfeasible` is an explicit nonfatal skip.
- Report/diagnostics now carry large/component-cap/unbalanced/non-bipartite/
  infeasible skip counts, component sizes/arc counts, selected
  pairs, total derived cost, added Q4 IDs and generation provenance. A perturbed
  real skew changes the generated component/cost signature, demonstrating that
  the production cost system is geometry-derived rather than hard-coded.

### PQ6 final gate evidence

```text
python -m pytest tests/quad_first_planar/test_pq6_geometry_mcf.py -q
  -> 15 passed in 0.91 s
python -m pytest tests/quad_first/test_q4_mcf.py -q
  -> 38 passed in 2.79 s
python -m pytest tests/quad_first_planar/test_pq3_public_driver.py \
             tests/quad_first_planar/test_pq4_general_domains.py \
             tests/quad_first_planar/test_pq4b_staged_domains.py \
             tests/quad_first_planar/test_pq5_real_optimization.py \
             tests/quad_first/test_q6_public_integration.py \
             tests/quad_first/test_q7_qualification.py -q
  -> 72 passed in 32.27 s
python -m pytest tests/quad_first_planar -q
  -> 66 passed in 45.98 s
python -m pytest tests/quad_first -q
  -> 275 passed in 5.63 s
git diff --check
  -> clean
```

Primary `C:\Github\ANYmesh` remains on `main` at `6834162`; its pre-existing
unrelated local changes were not modified. No push, merge, tag, release, PyPI,
reset or clean operation is part of PQ6.

## Next

Stop at the PQ6 planar-programme milestone after the local commit. Curved
surfaces, higher-order elements and unsupported anisotropy remain outside this
qualified scope unless a separate tranche is explicitly opened.
