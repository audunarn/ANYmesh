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

## Next

Run the bounded fresh PQ4a diff review, fix and rerun any affected gates, then
commit locally as `PQ4a: support holed and concave planar quad domains`. After
PQ4a acceptance, hand off automatically to the already-approved PQ4b
grading/narrow/parity/S3 tranche.
