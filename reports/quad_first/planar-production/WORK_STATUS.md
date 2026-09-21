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

## Next

PQ3: consume the committed PQ2 seed in the genuine planar quad driver; PQ3 has not been started.