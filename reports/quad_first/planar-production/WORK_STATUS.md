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

## Next

PQ2: target-size-driven one-face planar meshing on the Q4/T3 advancing front.
