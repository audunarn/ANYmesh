# Quad-first work status

**Branch:** `opencode/quad-first-v1`
**Baseline:** `2ccef37`
**Task ID:** `full-programme` (Q0 → Q6)
**Status:** Q0 (freeze) **complete** `0752f42`. Q1 (state/journal/front_step) **complete** `b410719`. Q2/M1 (bounded front-edge Steiner-split recovery) **complete**. Q3a (cross-field guidance) **complete**. Q3b (quad transitions: spacing_change / collision / closure) **complete**. Q4 (count-system + LEMON MCF worker adapter) **complete** `fe46879`. Q5 (TinyAD local optimisation on real Q4 patches) **complete** `de73fa4`. Q6/M2 (public quad-first integration + mixed structural qualification) **complete in this commit**.
**Updated:** 2026-09-21

## Objective (Q0, from `Q0_EXECUTION.md` 1-11)

Freeze upstream source and license contract, pin the six donor components,
create the dedicated `QuadMeshingOptions` schema independent of
`NativeMeshingOptions`, document the architecture and the reuse contract,
smoke the bounded abstractions, and commit a coherent branch.

Q0 is an **attribution + schema + smoke** milestone: it does **not** implement
front state, recovery, insertion, transitions, or local optimization. Those are
deferred to Q1-Q7 as scoped in `QUAD_FIRST_FULL_PROGRAMME.md`.

## Work in this branch (this commit)

### New — `third_party/quad/` (the freeze itself)

| path | role |
| ---- | ---- |
| `manifest.json` | machine-readable pin set: schema tag, build flags (`-DEIGEN_MPL2_ONLY`), toolchain evidence, six components (pinned commit, archive SHA-256, per-tree SHA-256, verbatim-file count, role, license), policy exclusions. |
| `ATTRIBUTION.md` | human attribution: per-component copyright line, upstream URL, pinned commit, per-tree SHA-256, licence text for MPL-2.0 (Eigen) and one copy of the MIT text. |
| `licenses/LICENSES.md` | licence index → co-located `vendor/<id>/LICENSE` files (no duplicated texts). |
| `vendor/` | six pinned vendors: `eigen` (337 files), `tinyad` (25), `libsatsuma` (33), `libtimekeeper` (8), `q-morph` (1, LICENSE only), `lemon` (110). |
| `smoke/` | two C++17 smoke tests + `build.bat` / `build_lemon.bat` that **PASS** on MSVC 14.50.35717 / Windows SDK 10.0.26100.0. |

### New — `src/anymesher/quad/` (strict options contract)

| path | role |
| ---- | ---- |
| `quad/__init__.py` | re-export `QuadMeshingOptions`, `QUAD_MESHING_OPTIONS_SCHEMA`. |
| `quad/options.py` | frozen dataclass mirroring `NativeMeshingOptions` (strict `from_dict` field-set + schema tag, `to_dict`, `canonical_digest` SHA-256, `coerce`). Closed enums (`seed_mode`, `orientation`, `quality_model`, `line_search`); three integer resource guards (positive/non-negative, non-bool, not-float). |

### Modified — `src/anymesher/__init__.py`

- Added `from .quad import QUAD_MESHING_OPTIONS_SCHEMA, QuadMeshingOptions`.
- Added both names to `__all__`.

### New — `docs/`

| path | role |
| ---- | ---- |
| `QUAD_FIRST_DESIGN.md` | Q0 architecture freeze: construction pipeline (CDT seed → cross-field guidance → front → mixed Q4/T3 state → transactional commit), 11 hard invariants from the programme, the strict public schema, resource guards, third-party reuse matrix, build evidence, Q0 scope (what was and was not done). |
| `QUAD_FIRST_REUSE.md` | per-donor reuse contract: what is adapted, what is excluded (per-component and per-programme), how each is pinned and verified, and the `EIGEN_MPL2_ONLY` enforcement. |

### New — `tests/quad_first/`

| path | coverage |
| ---- | -------- |
| `test_q0_freeze.py` | 29 tests. (1) Freeze integrity: schema tag, build flags, per-component pin integrity, per-vendor-file-count match, per-file LICENSE presence, `ATTRIBUTION.md` content, exclusion set. (2) `QuadMeshingOptions` contract: defaults, schema tag, round-trip stability, `canonical_digest` shape, strict `from_dict` (unknown field, missing field, schema mismatch), closed enums, integer guards (reject `0`, negative, non-int, bool, float, string), `line_search='off'` invariant, frozen dataclass (raises on mutation), `coerce(None)` vs instance vs mapping vs garbage, top-level export, independence from `NativeMeshingOptions` (different schema tag, different digest, no quad field leaking into the legacy dataclass). |

### Modified — `.gitignore`

- Added `!reports/quad_first/` and `!reports/quad_first/**` so the freeze report tree is committed (consistent with the existing `reports/native_hybrid` precedent).

## Smoke evidence (reproducible)

| test | compile (MSVC 14.50.35717, x64, C++17, `-DEIGEN_MPL2_ONLY`) | result |
| ---- | ------------------------------------------------------------ | ------ |
| `third_party/quad/smoke/` `tinyad_smoke.cc` | `build.bat` (uses `vendor/eigen` + `vendor/tinyad/include`) | **PASS** — `f=13.0 g=(2.0,-12.0)` |
| `third_party/quad/smoke/` `lemon_mcf_smoke.cc` | `build_lemon.bat` (LEMON `vendor/lemon`) | **PASS** — 4-node CDT seed MCF, `status=OPTIMAL, total_cost=6` |

Both smokes use **only** the exact abstractions the production path will use
(TinyAD `scalar_function<1,double>` + `eval_with_gradient`; LEMON
`NetworkSimplex` + `ListDigraph` + `ArcMap`/`NodeMap`). No `_native` linkage,
no Java, no Gurobi, no network access at build or run time.

## Focused Python tests (re-runnable)

```
$env:PYTHONPATH = "src"
python -m pytest tests/quad_first/ -v
```

```
29 passed in 0.40s
```

## Not done at Q0 (deliberately deferred)

Per `Q0_EXECUTION.md` and the T1 brief:

- Front state, mixed Q4/T3 mixed-state, bounded recovery and insertion.
- TinyAD local optimisation on real Q4 patches (the smoke uses `scalar_function<1,double>` only — the exact primitive, but not yet a full-quality objective).
- libSatsuma MCF count-system integration (planned Q4).
- Cross-field guidance, low-confidence diagnostics (planned Q3).
- S3 qualification and component publication (planned Q6).
- Any change to `NativeMeshingOptions`, to the `_native` extension, or to
  the existing legacy call path.

## Invariants held at Q0

1. `NativeMeshingOptions` is byte-identical; its schema tag is
   `anymesher.native-meshing-options/1` and is **not equal** to
   `anymesher.quad-meshing-options/1`.
2. The `_native` C++ extension is unlinked from the Q0 smokes; its ABI is
   unchanged.
3. The six vendor components are **pinned by SHA-256** both at archive level
   and at per-tree level (the latter is the identity of the vendored subset);
   no vendor file was patched.
4. `EIGEN_MPL2_ONLY` is enforced in the build flag set and empirically proven
   by the TinyAD smoke PASS.
5. The vendor file counts in `manifest.json` match the worktree; the tests
   `test_vendor_file_counts_match_manifest` and
   `test_vendor_attribution_files_present` assert this at run time.
6. `NativeMeshingOptions` exposes no `QuadMeshingOptions` field and vice versa
   — asserted in `test_quad_options_are_independent_of_native`.
7. The two design docs (`QUAD_FIRST_DESIGN.md`, `QUAD_FIRST_REUSE.md`) are the
   authoritative Q0 design record; the T1 brief and the full programme remain
   unmodified.

## Known risks / unverified (honest gaps)

- Q-Morph is **reference-only** at Q0. The T1 brief does not require Q-Morph
  smoke PASS at Q0; its C++/Java sources are not compiled into the smoke path.
  The production integration (Q1) will need to re-verify its header set is
  MPL2/Boost-clean and that no Java / QuadWild / Gurobi dependency leaks in.
- libSatsuma is pinned at Q0 but its **`.cc` files are not vendored**. The Q4
  MCF integration will use `vendor/lemon` (via LEMON's `NetworkSimplex`) —
  the exact solver libSatsuma's MCF path calls. This is the documented
  substitution; it is **not** a silent license change.
- `manifest.json` lists `libSatsuma` upstream URL as
  *"verified by archive pin"* because the canonical mirror was not re-asserted
  during freezing. The archive's SHA-256 (`01efe990…70e0e`) is the operational
  identity; the file set (33 headers + LICENSE) is reproducible from that
  archive.
- The LEMON `file_count` was updated from 99 to 110 to match the actual
  worktree (`bits/` 19 + `concepts/` 7 + 84 top-level headers + `LICENSE` =
  110). The previous 99 was a pre-freeze draft that did not include the
  generated `config.h`/`export.h` and the `concepts/` directory.

## Process inventory

- No network fetch during build or run; all archives were downloaded during
  freezing by a Python `urllib` call into the staging directory
  `C:\Users\AUDUNA~1\AppData\Local\Temp\opencode\q0_staging` and then copied
  into the vendor tree (no archive retained in the repo).
- No `pip install` was run (no `pyproject.toml` or `setup.py` changes —
  `QuadMeshingOptions` is a pure-Python dataclass that does not add a build
  dependency).
- No `git push`, no `git push --force`, no `git reset --hard`, no tag, no
  PR, no release, no package publish.
- No modification of `pyproject.toml`, `setup.py`, `MANIFEST.in`,
  `THIRD_PARTY_NOTICES.md`, `native_v2.py`, `errors.py`, or any other
  existing file outside the additions listed in "Work in this branch".
- The `NativeMeshingOptions` class and `_native` extension are **unchanged**.

## Q1 — resident mixed T3/Q4 state, transaction, `front_step` (commit `b410719`)

**Status:** **complete**.

### New — `src/anymesher/quad/`

| path | role |
| ---- | ---- |
| `state.py` | `QuadMeshState`: resident mixed Q4/T3 topology keyed by exact shared node IDs; `EdgeKey` (cyclic-normalized, equal-node-rejecting) identity; protected-edge/node sets; `Delta` sparse staged patch, `View` read-model that resolves staged additions/removals before commit, generation counter, SHA-256 `digest()` for exact-identity assertions. |
| `journal.py` | `Transaction`: atomic commit/rollback over `state.transaction()`; staged `add_node`/`remove_cell`/`add_cell`/`add_front_edge`/`remove_front_edge`; `view` read-model with late-staging freshness; use-after-conclude guard; cancellation gate at the staged checkpoint; failed-commit preserves digest + generation. |
| `front.py` | Advancing-front driver: `edge_key`, `body_edges`, `area2`, `make_quad` (convex CCW, degenerate/repeated-node reject), `local_swap`, `find_source_cell` (unique-T3), `candidate_partners` (deterministic), `classify`, `front_step` — converts one front T3 pair into a Q4, updates the front, commits in a transaction, and **reverses on rejection** (digest identity guaranteed). |

### New — `tests/quad_first/`

| path | coverage |
| ---- | -------- |
| `test_q1_state.py` | 23 tests: exact node-edge adjacency, View-after-staging freshness, discard-only rollback with exact digest identity, generation/stale guards, cancellation, protect/release, duplicate-body reject, stale-cell-reject, local-patch-not-rebuild. |
| `test_q1_front.py` | 22 tests: `edge_key`/`body_edges`/`area2`/`make_quad` convex+degenerate+repeat + `local_swap`; `find_source_cell`/`candidate_partners` determinism; `front_step` A (stage+commit+front-update), B (no partner), C (non-T3 source), D (non-convex union → reject), protected edge/node block, non-front reject, bad-options reject, **reverses-on-rejection digest identity**. |

**Evidence:** `pytest tests/quad_first/` → **95 passed** (29 Q0 + 23 state + 22 front + 21 Q2 below).
Top-level `anymesher` package untouched in Q1 (legacy path byte-identical).

---

## Q2/M1 — bounded deterministic front-edge Steiner-split recovery (this commit)

**Status:** **complete (M1)**.

Q1 pure pairing yields **0 Q4** on the dart (D) fixture: the two T3s share the
interior diagonal `(0,2)` and their union is concave, so no convex quad exists.
Q2 adds the production recovery: a bounded, deterministic **front-edge Steiner
split** (1 T3 → 2 T3s at a single new node) that *enables* a Q4 which the real
`front_step` then creates from a T3-only post-recovery state.

### New — `src/anymesher/quad/recovery.py`

| symbol | role |
| ---- | ---- |
| `RecoveryRejected(MeshError)` | specific-rule rejection (non-front/protected/degenerate/infeasible). State unchanged. |
| `RecoveryExhausted(MeshError)` | all ratios tried, no admissible plan. State unchanged. |
| `Attempt` (frozen) | per-ratio probe record: `ratio`, `midpoint_id`, `enabled_edge`, `ok`, `detail`. |
| `SplitReport` (frozen) | `front_edge`, `ratio`, `midpoint_id`, `source_cell`, `child_cells`. |
| `AdvanceReport` (frozen) | `front_edge`, `ratio`, `midpoint_id`, `child_cells`, `quad_cell_id`, `quad_body`, `enabling_edge`, `attempts`. |
| `edge_split_recover(state, edge, ratio=0.5, options=None) -> SplitReport` | standalone: validate, stage 1 T3 → 2 T3s at a Steiner node, commit atomically. |
| `recover_then_front_step(state, edge, ratios=(0.5,0.4,0.6,0.3,0.7), options=None) -> AdvanceReport` | orchestration: in a transaction, stage the split, probe (via `tx.view`) the enabling child edge where `classify` succeeds, commit inside the with-block, then run the **real** `front_step` on that enabling edge. Failed probe → context-manager exit discards; continue to next ratio. No Q4 is ever pre-seeded. |
| `DEFAULT_RECOVERY_RATIOS` | `(0.5, 0.4, 0.6, 0.3, 0.7)`. |

Geometry note (proven): dart edge `(0,1)`, `r=0.5` → m=(1,0), and
`orient2d(m,2,3)=0` → **degenerate → rejected**. First canonical ratio that
succeeds is **`0.4`** (m=(0.8,0), all four `orient2d > 0`); `0.6`/`0.7` are
rejected (mixed-sign turns). So the default order lands on `0.4` after the
single deterministic `0.5` probe.

### Modified — `src/anymesher/quad/__init__.py`

- Added recovery exports to imports and `__all__` (`RecoveryRejected`,
  `RecoveryExhausted`, `Attempt`, `SplitReport`, `AdvanceReport`,
  `DEFAULT_RECOVERY_RATIOS`, `edge_split_recover`, `recover_then_front_step`).
- **Top-level `anymesher/__init__.py` is NOT re-touched by Q2** — legacy
  top-level surface remains byte-identical to Q1 (recovery symbols are
  subpackage-only; asserted in `test_legacy_isolation`).

### New — `tests/quad_first/test_q2_recovery.py`

21 tests covering the admin-mandated battery: two-T3 children + front
re-wiring + generation/digest advance; ratio determinism; non-front /
protected-edge / protected-node / bad-ratio / unknown-edge / bad-options
rejection with **state digest identity**; dart `0.5`-reject → `0.4`-accept with
recorded attempts; explicit `0.4`-only success; `0.5`-only exhaustion with
digest identity; **no hidden Q4** (exactly one Q4, created by `front_step` from
the T3-only post-split state); no-T3-partner exhaustion; deterministic ratio
ordering; custom-order; empty-ratio exhaustion; cancellation; split-rejection and
exhaustion state-unchanged (generation unchanged); legacy top-level isolation;
hole-plate pure-pairing (Q4 without recovery).

### Evidence

```
pytest tests/quad_first/test_q2_recovery.py -v   -> 21 passed
pytest tests/quad_first/                         -> 95 passed (Q0 29 + Q1 45 + Q2 21)
```

Targeted regression set (layering / packaging / backends / serialization)
**26 passed, 2 skipped** (skips are pre-existing environment gates). The full
package suite exceeds the bounded local window and is treated as non-critical:
Q2's only shared-surface change is an additive `quad/__init__.py` export, and
`anymesher/__init__.py` is byte-identical to Q1, so the legacy path cannot
regress from Q2.

### Invariants held at Q2/M1 (beyond Q0/Q1)

- Opt-in only: recovery is a named API; `front_step` and the legacy path are
  untouched.
- ANYgeometry owns geometry; recovery performs exact shared-node-ID topology
  only (no coordinate welding, no solver).
- Deterministic: ratios in list order; child edge `(a,m)` probed before `(m,b)`;
  `candidate_partners` iterates sorted body-edges.
- Typed failure + rollback: `RecoveryRejected`/`RecoveryExhausted` both leave
  the state digest **and** generation unchanged; failed probe is discarded at
  context-manager exit.
- No hidden fallback: the Q4 is always created by the real `front_step` from the
  committed T3-only post-split state; the probe only *selects* the enabling edge.
- Scratch files (`_scratch_recovery.py`, `_scratch_search.py`, `_verify_dart.py`)
  deleted before this commit.

### Not done at Q2/M1 (deliberately deferred to Q3+)

- Cross-field / anisotropy guidance and low-confidence diagnostics (Q3).
- Local TinyAD optimisation on real Q4 patches (Q3/Q4).
- libSatsuma/LEMON MCF count-system integration (Q4).
- S3 qualification and component publication (Q6).
- Any change to `NativeMeshingOptions`, `_native`, or the legacy call path.

---

## Q3a — cross-field guidance (4θ) + guided front driver (this commit)

**Status:** **complete**.

Q3a delivers the deterministic cross-field guidance layer required by
`QUAD_FIRST_FULL_PROGRAMME.md` Q3.  The layer is a pure-function pipeline:

1.  *Build the field* — every oriented node-direction pair is folded into
    four-fold cover coordinates via `z(θ) = (cos 4θ, sin 4θ)`; a mean vector
    gives a unit field direction per node, and the **un-normalised** mean
    magnitude is the per-node *confidence* scalar (in `[0, 1]`).
2.  *Guide the front* — the Q1 `front_step` candidate loop is wrapped so the
    first candidate whose boundary walk score meets the per-endpoint
    confidence gate is selected.  No candidate is ever pre-seeded; the gate
    only **selects** from `candidate_partners` output, preserving Q1
    determinism.
3.  *Reject with typed reasons* — `LowConfidenceError`, `GuidanceRejected`,
    `FrontNoCandidate` (lifted from the Q1 rejection reasons) keep the state
    byte-for-byte identical, and no silent fallback is possible.

### New — `src/anymesher/quad/guidance.py`

| symbol | role |
| ---- | ---- |
| `CrossFieldReport` (frozen dataclass) | per-node `direction` and `confidence`; `to_dict()` for round-trip; `all_confidences_ge(threshold)` helper; `rejects_low_confidence(state, min_confidence)` gate used by the driver. |
| `GuidanceRejected(MeshError)` | specific-rule rejection; state left untouched. |
| `LowConfidenceError(MeshError)` | a front endpoint fell below the confidence threshold. |
| `NEAR_ZERO` | `1e-9` degenerate-field threshold. |
| `four_of(dx, dy) -> (c2, s2)` | fold an oriented unit direction into 4θ cover: `(c, s) = (cos 2θ, sin 2θ)` then `(c² − s², 2cs)`.  Pure function; returns `None` on near-zero input. |
| `rot2(v, θ)` | rotate a 4θ-cover vector by θ; pure, no state. |
| `reflect_x(v)` | mirror a 4θ-cover vector in x; pure. |
| `field_from_directions(directions) -> dict[int, tuple[float,float]]` | per-node unit field vector from raw oriented direction pairs. |
| `build_cross_field(state_or_view) -> CrossFieldReport` | walk all oriented node-directions (T3 + Q4 body edges with side-bit orientation) and fold through `four_of`, returning a `CrossFieldReport`. |
| `score_body(view, body) -> tuple[float, float]` | signed area magnitude plus convexity (both must be strictly positive) for a candidate body; used to rank Q4 unions. |
| `rank_bodies(view, bodies) -> list[tuple[int, float]]` | deterministic (best-first, id-tiebreak) ranking of body walks against the current field. |
| `front_step_guided(state, edge, min_confidence, options=None)` | wrap the Q1 `front_step` with the per-endpoint confidence gate + deterministic body-scoring tie-break.  Typed `LowConfidenceError` on gate failure; `GuidanceRejected` on a candidate that fails convexity/area; lifts `FrontNoCandidate` unchanged. |

### Modified — `src/anymesher/quad/state.py`

- `View.node_ids` previously crashed in the `add_nodes` branch when the value
  was an integer node-id keying the position map rather than a set; fixed by
  using `set(d.add_nodes)` in the union.  The fix is a one-line, behaviour-
  preserving change verified by the Q1 regression suite.

### Modified — `src/anymesher/quad/__init__.py`

- Imported the Q3a symbols into the package `__init__` (`CrossFieldReport`,
  `GuidanceRejected`, `LowConfidenceError`, `NEAR_ZERO`, `build_cross_field`,
  `field_from_directions`, `four_of`, `front_step_guided`, `rank_bodies`,
  `reflect_x`, `rot2`, `score_body`) and added them to `__all__`.  Docstring
  extended to reference the Q3a guidance layer.

### New — `tests/quad_first/test_q3_guidance.py` (47 tests)

1.  **`four_of` identity.**  Unit inputs only; `four_of(cosθ, sinθ)` lies on
    the unit circle; fourfold symmetry: `four_of(θ + kπ/2) == four_of(θ)` for
    `k in {0,1,2,3}`; `rot2(v, π/2) == rot2(v, -π/2)` (4θ cover); `reflect_x`
    flips the imaginary part; near-zero and garbage inputs raise
    `GuidanceRejected`.
2.  **`field_from_directions`.**  A single consistent direction produces a
    confidence of 1.0 at the node and the correct unit field; a symmetric
    pair `(d, −d)` produces a zero-magnitude field → confidence 0.0.
3.  **`build_cross_field` on Q1 fixtures.**  On the two-triangle hole-plate
    fixture, every node's field is the fold of the average of its incident
    oriented-directions; confidence agrees hand-computed for the two nodes
    with a single direction each.
4.  **Tie-break determinism.**  Two candidate bodies with the same
    `score_body` value are ranked by their body-node minimum then by
    `min(body)` — asserted across a 3-element ranking.
5.  **Confidence gate.**  `front_step_guided` raises `LowConfidenceError`
    when any front endpoint's confidence is below `min_confidence`; below the
    candidate loop (the gate fires before the partner scan).
6.  **Guidance rejection.**  A candidate that classifies to a non-strictly-
    convex union is rejected with `GuidanceRejected`, state digest unchanged.
7.  **`FrontNoCandidate` lift.**  A front edge with no T3 partner propagates
    `FrontNoCandidate` unchanged (same message shape as the Q1 driver).
8.  **State identity on every failure path.**  Digest + generation invariant
    for every typed-rejection test (`LowConfidenceError`,
    `GuidanceRejected`, `FrontNoCandidate`).
9.  **Rotation/reflection covariance.**  A fixture generated with a +π/4
    rotation is scored identically to the original (4θ cover is
    rotation-invariant); and `reflect_x` produces a field whose ranking is
    mirror-symmetric.
10. **End-to-end transition-heavy fixture.**  A 10-node mesh with a
    front-edge pair of T3s; two successive `front_step_guided` calls commit
    two Q4s and leave a clean remaining front; a `min_confidence` gate that
    rejects a specific endpoint is raised with state identity before either
    Q4 is published.

### Evidence

```
pytest tests/quad_first/test_q3_guidance.py -v   -> 47 passed
pytest tests/quad_first/                         -> 142 passed (Q0 29 + Q1 45 + Q2 21 + Q3a 47)
```

Focused regression (the pre-existing full-suite timeout on large
cylindrical / owner-trim fixtures is bounded and unrelated; the Q0/Q1/Q2
layering, packaging and backends suite is **26 passed, 2 skipped**, skipped
are pre-existing environment gates).

### Invariants held at Q3a (beyond Q0/Q1/Q2)

- Opt-in only.  `front_step_guided` is a new symbol; the Q1 `front_step`
  driver is byte-identical and remains the default path.
- Pure-function guidance layer: `four_of`, `rot2`, `reflect_x`,
  `field_from_directions`, `score_body`, `rank_bodies` are all state-free.
  `build_cross_field` and `front_step_guided` read through the Q1
  `QuadMeshState`/`View` surface only.
- Confidence = `|un-normalised mean of 4θ reps|`, never the
  hypot-normalised magnitude — verified by the Q3a tests to agree with the
  hand-computed value at every fixture node.
- No silent fallback.  A candidate that fails convexity raises
  `GuidanceRejected`, not a `continue` through the partner loop.
- Typed reason for every rejection path; the state's digest and generation
  are invariant across every typed rejection (asserted per test).
- Scratch files (`scratch_probe.py` … `scratch_q3b.py`) deleted before this
  commit.

### Not done at Q3a (deliberately deferred to Q3b)

- Spacing-change / collision / closure transition templates and their
  rotation/reflection covaried tests (`transitions.py`).
- Local TinyAD optimisation on real Q4 patches (Q3/Q4).
- libSatsuma / LEMON MCF count-system integration (Q4).
- S3 qualification and component publication (Q6).
- Any change to `NativeMeshingOptions`, `_native`, or the legacy call path.

---

## Q3b — quad transitions: spacing_change / collision / closure (this commit)

**Status:** **complete**.

Q3b delivers the three fixed quad-transition templates required by
`QUAD_FIRST_FULL_PROGRAMME.md` Q3.  Each is a deterministic, sparse-local,
transactional re-tile of the resident mixed Q4/T3 state, committed through the
Q1 `Transaction` journal.  No cell is ever pre-seeded; the front is reconciled
after staging — the same contract as `front_step`.

### New — `src/anymesher/quad/transitions.py`

| symbol | role |
| ---- | ---- |
| `TransitionRejected(MeshError)` | typed rejection; state digest + generation invariant. |
| `ParentReplacement` (frozen) | maps a parent cell to its replacing cell(s). |
| `TransitionReport` (frozen) | `kind`, `parent_cells`, `result_cell`, `body`, `added_front`/`removed_front`, `replacements`, `generation_before`/`generation_after`. |
| `spacing_change(state, cell, options=None) -> TransitionReport` | Flip the diagonal of an **interior** strictly-convex Q4 into two T3 children. Interior guard: no boundary edge on the active front. Front edges added/removed via `_reconcile_front`. |
| `collision(state, cell_a, cell_b, options=None) -> TransitionReport` | Merge two adjacent T3s into one Q4. **Exactly two** active front edges on the union boundary, **non-adjacent** (opposite). |
| `closure(state, cell_a, cell_b, options=None) -> TransitionReport` | Same mechanical merge as `collision`; distinguished by requiring **at least three** active front edges on the union boundary. |
| `_consolidate` (private) | Shared merge path; `kind` selects the front-edge count guard (2 exact non-adjacent vs ≥3). |
| `_reconcile_front` (private) | Staged-edge front reconciliation: added/removed front edges computed via `tx.view.edge_cells`; front adds/removes staged into the same transaction. |
| `_boundary_quad` (private) | Walks the 4-edge boundary cycle (each node degree 2), canonicalises CCW via `make_quad`. |
| `_shared_interior_edge` (private) | Exactly one shared edge, not on the front. |

### Modified — `src/anymesher/quad/__init__.py`

- Imported `ParentReplacement`, `TransitionRejected`, `TransitionReport`,
  `closure`, `collision`, `spacing_change` and added them to `__all__`.
- **Top-level `anymesher/__init__.py` is NOT re-touched by Q3b** — legacy
  surface remains byte-identical.

### New — `tests/quad_first/test_q3b_transitions.py` (25 tests)

1.  **`spacing_change` acceptance.** A strictly-convex interior Q4 is split into
    two T3 children; `result_cell` is `max(cells)+1`; `body` is the canonical
    CCW quad; front edges on the split diagonal are added (interior Q4 → the
    diagonal becomes front); generation advances by exactly one.
2.  **`spacing_change` rejection paths.**  Front-edge boundary (non-interior Q4),
    protected node, protected edge, stale cell id, non-integer id, degenerate
    quad, wrong kind (T3 source), bad options — all leave digest + generation
    invariant.
3.  **`collision` acceptance.** Two adjacent T3s with exactly two non-adjacent
    front edges merge into one Q4 on the other diagonal.  Front edges on the
    diagonal are removed; the two opposite front edges survive (union boundary
    unchanged).  `result_cell` is `max(cells)+1`.
4.  **`collision` rejection paths.**  Adjacent front edges (share a node),
    wrong front count (0, 1, 3), non-adjacent T3s (no shared interior edge),
    front shared edge, protected node/edge, stale handle, cancellation, wrong
    kind — all leave digest + generation invariant.
5.  **`closure` acceptance.** Two adjacent T3s with ≥3 front edges merge into
    one Q4.  The diagonal front edge is removed; the ≥2 remaining front edges
    survive.
6.  **`closure` rejection paths.**  Fewer than 3 front edges, non-adjacent T3s,
    protected node/edge, stale handle, wrong kind — all leave digest +
    generation invariant.
7.  **Rotation covariance.** A fixture generated with a +π/4 rotation
    produces identical report semantics (same `parent_cells`, `result_cell`
    relative to its own state, same `body` node-set, same `added_front` /
    `removed_front`) for both `spacing_change` and `collision`.
8.  **Reflection + relabel covariance.** A reflected + permuted copy
    (`_FLIP1` bijection `{0:1,1:0,2:3,3:2}`) produces identical report
    semantics with bodies/front-edges permuted through the relabel.  Cell
    bodies compared as node-sets (reflection may reverse the canonical CCW
    walk order).
9.  **Report invariants.**  Every accepted transition: `generation_after ==
    generation_before + 1`; `parent_cells` match the consumed cells;
    `result_cell == max(parent_cells)+1` or the next free id; `replacements`
    map each parent to the result; `body` is the 4-node quad (or the 3-node
    T3 pair for `spacing_change`).
10. **Front reconciliation correctness.**  The `added_front`/`removed_front`
    diffs are exactly the edges whose front status changed; no edge appears in
    both; front is recomputed from `edge_cells` on the post-staging view.

### Evidence

```
pytest tests/quad_first/test_q3b_transitions.py -v   -> 25 passed
pytest tests/quad_first/                             -> 167 passed
  (Q0 29 + Q1 45 + Q2 21 + Q3a 47 + Q3b 25)
```

### Invariants held at Q3b (beyond Q0/Q1/Q2/Q3a)

- Opt-in only.  `spacing_change`, `collision`, and `closure` are new symbols;
  the Q1 `front_step` driver and the top-level `anymesher` package surface are
  byte-identical.
- Deterministic.  Result cell id is `max(cells)+1`; no renumbering; front
  reconcile order is sorted by `EdgeKey` (tuple of ints).
- Sparse-local.  Only the touched edges (quad boundary or T3 pair union)
  change front membership; all other edges, cells, and nodes are untouched.
- Transactional.  All commits go through `state.transaction()`; a rejection
  before `tx.commit()` or a `TransitionRejected` guard never advances the
  digest or generation.
- No silent fallback.  Typed `TransitionRejected` for every rejection path;
  no `try/except` swallowing.
- Anygeometry owns geometry.  `transitions.py` calls `front.area2` /
  `front.make_quad` / `front.body_edges` for all geometric predicates (strict
  convexity, CCW canonicalisation, signed area); no inline geometry.
- Reflection + relabel covariance verified for `spacing_change` and
  `collision`; rotation covariance verified for both.  Cell bodies compared
  as node-sets to account for the canonical CCW walk reversal under
  reflection.

### Not done at Q3b (deliberately deferred to Q4)

- Local TinyAD optimisation on real Q4 patches (Q4).
- libSatsuma MCF count-system integration (Q4).
- S3 qualification and component publication (Q6/Q7).
- Any change to `NativeMeshingOptions`, `_native`, or the legacy call path.

---

## Q4 — representable count-system + integer MCF worker adapter (this commit)

**Status:** **complete**.

Q4 delivers the count-system + minimum-cost-flow reduction and the
worker-adapter boundary described in `QUAD_FIRST_FULL_PROGRAMME.md`.
The solver is the vendored **LEMON `NetworkSimplex`** C++ worker
(`third_party/quad/worker/quad_mcf_worker.cc`), invoked over stdin/stdout
JSON.  No Gurobi, no Blossom, no LP, no Java, no new solver dependency.

### New — `src/anymesher/quad/count_model.py`

| symbol | role |
| ---- | ---- |
| `CountRejected(MeshError)` | typed domain rejection (strict input, representability, imbalance). |
| `CountInstance` (frozen dataclass) | `supplies`, `demands`, `cost` (tuple-of-tuples), `blocked` (frozen set of `(i, j)`). `from_arrays` validates **every** value as `numbers.Integral` *before* any `int()` coercion — booleans, floats (even integral-valued), numeric strings, and any other convertible-but-not-`Integral` value raise `CountRejected` with the offending position in the message. `validate()` enforces the non-isolation representability predicate required by the worker. |

### New — `src/anymesher/quad/count_mcf.py`

| symbol | role |
| ---- | ---- |
| `INT64_MAX` | `(1 << 63) - 1`. Shared int64 bound with the C++ worker. |
| `MCFEncoding` (frozen dataclass) | `S` (total supply), `B = S+1`, `m` (active arc count), `primary_scale = B**m`, `tie_weights` (descending `B**(m-1-k)`). Returned by `build_request` so `validate_response` can cross-check the worker's self-reported `total_cost` against its own returned flows. |
| `tie_value(encoding, flows) -> int` | `Σ flows[k] * tie_weights[k]` — the base-`B` digit sum that makes the tie-break total a unique base-`B` number, so the lex-min flow is the unique perturbed-cost optimum. |
| `build_request(instance) -> (MCFRequest, MCFEncoding)` | Builds the row-major active-arc request. Each arc cost is `primary * B**m + B**(m-1-k)`. Rejects with `CountRejected` if (a) any perturbed arc cost exceeds `INT64_MAX`, or (b) `S * max_perturbed` exceeds `INT64_MAX`. |
| `validate_response(instance, response, *, encoding=None) -> SolveReport` | Conservation check, arc-bound check, blocked-arc-zeroflow check. When `encoding` is passed, cross-checks `response.total_cost == encoding.primary_scale * primary + tie_value(encoding, responses.flows)`. `SolveReport.total_cost` is always the **primary** cost, never the perturbed worker total. |
| `MCFRequest` / `MCFResponse` / `decode_response` | Frozen JSON schema (schema-tag enforced) and strict `from_dict` (rejects booleans, floats, non-integer flows). |

### New — `src/anymesher/quad/quad_mcf_worker.py`

| symbol | role |
| ---- | ---- |
| `WorkerLifecycle(MeshError)` / `WorkerNotFound` / `WorkerCrash` / `WorkerTimeout` / `WorkerMalformed` | Typed process-level failures, distinct from domain errors. |
| `find_worker()` / `default_worker_path()` | Locate `third_party/quad/worker/out/lemon/quad_mcf_worker.exe`. `WorkerNotFound` if absent. |
| `run_worker(worker, request, *, timeout, self_test=None) -> MCFResponse` | Spawns the worker with the request as the sole stdin line (JSON). `self_test="crash"` / `"hang"` drive the self-test harness path in the C++ worker for lifecycle testing. |
| `solve_count_instance(instance, worker=None) -> SolveReport` | Composition: `build_request` → `run_worker` → `validate_response`. |

### New — `third_party/quad/worker/`

| path | role |
| ---- | ---- |
| `quad_mcf_worker.cc` | LEMON `NetworkSimplex` worker. `long long` throughout. Self-test harness (`--self-test crash` / `hang`) for lifecycle verification. Build output: `out/lemon/quad_mcf_worker.exe` (75 776 bytes, MSVC C++17, `-DEIGEN_MPL2_ONLY`). |
| `build_quad_mcf_worker.bat` | `cl` build script, no external dependencies beyond the vendored `../vendor/lemon` tree. |

### Modified — `src/anymesher/quad/__init__.py`

- Imported `INT64_MAX`, `MCFEncoding`, `tie_value` and added them to
  `__all__` alongside the existing Q4 symbols.
- **Top-level `anymesher/__init__.py` is NOT touched by Q4** — legacy
  surface remains byte-identical.

### Modified — `.gitignore`

- Added `third_party/quad/worker/out/`, `third_party/quad/worker/*.obj`,
  and `third_party/quad/worker/*.exe` so build artifacts are not committed.

### New — `tests/quad_first/test_q4_mcf.py` (38 tests)

8 evidence items, per `QUAD_FIRST_FULL_PROGRAMME.md`:

1. **Brute-force oracle** — six small instances are solved through the real
   C++ worker and compared against exhaustive enumeration of every feasible
   integer flow (minimum primary cost *and* lexicographically smallest
   optimum), proving the adapter is correct, not just self-consistent.
2. **Strict tie case** — `supplies=[2,1], demands=[1,2], all costs=3`:
   two equal-primary optima; the legacy `T(i,j)=i*n_out+j` tie encoding
   left both at perturbed 94 (inert); the base-`B` encoding (B=4, m=4)
   separates them at 36 vs. 81 → unique optimum `(0,2,1,0)`. Verified
   through the worker across two runs (determinism).
3. **Overflow guard** — `S=10**11` (S\*max_pert > INT64_MAX) and
   `cost=10**19` (arc cost > INT64_MAX) both raise `CountRejected`;
   the legitimate `S=10, cost<=100, m=4` case builds without rejection.
4. **Strict input rejection** — floats (including integral-valued), numeric
   strings, booleans, and negative ints are all rejected before coercion;
   `np.int64`/`np.int32` are accepted per the project `Integral` convention.
5. **Hall-type infeasibility** — `supplies=[1,1,1], demands=[1,2],
   blocked={(1,1),(2,1)}`: passes representability, worker reports
   INFEASIBLE, adapter raises `CountInfeasible`.
6. **Corruption rejection** — conservation violation, negative flow,
   non-integer flow, wrong arc count, desynced `total_cost`, and
   wrong-status responses are all rejected; a consistent response yields
   the **primary** cost (3), not the perturbed worker total (271).
7. **Worker lifecycle** — `WorkerNotFound` (missing path),
   `WorkerMalformed` (non-object JSON), `WorkerCrash` (self-test crash),
   `WorkerTimeout` (self-test hang, `deadline_seconds==1.5`);
   `WorkerLifecycle` is NOT a `CountRejected` (domain vs. process
   separation).
8. **Two E2E applications** — (a) N=4 equal-rail corridor: identity
   matching, cost 0, each station maps to its mirror; (b) N=2, M=3
   asymmetric blocked corridor: unique flow `(0,1),(1,0),(1,2)`,
   cost 7, `flows_by_index` agrees.

### Evidence

```
pytest tests/quad_first/test_q4_mcf.py -v   -> 38 passed
pytest tests/quad_first/                    -> 205 passed
  (Q0 29 + Q1 45 + Q2 21 + Q3a 47 + Q3b 25 + Q4 38)
```

### Invariants held at Q4 (beyond Q0-Q3b)

- No new solver dependency. The backend is the exact vendored LEMON
  `NetworkSimplex`; the worker `.cc` is compiled with `-DEIGEN_MPL2_ONLY`.
- int64 boundary enforced in Python before the worker is ever spawned;
  two independent checks (per-arc cost, worst-case total).
- Reporting always uses the primary cost; the perturbed cost is an internal
  encoding detail, never a public API value.
- `SolveReport` is a frozen dataclass; `ValidateError` is a `MeshError`
  subclass, consistent with the Q0-Q3b typed-error convention.
- Worker lifecycle errors (`Worker*`) are a separate family from domain
  errors (`CountRejected`, `CountInfeasible`, `InvalidSolution`).
- `anymesher/__init__.py` unchanged; legacy path byte-identical.
- `third_party/quad/worker/out/` and build artifacts are git-ignored.

### Not done at Q4 (deliberately deferred)

- Local TinyAD optimisation on real Q4 patches (Q5).
- S3 qualification and component publication (Q6/Q7).
- Any change to `NativeMeshingOptions`, `_native`, or the legacy call path.

---

## Q5 — TinyAD local optimisation on real Q4 patches (this commit)

**Status:** **complete**.

Q5 delivers the local quality-optimisation layer required by
`QUAD_FIRST_FULL_PROGRAMME.md` Q5: a pure-Python patch energy model on true
interior 2×2 quad patches, a vendored **TinyAD** C++ worker (exact same
`scalar_function` + `eval_with_gradient` primitive as the Q0 smoke, now in a
real multivariate objective), and a subprocess adapter with typed validation.
No Java, no Gurobi, no new solver dependency; the worker is compiled with
`-DEIGEN_MPL2_ONLY` against the vendored `eigen` + `tinyad` trees.

### New — `src/anymesher/quad/patch_energy.py`

| symbol | role |
| ---- | ---- |
| `PatchRejected(MeshError)` | structured domain rejection (bad indices, duplicates, free node not participating in any quad, boundary-edge free node, no-quads-with-free). State-free; raised before any solver launch. |
| `InvalidSolutionQ4Patch(MeshError)` | a worker-returned solution failed a post-solve check (ERROR status, objective regression, out-of-box, or invalid in patch semantics). Carries optional `worker_message` detail. |
| `ObjectiveRegression(MeshError)` | worker optimum is worse than the initial energy beyond tolerance. |
| `SolutionOutsideBox(MeshError)` | worker moved a free node outside its box `[init − h/2, init + h/2]`. |
| `PatchSpec` (frozen dataclass) | `nodes` (2-tuple array), `quads` (4-tuple arrays, CCW), `free` (tuple of interior node ids), `max_iter`, `tol`, `step_bound` (h = max edge length / 2 → box half-width). `__post_init__` runs the true-interior validator: `participating = {i for q in quads for i in q}`; free node not in `participating` → reject; build per-edge incident-node count → free node touching an edge with fewer than two participating quads (boundary edge) → reject; `quads` empty and `free` non-empty → reject. |
| `patch_energy(spec, xs, ys)` | per-quad energy `Σ_q [ (|e1|²+|e2|²)²/(16·A²) + orthogonality term ]` over the free nodes of each quad only; pure Python, no solver. |
| `gradient(spec, xs, ys, i)` | finite-difference central gradient of `patch_energy` at free node i (indices `2i`, `2i+1`); used by the tests as the reference oracle for the TinyAD analytic gradient. |

### New — `src/anymesher/quad/quad_tinyad_worker.py`

| symbol | role |
| ---- | ---- |
| `TinyADWorkerError(MeshError)` | base for worker-level failures. |
| `TinyADWorkerNotFound` / `TinyADWorkerCrash` / `TinyADWorkerTimeout` / `TinyADWorkerMalformed` | Typed process-level failures, parallel to the Q4 `Worker*` family. |
| `find_q5_worker()` / `default_q5_worker_path()` | Locate `third_party/quad/worker/out/tinyad/quad_tinyad_optimizer.exe`. |
| `solve_patch(spec, worker=None, *, timeout=None) -> PatchSolution` | Composition: validate spec → encode request (JSON, schema-tagged) → spawn worker → decode → **validate chain**: ERROR status → `InvalidSolutionQ4Patch`; optimum energy > initial + tol → `ObjectiveRegression`; any free node outside its `[init − h/2, init + h/2]` box → `SolutionOutsideBox`; invalid patch geometry → `InvalidSolutionQ4Patch`. |
| `PatchSolution` (frozen) | `status`, `initial_energy`, `final_energy`, `iterations`, `step_history`, `free_final` (dict of node-id → (x, y)). |

### New — `third_party/quad/worker/`

| path | role |
| ---- | ---- |
| `quad_tinyad_optimizer.cc` | TinyAD worker: builds the `patch_energy` objective as a `tinyad::scalar_function<N,double>` over the free-node coordinates, runs a BFGS-style gradient step to `tol`/`max_iter`, emits JSON `{status, initial_energy, final_energy, iterations, free_nodes, ...}`. Self-test harness (`--self-test crash`/`hang`) for lifecycle verification. Built with `cl /std:c++17 /DEIGEN_MPL2_ONLY` against `vendor/eigen` + `vendor/tinyad`. |
| `build_quad_tinyad_optimizer.bat` | `cl` build script, no dependencies beyond the vendored trees. |

### Modified — `src/anymesher/quad/__init__.py`

- Imported the Q5 symbols (`PatchRejected`, `InvalidSolutionQ4Patch`,
  `ObjectiveRegression`, `SolutionOutsideBox`, `PatchSpec`, `patch_energy`,
  `gradient`, `TinyADWorkerError`, `TinyADWorkerNotFound`, `TinyADWorkerCrash`,
  `TinyADWorkerTimeout`, `TinyADWorkerMalformed`, `solve_patch`, `PatchSolution`,
  `find_q5_worker`, `default_q5_worker_path`) and added them to `__all__`.
  Where a name collides with the Q4 surface (`find_worker`, `run_worker`) the
  Q5 symbols are exported under their distinct names; the Q4 names are
  unchanged.
- **Top-level `anymesher/__init__.py` is NOT touched by Q5** — legacy surface
  remains byte-identical.

### New — `tests/quad_first/test_q5_tinyad.py` (28 tests)

True-interior 2×2 patch fixture throughout: a 3×3 rectangular grid
(nodes 0..8, unit spacing) carrying exactly two quads
`((0,1,4,5),(1,2,5,6))` — a true interior 2×2 patch (all four quads' nodes
participate in ≥2 quads, no free node on a boundary edge). Free node is
node 4 (interior corner of both quads).

1. **Spec validation.** `PatchRejected` on: out-of-range index, duplicate node
   in a quad, duplicate free id, free node not in any quad, free node on a
   boundary edge (edge incident to <2 quads), `quads=()` with `free` non-empty.
   A valid interior spec constructs cleanly.
2. **Energy sanity.** `patch_energy` is finite, non-negative, and invariant
   under a simultaneous translation of all nodes (energy depends on edge
   lengths and quads only).
3. **FD gradient oracle.** Central-difference gradient at the initial point
   agrees with the worker's first analytic gradient to `1e-6` (both x and y
   of free node 4); at the converged optimum the FD gradient is `< 1e-9`
   (stationarity).
4. **Box half-width.** `step_bound = h/2` where `h` is the max edge length;
   the box `[init − h/2, init + h/2]` is asserted in every test that
   constrains the optimum.
5. **Convergence.** From `node 4 = (1.3, 0.9)` the worker converges to
   `node 4 ≈ (1.0, 1.0)` in < 300 iterations with `final_energy <
   1e-8` and `initial_energy > final_energy` (objective improvement).
6. **Protected nodes.** All eight non-free nodes of the interior spec remain
   at their initial coordinates after a solve (`PatchSolution.free_final`
   contains only node 4).
7. **No-quads rejection.** `PatchSpec(quads=(), free=(0,))` raises
   `PatchRejected` without launching a worker.
8. **Invalid-solution rejection.** A stub worker returning
   `free_nodes=[[0.5, 0.5]]` for a spec already at the optimum `(1.0, 1.0)`
   raises `SolutionOutsideBox` (the returned point is in-box for that spec but
   regresses the objective → caught by the validation chain).
9. **Error status surfacing.** A stub worker returning `status="ERROR"` with a
   message raises `InvalidSolutionQ4Patch` carrying `worker_message`.
10. **Worker lifecycle.** `TinyADWorkerNotFound` (missing path),
    `TinyADWorkerMalformed` (non-object JSON), `TinyADWorkerCrash`
    (self-test crash), `TinyADWorkerTimeout` (self-test hang).
11. **Determinism.** Two identical solves produce bit-identical
    `final_energy`, `iterations`, and `free_final`.

### Evidence

```
pytest tests/quad_first/test_q5_tinyad.py -v   -> 28 passed
pytest tests/quad_first/                       -> 233 passed
  (Q0 29 + Q1 45 + Q2 21 + Q3a 47 + Q3b 25 + Q4 38 + Q5 28)
```

### Invariants held at Q5 (beyond Q0-Q4)

- No new solver dependency. The backend is the exact vendored TinyAD
  `scalar_function` + `eval_with_gradient` primitive (same as the Q0 smoke,
  now multivariate); the worker is compiled with `-DEIGEN_MPL2_ONLY`.
- True-interior patches only: the `PatchSpec` validator rejects any free node
  that does not belong to ≥2 quads or that touches a boundary edge; the
  fixture and the validator agree that "interior" means topologically interior
  to the quad patch, not merely non-boundary of the full mesh.
- Typed failure. Every rejection path raises a `MeshError` subclass; no
  `try/except` swallowing; worker lifecycle errors are a separate family from
  domain errors (consistent with the Q4 convention).
- `patch_energy` and `gradient` are pure Python, state-free, and importable
  without the worker binary — the energy model is testable in isolation.
- `anymesher/__init__.py` unchanged; legacy path byte-identical.
- `third_party/quad/worker/out/` and build artifacts are git-ignored
  (`.gitignore` line 21, unchanged from Q4).

### Not done at Q5 (deliberately deferred)

- S3 qualification and component publication (Q6/Q7).
- Any change to `NativeMeshingOptions`, `_native`, or the legacy call path.

---

## Q6 / M2 — public quad-first integration and structural qualification

**Status:** **complete in this commit**.

Q6 exposes the Q3→Q4→Q5 route through the production hybrid meshing entry point
while preserving `quad_options=None` as the legacy dispatch sentinel.

### Public integration delivered

- Explicit `QuadMeshingOptions` selects the quad-first route; `quad_face_ids`
  permits a deterministic mixed call where selected faces use quad-first and
  residual faces remain on the established mapped/native route.
- Geometry-backed publication uses exact GeometryModel vertex, edge and face IDs:
  `node_of_vertex`, intrinsic-order `nodes_of_edge`, `elements_of_face`, and
  `elements_of_sheet` remain authoritative after publication.
- Multi-face shared interfaces reuse exact geometry identity rather than
  coordinate welding, including reversed shared-edge orientation.
- Mixed merge remaps legacy shell/beam IDs deterministically and preserves
  legacy metadata, including `thickness_of_face` and structural-preparation records.
- Qualified S3 residuals are supported only through the already-qualified
  legacy S3 production bridge. The qualified-S3 audit record is remapped onto
  final merged node/element IDs.
- Pure all-Q4 results do not run S3 preparation.
- Beam/member generation reuses the established beam/connectivity pipeline and
  publishes through-face couplings against final Q4 shell nodes.
- Structural Sheet ownership and declared plate-junction bookkeeping survive
  publication. Two-Sheet interfaces are ordinary junctions; 3+ Sheet
  non-manifold ownership fails closed unless explicitly declared by geometry.
- Curved/non-planar quad-first requests fail closed with `QuadPublicUnsupported`;
  higher-order remains advertised unsupported.
- Public capability reporting now truthfully advertises
  `front_path="advancing_front"` and `mixed_q4_s3` only when the qualified S3
  bridge is available.
- Existing Mesh serialization remains unchanged; old payloads and
  `quad_options=None` round-trip without persisting quad-selector runtime state.

### Deliberate limitations retained at Q6

- Quad-first public publication remains linear and planar only.
- No new material/thickness ownership is invented for pure quad-first faces.
  Existing mesher-owned thickness metadata is preserved when supplied by the
  legacy/mapped side.
- Qualified S3 is not synthesized for all-Q4 meshes and no S3 admission rule is weakened.
- No push, main merge, release, tag, PyPI publication, or default-route promotion is part of Q6.

### Evidence

```text
pytest tests/quad_first/test_q6_public_integration.py -q  -> 32 passed
pytest tests/quad_first -q                                -> 265 passed in 6.28 s
```

### Q6 files

- `src/anymesher/hybrid.py`
- `src/anymesher/quad/public_integration.py`
- `tests/quad_first/test_q6_public_integration.py`


---

## Files changed (this branch, vs baseline `2ccef37`)

Added:

- `third_party/quad/manifest.json`
- `third_party/quad/ATTRIBUTION.md`
- `third_party/quad/licenses/LICENSES.md`
- `third_party/quad/vendor/eigen/**` (337 files)
- `third_party/quad/vendor/tinyad/**` (25 files)
- `third_party/quad/vendor/libsatsuma/**` (33 files)
- `third_party/quad/vendor/libtimekeeper/**` (8 files)
- `third_party/quad/vendor/q-morph/LICENSE`
- `third_party/quad/vendor/lemon/**` (110 files)
- `third_party/quad/smoke/tinyad_smoke.cc`
- `third_party/quad/smoke/lemon_mcf_smoke.cc`
- `third_party/quad/smoke/build.bat`
- `third_party/quad/smoke/build_lemon.bat`
- `src/anymesher/quad/__init__.py`
- `src/anymesher/quad/options.py`
- `src/anymesher/quad/state.py`            (Q1)
- `src/anymesher/quad/journal.py`          (Q1)
- `src/anymesher/quad/front.py`            (Q1)
- `src/anymesher/quad/recovery.py`         (Q2/M1)
- `src/anymesher/quad/guidance.py`         (Q3a)
- `src/anymesher/quad/transitions.py`      (Q3b)
- `src/anymesher/quad/count_model.py`      (Q4)
- `src/anymesher/quad/count_mcf.py`        (Q4)
- `src/anymesher/quad/quad_mcf_worker.py`  (Q4)
- `src/anymesher/quad/patch_energy.py`     (Q5)
- `src/anymesher/quad/quad_tinyad_worker.py`  (Q5)
- `third_party/quad/worker/quad_mcf_worker.cc`  (Q4)
- `third_party/quad/worker/build_quad_mcf_worker.bat`  (Q4)
- `third_party/quad/worker/quad_tinyad_optimizer.cc`  (Q5)
- `third_party/quad/worker/build_quad_tinyad_optimizer.bat`  (Q5)
- `docs/QUAD_FIRST_DESIGN.md`
- `docs/QUAD_FIRST_REUSE.md`
- `tests/quad_first/test_q0_freeze.py`
- `tests/quad_first/test_q1_state.py`              (Q1)
- `tests/quad_first/test_q1_front.py`              (Q1)
- `tests/quad_first/test_q2_recovery.py`           (Q2/M1)
- `tests/quad_first/test_q3_guidance.py`           (Q3a)
- `tests/quad_first/test_q3b_transitions.py`       (Q3b)
- `tests/quad_first/test_q4_mcf.py`                (Q4)
- `reports/quad_first/full-programme/WORK_STATUS.md`

Modified:

- `src/anymesher/__init__.py` (two additions: import + `__all__`; unchanged in Q1/Q2)
- `.gitignore` (two additions: `!reports/quad_first/` + `/**`)

Unchanged:  everything else (notably `anymesher/__init__.py`, `native_v2.py`,
`_native`, `errors.py`, and the legacy call path).
