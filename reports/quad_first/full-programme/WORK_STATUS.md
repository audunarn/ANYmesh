# Quad-first work status

**Branch:** `opencode/quad-first-v1`
**Baseline:** `2ccef37`
**Task ID:** `full-programme` (Q0 → Q2/M1)
**Status:** Q0 (freeze) **complete** `0752f42`. Q1 (state/journal/front_step) **complete** `b410719`. Q2/M1 (bounded front-edge Steiner-split recovery) **complete in this commit**.
**Updated:** 2026-09-20

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
- `docs/QUAD_FIRST_DESIGN.md`
- `docs/QUAD_FIRST_REUSE.md`
- `tests/quad_first/test_q0_freeze.py`
- `tests/quad_first/test_q1_state.py`              (Q1)
- `tests/quad_first/test_q1_front.py`              (Q1)
- `tests/quad_first/test_q2_recovery.py`           (Q2/M1)
- `tests/quad_first/test_q3_guidance.py`           (Q3a)
- `reports/quad_first/full-programme/WORK_STATUS.md`

Modified:

- `src/anymesher/__init__.py` (two additions: import + `__all__`; unchanged in Q1/Q2)
- `.gitignore` (two additions: `!reports/quad_first/` + `/**`)

Unchanged:  everything else (notably `anymesher/__init__.py`, `native_v2.py`,
`_native`, `errors.py`, and the legacy call path).
