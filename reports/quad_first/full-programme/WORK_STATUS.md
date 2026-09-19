# Quad-first Q0 work status

**Branch:** `opencode/quad-first-v1`
**Baseline:** `2ccef37`
**Task ID:** `full-programme` (Q0 scope)
**Status:** Q0 freeze **complete**. Q1+ not yet started.
**Updated:** 2026-09-19

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
- `docs/QUAD_FIRST_DESIGN.md`
- `docs/QUAD_FIRST_REUSE.md`
- `tests/quad_first/test_q0_freeze.py`
- `reports/quad_first/full-programme/WORK_STATUS.md`

Modified:

- `src/anymesher/__init__.py` (two additions: import + `__all__`)
- `.gitignore` (two additions: `!reports/quad_first/` + `/**`)

Unchanged:  everything else.
