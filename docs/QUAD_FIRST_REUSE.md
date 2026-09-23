# Quad-first reuse and attribution (Q0 freeze)

This document is the *reuse contract* for the quad-first programme. It states,
per donor, **what may be adapted, what is excluded, and how it is pinned and
attributed**. The authoritative machine-readable pins are in
[`third_party/quad/manifest.json`](../third_party/quad/manifest.json); the
verifiable human attribution is in
[`third_party/quad/ATTRIBUTION.md`](../third_party/quad/ATTRIBUTION.md) and
[`third_party/quad/licenses/LICENSES.md`](../third_party/quad/licenses/LICENSES.md).

## Ground rules

1. **Pin, don't track.** Every donor is pinned to a specific commit. We do not
   silently follow upstream branches. Re-adopting a newer commit is a deliberate
   Q-maintenance action, not a default.
2. **No install-time network.** `third_party/quad/` is committed. Nothing in the
   vendored set is downloaded at build or run time.
3. **Permissive-only.** The allowed licences are:
   MPL-2.0 (Eigen, chosen over its LGPL/BSD branches), MIT (TinyAD, libSatsuma,
   libTimekeeper, Q-Morph), and Boost-1.0 (LEMON). No GPL/AGPL/LGPL/EPL/CDDL,
   no commercial, no source-available.
4. **Selectivity.** We port/adapt only the routines the programme actually uses.
   We do not drag in unrelated solver/IO/plotting code from a donor.
5. **Attribution travels.** Where a header is copied into `src/`, the original
   copyright line is preserved on the file. The donor's LICENSE is carried under
   `third_party/quad/vendor/<id>/LICENSE`.

## Per-donor reuse contract

### Eigen 3.4.0 — `3147391` (MPL-2.0, `EIGEN_MPL2_ONLY`)

- **Adapted for:** dense linear algebra inside TinyAD, and the `Eigen::MatrixXd`
  numerics behind `lemon` and (at Q4) our count-system solver.
- **Excluded:** the iterative-linear-solvers headers that ship non-MPL-2.0 code
  are *not* exercised (TinyAD's LinearSolver path is MPL2-only in our use); the
  sparse modules are not linked.
- **Pinned & verified:** the `-DEIGEN_MPL2_ONLY` flag is enforced in
  `third_party/quad/smoke/build.bat`; the Q0 TinyAD smoke **PASSES** under it
  (proof the chain is MPL2-clean). The pinned source is `EIGEN_WORLD_VERSION 3,
  MAJOR 4, MINOR 0`.
- **Attribution:** MPL-2.0 text in [`third_party/quad/ATTRIBUTION.md`](../third_party/quad/ATTRIBUTION.md).

### TinyAD — `4b48d1a` (MIT)

- **Adapted for:** `scalar_function<1,double>`, `x_from_data(...)` and
  `eval_with_gradient(...)` — the bounded local Q4 quality optimisation loop
  (Q5).
- **Excluded:** higher-order tensor machinery we do not use is not exercised.
  The `Utils/LinearSolver.hh` path is used only for the dense solve inside our
  bounded local patch; it is not a global solver.
- **Verified:** `third_party/quad/smoke/tinyad_smoke.cc` computes `f` and `g`
  for a small polynomial objective and asserts both. **PASS.**

### libSatsuma — `4e96979` (MIT)

- **Adapted for (planned Q4):** the integer min-cost-flow reduction for the
  nontrivial count systems. At Q0 we **do not vendor the `.cc` files**; we
  vendor the headers so the production path can include them.
- **Excluded:** Gurobi-backed solvers (banned), Blossom-V (banned), Java glue.
- **Verification strategy at Q0:** because the exact reduction libSatsuma's MCF
  path uses is `lemon::NetworkSimplex` (an integer MCF solver), the Q0 evidence
  is the `lemon_mcf_smoke` PASS (§ lemon below) — this is the *same algorithm*
  libSatsuma calls.
- **Attribution:** `vendor/libsatsuma/LICENSE` (MIT).

### libTimekeeper — `f4f4866` (MIT)

- **Adapted for:** deterministic wall-clock + CPU reporting around solver loops.
- **Excluded:** OS-specific timer backends we don't use are not linked (we use
  the Posix/Windows paths already present).
- **Attribution:** `vendor/libtimekeeper/LICENSE` (MIT).

### Q-Morph (KarlLevik/qmorph) — `bb79875` (MIT)

- **Adapted for (planned Q1):** front classification, bounded candidate ranking,
  side/edge recovery/insertion templates, collision/no-progress handling.
- **Excluded (explicitly, per policy):**
  - production Java runtime;
  - viewer/file-loader;
  - static global state;
  - fake quads (quads that "look" right but have no valid residual T3 closure);
  - unbounded retry logic.
- **Q0 state:** **reference only.** The Q0 vendored set carries only the MIT
  LICENSE for Q-Morph; the source is not compiled into the smoke path. This is
  deliberate — the T1 brief states: "Exercise bounded libSatsuma/TinyAD smoke
  calls only if feasible; **do not claim later production integration.**"
- **Attribution:** `vendor/q-morph/LICENSE` (MIT). Full Q-Morph copyright and
  provenance are recorded in `manifest.json`.

### LEMON — `3c6aa54` (Boost-1.0)

- **Adapted for:** `lemon::NetworkSimplex` (the exact MCF algorithm used by
  libSatsuma's MCF path), `ListDigraph`, `ArcMap`/`NodeMap`, and the
  `ProblemType` enum.
- **Excluded (explicitly, per policy):** all LP/MIP solvers (Soplex, Clp,
  GMP-LPK, Minlp). The generated `vendor/lemon/config.h` has every LP/MIP
  `#define` **disabled**, so no LP runtime is linked.
- **Verified:** `third_party/quad/smoke/lemon_mcf_smoke.cc` builds a 4-node CDT
  seed network with integer supplies, costs and capacities and runs
  `NetworkSimplex::run()`. Expected total cost `6`; smoke asserts
  `status==OPTIMAL`, `flow` on each arc, and `totalCost()==6`. **PASS.**

## Excluded by programme policy (no vendor, no pin, no license)

- **QuadWild** — out of scope.
- **Gurobi** — commercial; banned from the runtime path.
- **Blossom-V** — not the selected matching kernel.
- **Soplex / Clp / GMP-LPK** — LP solvers; disabled in LEMON `config.h`.
- **VCGLib, Qt, OpenVolumeMesh** — banned at the programme level.
- **Java VM / any Java runtime** — banned at the programme level.

## Provenance verification (re-runnable)

Each donor archive is pinned by SHA-256 in `manifest.json`; the per-vendor
directory is additionally pinned by a per-tree SHA-256 (over the sorted
`<repo-relative-path>:<file-sha256>` pairs). To reproduce the freeze

from the staging area:

```powershell
# 1. re-download each archive at its pinned commit (URL in manifest.json)
# 2. compare Get-FileHash of the archive to manifest.json
# 3. re-extract, re-copy into third_party/quad/vendor/<id>/
# 4. recompute the per-tree SHA-256 (sorted path:hash join, SHA-256) and compare
```

The per-tree hash is the **operational identity** of the vendored subset: two
different archives whose headers hash to the same per-tree value are
*indistinguishable* for compilation — that is what makes the vendor subset
reproducible without re-tracking the upstream project.

## Patches

**None.** At Q0 no vendor component was patched.
