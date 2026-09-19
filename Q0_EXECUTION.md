# Q0 — Freeze contracts and reusable sources

Work only in `opencode/quad-first-v1` at the isolated quad-first worktree. Preserve existing defaults, `NativeMeshingOptions`, ANYgeometry ownership, exact shared topology/stations, deterministic behavior, cancellation, and the existing C++17 `_native` ABI.

Audit and pin: Q-Morph `bb798752b998489497f4f38b5989c42d13ac2bc0` (MIT), libSatsuma `4e96979ecb11bbfe8d9c05e8f8be1ecb992ca5fd` (MIT), TinyAD `4b48d1a1a588874556a692a3abbdecd0db4c23e1` (MIT), libTimekeeper `f4f486648faac7c740535678b45b387078e26918` (MIT), LEMON `3c6aa54c62a9b524bb9502872fa172776c1f8244` (Boost-1.0), and resolve an immutable Eigen MPL2-compatible pin/hash with `EIGEN_MPL2_ONLY`.

Create `third_party/quad/manifest.json` and required licence/attribution files; `docs/QUAD_FIRST_DESIGN.md`, `docs/QUAD_FIRST_REUSE.md`; strict separate `QuadMeshingOptions` schema; `reports/quad_first/full-programme/WORK_STATUS.md`; focused Q0 tests. A separate scoped quad-native build skeleton is allowed if useful, but do not change the current `_native` ABI.

Run tiny real libSatsuma/TinyAD build/call smoke tests if feasible. No production Java, Gurobi, Blossom V, QuadWild, GPL dependencies, unpinned install-time fetching, 500k/workstation/equivalent workloads, or slow 10k-hole diagnostic.

Run focused tests, investigate failures, inspect diff, and commit coherent accepted Q0 changes only on this branch. Do not push, update main, tag, release, publish, reset, force-push, or destructively clean. Do not spawn a broad exploration subagent. Stop only for a genuinely critical blocker; otherwise complete Q0 and record evidence in WORK_STATUS.