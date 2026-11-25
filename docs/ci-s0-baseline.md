# S0 frozen-source baseline in CI

`tools/ci_s0_baseline.py` meshes six fixed faces (planar, cylinder, cone, ruled, coons, native NURBS) twice on
the same CI runtime. The first run uses a frozen old source: ANYgeometry and ANYmesh at the refs in
`tests/_s0_baseline_contract.py::REFS`. The second run uses the current source. The tests in
`tests/test_qf_nurbs_s0_characterisation.py` and `tests/test_worker_isolation.py` then compare the current
bytes with that sealed baseline. Before it meshes anything, the script proves that the native build inputs
(`BUILD_FILES` in `tools/diagnostic_s0_mac.py`) are byte-identical at the frozen mesher ref and at `HEAD`. It
also checks that `pyproject.toml` differs only by the listed exceptions.

## Result handling (owner decision 2026-11-25)

The S0 steps run in both full profiles (`pytest` matrix and `wheel-full-linux`). None of them can stop a test
leg:

- The frozen-source checkouts (`.s0-old-geometry`, `.s0-old-mesher`) have `continue-on-error`. If one fails, the
  S0 step is skipped and the outcome is `skipped`. The current-source checkouts stay required.
- The S0 step has `continue-on-error` and writes its baseline binding to `s0-binding.env`, never to the job
  environment.
- `tools/ci_s0_report.py` always runs before pytest, and it is the only S0 step that can fail the leg. First it
  kills every S0 process still running and fails if one survives. Then it exports the binding together with
  `ANYMESHER_REQUIRE_S0_BASELINE=1`, but only after a successful outcome and a well-formed binding. Otherwise
  it exports `ANYMESHER_S0_BASELINE_UNAVAILABLE=<outcome>`. It writes the outcome to the job summary, and a
  warning for anything but success.
- With `UNAVAILABLE` set, the S0 byte comparisons are skipped with that reason and every other test runs. A
  partial binding, or a required but unavailable baseline, still fails.
- The evidence upload has `continue-on-error`.

The full matrix runs only on `workflow_dispatch`, the weekly schedule and release tags. Pull requests run the
smoke jobs only. Dispatch the workflow on the PR branch to see the S0 result and the Linux/macOS legs.

## Updating the frozen reference with a build-file change

A reviewed change to any file in `BUILD_FILES` (for example `setup.py`, `MANIFEST.in`, `_native.cpp`, the
native headers, `native.py` or `native_cpp.py`) makes the compiler proof fail until the frozen reference moves.
Move it in the same pull request:

1. Commit the build-file change. Call that commit X. Later commits in the PR must not touch `BUILD_FILES`, or
   X has to be moved again.
2. In a following commit, set the frozen mesher ref of the full CI to X wherever it is pinned:
   - `tests/_s0_baseline_contract.py`: `REFS['mesher']`
   - `.github/workflows/ci.yml`: both `.s0-old-mesher` checkouts (`pytest` and `wheel-full-linux`)
   - `tests/test_mesh_caller_ci_contract.py`: `OLD_MESHER_REF`
3. Move the `pyproject.toml` proof together with the ref. `compiler_proof` in `tools/ci_s0_baseline.py`
   hard-codes the old/new runtime dependencies and the two dev additions as differences between the frozen ref
   and `HEAD`. Rewrite those assertions to the differences between X and `HEAD`, which are none when X is the
   build-file commit itself. Update any test that pins them in the same commit.
4. If the old ANYgeometry ref cannot run with X, move it the same way. That means `REFS['geometry']`, both
   `.s0-old-geometry` checkouts in `ci.yml`, and `OLD_GEOMETRY_REF` in `tests/test_mesh_caller_ci_contract.py`.
5. Do not move the manual Mac diagnostic as part of this. `.github/workflows/s0-mac-diagnostic.yml` and
   `tools/diagnostic_s0_mac.py` (`REFS` `old_g` / `current_g` / `old_m` / `current_m`) pin a reviewed set of
   old and current sources plus one wheel artifact run. Its four refs and the artifact provenance move only
   together, when that diagnostic is rebased against matching source and wheel evidence.
6. Dispatch the full workflow on the PR branch. State the S0 outcome of every leg in the PR before merging.
   With a normal merge commit, X stays reachable from `main`.

A build-file change merged without these steps leaves S0 reported as failing on every leg until a later PR
moves the reference. Status at the time of writing: the frozen mesher ref `aa19c0f` predates the
`native_cpp.py` changes up to `210e56a`, so S0 reports a compiler-proof failure on `main`.
