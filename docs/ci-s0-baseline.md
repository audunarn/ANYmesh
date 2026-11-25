# S0 frozen-source baseline in CI

`tools/ci_s0_baseline.py` meshes six fixed faces (planar, cylinder, cone, ruled, coons, native NURBS) twice on
the same CI runtime. The first run uses a frozen old source: ANYgeometry and ANYmesh at the refs in
`tests/_s0_baseline_contract.py::REFS`. The second run uses the current source. The tests in
`tests/test_qf_nurbs_s0_characterisation.py` and `tests/test_worker_isolation.py` then compare the current
bytes with that sealed baseline. Before it meshes anything, the script proves that the native build inputs
(`BUILD_FILES` in `tools/diagnostic_s0_mac.py`) are byte-identical at the frozen mesher ref and at `HEAD`. It
also checks that `pyproject.toml` differs only by the listed exceptions.

## Result handling (owner decision 2026-11-25)

The S0 step runs in both full profiles (`pytest` matrix and `wheel-full-linux`) with `continue-on-error`. The
next step reports its outcome in the job summary and as a warning annotation:

- success: `ANYMESHER_REQUIRE_S0_BASELINE=1` is set, and the byte comparisons run against the baseline.
- failure: `ANYMESHER_S0_BASELINE_UNAVAILABLE=<outcome>` is set. The tests that need the baseline are skipped
  with that reason and every other test runs. A failed S0 never stops a test leg again.

The full matrix runs only on `workflow_dispatch`, the weekly schedule and release tags. Pull requests run the
smoke jobs only. Dispatch the workflow on the PR branch to see the S0 result and the Linux/macOS legs.

## Updating the frozen reference with a build-file change

A reviewed change to any file in `BUILD_FILES` (for example `setup.py`, `MANIFEST.in`, `_native.cpp`, the
native headers, `native.py` or `native_cpp.py`) makes the compiler proof fail until the frozen reference moves.
Move it in the same pull request:

1. Commit the build-file change. Call that commit X. Later commits in the PR must not touch `BUILD_FILES`, or
   X has to be moved again.
2. In a following commit, set the frozen mesher ref to X everywhere it is pinned:
   - `tests/_s0_baseline_contract.py`: `REFS['mesher']`
   - `.github/workflows/ci.yml`: both `.s0-old-mesher` checkouts (`pytest` and `wheel-full-linux`)
   - `.github/workflows/s0-mac-diagnostic.yml`: the `.s0-old-mesher` checkout
   - `tools/diagnostic_s0_mac.py`: `old_m`
   - `tests/test_mesh_caller_ci_contract.py`: `OLD_MESHER_REF`
3. Review the `pyproject.toml` exceptions in `compiler_proof` (`tools/ci_s0_baseline.py`). They are relative to
   the frozen ref, so remove the ones X already contains.
4. If the old ANYgeometry ref cannot run with X, move `REFS['geometry']` and the `.s0-old-geometry` checkouts to
   a compatible commit in the same way.
5. Dispatch the full workflow on the PR branch. State the S0 outcome of every leg in the PR before merging.
   With a normal merge commit, X stays reachable from `main`.

A build-file change merged without these steps leaves S0 reported as failing on every leg until a later PR
moves the reference. Status at the time of writing: the frozen mesher ref `aa19c0f` predates the
`native_cpp.py` changes up to `210e56a`, so S0 reports a compiler-proof failure on `main`.
