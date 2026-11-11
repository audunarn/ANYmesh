# Installed controlled volume delivery — PASS

Fresh Windows CPython 3.11.9 owner/mesher wheels and the registered unchanged
ANYsolid wheel passed the headless controlled example and **3 tests / 0 failures /
0 skips**. This is delivery evidence for an explicit six-Tet4 box fixture: 8 nodes,
12 complete exterior facets, 24 P1 and 81 P2 coefficients. It makes zero native
meshing calls and establishes no generated-mesh or platform qualification.
Public provider capability remains blocked pending the remaining gates.

The example checks owner snapshot readmission, P1/P2 stiffness symmetry, affine
traction equilibrium, full strain/stress tensor recovery, positive analytical
strain energy and body-load resultants. Focused tests additionally check immutable
mesh storage and refusal of stale source provenance. The existing eight-cell
Windows/Linux Python 3.11–3.14 CI owner/consumer lane was inspected and left intact;
no concrete defect or duplicate lane was introduced.

The candidate package inputs were copied outside both source trees and hashed
before building (342 input files). ANYgeometry base commit:
`433ac80da3909630a4b03be219232978a86a0c7d`; ANYmesh base commit:
`bb5e8b7e9c6c45ff965f4c5f8e4506d0017b179f`, with this delivery slice uncommitted.
Package inputs are preserved under
`C:/ANYworkspaceWork/workflow2-20261027/evidence/installed-delivery-20261028-a1/sources`.
Both builds explicitly disabled native extensions; no version bump occurred.

| Wheel | SHA256 |
|---|---|
| anygeometry-0.4.6-py3-none-any.whl | `0b56f76ef50b3a2075acfadd0d07c19b9459f482a5fb3cb805cbc362848711ac` |
| anymesher-0.5.1-py3-none-any.whl | `ea85486c54751c7676c379232c0a2420a0ab85ec0f837f4615c41ee4d8806be9` |
| anysolid-0.1.0-py3-none-any.whl | `601bd500625e6cd95f3a741f4f19b87bdcb4a72a7c7650278fea8be3efc56c3c` |

The fresh venv received only offline `--no-deps` wheel installs. Dependencies were
copied from the existing runtime with versions and 3431 file hashes retained:
NumPy 2.4.6, SciPy 1.17.1, scikit-fem 12.0.2, Shapely 2.2.0 and pytest 9.1.1,
plus the recorded pytest dependencies. The shared runtime was not modified.
`pip check` passed. The six measured package origins (anygeometry, anymesher,
anysolid, numpy, scipy, skfem) all resolve under
`.../installed-delivery-20261028-a1/venv/Lib/site-packages/`.
Every installed wheel payload file matched its archive hash (excluding generated
RECORD and wheel `.data` installation paths). Isolated ordinary owner/mesher
imports left pytetwild, PyVista, VTK, ANYsolid, ANYsolver and NGSolve unloaded.

| Example result | P1 | P2 |
|---|---:|---:|
| Affine equilibrium residual | 3.846150458379885e-15 | 1.2680377731759719e-14 |
| Stiffness symmetry error | 0 | 6.274622939669635e-17 |
| Positive strain energy | 1.615384615384598 | 1.6153846153844142 |

All complete tensor values and load resultants are retained in `example.json`.
The fixture result is deterministic within floating-point comparisons; model
identity metadata uses the native owner's model identity.

The reproducible initial command (working directory the candidate ANYmesh) was:

```powershell
& C:/ANYworkspaceWork/workflow2-20261027/runtime/Scripts/python.exe tools/workflow2_installed_delivery.py --evidence C:/ANYworkspaceWork/workflow2-20261027/evidence/installed-delivery-20261028-a1 --solid-wheel C:/ANYworkspaceWork/workflow2-20261027/evidence/installed-matrix-20261028-a1/wheels/anysolid-0.1.0-py3-none-any.whl
```

Use a new evidence path for a new run. This local run retained two failures and
used the harness's cumulative-budget continuation: first `--resume`, then
`--resume --refresh-example` with the same arguments. The first metadata build
failed with a sandbox-temp atomic-rename access denial; owned `TEMP`/`TMP` fixed
the build. The first example completed its physics assertions but JSON output
rejected NumPy's integer coefficient count; converting that count to Python `int`
fixed serialization. Failed logs and the original example are preserved.
Neither repair changed package source or scientific tolerances.

The exact terminal example and test commands were:

```powershell
$e = 'C:/ANYworkspaceWork/workflow2-20261027/evidence/installed-delivery-20261028-a1'
& "$e/venv/Scripts/python.exe" -I "$e/run/examples/solid_volume_controlled.py" --output "$e/example.json"
& "$e/venv/Scripts/python.exe" -I -m pytest -q -o pythonpath= "$e/run/tests/test_installed_volume_example.py" "--junitxml=$e/junit.xml"
```

Both ran with working directory `$e/run`, empty `PYTHONPATH` and one BLAS/OpenMP
thread. All build/install/venv/origin/check commands and effective environments
are in `processes/*/command.json`; full stdout/stderr and terminal cleanup receipts
are retained alongside them. `manifest.json`, `input-revisions.json`,
`dependency-inputs.json`, `wheel-hashes.json`, `installed-origins.json`,
`junit.xml`, `budget.json` and `handoff.json` provide machine-readable evidence.

Total supervised process-wall use, including the negative check below, was
**77.05900000000838 seconds of 600**; 522.9409999999916 seconds remained. Every process was
bounded to at most 180 seconds and every terminal receipt verified zero owned
active processes. The 60-second reserve was preserved. The venv, artifacts,
frozen sources and diagnostics remain for review; no owned child remains running.

Review status: author inspected, root independent review pending. Not run:
native tetrahedralization, new native rebuilds, broad regressions, eight-cell CI,
macOS/native-core qualification or release/public-provider promotion. Those are
separate gates. No shared installation, primary-checkout edit, commit, push,
publication or numerical acceptance decision was made by this delivery slice.

The final harness's completed-input refresh refusal was then verified through its
actual CLI with `--resume --refresh-example` against this completed evidence under
the same Windows Job supervisor (30-second maximum). It exited with the expected
code **2** and exact message `completed example inputs cannot be refreshed in this
evidence run`. All **350 protected hashes**, including the frozen source inputs,
original manifest, input revisions, example/test scripts, wheels, handoff and
budget, matched before and immediately after the invocation. Only afterward was
its **0.21799999999348074 seconds** charged to the original ledger. Cleanup
verified zero owned active processes; no passed build, example or test was rerun.

The original ledger and handoff remain preserved in
`completed-refresh-refusal/before/`; matching post-invocation snapshots are in
`completed-refresh-refusal/after-before-accounting/`. The exact command,
environment, final harness SHA256, stdout/stderr, raw terminal receipt and fresh
`completed-refresh-refusal/final-addendum.json` retain this negative-path evidence.
The original `handoff.json` remains unchanged as the prior smoke checkpoint; use
the addendum and current `budget.json` for the final resource totals.

## Root finite-delivery reconciliation

Root inspected the authored implementation and reconciled its artifacts with
independent non-author same-family review. The finite local delivery slice is
accepted with the report's explicit limits; no provider/release promotion or
historical failure is waived. Root record:
`evidence/delivery-root-review-20261028.json`; static receipt:
`evidence/w2-final-local-delivery-review-d53851b103` (0.312999999995s, no new
builds/tests/imports/physics). Worker handoffs retain their original author-time
review status; these separate root/non-author records provide final disposition.
