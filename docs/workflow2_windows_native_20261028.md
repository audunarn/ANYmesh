# Windows native Workflow 2 compatibility

State: verified against the assigned Windows native W2 compatibility criteria after root release. Final root acceptance and independent non-author final inspection pending.

Plan and acceptance:

1. Freeze copied ANYgeometry6a72e9d and ANYmesh2a9268e build inputs and focused test files, plus the unchanged registered ANYsolid wheel. Build native Windows Python3.11 wheels offline. Accept only geometry ABI3 and mesher CP311 wheels with installed `.pyd` modules, native enabled, and source imports absent.
2. Collect the frozen owner/serialization, volume validation/process/consumer and three controlled example tests. Root reviews the exact manifest and collection before contract execution. Accept only the frozen selection with zero real native-provider meshing calls.
3. Run released contracts and the controlled P1/P2 example outside source trees. Retain exact commands, JUnit, outputs, wheel hashes, process-wall ledger and verified Windows Job cleanup. Update the existing candidate CI lane to describe pure/native profiles without claiming remote execution.

The preparation and execution share1200 supervised process seconds, at most300 seconds per process with120 seconds reserved. Old qualification failures, closed budgets, scientific tolerances and W2-DEV-001 remain unchanged. No real TetWild generation, large PDE, unrelated shell sweep, publication or merge is authorized.

Evidence directory: `../evidence/windows-native-w2-20261028-a1`. Tool: `tools/workflow2_windows_native.py`.

Open risks are native compilation/toolchain compatibility, installed native owner behavior on Windows, exact frozen collection, and independent review. Build/setup failures remain evidence and may be repaired within the same budget; product-source defects require root adjudication.

Observed preparation: CPython3.11.9 x64, native geometry ABI3 built with MSVC19.50, native mesher CP311. Both `.pyd` files import from the private installed runtime, all wheel payload bytes match, geometry reports native, disable flags are0, require flags are1, and `pip check` passes. No native-provider meshing occurred.

The prospective manifest records exact commands, all input hashes, compiler/build information through the installed-origin receipt, expected118 node IDs and the collected inventory hash. Four amendments preserve all previous drivers/manifests: actual kernel witnesses, explicit opt-in isolation from default suites, direct native predicate API correction, and independent review's exact-inventory/environment acceptance repair. Each changed selection was collected again; no contract tests were replayed.

Acceptance requires118 passed,0 failed,0 skipped,0 deselected, all setup/call/teardown phases, both named compiled-operation witnesses and complete matching JUnit counts. Missing/truncated session or XML evidence cannot produce a verified handoff. The driver and supervisor hashes must match the prospective manifest. External pytest options/plugins and Python optimization/startup controls are sanitized.

Synthetic refusal controls ran with:

```text
../runtime/Scripts/python.exe -I tools/workflow2_windows_native.py guard-controls --evidence ../evidence/windows-native-w2-20261028-a1
```

Result:12 passed,0 failed. Controls cover deselection, skipped phase, missing witness, missing phase, incomplete session, bad exit, missing/truncated/skipped/incomplete JUnit and hostile inherited environment. Full receipts and `guard-controls.xml` are retained. Preparation/collection/control supervised process wall is85.984 seconds of1200, with verified zero-active-process cleanup for every completed owned Job.

Wheel SHA256 identities:

| Artifact | SHA256 |
|---|---|
| ANYgeometry0.4.6 cp311-abi3-win_amd64 | `eb8d8df54bd474ce137f1c12c69dc62b80088ad883668ac5131646cd6ade8e71` |
| ANYmesher0.5.1 cp311-cp311-win_amd64 | `7e8fc75cc4e1cc382f995e275002713e1d75ad852ba33d167e699184ea91ea18` |
| unchanged ANYsolid0.1.0 pure wheel | `601bd500625e6cd95f3a741f4f19b87bdcb4a72a7c7650278fea8be3efc56c3c` |

The existing candidate CI lane now declares pure/native Windows/Linux Python3.11–3.14 profiles with explicit expected imports and required native kernel operations. Pull requests use the immutable tested geometry candidate; manual dispatch validates its supplied40-character SHA. The lane has not been run remotely, and local Windows3.11 evidence does not establish the entire hosted matrix.

Released execution command, from the ANYmesh checkout:

```text
../runtime/Scripts/python.exe -I tools/workflow2_windows_native.py verify --evidence ../evidence/windows-native-w2-20261028-a1
```

The released manifest SHA256 is `217f47f08211c88484ac8f32c3364c5841b6465ed73de3fe1a899074a09126f8`. Root's prospective approval and release are retained. No code, test or manifest changed after release.

Observed result:118 passed,0 failed,0 skipped,0 deselected; exact118 selected IDs,354 successful setup/call/teardown reports, both required compiled witnesses and complete118-case JUnit independently satisfy the terminal guard. Contract process wall was8.844 seconds. `pytest.xml`, `test-inventory.json`, `test-phases.jsonl` and `validated-outcomes.json` retain evidence.

The controlled example passed for P1/P2 on8 nodes,6 tetrahedra and12 exterior facets. Equilibrium residuals were `3.846150458379885e-15` and `1.2680377731759719e-14`; native-provider meshing calls were0. Example process wall was1.203 seconds. No numerical tolerance changed.

Final supervised process wall is96.031 seconds of1200, remaining1103.969 seconds, with120 seconds reserve retained. All16 owned processes completed, all Windows Job cleanup checks verified,0 active owned processes remain and0 process attempts failed. Twelve synthetic guard controls plus118 contract tests passed. `budget.json` retains exact unrounded process-wall accounting and every command/output/receipt. There was one contract profile execution, with no passed profile replay.

Not run: real native volume generation, historical/full shell gates, a large PDE, remote hosted matrix, publication/release acceptance. Existing Linux native evidence and frozen campaign failures were not replayed or waived. Product source files, sibling repositories and primary C:/Github checkouts were not edited.
