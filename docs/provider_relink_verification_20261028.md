# Recipient relink verification, 2026-10-28

The copied recipient inputs successfully link and import on Windows Python 3.11
and explicit WSL `Ubuntu-24.04` Python 3.12. These are new **unqualified native
derivatives**, separate from all delivered/qualified wheels. No native meshing
function was called. This closes the tested technical link/import gap only.
Root independent review is pending.

Plan/acceptance: (1) copy the two delivered relink directories and matching
notice-refreshed wheels into fresh evidence work directories, verifying their
input identities; (2) execute the supplied recipes with compatible installed
toolchains, preserve any failure and repair only a copied recipient recipe;
(3) replace only the copied extension, retain the copied wrapper/bootstrap and
dependencies, and import the marker and callable exports without calling them.
No scientific thresholds, failed evidence or original artifacts may change.

Evidence: `../evidence/provider-relink-20261028-a1/`. `handoff-final.json` indexes the
results; `input-identities.json`, `work/*/effective-link*.json`,
`work/*/raw-link-output.json`, `work/*/derivative.json` and `work/*/import.json`
record input/output hashes, compiler/runtime, exact commands and import origins.
`budget.json` and `processes/*/result.json` preserve statuses, precise supervised
process-wall durations and verified owned-process cleanup. Full stdout/stderr
are retained beside each receipt. Original exact-envelope and notice evidence
were read only. The two original delivered wheels still match their prior hashes.

| Tested recipient | Effective tools/runtime | Result |
|---|---|---|
| Windows311 | MSVC linker 14.50.35725.0, Python 3.11.9, Python311 import library, retained MPIR SDK gmp.lib | Original recipe links; derivative imports after new-extension DLL-name adaptation |
| Linux312abi3 | Ubuntu 24.04, GCC 13.3.0, Python 3.12.3, installed GMP/OpenMP | Separate LF recipe copy links; derivative imports |

Both imports expose `_workflow2_exact_envelope == True` and callable
`tetrahedralize_mesh`/`tetrahedralize_csg`; neither export was called. The imported
extension path/hash is the derivative's, and package metadata remains
`0.4.2+anyworkflow2.exact1`. Twelve Windows and eight Linux other runtime files
retain their delivered hashes, including wrapper/bootstrap and bundled DLLs.
The package directories are import-only work products, not regenerated wheels;
their preserved original RECORD is not a claim that the substituted extension
has its original hash.

Raw Windows native SHA256:
`2b1758c4d09e952c4360d17f13c68973c75a43d68e9e6c05dcc5aa48389e193d`.
Adapted Windows native SHA256:
`b440b52b1934a83736d16a6ed84ad0349f07edf91f22200a040d0f24cde21fe7`.
Linux native SHA256:
`be9b2f9e34c0429e1cd61fac92c8fec969e171b4280f469ad647cca8ca834541`.
Different bytes are expected from this recipient relink and establish no
numerical equivalence. Prior qualified native hashes remain in `derivative.json`.

The delivered Linux `relink.sh` contains four CRLF line endings and failed under
`sh` with `set: Illegal option -`. The failing receipt/logs and copied original
remain. The separate `work/linux312abi3/relink/relink.lf.sh` replaces exactly
`b'\r\n'` with `b'\n'` (569 to 565 bytes); no other text changed.
`linux-recipe-repair.json` is the exact byte transformation and before/after
hash receipt. The final audit asserts this equality against the preserved copy.
The original payload was not repaired or relabelled.

The canonical generator is now fixed: `write_posix_relink_recipe()` explicitly
writes UTF-8/LF and `export_relink()` uses that helper. The executed pre-fix
generator is preserved as `package_exact_provider_notices.pre-fix.py` (SHA256
`b68cbb8af025a68aa98d2afa000619505a482d4662923de9f7c8191ae57ece30`).
The helper-generated Linux312 recipe matches the already-successfully-linked LF
copy byte for byte (SHA256
`06a00679fcac957d9cc5974f95edc03c6257acca88272a8d0fcf1bb173d82120`).
The Windows-host regression calls the complete export path against synthetic
input archives/build directories and checks both Linux ABI recipes' exact LF
shell boundaries, output names and UTF-8 round trip. Historical payloads and
wheels are unchanged; the separate LF file remains the executed overlay.

Windows links against ordinary `mpir.dll`, `MSVCP140.dll` and `CONCRT140.dll`
names; the delivered wheel bootstrap resolves its hashed DLL names. The harness
uses installed delvewheel 1.13.2/pefile solely to adapt the **new extension's**
three import names to the existing copied DLLs. It verifies the resulting import
table and all other runtime hashes. `windows-native-adaptation.json` records
the exact mapping and tool source hashes. No dependency/bootstrap was rewritten.

For recipient use, copy the matching `payload-final/relink/<ABI>` directory and
extract the matching delivered wheel to a separate directory. On Windows enter
an MSVC14.50 x64 developer environment, set `PYTHON_LIB_DIR` to compatible
Python311 `libs` and `MPIR_LIB_DIR` to the MPIR development `lib` directory, then
run `relink.cmd` from its directory. The resulting extension requires the three
recorded DLL-name adaptations before substituting it into this delivered wheel's
bootstrap. The `package()` implementation in `tools/verify_provider_relink.py`
shows that bounded operation using already-installed delvewheel; it does not
install dependencies. On Linux, make a separate LF copy with
`Path('relink.lf.sh').write_bytes(Path('relink.sh').read_bytes().replace(b'\r\n', b'\n'))`
and run `sh relink.lf.sh` from the copied directory with GCC13, GMP and OpenMP.
Substitute the new `.abi3.so` only into the separately extracted matching wheel.
Keep original wheel and raw/native outputs separately identified.

Reproduction commands from this checkout (the one-time evidence root must be
fresh; existing failed evidence is deliberately not overwritten):

```powershell
& ../runtime/Scripts/python.exe -B tools/verify_provider_relink.py run checks
& ../runtime/Scripts/python.exe -B tools/verify_provider_relink.py run prepare
& ../runtime/Scripts/python.exe -B tools/verify_provider_relink.py run link --abi windows311
& ../runtime/Scripts/python.exe -B tools/verify_provider_relink.py run package --abi windows311
& ../runtime/Scripts/python.exe -B tools/verify_provider_relink.py run smoke --abi windows311
& ../runtime/Scripts/python.exe -B tools/verify_provider_relink.py run repair
wsl.exe -d Ubuntu-24.04 --exec /mnt/c/ANYworkspaceWork/workflow2-20261027/linux-runtime/bin/python -B /mnt/c/ANYworkspaceWork/workflow2-20261027/ANYmesh/tools/verify_provider_relink.py run link-repaired --abi linux312abi3 --child-python /mnt/c/ANYworkspaceWork/workflow2-20261027/exact-runtime-linux312/bin/python
# Same Linux controller/child with run package, then run smoke, --abi linux312abi3.
& ../runtime/Scripts/python.exe -B tools/verify_provider_relink.py run audit
& ../runtime/Scripts/python.exe -B tools/verify_provider_relink.py run generator-check
# The final focused retry was run focused-tests; both receipts are preserved.
& ../runtime/Scripts/python.exe -B tools/verify_provider_relink.py run finalize
```

Checks actually run: six focused unsafe-path/input-identity/mutation-refusal
checks passed; Windows link exit zero, package pass and isolated import pass;
Linux repaired recipe link exit zero, package pass and isolated import pass;
final manifest/runtime/repair/cleanup audit passed. Two failed attempts remain:
the first combined Windows link/package action linked successfully but failed
because delvewheel globally enables pefile fast loading; explicit
`fast_load=False` fixed package inspection without repeating the successful
link. The original Linux recipe failed on CRLF. Its controller then failed
updating the ledger because its receipt already included a `command` field;
the exact 0.148125451-second child receipt was recovered without rerunning it.
That first Linux runner hash is unavailable; this evidence limit is explicit.
Later attempts retain runner snapshots and hashes. The small unsupervised ledger
recovery has no separately measured process duration; its encompassing shell
call's tool-reported wall was 0.8284962 seconds, including the already-accounted
repair action. It is not represented as precise child process-wall timing.

Generator validation: `runtime/Scripts/python.exe -I -B -m pytest -q
tests/test_exact_provider_notices.py --junitxml=<evidence>/generator-focused-repaired.xml`
finished with **12 passed, zero failures/errors/skips**. The prior attempt had
11 passed and one new-test failure because its synthetic fixture included both
nanobind ABI library variants in the nine-library closure. Correcting that fixture
resolved the failure; both XML reports and full logs remain. The generator-to-LF
recipe comparison passed before the first test invocation and was not repeated.

Supervised child process usage: **10.677205986963447 / 600 seconds**, leaving
589.3227940130365 seconds before any root review work. Each process limit was
180 seconds; non-smoke/check actions reserved 90 seconds. All cleanup receipts
verify no remaining owned processes. No timeouts or denied WSL launches occurred.

Not verified: Windows312abi3/Linux311 recipient relinks, modified i_pred source
rebuild, numerical/physics equivalence, archive-member/header final retention,
licensing/source closure or Microsoft redistribution conditions. No full C++
build, downloads/installs, native generation, qualification rerun, provider/release
promotion, merge, push or publication occurred. The accepted case-specific thin
discrepancy does not alter these technical/release limits or historical thresholds.

## Root finite-delivery reconciliation

Root inspected the authored implementation and reconciled its artifacts with
independent non-author same-family review. The finite local delivery slice is
accepted with the report's explicit limits; no provider/release promotion or
historical failure is waived. Root record:
`evidence/delivery-root-review-20261028.json`; static receipt:
`evidence/w2-final-local-delivery-review-d53851b103` (0.312999999995s, no new
builds/tests/imports/physics). Worker handoffs retain their original author-time
review status; these separate root/non-author records provide final disposition.
