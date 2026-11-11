# Exact-envelope source candidate build

State: first Linux CPython 3.12+ ABI3 and Windows CPython 3.11 candidates built,
packaged and import-verified. This is build feasibility, not qualification.
Scope: tools/build_* and this note only; external evidence/exact-envelope-*.

The native profile retained 1386 samples; 1349 include
`sample_triangle_and_check_is_out` under `mark_surface_fs`. This motivates a
source candidate using the documented exact-envelope CMake option, not a changed
physical tolerance or a change to upstream algorithms. The candidate is separately
versioned `0.4.2+anyworkflow2.exact1`; its native bytes will differ. The previously
approved optional-PyVista accessor guard remains the only Python behavior patch.
Root additionally authorized the minimal raw-wrapper initialization immediately
after successful `params.init`: `tree.init_sf_tree(input_points, input_faces,
params.eps)` under `NEW_ENVELOPE`, and an NB_MODULE marker derived from that
actual preprocessor macro. fTetWild/FastEnvelope algorithms are not edited.

Plan and acceptance:
1. Capture pinned wrapper v0.4.2, fTetWild d7d99bb4387a07895b9adce058dc7305f6b6e5ab
   and FastEnvelope 520ee04b6c69a802db31d1fd3a3e6e382d10ef98, dependency revisions,
   source hashes/licenses and effective build recipe.
2. Build the smallest viable Linux wheel with exact envelope ON, <=4 compile jobs.
   Preserve configure/compile/link/wheel evidence and the exact flag proof.
3. One import-only smoke in an isolated environment. Do not install into an
   existing runtime or invoke meshing. Assess Windows source build feasibility.
4. Handoff wheel/native hashes, license inventory, command/process receipts and
   remaining gates to root. Root alone adopts artifacts/qualifies execution.

Budget: 1200s per process, 2400s cumulative build wall, including failures.
Owned WSL Ubuntu-24.04 user audunarn; no global settings/credentials changes.
Initial read: canonical solid_volume_contract.md and native profile summary.
Read-only shared Linux/Windows supervisor: ANYsolid/tools/adapter_platform_matrix.py.

Early checkpoint (source preparation complete, build active):
- Evidence root: `C:/ANYworkspaceWork/workflow2-20261027/evidence/exact-envelope-20261028-a1`.
- Wrapper v0.4.2 resolves `85c28acd6e1546b7c19b96fd6a5ee5bc2184b176`;
  fTetWild resolves the requested `d7d99bb4387a07895b9adce058dc7305f6b6e5ab`.
- GCC 13.3.0, CMake 3.28.3, Ninja 1.11.1, Python 3.12; system GMP development
  6.3.0 and MPFR 4.2.1 are present. Build venv is isolated in explicitly owned
  `/tmp/exact-envelope-20261028-a1-build-venv`; build artifacts temporarily use
  `/tmp/exact-envelope-20261028-a1-build`. Durable source/receipts are on C:.
- Source/accessor and raw-wrapper glue diffs retained in `source-preparation.json`,
  `wrapper-glue-patch.json` and `candidate-source.patch`.
- Missing Ubuntu ensurepip caused a retained setup failure; provisioned a fresh
  venv with the already-installed pip `--python` route, no existing runtime edits.
  Downloaded build-tool wheels and exact hashes retained (`build-environment.json`).
- First configure intentionally stopped on root's new wrapper-patch instruction;
  its 39.588s and verified zero-member cleanup are retained, not a build success.
- Revised Linux wheel build active with exact option ON and <=4 compile jobs.
  Last terminal cumulative wall: 66.443s used / 2333.557s remaining (active run
  accounted on termination). No meshing or numerical check has run.
- Windows has Visual Studio Build Tools 18 / MSVC 14.50 installed; Windows GMP/
  compiler/configure feasibility still to inspect after Linux outcome.
- Automatic review rejected one command with a mistyped unauthorized evidence
  root before execution. Corrected command ran in the authorized root; no blocker.

## Linux candidate result

Wheel: `evidence/exact-envelope-20261028-a1/wheels/`
`pytetwild-0.4.2+anyworkflow2.exact1-cp312-abi3-linux_x86_64.whl`.
SHA-256: `56886d352d99175ccb178c491e7845befd178b280144c30e0e598a953a413146`.
Wheel native `PyfTetWildWrapper.abi3.so` SHA-256:
`cd6f2fa00f42c18b03b628e85c492fd11bead3caaf6fa8932001abc95e8c0cd5`.

`linux-wheel-durable-exact1`: 237/237 compile/link steps completed in
303.605221012s process-wall, cleanup verified with zero remaining owned members.
One import-only smoke passed in 1.264259055s in a fresh Python process against
the extracted wheel using the existing NumPy runtime; no installation into that
runtime and no meshing. Actual native `_workflow2_exact_envelope is True`.
`native-flag-proof.json` retains actual Ninja wrapper compilation with
`-DNEW_ENVELOPE`, cache option ON and unchanged fTetWild source status. CMake's
compile_commands database excludes the parent wrapper because the upstream
export flag is enabled only in the fTetWild subdirectory; the first inventory
attempt failed on that bookkeeping assumption, preserving 109.231s evidence.
Inventory completion used actual Ninja rules without rebuilding or re-importing.

## Licenses and corresponding inputs

Do not label the native binary MIT-only or MPL-only. FastEnvelope's top-level
MIT license coexists with compiled indirect-predicate sources carrying
LGPL-3.0-or-later. Nested `3rdparty/i_pred` is commit
`e144a34c85cf25ee64ae9983745b548b27f9d996`; its `lgpl.txt` SHA-256 is
`e3a994d82e644b03a792a930f574002658412f62407f5fee083f2555c5f23118`.
Its exact source/header notices and license text are retained.

`recursive-git-closure.json`: 19 fetched Git roots, including FastEnvelope's
own old TBB, i_pred and nested Geogram submodules. CLI11 1.6.1 is a tarball,
retained separately with archive/content hashes and BSD-3-Clause license.
`recursive-license-inventory.json`: 45 retained notices; build-tool-wheel,
embedded-notice and system-runtime license inventories are separate.
Fetched dependencies are not all necessarily incorporated into the binary;
compile/link records are retained for that future distribution review.

`source-files-sha256.json`, candidate source patch, exact dependency commits,
retained build-tool wheels/versions, durable source/build/venv directories,
`native-build-configuration/` (cache, compile commands, Ninja/link commands,
`ldd`) and `relink-inputs.json` preserve reproducible/relinkable inputs, including
9 static libraries. Linux runtime dependencies include system GMP, libgomp,
libstdc++, libm/libgcc/libc; this is an Ubuntu local `linux_x86_64` wheel, not an
audited manylinux or redistributable release claim. License/distribution
acceptance remains root-owned and pending. No external publication authorized.

## Windows progress

MSVC 19.50.35725.0 / Visual Studio Build Tools 18.3.11520.95, CMake 4.1.2-msvc8,
Ninja 1.12.1 are available. Two initial compiler probes failed C1902 because
CMake still selected Debug/PDB try-compiles; explicit
`CMAKE_TRY_COMPILE_CONFIGURATION=Release` passes, matching the candidate's
Release build without changing numerical compiler semantics or global settings.

Root explicitly adopted the upstream-CI MPIR route. Isolated `windows-sdk`
contains MPIR 3.0.0 `he025d50_1002` (package SHA-256
`c7243b2c3f8e849a9367eab8d77babcd8dd5b828d6b5068441297edc86843b69`),
zlib 1.3.1 and resolved VC/UCRT runtime packages. Exact URLs, package archives,
library/header hashes and shipped licenses are retained. MPIR is LGPL 3;
VC/UCRT redistribution has separate Microsoft terms. Conda registration,
shortcuts, auto-update and global package caches were disabled child-locally.
No existing conda/runtime/configuration was changed.

`windows-wheel-exact1` now builds the source candidate using its own Python 3.11
build venv, isolated MPIR SDK, Release try-compiles, CMake compatibility policy
floor 3.5 for old dependency build scripts and <=4 jobs. Its terminal receipt
will settle the remaining platform gate; no Windows candidate import or meshing
has run. Last terminal cumulative budget before this active process:
777.145024675s used / 1622.854975325s remaining of 2400s.

## First-two-artifact checkpoint before ABI variants

Windows raw wheel SHA-256:
`9adb8197b706be1f86ab573de7db50717ee5d95e2af56bae6e7b33bfa8f12618`.
Windows DLL-packaged wheel (upstream delvewheel 1.13.2 route):
`windows-repaired-wheels/pytetwild-0.4.2+anyworkflow2.exact1-cp311-cp311-win_amd64.whl`,
SHA-256 `6a4742c7719197e7c25e2100aefb377f97ee932e5d9285d978bb14aa28c2bebb`.
Raw and repaired artifacts remain distinct and retained. Repair adds the standard
DLL-loader Python patch and rewrites DLL dependency names; it is not an
unchanged-native/import-only patch. Repaired native Pyd SHA-256:
`9a229a95a3a57eae3cb5c9eaa91fc2f75eb63393427e4f6e0949d07e935b6170`.

Windows source wheel built 238/238 steps in 125.484s. Root separately authorized
one Windows import-only smoke after the Linux smoke was consumed. It passed in
0.422s with actual compiled marker True, version exact1, no PyVista, and all
three loaded mangled DLL paths/hashes proven inside the extracted wheel.
No native meshing/FE calls were made by this worker. SDK DLL origins are proven
using delvewheel's actual dependency-seeded hash algorithm, not guessed from
unseeded source hashes. See `windows-bundled-dll-origin-proven.json`.

`windows-compiler-attestation.json` records compile-time `_MSC_VER=1950`, a
passing signed-char static assertion, preprocessed ISVISUALSTUDIO branch,
no `/fp:fast` or `/fp:contract` in actual compile commands/environment, and
observed Geogram `/fp:strict`. Unchanged i_pred `numerics.h` SHA-256:
`049026be6463e5bbdf994fa13f11182af08e10f65983ad8cc2b8ff6d5162cc23`.
The MSVC-specific branch contains `#pragma fenv_access(on)` and `_controlfp`.
Root owns adjudication of floating-point assumptions; these are compile/header
facts, not numerical qualification. Initial DLL-origin bookkeeping and a missing
native-file-path inventory assumption were corrected with all failed logs kept.

All seven recipe scripts pass AST syntax validation; final hashes are in
`final-tools-sha256.json`. `git diff --check` passed for owned paths.
First-two-artifact terminal cumulative wall: **918.534024675s used**, original
2400s budget **1481.465975325s remaining**. No budget renewal.

Root now authorizes the missing Linux CPython 3.11 and Windows CPython 3.12+ ABI3
variants within this same remaining budget, one fresh import-only check each.
The existing two artifacts/imports will not be repeated. Before reconfiguring
an existing build directory, preserve its original ABI objects/static libraries
and exact build configuration as relink inputs. Reuse native core objects only
with before/after hash, timestamp and command evidence. Existing wheel and all
failed evidence paths remain protected. Four-ABI/platform artifacts still do not
authorize provider admission, broad numerical qualification or publication.

ABI-variant continuation checkpoint: a transient model-capacity interruption
did not restart preparation or renew authority. `linux311-abi-prepare` already
completed in 155.334047670s with zero remaining owned processes. The installed
managed CPython 3.11.15 was used; its isolated NumPy 2.4.6 wheel and build venv
are retained. Original Linux ABI relink inputs were archived before any build
directory reconfiguration. Actual cumulative wall is now 1073.868072345s used /
1326.131927655s remaining of the same 2400s; Linux 3.11 wheel build is next.

## Four-variant terminal checkpoint

All four native ABI artifacts are built and actual import-only checks passed.
Linux311 wheel SHA256 `7c9d42c9d7b86c1c8be8f2f7c129b39c5ef063b6d57744e0f4602f3969269cd4`;
Windows312+ABI3 repaired wheel SHA256
`3deeb27747d193c9ddd494b6f0d5916f835d284d98ab4fd37600aff892267bcb`.
Their per-variant `import-only-smoke.json` files record actual interpreter,
compiled marker, native hashes and no PyVista. Linux311 rebuild251.303s did
not reuse215 core objects (compiler invocation changed); Windows312 build20.875s
reused214 objects with identical hashes/timestamps and commands, while four
objects changed. Complete relink inputs are preserved for the previous ABIs.

Root finished final Windows312 packaging after the owner hit model capacity.
The first smoke reached import then failed DLL comparison because filesystem
paths used backslashes while ZIP inventory used slashes. The actual DLL hashes
were identical. `as_posix()` fixes key normalization; absolute loaded-DLL origin
and byte-hash checks still apply. Failed receipt and original extracted tree
remain intact; repaired import used a separate extraction folder. This is a
reporting repair, no native/algorithm/tolerance change. Independent non-author
same-family Read review accepted the delta.

Original cumulative build budget: **1379.061363631/2400s used**,
**1020.938636369s remaining**. No native meshing/FE in this campaign. All process
receipts verify owned cleanup. These local wheels are not release/manylinux
artifacts; LGPL/MPIR/Microsoft redistribution assessment remains open.
