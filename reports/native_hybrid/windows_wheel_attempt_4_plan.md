# ANYmesh Windows Installed-Wheel Qualification Attempt 4

## Status and authority

This document is a bounded qualification addendum to:

- Governing plan: `C:\Users\AudunArnesenNyhus\Downloads\ANYmesher_native_hybrid_mesher_plan.md`
- Compiled-triangulation addendum: `C:\Github\ANYmesh\reports\native_hybrid\compiled_triangulation_addendum.md`
- Historical compiled-triangulation approval baseline SHA-256: `A64E3DC1DC7733A6ED065E85C7475B49A1071840BE7A075D976F13935CFFBD95`
- Current compiled-triangulation addendum SHA-256: `E368269C7B9EC2C3E9912EA9BACCB78B3D004330A8F5297082AEB198D19F4A92`

It replaces the combined build/install design used by wheel-smoke Attempts 1-3 with two independently reviewed stages. It does not authorize either stage. Each stage requires a separately reviewed literal command and an explicit ecosystem performance lease.

The qualification source is pinned to:

- Repository: `C:\Github\ANYmesh`
- Branch: `native_hybrid_mesher`
- Commit: `574fac99db064cc447bdb3e91ff029047a3c2248`
- Git tree: `c3d6a66aaeeab4cc1c7770f2d3290112f0c55a33`
- Commit time: `2026-08-12T21:42:13+02:00`
- Sorted UTF-8/LF source-path manifest: 106 files, 3,250 bytes, SHA-256 `CB3AA53B849774B234AFF5342A224E16CDC4AE4FE4EEB8AF19B9FA743164E9C3`

The claim boundary is Windows AMD64 with CPython 3.13.9 only. Neither stage supports a performance, scaling, publication, non-Windows, wheel-matrix, or byte-reproducibility claim.

## Preserved prior attempts

The following artifacts are immutable inputs, not passing wheel evidence:

- Attempt 1: `C:\Github\ANYmesh\reports\native_hybrid\wheel_smoke_windows.json`, SHA-256 `FEC903E15C24A8A028C7A3B084F52A3A32192650E3E7637480E855EA0171E813`
- Attempt 2: `C:\Github\ANYmesh\reports\native_hybrid\wheel_smoke_windows_attempt_2.json`, SHA-256 `A6E3BA9E570986109FDED3BDD9C13F508DC5FDD4AE4549C0FC029DAEED2512F6`
- Attempt 3 expected report: `C:\Github\ANYmesh\reports\native_hybrid\wheel_smoke_windows_attempt_3.json`, required absent
- Attempt 3 command SHA-256: `B9146655853CD3822A1B6D49DA04FEF7BD8718D7DD936C86401DBF43CE9F42F2`
- Attempt 3 result: exit 124 after 191.5 seconds, no accepted report, no surviving TEMP root, no surviving run process, and no installed-behavior claim
- Durable Attempt 3 receipt to create before Stage B: `C:\Github\ANYmesh\reports\native_hybrid\wheel_smoke_windows_attempt_3_timeout_receipt.json`

Attempts 1 and 2 must hash identically before and after both stages. The Attempt 3 report must remain absent. The durable receipt must record only already accepted facts, must be content-hashed before Stage B, and must not imply behavior qualification.

## Stage A purpose

Stage A creates one immutable ANYmesh wheel bundle. It performs no installed-wheel behavior test. Its only positive claim is that the exact pinned source produced one structurally valid Windows CPython 3.13 wheel under the recorded build identity.

Stage A must not use NumPy, ANYgeometry, source-tree imports, an editable install, a package index, a pip cache, CMake, Ninja, a GPU, or a network connection.

## Stage A exact paths and fresh-target policy

All work occurs outside the repository except the already registered plan and command artifacts.

- Canonical LocalApplicationData TEMP parent: `C:\Users\AudunArnesenNyhus\AppData\Local\Temp`
- Fresh Stage A TEMP root: `C:\Users\AudunArnesenNyhus\AppData\Local\Temp\anymesher-wheel-attempt-4-stage-a-574fac99-v1`
- Bundle parent: `C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\anymesher\0.2.1\windows-cp313\attempt-4\bundles`
- Fresh same-volume staging bundle: `...\bundles\.stage-a-574fac99-v1.partial`
- Successful final bundle: `...\bundles\sha256-<bundle_index_sha256>`
- Failure parent: `C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\anymesher\0.2.1\windows-cp313\attempt-4\failures`
- Fresh atomic failure report: `...\failures\stage-a-574fac99-v1-failure.json`
- Fresh failure partial: `...\failures\stage-a-574fac99-v1-failure.json.partial`

The literal command must canonicalize every parent and exact leaf, reject reparse points, refuse every pre-existing TEMP, staging, final, failure, and failure-partial target, and remove only paths placed in an explicit ownership ledger by that run. It must never delete a computed or foreign path. Stage A has no overwrite policy and no retry policy.

## Stage A immutable offline build inputs

The build venv is created fresh without `--system-site-packages`. Its bundled pip is pinned by the CPython ensurepip artifact:

- `C:\Python\Python313\Lib\ensurepip\_bundled\pip-25.2-py3-none-any.whl`
- 1,752,557 bytes
- SHA-256 `6D67A2B4E7F14D8B31B8B52648866FA717F45A1EB70E83002F4331D07E953717`

The six acquired, non-yanked offline build wheels are:

1. `C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\pypi\build-stack\setuptools\83.0.0\setuptools-83.0.0-py3-none-any.whl`, 1,008,090 bytes, SHA-256 `29B23C360F22F414DC7336BB39178CC7BCBF6021ED2733CDE173F09DBA19ABB3`
2. `C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\pypi\build-stack\wheel\0.47.0\wheel-0.47.0-py3-none-any.whl`, 32,218 bytes, SHA-256 `212281CAB4DFF978F6CEDD499CD893E1F620791CA6FF7107CF270781E587ECED`
3. `C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\pypi\build-stack\packaging\25.0\packaging-25.0-py3-none-any.whl`, 66,469 bytes, SHA-256 `29572EF2B1F17581046B3A2227D5C611FB25EC70CA1BA8554B24B0E69331A484`
4. `C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\pypi\build-stack\pyproject-hooks\1.2.0\pyproject_hooks-1.2.0-py3-none-any.whl`, 10,216 bytes, SHA-256 `9E5C6BFA8DCC30091C74B0CF803C81FDD29D94F01992A7707BC97BABB1141913`
5. `C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\pypi\build-stack\colorama\0.4.6\colorama-0.4.6-py2.py3-none-any.whl`, 25,335 bytes, SHA-256 `4F1D9991F5ACC0CA119F9D443620B77F9D6B33703E51011C16BAF57AFB285FC6`
6. `C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\pypi\build-stack\build\1.5.0\build-1.5.0-py3-none-any.whl`, 26,018 bytes, SHA-256 `13F3EECB844759AB66EFEC90CA17639BBF14DC06CB2FDF37A9010322D9C50A6F`

The six wheels total exactly 1,168,346 bytes. Stage A rehashes every input before and after the run.

The offline install order is exactly the numbered order above. Every install uses the fresh venv interpreter with `pip install --no-index --no-deps --no-cache-dir`. Installed versions and origins must resolve under the fresh venv. `pyvenv.cfg` must state `include-system-site-packages = false`. Pip must remain version 25.2 and originate under the venv.

## Stage A toolchain identity

Stage A fails closed on drift from these identities:

- PowerShell: `C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe`, 454,656 bytes, SHA-256 `7600FFE12DA441FE89D035B13801E8E91D064BC544A27B19A5CF49F6AB8B18F5`, file version `10.0.26100.8875`
- Git: `C:\Program Files\Git\cmd\git.exe`, 46,920 bytes, SHA-256 `7B7971DD13F0C3A284E538601F2F9770B3A87DFACCB5FB52D68141C67ED22364`, version `2.55.0.windows.3`
- CPython executable: `C:\Python\Python313\python.exe`, 105,816 bytes, SHA-256 `08A64DC73AC3E3776B49F0097C6306BDB9C8F7990A037065213324D328467BF5`, version `3.13.9`
- CPython DLL: `C:\Python\Python313\python313.dll`, 6,125,912 bytes, SHA-256 `C9F98606D0D06F4E8AE75AE385021E58B57C90D4FD325C0313C8C42ABE1EBF63`
- CPython import library: `C:\Python\Python313\libs\python313.lib`, 368,882 bytes, SHA-256 `D4E5CA91FDDE3D8FAB4A2276CC329ABE4E63481279294634842B35673539A316`
- CPython public header anchor: `C:\Python\Python313\Include\Python.h`, 4,178 bytes, SHA-256 `1092F5E36A87909D0B0F5D0B0D8F8505454753C99A65C115DF396BC13CED8CD0`
- MSVC compiler: `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\HostX86\x64\cl.exe`, 604,744 bytes, SHA-256 `FD30D75E6AA319673CF3A4F56AEB3A1D6106AFF87360B78966A7C5783567B78A`, file version `19.50.35725.0`
- MSVC linker: `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\HostX86\x64\link.exe`, 2,921,032 bytes, SHA-256 `69EE768E3BA674087B8644E1D46B81379BEF1580470461E61535A1075D1A0957`, file version `14.50.35725.0`
- MSVC header anchor: `...\MSVC\14.50.35717\include\vcruntime.h`, 11,833 bytes, SHA-256 `C301388C27F581C3A85257A70E892705A5712F32391D9D0211B486907BE3D60E`
- MSVC runtime library anchor: `...\MSVC\14.50.35717\lib\x64\vcruntime.lib`, 285,180 bytes, SHA-256 `F2BD58D07CB4CF5A85978EE5807A2666742F9DA0AA71FEA1E3FE034997EC6653`
- MSVC C++ runtime library anchor: `...\MSVC\14.50.35717\lib\x64\libcpmt.lib`, 22,491,130 bytes, SHA-256 `3C6F83F971466C768B532B4A672A511ACF4B145E8A7DC38CB065A613F1E310A7`
- Windows SDK version: `10.0.26100.0`
- Windows SDK header anchors: `um\Windows.h`, 7,511 bytes, SHA-256 `B337D661D03A4ABEFB7B86A2742CE1AD5D19B57CD8B858BD13E7BBCC1DBEEAAA`; `ucrt\corecrt.h`, 127,273 bytes, SHA-256 `822E503B81DD7B3D7DF93CA22FCED3672A5154484FD42054D5941E619BCF6CBC`
- Windows SDK library anchors: `um\x64\kernel32.lib`, 311,908 bytes, SHA-256 `341C7D56125A03B458E4D5093E4C79B33123CCFDFD610FE236937B8E6F3134BB`; `ucrt\x64\ucrt.lib`, 285,588 bytes, SHA-256 `7EF4EAC926BF597D2F243F16CDFED7E0DB22CB3CA34A1D7E088A84C994A03D66`

The exact runtime reports `Windows-11-10.0.26200-SP0` and `MSC v.1944 64 bit (AMD64)`. CMake and Ninja must be absent and unused. The raw build log records the actual compiler/linker commands and options. Stage A uses `python -m build --wheel --no-isolation`, `ANYMESHER_REQUIRE_NATIVE=1`, `PYTHONHASHSEED=0`, and `SOURCE_DATE_EPOCH` derived from the pinned commit time. It does not add unreviewed optimization or linker flags.

The header/library entries are identity anchors for the selected installed MSVC/SDK roots, while the raw compiler/linker command lines are the authority for paths/options actually selected. This is an environment identity claim, not proof that every installed SDK file participated in the build.

## Stage A source and wheel checks

The worker creates a fresh Git ZIP archive from the pinned commit. It requires:

- Exact HEAD and tree IDs before and after
- Exactly 106 non-directory archive members
- No duplicate, absolute, parent-traversal, backslash, drive-qualified, or empty member path
- Exact sorted UTF-8/LF member-path manifest SHA-256 `CB3AA53B849774B234AFF5342A224E16CDC4AE4FE4EEB8AF19B9FA743164E9C3`
- Extracted file paths exactly equal archive file paths
- Archive bytes and SHA-256 recorded, with no cross-run byte-reproducibility claim

The wheel must be exactly `anymesher-0.2.1-cp313-cp313-win_amd64.whl`. It must contain one and only one `anymesher/_native*.pyd`, one METADATA record declaring version 0.2.1, one matching WHEEL tag, one RECORD covering every member, no duplicate or unsafe member, and no source-tree path. Normalizing names and specifiers through `packaging` must produce exactly these two base runtime requirements and no others:

- `numpy>=1.26`
- `ANYgeometry>=0.2,<0.3`

Duplicate requirements, extras on either base requirement, direct URLs, environment markers, and any unapproved runtime requirement fail closed. RECORD must contain exactly one row for every wheel member and no extra row. Every non-RECORD member must have a `sha256=` digest and exact decimal size that both validate against its bytes; the RECORD self-row alone must have empty digest and size. Stage A records the complete sorted member list, normalized requirement records, RECORD verification results, and wheel hash.

## Stage A watchdog and resource gate

The coordinator starts one `-NoProfile -NonInteractive` worker behind a named startup gate, assigns it to a Windows Job Object before releasing the gate, and captures complete stdout and stderr into one labeled UTF-8 build log.

- Internal worker-tree deadline: 210 seconds
- Internal Job Object memory limit: 1,610,612,736 bytes (1.5 GiB)
- Combined coordinator plus Job Object hard limit: 1,879,048,192 bytes (1.75 GiB)
- Cleanup/process-accounting margin: 60 seconds
- Requested outer lease: 300 seconds, under 2 GiB total RAM, one coordinator plus its one job-controlled worker tree, no GPU/network
- Poll/accounting: job PID list, total processes, active processes, terminated processes, total user/kernel time, coordinator working set, current summed job-process working set, peak process memory, peak job memory, and peak observed combined coordinator/job working set
- Enforcement: every poll sums the coordinator working set and current working sets for the exact Job Object PID list; crossing the 1.75-GiB combined threshold terminates the complete Job Object and fails the run even if the Job Object's independent 1.5-GiB limit has not fired
- Timeout or memory breach: terminate the complete Job Object, wait at most 15 seconds, close the job handle, clean only owned paths, and preserve failure truth without retry

A success requires worker exit 0, no active job process, no relevant owned process, peak job memory below its hard limit, peak observed combined memory below its hard limit, complete TEMP cleanup, cleared task environment, restored PATH, exact source inventories, and unchanged input/prior hashes.

## Stage A success bundle and hash DAG

The staging directory must contain exactly four files:

1. `anymesher-0.2.1-cp313-cp313-win_amd64.whl`
2. `build.log`
3. `build_report.json`
4. `bundle_index.json`

The hash graph is acyclic:

- Pinned source, command/worker hashes, offline wheel hashes, and toolchain identities feed the build.
- `build_report.json` records and hashes the wheel and `build.log`, but never names or hashes `bundle_index.json` and never contains the final bundle digest/path.
- `bundle_index.json` hashes the exact wheel, `build.log`, and `build_report.json`; it does not hash itself.
- The final directory leaf is `sha256-<SHA-256(bundle_index.json)>`.

Only after all source/input/prior/process/environment/cleanup gates and every staged member hash pass may one same-volume `Directory.Move` publish staging to the final content-addressed directory. A post-move rehash must pass. A wheel without its matching log, report, and index is invalid and must not survive.

## Stage A failure truth

No failed run may publish a bundle or wheel. A failure report is eligible only after command/worker/source/input/prior identities passed and final cleanup, process, environment, and source/input/prior rechecks are clean. The atomic JSON failure report embeds complete worker stdout/stderr, exit/timeout/resource accounting, and the primary error. If identity or finalization gates fail, the command reports to the console and writes no durable report. The failure path is fresh and separately registered; it never becomes Stage B input.

## Stage B purpose and prerequisites

Stage B performs no build. It begins only after independent acceptance of one Stage A bundle path, index SHA-256, index member hashes, and wheel SHA-256. The accepted values are then substituted into a frozen Stage B command and standalone probe; both exact byte hashes require separate review before a lease.

## Stage B exact paths and ownership policy

- Canonical LocalApplicationData TEMP parent: `C:\Users\AudunArnesenNyhus\AppData\Local\Temp`
- Fresh Stage B TEMP root: `C:\Users\AudunArnesenNyhus\AppData\Local\Temp\anymesher-wheel-attempt-4-stage-b-574fac99-v1`
- Final report: `C:\Github\ANYmesh\reports\native_hybrid\wheel_smoke_windows_attempt_4.json`
- Fresh report partial: `C:\Github\ANYmesh\reports\native_hybrid\wheel_smoke_windows_attempt_4.json.partial-attempt4-stage-b`

The Stage B coordinator canonicalizes every repository, artifact, TEMP, report-parent, final-report, and report-partial path. Every existing path component must be a plain non-reparse directory or the exact expected plain file. The exact TEMP leaf, final report, and report partial must all be absent before work. The run maintains separate prospective and owned-path ledgers, adds a file or directory to the owned ledger only immediately after successful create-new creation, and cleans only owned paths whose canonical parent and exact leaf revalidate. It has no overwrite or retry policy.

The registered final report is the sole permitted source-tree inventory delta. Pre/post tracked, untracked, and ignored inventories must otherwise be byte-identical after excluding only that exact final path and its exact partial path. A report partial may never survive.

Stage B rehashes:

- Every Stage A bundle member against the accepted index
- `bundle_index.json` against the accepted content-addressed directory leaf
- The accepted ANYmesh wheel
- NumPy 2.4.3 wheel: `C:\Users\AudunArnesenNyhus\AppData\Local\ANYmesh\qualification-artifacts\pypi\numpy\2.4.3\numpy-2.4.3-cp313-cp313-win_amd64.whl`, 12,312,824 bytes, SHA-256 `0A60E17A14D640F49146CB38E3F105F571318DB7826D9B6FEF7E4DCE758FAECD`
- ANYgeometry 0.2.1 wheel: `C:\Github\ANYgeometry\dist_gap_closure\anygeometry-0.2.1-py3-none-any.whl`, 274,758 bytes, SHA-256 `99D3035806E109341E92475B555D21CA89EBB12E6D9410C13132920122CA5E95`
- Attempts 1 and 2 and the durable Attempt 3 timeout/absence receipt

The fresh no-system-site venv install order is NumPy, ANYgeometry, then ANYmesh. Every install is `--no-index --no-deps --no-cache-dir`. All three versions and origins must lie under the fresh venv and outside every repository, pip cache, user site, and source checkout.

Before and after the worker, Stage B revalidates the exact PowerShell executable bytes/version, CPython executable bytes/version, CPython DLL bytes/version, and CPython ensurepip wheel bytes/hash recorded by Stage A. The fresh venv must report `include-system-site-packages = false`; pip must be exactly 25.2 and originate under that venv both before and after the three offline installs. Any tool, origin, version, environment, bundle, dependency, prior-evidence, or source-identity drift fails closed.

The frozen standalone probe must establish:

- Installed compiled capability is present and importable; absent extension and corrupt ABI remain distinct
- Public `auto` selects and records actual provenance `anymesher-cpp17`
- Explicit `python` runs the identical constrained fixture containing a hole
- Python and compiled results have equal canonical topology/coverage/boundary incidence and no outside, duplicate, overlapping, or nonmanifold cells
- Strict native mode does not silently fall back
- One-ULP near-degenerate orientation and incircle results match the Python oracle
- A second identical auto/native run has the same canonical hash and backend provenance
- NumPy, ANYgeometry, and ANYmesh origins/versions are recorded; ANYgeometry schema-4/public API availability is checked without parsing its documents

Stage B writes only the fresh atomic report `C:\Github\ANYmesh\reports\native_hybrid\wheel_smoke_windows_attempt_4.json`. It uses the exact path and ownership policy above, refuses overwrite, and preserves source and prior/bundle/artifact identities. The report records the accepted bundle path and index hash, every accepted bundle member path/bytes/hash, all three dependency wheel paths/bytes/hashes, Attempts 1 and 2 paths/hashes, the durable Attempt 3 receipt path/hash, command/probe hashes, pre/post PowerShell/CPython/DLL/ensurepip/pip identities, pre/post source inventories, final environment/cleanup/process checks, worker/job/coordinator resource accounting, installed origins/versions, and every behavior result and canonical hash. The report is written to its create-new partial, flushed, re-read, schema/identity checked, and atomically renamed only after every final eligibility, identity, resource, process, environment, cleanup, and source gate passes. It may truthfully record `success=false` only after all those gates pass; identity or finalization failure emits no report.

Stage B uses a job-controlled worker tree with a 60-second internal deadline, a 805,306,368-byte (768 MiB) hard job-memory limit, a 939,524,096-byte (896 MiB) combined coordinator/job hard limit, and a 30-second cleanup margin. Its requested outer lease is 120 seconds and under 1 GiB, leaving a separate 30-second outer slack beyond the worker deadline plus cleanup margin. It polls the exact Job Object PID list and enforces/evidences coordinator, job, and combined current/peak memory using the same method as Stage A. It uses no build, network, GPU, source import, editable install, retry, benchmark, or performance claim.

## Registered implementation artifacts and delivery

The following paths are mandatory delivery artifacts. At final delivery, an exact `git check-ignore` gate determines handling per path: ordinary untracked/unignored paths use a normal exact-path add, while only paths proven ignored use exact-path force-add. No blanket force-add is permitted:

- `C:\Github\ANYmesh\reports\native_hybrid\windows_wheel_attempt_4_plan.md`
- `C:\Github\ANYmesh\reports\native_hybrid\attempt4\stage_a.ps1`
- `C:\Github\ANYmesh\reports\native_hybrid\attempt4\stage_a_worker.ps1`
- `C:\Github\ANYmesh\reports\native_hybrid\attempt4\stage_b.ps1`
- `C:\Github\ANYmesh\reports\native_hybrid\attempt4\stage_b_probe.py`
- `C:\Github\ANYmesh\reports\native_hybrid\wheel_smoke_windows_attempt_3_timeout_receipt.json`
- `C:\Github\ANYmesh\reports\native_hybrid\wheel_smoke_windows_attempt_4.json`

Before execution, each existing command/probe/receipt artifact must have a registered UTF-8/LF SHA-256 and must rehash identically from disk. After completion, exact Git blob/content hashes must be recorded for every delivered artifact. Only an exact registered path proven ignored by the final gate may be force-added. Stage A's external bundle is delivered by its absolute path and accepted index/member hashes, not copied into Git.

## Review and execution order

1. Register this addendum path and SHA-256.
2. Create and register the exact Stage A coordinator/worker bytes, hashes, and literal invocation.
3. Obtain a Stage A performance lease and run exactly once.
4. Independently review and accept the Stage A content-addressed bundle.
5. Create the durable Attempt 3 receipt and finalize Stage B's exact accepted bundle substitutions, command/probe bytes, and hashes.
6. Obtain a Stage B performance lease and run exactly once.
7. Preserve passing or failing evidence without retry and report the remaining platform/wheel-matrix gates.

No stage, retry, wheel publication, default change, resolver edit, benchmark, or closeout follows merely from registration of this addendum.
