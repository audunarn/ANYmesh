# Remaining provider ABI recipient link/import proof

Scope: Windows `windows312abi3` and Linux `linux311`; technical relink/import only.
Exactly zero meshing calls, no numerical/physics qualification, dependency installs,
public enabling, version change, publication or Git delivery. W2DEV001, thresholds
and raw NO_GO evidence remain preserved.

Plan and acceptance:

1. Preserve prior executed runner; isolate a fresh 600 supervised process-second
   campaign (each process at most 180 seconds, 90-second integration reserve).
   Accept when snapshots and immutable input identities exist before each step.
2. Verify changed profile/refusal controls and prepare recipient copies. Reuse the
   prior 12 generator tests only with the same generator hash. Accept when copied
   inputs match the delivered manifest and Linux recipe changes only CRLF to LF.
3. Relink using existing compatible SDKs; adapt only the new Windows derivative's
   DLL dependency names. Import with matrix Python 3.12 Windows and Python 3.11
   Linux. Accept version, marker, callable exports and exact binary identity;
   preserve all wrapper/bootstrap/dependency bytes and call no exports.
4. Audit hashes, receipts, cumulative budget and zero-owned-process cleanup;
   hand off for independent non-author review by the parent. Failed evidence stays.

Dependencies: slice 3 uses slice 2 outputs; audit uses both ABI import records.
Risks: stable-ABI SDK mismatch, hashed Windows dependency names, Linux recipe
line endings and wrong WSL default distribution. Explicit Ubuntu-24.04 and a
Linux 3.12 pidfd-capable controller contain the latter two execution risks.

Evidence root: `C:/ANYworkspaceWork/workflow2-20261027/evidence/provider-relink-remaining-20261028-a1`.
Prior runner snapshot SHA256:
`fa1c5d64095446c6f53f4590df7a5c6efbf722e67e9842e8b2c0ca57064793d6`.

Implementation is directly owned by the assigned OpenAI worker because the parent
has cached Claude subscription and Mistral quota unavailability for this milestone.

Status: verified against assigned technical link/import criteria; independent
non-author parent review pending. Both imports report version
`0.4.2+anyworkflow2.exact1`, the exact-envelope marker and callable
`tetrahedralize_csg`/`tetrahedralize_mesh`; neither export was called.
Windows Python is 3.12.13; Linux Python is 3.11.15. The Windows SDK is the
existing matrix runtime's `libs/python3.lib`, with MSVC linker 14.50.35725.0
and Windows SDK 10.0.26100.0. Linux uses the existing GCC 13/system GMP/OpenMP
recipe. Exact paths, hashes and compiler output are in `work/*/effective-link*.json`.

| ABI | Raw link SHA256 | Imported derivative SHA256 | Other runtime files unchanged |
| --- | --- | --- | --- |
| windows312abi3 | `ad21fd960c357a863fdc71a0ec6b6307deaa5b0a20d6b72d7798016275ce8532` | `73945e802c4b8ecdc4039991126d5d062d2f4958ff9e92ec7a9f4ee09c990d56` | 12 |
| linux311 | `30e7be4493a6bb43cff604f626a3c9df2b5cc6e8fda5ee815fac4c6003de8ad8` | `30e7be4493a6bb43cff604f626a3c9df2b5cc6e8fda5ee815fac4c6003de8ad8` | 8 |

Only the new Windows extension's MPIR/MSVC dependency import names were adapted
to the delivered hashed DLL names. No wrapper, bootstrap or dependency byte
changed. Linux original recipe SHA256
`1aa63f0fb7a55263a93602ad12dcc618e1e7dd3fcd86edd455f6dc6ce6fd51c3`
was preserved. Its separate executed LF copy SHA256 is
`4ef9cfe6f1e4398f3d548438273bd47593ba2f00e1053fb18731a77df2009882`;
exactly four CRLF pairs were replaced, with no other content change.

Tests run: changed registration/refusal controls **10 passed, 0 failed**;
both link/package/import sequences and final audit passed. Ten supervised
processes completed with exit 0 and verified zero owned processes remaining.
There were no campaign failures. Precise process-wall total:
**10.067699122997904 seconds**, leaving **589.9323008770021 seconds** of 600.
This is supervisor process-wall time, not pytest duration or WSL startup time.
No timeout exceeded 180 seconds. The 90-second reserve was retained during
preparation/link/package stages and available for final import/audit.

The prior **12 generator tests (0 failures/errors/skips)** were reused after
matching generator SHA256
`a957c33fbe8b24b3ac9ba74753441b096438082af204ec86688a7cdf130662e3`.
No prior Windows311/Linux312 proof or generator suite was repeated. Final runner
SHA256 is `a935059e06e870a8fae821d44f6c4aaf2de65d597727745c589ff86fcab43cbc`;
all ten per-step executed snapshots have this exact hash. No post-pass runner
refactoring was performed.

Exact commands are retained in `budget.json` (`attempts[].command`) and the
effective link files. The outer command forms used these values:

```powershell
$py = 'C:\ANYworkspaceWork\workflow2-20261027\runtime\Scripts\python.exe'
$r = 'C:\ANYworkspaceWork\workflow2-20261027\ANYmesh\tools\verify_provider_relink.py'
& $py -I -B $r run prepare
& $py -I -B $r run checks
& $py -I -B $r run repair
& $py -I -B $r run link --abi windows312abi3
& $py -I -B $r run package --abi windows312abi3
& $py -I -B $r run smoke --abi windows312abi3 --child-python C:\ANYworkspaceWork\workflow2-20261027\evidence\installed-matrix-20261028-a1\cells\windows312\venv\Scripts\python.exe
wsl.exe -d Ubuntu-24.04 -- /mnt/c/ANYworkspaceWork/workflow2-20261027/linux-runtime/bin/python -I -B /mnt/c/ANYworkspaceWork/workflow2-20261027/ANYmesh/tools/verify_provider_relink.py run link-repaired --abi linux311
wsl.exe -d Ubuntu-24.04 -- /mnt/c/ANYworkspaceWork/workflow2-20261027/linux-runtime/bin/python -I -B /mnt/c/ANYworkspaceWork/workflow2-20261027/ANYmesh/tools/verify_provider_relink.py run package --abi linux311
wsl.exe -d Ubuntu-24.04 -- /mnt/c/ANYworkspaceWork/workflow2-20261027/linux-runtime/bin/python -I -B /mnt/c/ANYworkspaceWork/workflow2-20261027/ANYmesh/tools/verify_provider_relink.py run smoke --abi linux311 --child-python /mnt/c/ANYworkspaceWork/workflow2-20261027/evidence/installed-matrix-20261028-a1/cells/linux311/venv/bin/python
& $py -I -B $r run audit
```

Full receipts/stdout/stderr are in `processes/`; per-step prelaunch identities and
complete runner snapshots are in `step-identities/` and `runner-snapshots/`.
`input-identities.json` retains exact notice wheel, original native payload,
manifest and recipient object/archive input hashes. `handoff.json` combines
derivative/import records, preserved-runtime counts, recipe transformation and
the reused generator source/test evidence. Its budget reference deliberately
points to the completed `budget.json`, which includes the audit receipt.

Not run or established: numerical equivalence, real meshing/physics calls,
modified-library rebuild, final license/source/retained-symbol/header or Microsoft
redistribution closure, qualification promotion, release or publication approval.
Both new derivatives remain unqualified. No commit was created. Concurrent
workflow and Windows-native tooling edits belong to the parent/other owner and
were left untouched.
