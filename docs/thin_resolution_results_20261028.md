# Fresh thin-resolution study — NO_GO for the proposed envelope

The two-call study is terminal. Source/runtime/input identities were frozen and
verified in a fresh process before either native call. No limits were relaxed,
no geometry repaired and no retries performed. Both native calls are consumed.

| Case | Cells | Max axial span/thickness (limit0.5) | Max regular-tet condition (limit32) | Result |
|---|---:|---:|---:|---|
| Aligned0.08×0.03×0.01m, h0.0025m |12940|0.49754309315662026|5.019503770641241|PASS including22P1/P2 checks|
| Rotated negative extrusion,4mm offset, same size/h |11670|0.5021240286845365|6.522175924665449|NO_GO: sampling fails|

The rotated case passes production topology/coverage/ownership, volume, face
area, normals, physical boundary tolerance, groups and conditioning. Its
sampling excess is not a roundoff ambiguity. Physics is intentionally not run
after this geometric failure. The required rotated-offset sampling envelope is
therefore **not qualified**; the aligned pass cannot waive it.

Full-lifetime native-worker kernel peak RSS: aligned95002624bytes,
rotated98377728bytes. These include interpreter/NumPy/provider imports and
serialization, not incremental native allocations or the parent's FE memory.
They are measured local baselines; no speed/memory improvement over the older
unmeasured timeouts is claimed.

Evidence root: `evidence/thin-resolution-20261028-a1`. Frozen identity
`19908113aba71307a4ab63253a32d6df4ea13ec5f895b65c62c1c8effc621cf4`.
Exact commands and terminal cleanup are in its process receipts; per-case raw,
canonical arrays, face/group ownership, metric distributions, original native
logs, physics journals and full failure are preserved. The bookkeeping failure
after successful controlled tests was repaired from the retained receipt; no
test was rerun to recreate missing evidence. Instrumented process-wall total
304.8633017330033/1200s; unused time does not authorize extra native calls.

Public provider remains disabled. Source/criteria/cleanup received independent
non-author same-family review before execution; root adjudicated these terminal
predicates. This outcome does not supersede prior failed evidence or qualify
coupling, curved geometry, arbitrary mesh resolutions or a new sizing policy.
