# Native-v2 0.5.0 qualification: runtime deferral

User direction, 2026-09-09: finish qualification without stopping on runtime
overruns; performance improvement belongs to the next release.

For the current native-v2 and owner-certified cylindrical Frontal-Delaunay
scope, measured runtime regressions are reported limitations, not release
blockers. This supersedes the current-release enforcement of the mapped 3%
runtime ceiling and planar/cylindrical 100k 1.25x legacy timing ceilings.
Further optimization and repeated timing campaigns are deferred. Existing
benchmark algorithms, numerical thresholds, and failed evidence are retained
unchanged for the next release's performance work.

Remaining large-scale execution establishes actual mesh size, completed work,
validity, protected topology, ownership, memory use, and deterministic output.
A single correctness/scaling execution is not labeled a seven-sample timing
qualification. Timing and resource use are still recorded. Safety deadlines
and containment remain enforced; an interrupted or incomplete execution is
never a correctness pass.

This does not waive geometry or mesh correctness, coverage, positive physical
Jacobians, protected/shared identities, cancellation, deterministic behavior,
owner contracts, memory safety/registered memory ceilings, installed-wheel
checks, or terminal normal CI. Failed runtime comparisons must not be relabeled
as passing comparisons. Full performance qualification is deferred rather than
claimed complete.

Preserved mapped evidence:

- `reports/mapped_closeout_71430bf_20260909`: the original 10k request produced
  2,809 elements and was correctly rejected by the scale check.
- `reports/mapped_scale_fix_contract_20260909`: 32 tests passed for the
  benchmark-only sizing correction, committed as
  `da3983343eaf24b85ba5047aeff328716ee3de08`.
- `reports/mapped_scales_da39833_20260909/10k`: both routes produced exactly
  10,000 elements with identical canonical mesh hashes. The recorded medians
  were 1.105552399996668 and 1.2693957000010414 seconds; the original comparator
  failed only its mapped runtime limit. This remains a recorded timing failure.

Quad-first remains deferred and is not represented as implemented. Existing
mapped/automatic defaults remain unchanged. This decision authorizes no upload,
tag, release publication, downstream production edit, or topology relaxation.
