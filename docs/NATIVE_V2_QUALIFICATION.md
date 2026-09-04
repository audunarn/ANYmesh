# Native-v2 qualification

Native-v2 is opt-in in 0.5.0. The accepted legacy reference is commit
`27e428188a891705288fef82bab0b166e330aff2`; default options must preserve its
canonical points, topology, associations, quality decisions, serialization,
and cancellation boundaries.

The evidence runner is `benchmarks/native_v2_baseline.py`. CI runs
`check-evidence`, which first validates the immutable contract and then executes
the fixed bounded native test-node corpus. A `run` requires an exact source commit, install kind,
compiler identity, backend, case and scale; wheel runs additionally require the
wheel SHA-256. It performs exactly one warmup and seven measurements and records
all repetition hashes, topology and Q4 distribution, alignment, serialization
size, quality, cancellation, actual backend, environment, and peak RSS. Windows
and Linux retain paired source and installed-wheel evidence. macOS retains
normal build/smoke coverage. The `500k` and workstation scales additionally
require `--allow-large` and are never CI.

The registered scales and labels can be inspected without running heavy cases:

```text
python benchmarks/native_v2_baseline.py check-contract
```

The CI evidence gate builds the extension and executes the bounded corpus:

```text
python benchmarks/native_v2_baseline.py check-evidence
```

The command first resolves every declared corpus path. Frozen performance
records may be supplied with repeatable
`--evidence-pair LEGACY_JSON FRONTAL_JSON` arguments; every pair is consumed by
the acceptance comparator for backend, environment, target, configuration,
quality, protected topology, associations, ownership, activity, mapped
zero-use runtime/RSS, frontal runtime, and native-v2 RSS.

This bounded gate does not replace platform-scale qualification. Merge remains
blocked until reviewed Windows and Linux source and installed-wheel records
cover the registered 10k, 100k, and 500k scales, including mapped zero-use.
When no pairs are supplied the command reports `not_supplied_merge_blocker`;
missing evidence is never inferred or fabricated.

Each legacy/frontal pair is accepted with the explicit comparator:

```text
python benchmarks/native_v2_baseline.py compare --legacy LEGACY.json --frontal FRONTAL.json
```

Acceptance first requires exact backend, environment, target/configuration,
quality-policy, protected-topology, association, ownership, and activity
contracts. It then rejects any measured quality regression before comparing
performance: 100k Frontal median at most 1.25 times legacy, mapped zero-use median
regression at most 3%, mapped peak-RSS regression at most 5%, and native-v2
peak RSS at most twice legacy. Evidence directories are immutable and named by
source commit and configuration digest.

Compiled metric and topology kernels check Python signals at least every 4096
records throughout material native loops.
Application cancellation remains governed by `cancellation_interval`: the
orchestrator checks before the initial queue and after each configured block of
topology operations, while each native insertion is one bounded linear scan of
the current topology. Near-zero orientation or in-circle predicates decline
native dispatch and use the repository adaptive Python predicate rather than
making a platform-dependent `long double` decision.

The corpus covers planar and curved legacy paths, intersections, declared
junctions, holes, narrow ligaments, mixed mapped/native components, activity,
incremental components, cancellation, and deterministic transforms. Curved
Frontal-Delaunay, global matching, Q-Morph, multiblock expansion, and adaptive
remeshing remain outside 0.5.0.
