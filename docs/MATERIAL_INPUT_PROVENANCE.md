# Material-region primary input provenance

The invocation-local `MaterialInputProvenanceReceipt` records the exact
source-origin inputs submitted to the primary planar triangulator for a
qualified material region. It is an internal prerequisite for later row-origin
work, not a meshing, quality, solver-admission or publication certificate.

Each boundary and interior path is checked against its ANYgeometry owner and
the complete shared-edge station registry. The receipt records path occurrence,
source-edge direction, exact binary64 station fractions, global node IDs,
boundary/pin tokens, owner UV and the detached chart loops, constraints and
pins passed to the primary native call. It also records every effective
shared-edge selection predicate. A fresh check binds the owner identity,
revision, serialized content, registry, input arrays and immutable entry
signature. A missing retained-vertex mapping or node is refused before the
interior helper can create one.

The input receipt alone does not follow tokens through native insertion,
merging or row permutation. The chart matrix is checked for finite shape and consistency with
the emitted arrays, not independently certified against the physical surface
evaluator. The separate Plane parameter fallback uses other arrays and is not
covered. The receipt gives no permission to subdivide protected curved paths
or publish a connected component. These gaps require separate owner-bound
work before mixed-mesh acceptance can be claimed.

## Opt-in output-row development candidate

The subsequent unreviewed output slice forwards input global IDs to the
triangulator's existing protected-row contract. It shifts constraint input
indices past generated interior points, reconstructs lineage for quality
retries, and keeps the selected triangulation's ID-to-row pairs through
frontal insertion and recombination. The selected core receives a frozen
source-by-row ledger; all other rows are marked generated. Before material
lifting, every protected ID must occur once in active connectivity and the
input receipt is freshly checked. Protected identity is never inferred from
matching coordinates. Native shared-edge insertion still uses its separate
explicit split records and does not turn a generated row into a pre-existing
source station.

This output slice has only focused Python/emulated checks so far. It does not
certify the physical chart transform or the separate Plane parameter fallback,
and it does not establish accepted connected mixed meshing or solver behavior.

## Analytic chart-authority development candidate

For an eligible cone or extrusion, the chart's reference differential is
queried through the original ANYgeometry owner and validated collection. Its
Cholesky transform and inverse are checked for finite, nonsingular metric
consistency and captured with the original owner, face, source checksum,
collection definition and region context in a local entry signature before
the reference derivative callback. Evaluation and Jacobian batches copy submitted rows
before callbacks, then check the same selected chart before and after the
owner-guarded operation. The hybrid material input path binds the emitted
chart transform to that selected entry. This reference metric chooses a
discretization chart; it is not a claim that physical metric is uniform over
the whole surface. The material Plane parameter fallback remains refused
until its separate emitted inputs and output rows are proved.

Collection and region content tokens come from ANYgeometry's public
`chart_definition_fingerprint` API. They compare the original definition with
the live evidence object; public owner binding validation separately checks
that object against the current geometry.
