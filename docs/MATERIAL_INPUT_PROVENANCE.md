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

This receipt does not follow tokens through native insertion, merging or row
permutation. The chart matrix is checked for finite shape and consistency with
the emitted arrays, not independently certified against the physical surface
evaluator. The separate Plane parameter fallback uses other arrays and is not
covered. The receipt gives no permission to subdivide protected curved paths
or publish a connected component. These gaps require separate owner-bound
work before mixed-mesh acceptance can be claimed.
