# Held geometry capabilities

The owner excluded these two existing capabilities from this release's acceptance on 2026-11-04:

- Conformal meshing of the quarter-cylinder after an axial midpoint split (`test_curved_surface_operations_remain_exact_and_conformal`).
- Exact conformal meshing after dividing the quarter-cylinder into four axial strips (`test_stripping_a_cylinder_keeps_the_surface_exact`).

Decomposition retains circular boundaries, but preparation currently rejects the touching extruded supports as positive-area overlap. Supporting these cases requires geometry-owner work representing their curved boundary contact. Neither capability is qualified by this release. Existing overlap ownership rules and numerical tolerances still apply.

Both original positive test bodies remain executable. Only their exact reproduced MeshError, including the fixture's face pair, is classified as an expected failure. Any other exception or failed assertion still fails. An unexpected complete positive pass fails strictly so the exclusion must be reconsidered. This exclusion grants no waiver to other tests, other geometries, or process safety gates.

The same failures were reproduced with both current sources and the frozen old geometry/mesher sources. Detailed historical evidence remains in the programme's `r10-mesh-operation-overlap-ci` evidence directory.
