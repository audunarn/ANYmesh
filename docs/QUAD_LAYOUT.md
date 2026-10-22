# Quad-first layout and edge alignment

When quad options are supplied, an omitted `layout_policy` now selects
`"adaptive"`. An explicit `"existing"` remains available. Routes without quad
options retain their previous layout and continue to reject an explicitly
requested adaptive quad layout.

Adaptive planar rectangular bays use the canonical boundary stations to form
tensor rows. Collinear source edges split by stiffeners or intersections may
form one side when opposite station distributions match. The grid follows the
actual rectangle, including rigid rotations, rather than global axes. These
regular grids bypass local smoothing so their right angles survive the front
pipeline. Final coverage, topology, quality and high-order checks still run.

The additional split-side path does not add, move or remove boundary stations.
It declines holes, nonrectangular or curved boundaries, nonuniform size fields,
incompatible opposite stations, excessive aspect ratios and oversized grids.
Those regions retain the established adaptive triangulation/front route.
Admission has a bounded station scan and a 250,000-node tensor allocation cap.
Cancellation remains active during construction.

This is a structured-interior improvement, not an all-quad promise for arbitrary
geometry. Holes, oblique intersections and changes in boundary division count
may still require local triangles or distorted transition cells.
