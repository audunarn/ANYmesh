# Prospective thin-region acceptance

State: **criteria adopted by root after independent non-author review**. Execution
requires the READY manifest, successful immutable freeze and fresh-process
identity verification, within the two-call1200s registration. The previous five
geometry runs and eight installed cells remain immutable evidence; their
unregistered quality distributions are not reclassified as accepted.

The remaining question is whether a fresh, explicitly sized thin offset region
can meet a useful geometric sampling/conditioning envelope and the unchanged
finite-case elasticity checks. This study uses two fresh0.08×0.03×0.01m prisms,
target edge length0.0025m, physical boundary tolerance1e-6m. One is axis aligned
with zero offset; one uses negative extrusion,4mm offset, translated/rotated
coordinates and explicit millimetre-to-SI conversion. Source/input/runner/backend
identities and both snapshots are frozen before either native call.

Proposed additional criteria:

- Let d be the unit physical extrusion director and t the absolute extrusion
  thickness after SI conversion. Every tetrahedron must have
  (max_i(x_i dot d)-min_i(x_i dot d))/t <=0.5. A mesh-covered line segment with
  axial extent greater than t/2 requires at least two tetrahedral intervals.
  Boundary tolerance alone does not prove coverage of every exact-source line.
  This is geometric sampling,
  not a uniform layer count, an edge-length guarantee or mechanical accuracy.
- Maximum spectral condition number of the affine map from a unit regular
  tetrahedron is32. With edge-column matrix J and regular-reference edge
  matrix W, compute sigma_max(J W^-1)/sigma_min(J W^-1). This metric is invariant
  under vertex permutations, physical rotation/translation and uniform scaling.
  Use W columns from regular vertices(0,0,0),(1,0,0),(1/2,sqrt(3)/2,0),
  (1/2,sqrt(3)/6,sqrt(2/3)). Require finite positive singular values, with
  degeneracy/NaN/infinity refused. Verify invariances on controlled fixtures.
- Preserve positive/nondegenerate/nonoverlapping cells, complete outward face
  ownership/coverage, protected partitions, IDs and overlapping selections.
  Keep relative volume/face-area1e-3, normals1degree and supplied physical delta.
- Run the unchanged P1/P2 finite-case checks: residual1e-9, applicable work and
  tensor checks1e-10, resultants1e-8, exact tension displacement1e-7; independently
  establish physical rigid rank/span. Maximum100000cells/150000coefficients.

The condition-number32 limit is a new project engineering criterion, not a
published universal threshold. It limits local directional distortion relative
to a regular element and screens numerical sensitivity. It is not a proven
bound for assembled elasticity or a replacement for the actual checks.
The sampling target is similarly an explicit minimum for this geometry workflow;
problem-specific displacement/stress convergence remains downstream work.

Knupp's [tetrahedral shape measure](https://www.sandia.gov/research/publications/details/tetrahedral-element-shape-optimization-via-the-jacobian-determinant-and-con-1999-09-27/)
uses the transformation from a regular tetrahedron to measure shape/degeneracy.
Shewchuk's [finite-element analysis](https://people.eecs.berkeley.edu/~jrs/papers/elemj.pdf)
distinguishes interpolation, discretization and conditioning requirements;
favorable shape alone is not a general error guarantee.

Execution proposal: one native call per case,2maximum, no retries or automatic
sizing/tolerance changes. Per native child120s, complete case480s, total1200
supervised process seconds including checks/preparation. Record exact commands,
wall/cleanup, native-call claims and Linux full-lifetime kernel peak RSS of the
isolated native worker, read from RUSAGE_CHILDREN after its only child is reaped.
Require zero prior child usage in that fresh case process. The worker's own
earlier peak-through-observation, after streaming raw-file hashing, is retained
as a diagnostic and must not exceed the full-lifetime value. These measurements
include Python/NumPy/provider imports, not an incremental native allocation or
the parent's FE memory. Memory/timing establish an explicit
local baseline; no speedup claim is inferred against unmeasured old timeouts.
Either case failing a required predicate gives NO_GO for this proposed envelope.
No public-provider promotion or release follows solely from this study.

Root retains criterion adoption and cross-repository acceptance; a non-author
review must scrutinize this proposal and runner before native execution.

The governing Linux supervisor owns a dedicated child-free controller, becomes
a subreaper, tracks descendants across their sessions, and uses PID identities
and pidfds for cleanup/reaping on timeout or graceful SIGINT/SIGTERM/cancellation.
Controlled nested-session timeout/cancel/interrupt regressions are required.
Uncatchable controller death/host failure is outside this userspace contract.
Freeze the actual launcher/supervisor and all installed ANYsolid/skfem/NumPy/
SciPy/provider/Shapely implementation files/versions, loaded external shared
libraries and relevant owner source, then verify again in a fresh process.
