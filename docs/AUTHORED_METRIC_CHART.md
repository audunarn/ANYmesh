# Original-root metric adapter

`anymesher._authored_metric_chart.AuthoredMetricChart` is a private, read-only
adapter for the original support bound by ANYgeometry's authored boundary
correspondence. It requires the development geometry implementation at
`0859e6ae09a1811b2c9ae2b26c839a95994121fb` or a qualified successor. The existing
package version alone does not identify these APIs.

At original UV `(0.5, 0.5)`, the adapter obtains public owner derivatives,
forms `G = J.T @ J` and sets `L = cholesky(G)`. Metric rows are `UV @ L`.
Metric-row evaluation uses `UV = rows @ inv(L)`; physical Jacobians are
`J(UV) @ inv(L).T`. This constant reference metric is not globally isometric
on a curved support. Positions and derivatives come from the owner's original
support extension, which does not certify patch or material membership.

Input rows are copied before owner callbacks. The entry chart context is
captured before input conversion; replacing it through a custom array conversion
or a callback refuses. Immutable matrix storage, detached outputs and current
owner guards prevent a query from changing chart identity during execution.
Geometry errors and cancellation exceptions preserve their owner semantics.
The adapter performs no world-XYZ inversion and stores or changes no registered
node IDs, XYZ, elements or work ledgers.

`publication_qualified` is always false. There is no production routing in this
slice. Full original-domain constraint import, reference/scope qualification,
immutable shared-node handling, approximation and quality certification, work
budget accounting and detached atomic shared-component publication still need
integration. Existing native diagnostic authority is not renewed by this adapter.

`anymesher._authored_boundary_binding.bind_authored_exterior_stations` is a
separate read-only bridge for already registered exterior nodes. It requires a
current geometry-bound registry, one exact station per existing global node,
unchanged mesh XYZ, and the owner's original/current station coordinate
assertion. It retains original UV and owner evidence alongside the same node IDs
and coordinates. Its receipt also has `publication_qualified=False`; rerun the
binding after any candidate topology change. It grants no station insertion,
constraint remap or authored-root mesh publication permission.

`anymesher._authored_scope_binding.bind_authored_root_inputs` consumes the
owner's complete original/current prepared-model snapshot. It requires every
descendant of one authored face to be selected and refuses member/attachment
records, physical junctions, groups, tags, construction/isolated vertices,
extensions, feature records, changed child face metadata and adjacent shared
edges until their meaning has an explicit consumer. It revalidates the owner
receipt after inspection. The result remains `publication_qualified=False`:
FaceUse/Sheet interpretation, external ANYfem loads and supports, original work
accounting, shared-component transactions and final mesh gates are still open.

`anymesher._authored_work_ledger.AuthoredWorkLedger` carries the original native
insertion and topology-operation limits forward from an actual diagnostic
receipt. It requires the report's limits to equal the original options, charges
reserved shared-node reuses, and refuses incomplete or altered counts. Charges
are immutable and cannot exceed the original allowance. This accounting helper
does not reset the invocation's time/cancellation budget or authorize a retry.
It is not wired to authored-root publication yet.
