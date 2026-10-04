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
owner's complete typed original/current constraint inventory. It binds the
exact authored root, every current descendant and each owner trace, then
refuses member/attachment records, physical junctions, groups, tags,
construction/isolated vertices, extensions, feature records, changed child
face metadata and outside shared-trim roots until their meaning has an explicit
consumer. `Face.corners` are loop offsets; isolated-vertex detection now comes
from the owner inventory. The separate complete Sheet-joint binding retains
the same inventory for both selected roots and reports unqualified reference
categories, including occupied opaque metadata, without treating literal
record equality as a remap. Boundary planning rebinds the whole Sheet-joint
component, checks exact root/boundary coverage and registered edge sets against
each root's typed trace, then rebinds after callback-bearing station work.
The owner inventory is revalidated after inspection. These results remain
`publication_qualified=False`:
FaceUse/Sheet interpretation, external ANYfem loads and supports, original work
accounting, shared-component transactions and final mesh gates are still open.

`anymesher._authored_work_ledger.AuthoredWorkLedger` carries the original native
insertion and topology-operation limits forward from an actual diagnostic
receipt. It requires the report's limits to equal the original options, charges
reserved shared-node reuses, and refuses incomplete or altered counts. Charges
are immutable and cannot exceed the original allowance. This accounting helper
does not reset the invocation's time/cancellation budget or authorize a retry.
It is not wired to authored-root publication yet.

The private `_authored_associations` plan binds a complete standalone root-local
shell to its original face and original FaceUse/Sheet orientations. It checks
that every current descendant retains the same use and Sheet meaning, and
stages those source associations on a detached mesh. It never assigns a
root-born cell to a current child merely to satisfy a load.
It cannot see application-owned child-local loads, supports or sections and
therefore remains `publication_qualified=False`. A consuming application needs
to provide an explicit prepared-reference manifest and basis: which original
or current faces each external reference addresses, and whether each internal
child boundary must be imprinted so cells cannot straddle that reference.

The native shared-node allocator now uses a component-owned reservation pool.
It preserves the previous ID sequence and keeps failed reservations occupied.
The private `_authored_component_stage` copies the mesh and boundary stations,
then forks that pool and seed identities before provisional changes. Its
publication holder rejects stale state and swaps the mesh, station registry
and seed registry together only after a caller-provided validator admits them.
Failure, cancellation or stale owner validation discards the staged pool and
leaves original reservations unchanged. Arbitrary external allocator callbacks
still refuse staging because their reservations cannot be snapshotted.
Production authored-root routing does not call these private helpers. Owner
binding, complete source associations, strict mesh checks, resource limits and
external reference handling remain mandatory before any authored-root result.

`_authored_project_references` consumes ANYfem's optional, preparation-bound
reference manifest. It validates the complete owner/project scope, preserves
the explicit source namespace, and refuses root-only associations when a
child-local load, support or section exists. Its required-edge check needs
registered endpoint stations, unchanged node coordinates and an active shell
edge chain; required vertices need active source-linked nodes. A private stage
preflight rechecks the manifest immediately before a separate full mesh gate
and refuses detached source namespaces until an output remap is qualified.
It returns no admission token and is not wired to production publication.

Exact coordinated development inputs: ANYfem candidate wheel SHA-256
`ea1a7d2daf0cb269eb8271ae48c8ad35918afded7046a3f0f1089f028cf9def5`
and ANYgeometry wheel SHA-256
`2d7468767d0cbca26185dcab543912582048d828d435a15fd3c6c6736cb4b553`.
The owner original-domain triangle validator explicitly does not certify the
literal current-fragment partition. A conforming child boundary plus an
input manifest is not a proof of each cell's current-child membership; that
proof and the strict quality/high-order gates remain required for routing.

The later private `validate_authored_child_project_cells` guard uses the
owner's literal prepared-child triangle API for **linear** root-owned T3/Q4
cells when ANYfem declares a child-local reference. The caller must provide
one current child ID per cell and original UV per corner from the authored
chart. Node XYZ is checked against the original owner support; each whole
triangle (or both halves of a convex Q4) is validated inside the named actual
child. The guard retains original-root `elements_of_face` and original Sheet
scope, verifies required child-boundary station chains, and revalidates owner
and project inputs at completion. The final exact owner partition call requires
triangles for **every** authenticated child and certifies complete closed
material equality and disjoint interiors in its narrow straight-planar scope.
Missing, partial, duplicated, overlapping, crossing or wrong-child cells
refuse. It remains `publication_qualified=False`: partition coverage does not
certify node/joint conformity, structural attachments, quality, high-order
mapping or application/solver consumption. Quadratic cells, curved/holed and
rounded split-spline/BQC/QIC trims, detached project output remapping and
active production routing still refuse.

Exact final owner development input for this guard is ANYgeometry candidate
`931916ab9ee399ff59013c01e622eb8da20cffe9`, wheel SHA-256
`a7c64ad31cffc463b8d4e3b3d8f2084700465bafbeec3b9ba80441011d838cea`.
It supersedes the earlier tested `a2ea7eb` candidate after an owner fix for
reversed polynomial trims; both identities remain in the task evidence.
The previous exact ANYfem wheel remains `ea1a7d2daf0cb269eb8271ae48c8ad35918afded7046a3f0f1089f028cf9def5`.

The exact partition owner input is later candidate
`be2dd134887e2d245c194e7011124c230aea4d1e`, wheel SHA-256
`94c83f919852210ee22c8f3f72e78df47f181eeaca8b7f30e0defad713914d19`.
It extends the child-containment guard only for exact coplanar Plane domains
with one simple straight outer loop per face and no holes. Previous wheel
identities remain in the living task note as historical inputs.

The private `_authored_planar_stations` preflight uses the owner's exterior
station and candidate straight-interior station queries to retain exact
original UV, source parameters, and registered current node IDs/XYZ before
any opt-in root meshing. It validates both exact original and current XYZ
representations and retains the owner's complete vertex-preimage receipt. In
the 4-by-4 planar split at `x=3`, current joint edge 25 now has an authenticated
ordered original-UV chain and child incidence; missing or altered stations
refuse. The interior-station candidate is ANYgeometry `fe1ed812` (wheel SHA256
`17b6d6450a1731345ebc47e166508194ec97543a2bd08fe36bb6bc63fbbb978c`),
not a released dependency. Its vertex receipt says which current vertices have
original *vertex* ancestors; absent ancestry does not prove absence of a
source-edge, member, attachment or junction reference. The physical split has
a sheet joint and attachments on this generated interior edge. Complete
structural association semantics and a current-keyed FEM output receipt remain
unproven, so no native root call or publication route is activated.

The candidate ANYgeometry Sheet-joint component receipt at `da101adc`
(wheel SHA256 `1bc9121b19d73b6ebcfad6ffd6e08c3c42dcc497fd20a319cea9dfbfc9569cce`)
allows a separate private `_authored_component_binding` preflight. It requires
both authored roots, all current descendants, both Sheets and the complete
qualified original-root/Sheet FaceUse occurrence correspondence. Root-only
selection, missing root boundaries, source-less Sheets and stale bindings
refuse. The two root station plans can share one unchanged global edge
registry. The owner explicitly leaves arbitrary structural/reference semantic
remapping unqualified; this component binding does not construct a current
mesh association receipt, invoke a native route or permit publication.

The later exact-current-material owner candidate `e509862b` (wheel SHA256
`2761f6d62611ea62b69f80d6ec47659c4cfa1142d3ac07ed460e54fc191165ed`)
adds current edge traces in the original Plane chart and accepts exact rational
cell UV. Keep these material stations separate from ancestral source-edge UV:
on the cutter, the same current vertex can legitimately have different
ancestral and current-material fractions. A source-only two-Sheet fixture now
passes the owner's exact partition for both roots without moving a node or
casting the cell fractions to binary64.

`query_prepared_current_component_associations` exposes a versioned,
**current-only** receipt for an already staged, complete linear T3/Q4
Plane/Straight component. It revalidates the owner component, all root station
receipts, exact cell partition and current face/Sheet/FaceUse buckets, every
ordered edge and joint chain, all current vertex nodes and the known Mesh field
schema. Original source-boundary chains retain the owner's loop direction and
the exact ordered current edge/station sequence; an interior joint is not
mislabelled as a source exterior edge. It refuses members, offsets, grids,
seeding, thickness, activity,
diagnostic records and nonzero automatic-connection counters until those
semantics are implemented. Empty ancestral vertex preimages and interior edge
ancestry are recorded without calling them generated-only. This narrow API is
not an actual meshing route, does not transfer application source references,
and marks solver admission and publication false. Its validator rederives the
owner proofs; the mesh digest alone is not an admission certificate.

The private `_authored_route_boundary` packet is the first engine-input slice
for an opt-in authored-root route. For every root in the qualified whole
component it joins owner-ordered exterior station IDs into one closed loop,
retains each paired interior edge as protected constraint segments, and maps
their exact current-material UV into the original-root metric chart. Exact
rational UV remains attached to the unchanged global node IDs beside the float
metric coordinates needed by the existing surface mesher. The source-only
fixture proves both roots share the same joint IDs and rejects altered or stale
stations. This packet does not invoke triangulation. A native-result adapter
must still retain each created node's original UV and work-ledger charge before
the exact partition, conformity, quality and current-association gates can
admit a staged result; no current production route calls this packet.
