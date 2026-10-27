# Convex-prism volume mesh development API

`VolumeMesh` is a separate immutable tet4 sibling of the existing shell `Mesh`.
This is Workflow 2 development work, not volume-provider qualification or release
approval. The canonical geometry handoff is ANYgeometry's
`docs/solid_volume_contract.md` (v1).

**Current real-provider capability: UNQUALIFIED / blocked.** Public
`generate_volume_mesh(..., backend='pytetwild')` refuses with
`VolumeBackendUnavailable` before launching a process. W2.0 did not establish
viable provider execution. There is no environment variable or public parameter
that bypasses this decision. Root's `tools/volume_feasibility.py` is a separately
governed research route, not a production API. Controlled tests substitute fake
providers and establish the neutral mesh, admission and adapter contracts only.
The example below documents the interface for a future explicitly adopted
provider capability; it does not currently enable real volume generation.

```python
from anymesher import generate_volume_mesh

snapshot = model.solid_region_snapshot(region_id)
mesh = generate_volume_mesh(
    snapshot,
    target_edge_length=0.1,       # metres; explicit physical sizing request
    boundary_tolerance=1e-6,      # metres; explicit maximum admission distance
    timeout=120,
    backend='pytetwild',
)
```

Only one convex normal extrusion of a strictly convex polygon is supported:
boxes, triangular/prismatic polygon caps and thin signed-offset prisms, with
planar straight-edged faces. Curved, concave, holed and multipart sources,
interfaces, oblique-only prisms and malformed owner snapshots fail closed.
Tolerance must resolve source floating-point coordinates and be less than one
eighth of the smallest source edge/face separation. It is never adjusted.

`VolumeMesh` implements unchanged ANYsolid `SolidMesh.snapshot(mesh)` structural
attributes: coordinates, tet4 cells, node/cell/exterior IDs, node/cell/boundary
groups, SI units, first geometry order and frozen JSON-compatible provenance.
Connectivity contains zero-based rows. All groups contain external IDs. The
complete outward exterior is mandatory; partial exterior lists are refused.
`face_owners` maps each boundary external ID to exactly one owner-bound
ANYgeometry face handle. Source groups can overlap; every selection is preserved
without duplicating an ownership assignment. Arrays own immutable byte-backed
storage and cannot have their WRITEABLE flag re-enabled. No consumer import or
dependency is required.

Source identity, revision, region and content signature are recorded. Call
`validate_volume_mesh(mesh, snapshot, boundary_tolerance=...)` before reuse
against an owner snapshot: a stale identity/revision/signature is refused.
This validator also re-admits structural objects and all their IDs/groups.

The optional `solids-pytetwild` extra pins `pytetwild==0.4.2`. Ordinary imports never load
that provider. Upstream 0.4.2's public import unconditionally imports its optional
PyVista accessor despite declaring NumPy as its sole base dependency. Root's
Workflow 2 artifact `0.4.2+anyworkflow2.1` contains the explicitly approved
import-only guard; native bytes remain unchanged. This local version is accepted
and recorded exactly. No private-module import bypass, silent provider change,
PyVista installation or automatic package repair occurs at runtime. Follow the
root Workflow 2 handoff for the exact local artifact identity and feasibility
disposition. The local import patch by itself establishes no meshing capability.

Generation uses the pinned public raw-array provider in an owned subprocess.
Effective options are recorded in provenance and `request.json`:

- `edge_length_abs=target_edge_length`, `edge_length_fac=0.05` (ignored by provider
  when absolute sizing is present), `epsilon=boundary_tolerance/source_diagonal`;
- `simplify=False`, `coarsen=False`, `optimize=True`, `num_threads=1`;
- provider's explicit remaining defaults: `stop_energy=10`, `num_opt_iter=80`,
  `loglevel=3`, `quiet=True`, `vtk_ordering=False`, `disable_filtering=False`.

The raw coordinates/topology are never projected, smoothed, merged or repaired.
Negative provider tetrahedron ordering is deterministically swapped into positive
orientation, and the swap count is recorded. Degenerate, duplicate, crossing,
overlapping or otherwise invalid output is refused atomically. Coordinate rows
are lexicographic, connectivity rows use canonical positive local ordering and
lexicographic cell order, and exterior enumeration uses sorted connectivity keys.
Provenance retains raw-node-row-to-row and raw-cell-row-to-row maps. IDs are
consecutive external IDs for those canonical rows. This does not promise
cross-platform identical remeshing.

Independent checks include positive scale-normalized Jacobians, no orphan or
duplicate rows/cells, oriented manifold incidence, face-connected cells and
closed outward exterior. A sweep of tetrahedron AABBs feeds exhaustive candidate
SAT/intersection contact checks. Ambiguous contact clipping in thin prisms uses
exact rational predicates on the unchanged binary coordinates. Adjacent positive
cells with opposite shared facets have a direct separating-plane proof.

Every boundary triangle must match exactly one complete source plane and convex
loop, with outward normal. Independent coverage uses projected triangle
non-overlap, per-face area and explicit protected edge chains and corners.
Convex halfspace containment and total volume are checked too. Coverage allows
only the analytically bounded boundary strip implied by the requested geometric
tolerance (perimeter times tolerance plus the corner term); cell-intersection
roundoff predicates use separate machine precision bounds. No nearest-face-only
or sampled acceptance is used. This validator favors correctness; its worst-case
candidate-pair cost and exact ambiguity resolution are not scalability claims.

`VolumeBackendUnavailable`, `UnsupportedVolumeGeometry`, `InvalidVolumeMesh`,
`VolumeMeshingTimeout` and `VolumeMeshingCancelled` distinguish refusal causes.
Cancellation accepts a callable or an `is_set()` object, is polled during the
provider run and admission and is checked before publication. Timeout bounds the
provider process, including startup; postchecks remain cancellable. Windows
workers await a start signal until assigned to a kill-on-close Job Object.
Cleanup explicitly terminates the job, verifies zero owned active processes and
reaps the parent. POSIX workers use a new session/process group.
POSIX cleanup also checks the process table until no active owned group members
remain; terminated zombies are excluded from the active count.

Each provider attempt retains `input.npz`, `request.json`, full `stdout.txt` and
`stderr.txt`, terminal `process.json`, response metadata and raw output (if any)
in a unique system temporary `anymesher-volume-*` evidence directory. Successful
provenance records that path; timeout/cancel/unavailability errors expose
`evidence_directory`. Other invalid-response messages include the path. Input
NPZ bytes have an exact SHA-256; effective options, executable/command, source
signature, version, wall duration, exit and cleanup status are retained. No
mesh object escapes on refusal. Evidence directories require caller retention
or cleanup according to their development/qualification workflow.

Focused controlled checks (no real tetrahedralizer) are available as:

```text
python tools/run_dev_smoke.py --profile volume
python -m pytest -q tests/test_volume_mesh.py tests/test_volume_process.py
```

The tests include explicitly controlled fake provider distributions in isolated
child environments to exercise the real adapter and process lifecycle. Those
fixtures provide no real-provider evidence. The affected-test runner maps the
file-launched worker explicitly; ordinary import analysis cannot infer it.
Real-provider feasibility, installed cross-repository consumption, independent
review and release gates remain root-owned and are reported separately.
