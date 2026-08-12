# Native hybrid mesher persistence handoff

## Geometry document boundary

- ANYmesher consumes `ANYgeometry>=0.2,<0.3` through the live public Python API and public codecs only. It does not parse ANYgeometry JSON records.
- ANYgeometry 0.2.1 reads geometry schemas v1-v4 and writes canonical schema v4.
- Loading v1-v3 is a deterministic one-way migration. ANYgeometry 0.2.0 intentionally rejects v4 documents, so package-range compatibility does not imply backward document readability.
- Legacy zero-valued attachment or junction evidence is `UNVERIFIED`; it never implies exactness, structural connectivity, or certification.

The authoritative upstream documents are:

- `C:\Github\ANYgeometry\MIGRATION.md`
- `C:\Github\ANYgeometry\docs\FUTURE_MESHER_KERNEL_CONTRACT.md`

The upstream handoff reports 116 focused persistence tests covering v1-v4 reads, canonical v4 writes, deterministic migration, current-record checksums, and fail-closed unverified relationship evidence.

## Mesh and project documents

- ANYmesher writes `anymesher.mesh` v2 and reads v1-v2. Version 2 persists source model identity/revision, sheet/member associations, member nodes, and element activity. Version 1 migrates these additions to explicit empty or unknown values.
- ANYfem writes project format v5 and reads earlier supported project formats. Version 5 persists native meshing settings under `meshing.native` while geometry remains owned by ANYgeometry's codec.
- Model-bound mesh queries reject `EntityHandle` values from a different geometry model.
- Future unknown mesh versions fail closed rather than being interpreted as an older schema.

Downstream contract evidence is maintained in:

- `C:\Github\ANYmesh\tests\test_mesh_persistence_contract.py`
- `C:\Github\ANYmesh\tests\test_kernel_intersection_contract.py`
- `C:\Github\ANYfem\tests\test_native_meshing_project.py`
# Downstream qualification update (2026-08-12)

ANYgeometry 0.2.1 remains the authoritative geometry codec: schemas v1-v4 are readable and canonical writes use v4. ANYfem currently writes project format 5. Package compatibility (`ANYgeometry>=0.2,<0.3`) does not imply that ANYgeometry 0.2.0 can read documents containing schema v4 geometry.

ANYfem transaction rollback now snapshots and restores through `project_to_dict` / `project_from_dict`, which in turn use the public geometry codec. The live `Project` object identity is preserved, but read-only ANYgeometry mappings are never deep-copied, reopened, or assigned directly.

Focused downstream commands and results:

```powershell
# C:\Github\ANYfem
python -m pytest tests\test_io.py tests\test_legacy_geometry_owner_migration.py tests\test_imported_persistence.py tests\test_native_meshing_project.py tests\test_output_requests.py tests\test_section_assignments.py tests\test_document_geometry_snapshot.py -q
```

Result: `60 passed in 5.67s`.

```powershell
# C:\Github\ANYmesh
python -m pytest tests\test_mesh_persistence_contract.py tests\test_quality_and_serialize.py -q
```

Result: `18 passed in 0.11s`.

The dedicated ANYfem legacy migration module was also run independently of the
focused matrix:

```powershell
# C:\Github\ANYfem
python -m pytest tests\test_migration.py -q
```

Result: `30 passed in 9.13s`, including deterministic conversion diagnostics,
wrong-format refusal, and the conservative overall-readiness sentinel.

The qualified scope includes current round trips, geometry IDs/attributes/groups/tags/lineage, public-owner migration of non-contiguous legacy geometry IDs, imported-mesh provenance, native settings and model-bound handles, output requests, deterministic region-backed section migration, transaction rollback, mesh schema v1-to-v2 migration, future mesh-format refusal, and mesh quality persistence.

Upstream persistence evidence remains separately owned by ANYgeometry: 116 focused tests were reported for v1-v4 reads, canonical v4 writes, deterministic one-way migration, strict checksums, and fail-closed legacy relationship evidence. Its contract documents are `C:\Github\ANYgeometry\MIGRATION.md` and `C:\Github\ANYgeometry\docs\FUTURE_MESHER_KERNEL_CONTRACT.md`.

Legacy relationship residual/tolerance zeros remain `UNVERIFIED`; downstream code must not interpret them as exactness or certification.
