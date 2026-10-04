"""Prebuilt linear cells exercise detached joint integration without meshing."""

from dataclasses import replace
import numpy as np
import pytest

from test_authored_component_cells import candidate, created_candidate
from anymesher._authored_component_binding import BoundAuthoredSheetJointComponent
from anymesher._authored_component_stage import _mesh_digest, _registry_receipt
from anymesher._authored_route_boundary import (
    AuthoredRootChildBinding, AuthoredRootTriangulation,
    plan_authored_component_boundaries,
)
from anymesher._authored_staged_pair import stage_authored_root_pair
from anymesher.errors import MeshError
from anymesher.native_v2 import ComponentSeedRegistry
from anymesher.prepared_current_associations import (
    PREPARED_CURRENT_ASSOCIATIONS_CREATED_UV_SCHEMA,
    PREPARED_CURRENT_ASSOCIATIONS_SCHEMA,
)
from anymesher.triangulation import PlanarTriangulation


def _prebuilt_pair(*, created=False):
    if created:
        geometry, component, mesh, registry, cell_faces, created_uv, _node = (
            created_candidate()
        )
    else:
        geometry, component, mesh, registry, cell_faces = candidate()
        created_uv = {}
    packets = plan_authored_component_boundaries(
        geometry, component, mesh, registry,
    )
    roots, children = [], []
    for packet, correspondence in zip(packets, component.boundary_correspondences):
        face_ids = set(correspondence.descendants)
        cells = tuple(cell for cell in sorted(mesh.tris)
                      if cell_faces[cell] in face_ids)
        protected = dict(packet.material_uv_by_node)
        extra = dict(created_uv.get(packet.authored_face_id, {}))
        node_ids = tuple(sorted(protected)) + tuple(sorted(extra))
        rows = {node: row for row, node in enumerate(node_ids)}
        uv = tuple(protected.get(node, extra.get(node)) for node in node_ids)
        points = packet.chart.to_metric(
            np.asarray([[float(u), float(v)] for u, v in uv], dtype=float)
        )
        triangles = np.asarray(
            [[rows[node] for node in mesh.tris[cell]] for cell in cells],
            dtype=np.int64,
        )
        outer = np.asarray([rows[node] for node in packet.outer_node_ids],
                           dtype=np.int64)
        boundary = np.asarray(
            [(outer[index], outer[(index + 1) % len(outer)])
             for index in range(len(outer))], dtype=np.int64,
        )
        mandatory = np.asarray(
            [(rows[a], rows[b]) for a, b in packet.constraint_node_pairs],
            dtype=np.int64,
        ).reshape((-1, 2))
        segments = np.vstack((boundary, mandatory))
        triangulation = PlanarTriangulation(
            points, triangles, segments, boundary, mandatory, outer, (),
            protected_node_rows=tuple((node, rows[node]) for node in protected),
        )
        roots.append(AuthoredRootTriangulation(triangulation, uv, packet))
        children.append(AuthoredRootChildBinding(
            packet.authored_face_id, tuple(cell_faces[cell] for cell in cells),
        ))
    seeds = ComponentSeedRegistry(max(mesh.nodes) + 1)
    return (geometry, component, mesh, registry, seeds, packets,
            tuple(roots), tuple(children))


@pytest.mark.parametrize("created", (False, True))
def test_prebuilt_pair_retains_narrow_joint_receipt_and_source(created):
    args = _prebuilt_pair(created=created)
    source, registry, seeds = args[2:5]
    before = (_mesh_digest(source), _registry_receipt(registry),
              seeds.committed_snapshot())
    result = stage_authored_root_pair(*args)
    joint = result.current_only_joint_receipt
    assert joint.current_associations == result.current_receipt
    assert joint.mesh_digest == _mesh_digest(result.mesh)
    assert joint.attachment_ids == args[1].owner_receipt.attachment_ids
    assert len(joint.sheet_use_segment_cells) == len(joint.joint_chain) - 1
    assert result.current_only_joint_cell_binding_qualified
    assert not result.source_reference_transfer_qualified
    assert not result.solver_admitted and not result.publication_qualified
    assert result.current_receipt.schema == (
        PREPARED_CURRENT_ASSOCIATIONS_CREATED_UV_SCHEMA if created
        else PREPARED_CURRENT_ASSOCIATIONS_SCHEMA
    )
    assert (_mesh_digest(source), _registry_receipt(registry),
            seeds.committed_snapshot()) == before


def test_joint_proof_refusal_and_cancel_keep_source_unchanged(monkeypatch):
    args = _prebuilt_pair()
    source, registry, seeds = args[2:5]
    before = (_mesh_digest(source), _registry_receipt(registry),
              seeds.committed_snapshot())
    original = BoundAuthoredSheetJointComponent.bind_current_only_joint_cells

    def unqualified(self, *values, **options):
        return replace(original(self, *values, **options),
                       current_only_joint_cell_binding_qualified=False)

    monkeypatch.setattr(BoundAuthoredSheetJointComponent,
                        "bind_current_only_joint_cells", unqualified)
    with pytest.raises(MeshError, match="joint receipt widened"):
        stage_authored_root_pair(*args)
    monkeypatch.setattr(BoundAuthoredSheetJointComponent,
                        "bind_current_only_joint_cells", original)

    def cancel(phase):
        if phase == "authored pair before validation":
            raise LookupError("cancelled")

    with pytest.raises(LookupError, match="cancelled"):
        stage_authored_root_pair(*args, cancellation_check=cancel)
    assert (_mesh_digest(source), _registry_receipt(registry),
            seeds.committed_snapshot()) == before
