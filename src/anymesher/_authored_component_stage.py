"""Detached shared-component staging; no production route calls this yet."""

from copy import deepcopy
from hashlib import sha256
import json
from threading import RLock

from .boundary import GlobalEdgeBoundaryRegistry
from .errors import MeshError
from .native_v2 import ComponentSeedRegistry
from .serialize import mesh_to_dict


def _mesh_digest(mesh):
    return sha256(json.dumps(mesh_to_dict(mesh), sort_keys=True,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


def _registry_receipt(registry):
    return tuple((entry.key.edge_id, entry.key.parameter, entry.node_id,
                  tuple(entry.owners), tuple(float(value) for value in entry.point))
                 for entry in registry.entries())


class AuthoredComponentStage:
    """Keep provisional mesh, stations and seed IDs off published objects.

    `finish` returns a new artifact triple only after caller-supplied validation.
    It deliberately does not mutate or replace its inputs. The caller must
    still revalidate owner bindings and run all mesh gates before publishing.
    """

    def __init__(self, mesh, boundary_registry, seed_registry):
        if not isinstance(boundary_registry, GlobalEdgeBoundaryRegistry):
            raise MeshError("authored component needs a boundary registry")
        if not isinstance(seed_registry, ComponentSeedRegistry):
            raise MeshError("authored component needs a seed registry")
        self.mesh = deepcopy(mesh)
        self.boundary_registry = GlobalEdgeBoundaryRegistry(boundary_registry.view)
        for entry in boundary_registry.entries():
            owners = entry.owners or (None,)
            for owner in owners:
                self.boundary_registry.register(
                    entry.key.edge_id, entry.key.parameter, entry.point,
                    node_id=entry.node_id, owner=owner,
                )
        source_pool = seed_registry.reservation_pool
        if source_pool is not None and source_pool.mesh is not mesh:
            raise MeshError("component reservations bind a different mesh")
        self.source_reservations = None if source_pool is None else source_pool.snapshot()
        staged_pool = (
            None if source_pool is None else source_pool.fork_detached(
                self.mesh, expected=self.source_reservations,
            )
        )
        reserved = () if staged_pool is None else staged_pool.snapshot().reserved_ids
        next_seed_id = seed_registry.committed_snapshot()[0]
        first = max(max((*mesh.nodes, *seed_registry.assigned_node_ids, *reserved),
                        default=0) + 1, next_seed_id)
        self.seed_registry = seed_registry.fork_detached(
            first, reservation_pool=staged_pool,
        )
        self._closed = False

    def reserve_interior_nodes(self, count):
        if self._closed:
            raise MeshError("authored component stage is already closed")
        ids = self.seed_registry._reserve_unshared_nodes(count)
        if any(node in self.mesh.nodes for node in ids):
            self.abort()
            raise MeshError("authored component interior ID overlaps staged mesh")
        return ids

    def finish(self, validate):
        if self._closed:
            raise MeshError("authored component stage is already closed")
        if not callable(validate):
            raise MeshError("authored component needs a final validator")
        try:
            admitted = validate(self.mesh, self.boundary_registry, self.seed_registry)
            if admitted is not True:
                raise MeshError("authored component final validator did not admit result")
        finally:
            self._closed = True
        return self.mesh, self.boundary_registry, self.seed_registry

    def abort(self):
        self._closed = True


class AuthoredComponentPublication:
    """Swap a whole validated component result as one guarded pointer update.

    This holder is private and not used by the production meshing route yet.
    Readers must obtain all three objects from one `snapshot` call.
    """

    def __init__(self, mesh, boundary_registry, seed_registry):
        self._state = (mesh, boundary_registry, seed_registry)
        self._generation = 0
        self._lock = RLock()

    def snapshot(self):
        with self._lock:
            return self._generation, self._state

    def begin(self):
        with self._lock:
            mesh, boundary_registry, seeds = self._state
            mesh_digest = _mesh_digest(mesh)
            registry_receipt = _registry_receipt(boundary_registry)
            seed_receipt = seeds.committed_snapshot()
            stage = AuthoredComponentStage(mesh, boundary_registry, seeds)
            if (_mesh_digest(mesh) != mesh_digest
                    or _registry_receipt(boundary_registry) != registry_receipt
                    or seeds.committed_snapshot() != seed_receipt):
                stage.abort()
                raise MeshError("authored component source changed during staging")
            stage._publication_owner = self
            stage._base_generation = self._generation
            stage._source_mesh_digest = mesh_digest
            stage._source_registry_receipt = registry_receipt
            stage._source_seed_receipt = seed_receipt
            return stage

    def publish(self, stage, validate, cancellation_check=None):
        with self._lock:
            if (not isinstance(stage, AuthoredComponentStage)
                    or getattr(stage, "_publication_owner", None) is not self
                    or stage._closed or stage._base_generation != self._generation):
                raise MeshError("authored component publication is stale")
            if not callable(validate):
                raise MeshError("authored component needs a final validator")
            mesh, boundary_registry, seeds = self._state
            source_pool = seeds.reservation_pool

            def source_unchanged():
                return (
                    _mesh_digest(mesh) == stage._source_mesh_digest
                    and _registry_receipt(boundary_registry) == stage._source_registry_receipt
                    and seeds.committed_snapshot() == stage._source_seed_receipt
                    and (source_pool is None or source_pool.snapshot() == stage.source_reservations)
                )

            try:
                if not source_unchanged():
                    raise MeshError("authored component source changed before publication")
                if cancellation_check is not None:
                    cancellation_check("authored component before validation")

                def final_validator(candidate_mesh, candidate_edges, candidate_seeds):
                    if validate(candidate_mesh, candidate_edges, candidate_seeds) is not True:
                        return False
                    if cancellation_check is not None:
                        cancellation_check("authored component before publication")
                    if not source_unchanged():
                        raise MeshError("authored component source changed during validation")
                    return True

                candidate = stage.finish(final_validator)
                candidate_pool = candidate[2].reservation_pool
                if candidate_pool is not None:
                    candidate_pool.release_origin()
                self._state = candidate
                self._generation += 1
                return self.snapshot()
            except BaseException:
                stage.abort()
                raise
