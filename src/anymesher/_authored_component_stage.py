"""Detached shared-component staging; no production route calls this yet."""

from copy import deepcopy

from .boundary import GlobalEdgeBoundaryRegistry
from .errors import MeshError
from .native_v2 import ComponentSeedRegistry


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
        first = max((*mesh.nodes, *seed_registry.assigned_node_ids), default=0) + 1
        self.seed_registry = seed_registry.fork_detached(first)
        self._closed = False

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
