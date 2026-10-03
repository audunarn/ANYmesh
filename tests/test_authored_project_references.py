"""ANYfem input visibility is a prerequisite, not authored-root permission."""

from dataclasses import replace

import numpy as np
import pytest

import anygeometry as owner
from anygeometry.entities import EntityRef
from anyfem import Project
from anyfem.model.attributes import LineLoad
from anyfem.model.regions import ManualRegion, Region, RegionDomain, RegionRef
from anyfem.prepared_reference_scope import (
    PreparedReferenceScopeError, query_prepared_project_reference_scope,
)
from anymesher._authored_project_references import (
    bind_authored_project_references, validate_authored_project_stage,
    validate_required_project_constraints,
)
from anymesher._authored_associations import plan_authored_root_associations
from anymesher._authored_component_stage import AuthoredComponentPublication
from anymesher._authored_scope_binding import (
    BoundAuthoredRootInputs, bind_authored_root_inputs,
)
from anymesher.boundary import GlobalEdgeBoundaryRegistry
from anymesher.errors import MeshError
from anymesher.mesh import Mesh
from anymesher.meshing_view import GeometryMeshingView
from anymesher.native_v2 import ComponentSeedRegistry


def simple_project(*, line_load=False):
    geometry = owner.GeometryModel()
    root = geometry.add_plate(geometry.add_points(
        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))))
    geometry.add_sheet((root,))
    plan = owner.plan_intersections(geometry, tuple(geometry.faces), policy="connect")
    owner.apply_intersections(geometry, plan, policy="connect")
    project = Project(geometry=geometry)
    edge = geometry.faces[root].loop[0].edge
    if line_load:
        project.load_case().line_loads.append(
            LineLoad(EntityRef("edge", edge), np.array((0., 0., 2.))))
    correspondence = owner.query_prepared_authored_boundary_correspondence(
        geometry, root,
    )
    bound = bind_authored_root_inputs(
        geometry, correspondence, correspondence.descendants,
    )
    manifest = query_prepared_project_reference_scope(project, geometry)
    return project, bound, manifest, edge


def test_root_only_input_scope_binds_exact_project_source():
    project, bound, manifest, _edge = simple_project()
    before = owner.to_dict(project.geometry)
    reference = bind_authored_project_references(
        project, project.geometry, bound, manifest,
    )
    assert (reference.authored_face, reference.source_namespace,
            reference.source_face) == (bound.authored_face, "project_authored",
                                       bound.authored_face)
    assert reference.descendants == bound.descendants
    assert reference.publication_qualified is False
    assert owner.to_dict(project.geometry) == before


def test_required_line_edge_must_be_a_registered_active_shell_chain():
    project, bound, manifest, edge = simple_project(line_load=True)
    reference = bind_authored_project_references(
        project, project.geometry, bound, manifest,
    )
    assert edge in reference.required_boundary_edge_ids
    registry = GlobalEdgeBoundaryRegistry(GeometryMeshingView(project.geometry))
    registry.register(edge, 0., node_id=1)
    registry.register(edge, 1., node_id=2)
    mesh = Mesh(nodes={1: np.array((0., 0., 0.)),
                       2: np.array((1., 0., 0.)),
                       3: np.array((0., 1., 0.))}, tris={7: (1, 2, 3)})
    validate_required_project_constraints(
        project, project.geometry, reference, mesh, registry,
    )
    mesh.tris[7] = (1, 3, 2)
    # Orientation is irrelevant to exact station membership.
    validate_required_project_constraints(
        project, project.geometry, reference, mesh, registry,
    )
    mesh.tris[7] = (1, 3, 3)
    with pytest.raises(MeshError, match="not a conforming shell chain"):
        validate_required_project_constraints(
            project, project.geometry, reference, mesh, registry,
        )
    with pytest.raises(MeshError, match="binding was altered"):
        validate_required_project_constraints(
            project, project.geometry, replace(reference, required_boundary_edge_ids=()),
            mesh, registry,
        )
    mesh.tris[7] = (1, 2, 3)
    mesh.nodes[2] = np.array((1., 0., .01))
    with pytest.raises(MeshError, match="altered node coordinates"):
        validate_required_project_constraints(
            project, project.geometry, reference, mesh, registry,
        )


def test_child_pressure_refuses_root_only_association():
    geometry = owner.GeometryModel()
    root = geometry.add_plate(geometry.add_points(
        ((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0))))
    cutter = geometry.add_plate(geometry.add_points(
        ((1, -1, -1), (1, 2, -1), (1, 2, 1), (1, -1, 1))))
    owner.apply_intersections(geometry,
        owner.plan_intersections(geometry, (root, cutter), policy="connect"),
        policy="connect")
    children = tuple(ref.id for ref in geometry.resolve_ref(EntityRef("face", root)))
    left = next(face for face in children if all(
        geometry.vertex_position(geometry.edges[use.edge].start)[0] <= 1
        for use in geometry.faces[face].loop))
    project = Project(geometry=geometry)
    region = Region("Left", RegionDomain.GEOMETRY, "face",
                    ManualRegion((EntityRef("face", left),)))
    project.regions.add(region)
    project.load_case().add_pressure(EntityRef("face", left), 1200,
                                     region=RegionRef(region.id))
    manifest = query_prepared_project_reference_scope(project, geometry)
    # The existing owner preflight also rejects this junction; construct only
    # its validated scope receipt to isolate the external-reference check.
    scope = owner.query_prepared_model_scope(geometry)
    bound = BoundAuthoredRootInputs(root, children, scope)
    with pytest.raises(PreparedReferenceScopeError, match="child-local"):
        bind_authored_project_references(project, geometry, bound, manifest)


def test_changed_project_and_stale_owner_refuse_before_retention_check():
    project, bound, manifest, edge = simple_project()
    reference = bind_authored_project_references(
        project, project.geometry, bound, manifest,
    )
    project.load_case().line_loads.append(
        LineLoad(EntityRef("edge", edge), np.array((0., 0., 3.))))
    with pytest.raises(PreparedReferenceScopeError):
        validate_required_project_constraints(
            project, project.geometry, reference, Mesh(),
            GlobalEdgeBoundaryRegistry(GeometryMeshingView(project.geometry)),
        )


def test_private_stage_preflight_retains_root_source_and_rechecks_before_swap():
    project, bound, manifest, edge = simple_project(line_load=True)
    local = Mesh(nodes={1: np.array((0., 0., 0.)),
                        2: np.array((1., 0., 0.)),
                        3: np.array((0., 1., 0.))}, tris={7: (1, 2, 3)})
    association = plan_authored_root_associations(
        project.geometry, bound, local, [7],
    )
    staged = association.stage_mesh(local)
    registry = GlobalEdgeBoundaryRegistry(GeometryMeshingView(project.geometry))
    registry.register(edge, 0., node_id=1)
    registry.register(edge, 1., node_id=2)
    assert validate_authored_project_stage(
        project, project.geometry, bound, manifest, association, staged, registry,
    ) is None
    staged.elements_of_face = {bound.descendants[0]: [7], 999: [7]}
    with pytest.raises(MeshError, match="exact root-local source associations"):
        validate_authored_project_stage(
            project, project.geometry, bound, manifest, association, staged, registry,
        )
    staged = association.stage_mesh(local)
    holder = AuthoredComponentPublication(staged, registry, ComponentSeedRegistry(4))
    candidate = holder.begin()
    project.load_case().line_loads.append(
        LineLoad(EntityRef("edge", edge), np.array((0., 0., 3.))))

    def preflight_only(mesh, stations, _seeds):
        validate_authored_project_stage(
            project, project.geometry, bound, manifest, association, mesh, stations,
        )
        return True

    with pytest.raises(PreparedReferenceScopeError):
        holder.publish(candidate, preflight_only)
    assert holder.snapshot()[0] == 0
