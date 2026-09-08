"""Owner-certified cylindrical components in the public hybrid orchestrator."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
from anygeometry.curves import Straight
from anygeometry.surfaces import Cylinder

from ._cylindrical_atlas import prepare_cylindrical_atlas
from ._cylindrical_patch import prepare_cylindrical_patch
from ._cylindrical_quadratic_refine import refine_quadratic_component
from .errors import MeshError
from .metric import IsotropicMetricControl, MetricFieldSpec
from .surface_mesh import SurfaceMeshOptions


def prepare_bindings(geometry, native_faces, native_options, cancellation_check=None):
    """Require an owner patch for one face or an owner atlas for a component."""
    if native_options.point_placement != "frontal_delaunay":
        return {}
    remaining = {int(face) for face in native_faces
                 if isinstance(geometry.faces[face].surface, Cylinder)}
    bindings = {}
    while remaining:
        component = set()
        pending = [min(remaining)]
        while pending:
            face = pending.pop(0)
            if face in component:
                continue
            component.add(face)
            owner = geometry.faces[face]
            neighbours = {int(other) for loop in (owner.loop, *owner.holes)
                          for use in loop for other in geometry.faces_using_edge(use.edge)}
            pending.extend(sorted((neighbours & remaining) - component - set(pending)))
        selected = []
        for face in sorted(component):
            uses = sorted(identifier for identifier, use in geometry.face_uses.items()
                          if use.face_id == face)
            if len(uses) != 1:
                raise MeshError("cylindrical native face requires one qualified FaceUse")
            selected.append(geometry.handle("face_use", uses[0]))
        selected = tuple(selected)
        if len(selected) == 1:
            binding = prepare_cylindrical_patch(
                geometry, selected, cancellation_check=cancellation_check,
            )
        else:
            # An incomplete periodic family must not be disguised as patches.
            binding = prepare_cylindrical_atlas(
                geometry, selected, reference_face_use=selected[0],
                cancellation_check=cancellation_check,
            )
        for face in component:
            bindings[face] = binding
        remaining.difference_update(component)
    return bindings


def component_local_split_edges(geometry, edges, bindings):
    """Freeze interfaces whose incident faces cannot share one cylinder trial.

    Apply to every native face, including a planar neighbour. Owner-certified
    external incidences are allowed, but cannot acquire one-sided split nodes.
    """
    if not bindings:
        return edges
    members = {}
    for binding in bindings.values():
        if id(binding) not in members:
            members[id(binding)] = frozenset(record.face.id for record in binding.face_records)
    permitted = []
    for edge in sorted(edges):
        incident = frozenset(geometry.faces_using_edge(edge))
        if all(incident.issubset(members[id(bindings[face])])
               for face in incident if face in bindings):
            permitted.append(edge)
    return frozenset(permitted)


def _quadratic_settings(size_field, native_options, quality_options, recombine, backend):
    size_metric = MetricFieldSpec.from_size_field(size_field)
    options = native_options
    if options.metric_mode == "isotropic_spatial":
        explicit = options.metric_field
        if explicit is None:
            combined = size_metric
        else:
            combined = MetricFieldSpec(
                IsotropicMetricControl(min(explicit.global_control.target_size,
                                           size_metric.global_control.target_size)),
                feature_controls=(*explicit.feature_controls, *size_metric.feature_controls),
                imported_samples=explicit.imported_samples,
                maximum_anisotropy=min(explicit.maximum_anisotropy, size_metric.maximum_anisotropy),
                maximum_gradation=min(explicit.maximum_gradation, size_metric.maximum_gradation),
            )
        options = replace(options, metric_field=combined)
    elif size_metric.feature_controls:
        # Preserve named SizeField refinements in physical coordinates.
        options = replace(options, metric_mode="isotropic_spatial", metric_field=size_metric)
    values = {}
    if quality_options is not None:
        policy = quality_options.quality_policy
        values = dict(
            min_scaled_jacobian=policy.minimum_scaled_jacobian,
            max_aspect_ratio=policy.maximum_aspect_ratio,
            min_angle=policy.minimum_angle, max_angle=policy.maximum_angle,
            max_warpage=policy.maximum_warpage,
            max_element_growth=quality_options.max_element_growth,
        )
    return SurfaceMeshOptions(
        order="quadratic", recombine=bool(recombine),
        target_size=size_metric.global_control.target_size, backend=backend,
        native_options=options, prefer_quality_policy=True, enforce_quality=True, **values,
    )


def finish_quadratic_components(geometry, mesh, bindings, boundary_registry, *,
                                native_options, size_field, quality_options,
                                recombine, backend, pinned_edges=(), beam_edges=(),
                                declared_junction_edges=(), supplied_seeding=False,
                                metric_model_uuid=None, metric_geometry_revision=None,
                                face_diagnostics, cancellation_check=None):
    """Qualify detached components, then merge using fresh global element IDs."""
    if not bindings:
        return mesh
    settings = _quadratic_settings(size_field, native_options, quality_options, recombine, backend)
    handled = set()
    for face in sorted(bindings):
        if face in handled:
            continue
        binding = bindings[face]
        faces = {sector.face.id for sector in binding.face_records}
        handled.update(faces)
        binding.validate(cancellation_check=cancellation_check)
        old_elements = {element for owner in faces for element in mesh.elements_of_face[owner]}
        component_edges = {use.edge for owner in faces
                           for loop in (geometry.faces[owner].loop, *geometry.faces[owner].holes)
                           for use in loop}
        external_nodes = {node for element, nodes in mesh.shells.items()
                          if element not in old_elements for node in nodes}
        external_nodes.update(node for nodes in mesh.beams.values() for node in nodes)
        for coupling in mesh.couplings.values():
            external_nodes.update((coupling.beam_node, *coupling.plate_nodes))
        excluded = set(pinned_edges) | set(beam_edges) | set(declared_junction_edges)
        eligible = tuple(sorted(
            edge for edge in component_edges
            if not supplied_seeding and edge not in excluded
            and isinstance(geometry.edges[edge].curve, Straight)
            and len(geometry.faces_using_edge(edge)) == 2
            and set(geometry.faces_using_edge(edge)).issubset(faces)
            and not external_nodes.intersection(mesh.nodes_of_edge[edge])
        ))
        seed = deepcopy(mesh)
        seed.tris = {e: nodes for e, nodes in seed.tris.items() if e in old_elements}
        seed.quads = {e: nodes for e, nodes in seed.quads.items() if e in old_elements}
        seed.beams = {}
        seed.couplings = {}
        seed.elements_of_face = {owner: list(mesh.elements_of_face[owner]) for owner in sorted(faces)}
        seed.nodes_of_edge = {edge: list(mesh.nodes_of_edge[edge]) for edge in sorted(component_edges)}
        # Keep the global node inventory for allocation, but no external cells.
        trial = refine_quadratic_component(
            geometry, seed, binding, boundary_registry.entries(), settings,
            eligible_edges=eligible, metric_model_uuid=metric_model_uuid,
            metric_geometry_revision=metric_geometry_revision,
            cancellation_check=cancellation_check,
        )
        if not trial.accepted:
            error = MeshError("cylindrical frontal_delaunay component rejected: "
                              + str(trial.diagnostics.get("reason", "quality policy")))
            error.diagnostics = deepcopy(trial.diagnostics)
            raise error
        for node in external_nodes:
            if node in trial.mesh.nodes and np.asarray(trial.mesh.nodes[node]).tobytes() != np.asarray(mesh.nodes[node]).tobytes():
                raise MeshError("cylindrical refinement changed an external node")
        for edge in component_edges - set(eligible):
            if trial.mesh.nodes_of_edge[edge] != mesh.nodes_of_edge[edge]:
                raise MeshError("cylindrical refinement changed a protected boundary sequence")
        candidate = deepcopy(mesh)
        first = max((*mesh.shells, *mesh.beams), default=0) + 1
        element_map = {old: first + index for index, old in enumerate(sorted(trial.mesh.shells))}
        for element in old_elements:
            candidate.tris.pop(element, None)
            candidate.quads.pop(element, None)
            candidate.activity.pop(element, None)
        candidate.nodes.update({node: np.asarray(point).copy() for node, point in trial.mesh.nodes.items()})
        candidate.tris.update({element_map[e]: tuple(nodes) for e, nodes in trial.mesh.tris.items()})
        candidate.quads.update({element_map[e]: tuple(nodes) for e, nodes in trial.mesh.quads.items()})
        candidate.activity.update({element_map[e]: trial.mesh.activity.get(e, 1.) for e in element_map})
        for owner in sorted(faces):
            candidate.elements_of_face[owner] = [element_map[e] for e in trial.mesh.elements_of_face[owner]]
            candidate.grid_of_face.pop(owner, None)
            candidate.block_grids_of_face.pop(owner, None)
        for name in ("elements_of_sheet", "elements_of_member"):
            groups = getattr(candidate, name)
            for key, members in list(groups.items()):
                if not old_elements.intersection(members):
                    continue
                selected_faces = [owner for owner in sorted(faces)
                                  if set(mesh.elements_of_face[owner]).intersection(members)]
                if any(not set(mesh.elements_of_face[owner]).issubset(members) for owner in selected_faces):
                    raise MeshError("ambiguous cylindrical component group ownership")
                groups[key] = [e for e in members if e not in old_elements] + [
                    e for owner in selected_faces for e in candidate.elements_of_face[owner]]
        candidate.nodes_of_edge.update({edge: list(nodes) for edge, nodes in trial.mesh.nodes_of_edge.items()})
        binding.validate(cancellation_check=cancellation_check)
        for edge, parameter, node, raw in trial.boundary_stations:
            point = np.frombuffer(raw, dtype=np.float64).copy()
            if np.asarray(candidate.nodes[node]).tobytes() != point.tobytes():
                raise MeshError("cylindrical promotion station disagrees with its owner point")
            boundary_registry.register(edge, parameter, point, node_id=node, owner=geometry.handle("edge", edge))
        mesh = candidate
        for owner in sorted(faces):
            entry = face_diagnostics[owner]
            entry["cylindrical_seed"] = {"point_placement": "legacy_lattice", "role": "detached_quadratic_background"}
            entry["cylindrical_refinement"] = trial.diagnostics["faces"][str(owner)]
            entry["cylindrical_component"] = {
                "accepted": True, "faces": sorted(faces), "point_placement": "frontal_delaunay",
                "owner_contract": binding.certification_kind,
                "order": "quadratic", "physical_metric": True,
                "published_insertions": trial.diagnostics["published_insertions"],
                "reserved_node_reuses": trial.diagnostics["reserved_node_reuses"],
                "quality": trial.diagnostics["promoted_quality"],
            }
    return mesh
