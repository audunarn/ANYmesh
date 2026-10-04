"""Owner-bound pre-native INPUT identities, without native triangulation."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from anygeometry import EntityRef, GeometryError, to_dict

from anymesher._material_input_provenance import _TERMS
from anymesher.boundary import GlobalEdgeBoundaryRegistry
from anymesher.errors import MeshError
from anymesher.meshing_view import GeometryMeshingView
from test_material_boundary_receipts import owner_boundary


def inputs():
    model, binding, mesh, registry, curved = owner_boundary()
    loops = tuple(SimpleNamespace(node_ids=nodes, uv=uv, segment_specs=specs)
                  for nodes, uv, specs in binding.loops(mesh, registry, ()))
    edge_ids = {int(path.source_edge) for paths in binding.region.boundaries for path in paths}
    context = {"model_id": str(model.model_id), "revision": model.revision,
               "edges": {edge: {**{name: True for name in _TERMS},
                                  "default_seeding": False, "selected": False}
                         for edge in edge_ids}}
    emitted = {"transform": np.eye(2),
               "chart_loops": tuple(np.array(loop.uv, copy=True) for loop in loops),
               "chart_constraints": (), "chart_pinned": np.empty((0, 2))}
    return model, binding, mesh, registry, loops, context, emitted, curved


def test_bound_paths_tokens_exact_stations_and_no_mutation():
    model, binding, mesh, registry, loops, context, emitted, curved = inputs()
    def entries():
        return tuple((item.key.edge_id, item.key.parameter, item.node_id,
                      item.point.tobytes()) for item in registry.entries())
    before = to_dict(model), deepcopy(mesh.nodes), entries()
    receipt = binding.input_provenance(mesh, registry, loops, (), context, emitted)
    payload = receipt.to_dict()
    assert payload["schema"] == "anymesher.material-input-provenance-v1"
    assert payload["owner_model_id"] == str(model.model_id)
    assert len(payload["boundary_inputs"]) == len(loops[0].node_ids)
    assert len({tuple(row["token"]) for row in payload["boundary_inputs"]}) == len(loops[0].node_ids)
    assert any(path["source_edge"] == curved for path in payload["boundary_paths"])
    assert all(row["reasons"] == ["default_seeding"] for row in payload["boundary_inputs"])
    assert any(item["stations"] == [[0, 1], [1, 1]] for item in payload["boundary_paths"])
    # Adjacent path occurrences may share one global node, never an invented duplicate.
    paths = payload["boundary_paths"]
    assert paths[0]["global_nodes"][-1] == paths[1]["global_nodes"][0]
    assert receipt.validate() is receipt
    assert to_dict(model) == before[0]
    assert mesh.nodes == before[1]
    assert entries() == before[2]


def test_missing_conflicting_and_stale_owner_registry_context_refuse():
    model, binding, mesh, registry, loops, context, emitted, curved = inputs()
    receipt = binding.input_provenance(mesh, registry, loops, (), context, emitted)
    incomplete = (SimpleNamespace(node_ids=loops[0].node_ids[:-1], uv=loops[0].uv[:-1],
                                  segment_specs=loops[0].segment_specs[:-1]),)
    with pytest.raises(MeshError, match="loop lost|segment"):
        binding.input_provenance(mesh, registry, incomplete, (), context, emitted)
    duplicate_pin = (loops[0].node_ids[0], loops[0].node_ids[0])
    with pytest.raises(MeshError, match="duplicate pinned"):
        binding.input_provenance(mesh, registry, loops, duplicate_pin, context, emitted)
    changed = deepcopy(context)
    changed["edges"].pop(curved)
    with pytest.raises(MeshError, match="complete route context"):
        binding.input_provenance(mesh, registry, loops, (), changed, emitted)
    foreign, *_ = owner_boundary()
    with pytest.raises(MeshError, match="different owner"):
        binding.input_provenance(mesh, GlobalEdgeBoundaryRegistry(GeometryMeshingView(foreign)),
                                 loops, (), context, emitted)
    context["edges"][curved]["default_seeding"] = True
    with pytest.raises(MeshError, match="contradicts|changed"):
        receipt.validate()
    context["edges"][curved]["default_seeding"] = False
    context["edges"][999] = {**{name: False for name in _TERMS}, "selected": False}
    with pytest.raises(MeshError, match="changed"):
        receipt.validate()
    context["edges"].pop(999)
    model.add_point(9, 9, 9)
    with pytest.raises((MeshError, GeometryError)):
        receipt.validate()


def test_wrong_model_identity_and_registered_chain_change_refuse():
    model, binding, mesh, registry, loops, context, emitted, curved = inputs()
    wrong = deepcopy(context)
    wrong["model_id"] = "not-this-owner"
    with pytest.raises(MeshError, match="stale owner identity"):
        binding.input_provenance(mesh, registry, loops, (), wrong, emitted)
    receipt = binding.input_provenance(mesh, registry, loops, (), context, emitted)
    node = max(mesh.nodes) + 1
    registry.register(curved, .25, node_id=node)
    mesh.nodes[node] = tuple(model.sample_edge(curved, [.25])[0])
    mesh.nodes_of_edge[curved].insert(1, node)
    with pytest.raises((MeshError, GeometryError)):
        receipt.validate()


def test_missing_registered_mesh_coordinate_is_typed_and_nonmutating():
    model, binding, mesh, registry, loops, context, emitted, _ = inputs()
    before = to_dict(model)
    node = loops[0].node_ids[0]
    mesh.nodes.pop(node)
    with pytest.raises(MeshError, match="missing registered node"):
        binding.input_provenance(mesh, registry, loops, (), context, emitted)
    assert to_dict(model) == before


def test_selected_interval_requires_exact_owner_station_pair():
    model, binding, mesh, registry, _, context, _, curved = inputs()
    straight = next(edge for edge in context["edges"] if edge != curved)
    context["edges"][straight] = {**{name: True for name in _TERMS}, "selected": True}
    loops = tuple(SimpleNamespace(node_ids=nodes, uv=uv, segment_specs=specs)
                  for nodes, uv, specs in binding.loops(mesh, registry, {straight}))
    emitted = {"transform": np.eye(2), "chart_loops": (np.array(loops[0].uv, copy=True),),
               "chart_constraints": (), "chart_pinned": np.empty((0, 2))}
    assert binding.input_provenance(mesh, registry, loops, (), context, emitted).validate()
    altered = list(loops[0].segment_specs)
    index = next(index for index, spec in enumerate(altered) if spec is not None)
    altered[index] = (straight, .25, 1.)
    changed = (SimpleNamespace(node_ids=loops[0].node_ids, uv=loops[0].uv,
                               segment_specs=tuple(altered)),)
    with pytest.raises(MeshError, match="changed source interval"):
        binding.input_provenance(mesh, registry, changed, (), context, emitted)


def test_complete_registry_and_pin_sets_are_required(monkeypatch):
    model, binding, mesh, registry, loops, context, emitted, curved = inputs()
    extra = max(mesh.nodes) + 1
    registry.register(curved, .25, node_id=extra)
    mesh.nodes[extra] = tuple(model.sample_edge(curved, [.25])[0])
    with pytest.raises(MeshError, match="omitted"):
        binding.input_provenance(mesh, registry, loops, (), context, emitted)
    model, binding, mesh, registry, loops, context, emitted, curved = inputs()
    node = mesh.nodes_of_edge[curved][0]
    registry.register(curved, .25, node_id=node)
    with pytest.raises(MeshError, match="multiple source stations|omitted|does not bind"):
        binding.input_provenance(mesh, registry, loops, (), context, emitted)
    model, binding, mesh, registry, loops, context, emitted, _ = inputs()
    from anymesher._material_region_binding import MaterialRegionBinding
    monkeypatch.setattr(MaterialRegionBinding, "interior",
                        lambda *_args, **_kwargs: ((), (99,), np.asarray(((.3, .3),))))
    with pytest.raises(MeshError, match="pin set"):
        binding.input_provenance(mesh, registry, loops, (), context, emitted)


@pytest.mark.parametrize("changed", ("loop_uv", "chart_loop", "transform",
                                      "constraint", "pin"))
def test_actual_emitted_inputs_cannot_change_without_refusal(changed):
    model, binding, mesh, registry, loops, context, emitted, _ = inputs()
    if changed == "loop_uv":
        loops[0].uv[0, 0] += .125
    elif changed == "chart_loop":
        emitted["chart_loops"][0][0, 0] += .125
    elif changed == "transform":
        emitted["transform"][0, 0] *= 2
    elif changed == "constraint":
        emitted["chart_constraints"] = (np.asarray(((0., 0.), (1., 1.))),)
    else:
        emitted["chart_pinned"] = np.asarray(((.3, .3),))
    with pytest.raises(MeshError, match="UV|emitted|constraints|pins"):
        binding.input_provenance(mesh, registry, loops, (), context, emitted)


def test_callback_substitution_and_coordinated_payload_rebase_refuse():
    model, binding, mesh, registry, loops, context, emitted, _ = inputs()
    receipt = binding.input_provenance(mesh, registry, loops, (), context, emitted)
    stable = receipt.to_dict()
    receipt.payload["boundary_inputs"][0]["global_node"] = 999
    with pytest.raises(MeshError, match="payload changed"):
        receipt.validate()
    assert receipt.to_dict() == stable
    model, binding, mesh, registry, loops, context, emitted, _ = inputs()
    def mutate(_stage):
        emitted["chart_loops"][0][0, 0] += .125
        return False
    with pytest.raises(MeshError, match="emitted"):
        binding.input_provenance(mesh, registry, loops, (), context, emitted,
                                 cancellation_check=mutate)


def test_callback_cannot_rebase_entry_baseline():
    model, binding, mesh, registry, loops, context, emitted, _ = inputs()
    receipt = binding.input_provenance(mesh, registry, loops, (), context, emitted)
    called = False

    def substitute(_stage):
        nonlocal called
        if called:
            return
        called = True
        context["callback_marker"] = "changed"
        replacement = binding.input_provenance(mesh, registry, loops, (), context, emitted)
        object.__setattr__(receipt, "payload", replacement.payload)
        object.__setattr__(receipt, "_entry_bytes", replacement._entry_bytes)

    with pytest.raises(MeshError, match="changed since capture"):
        receipt.validate(substitute)
    assert called


@pytest.mark.parametrize("missing", ("mapping", "node"))
def test_missing_retained_vertex_refuses_before_interior_mutation(monkeypatch, missing):
    model, binding, mesh, registry, loops, context, emitted, _ = inputs()
    retained_id = next(iter(model.vertices))
    region = replace(binding.region, retained_vertices=(EntityRef("vertex", retained_id),))
    binding = replace(binding, region=region)
    # Isolate the preflight: this synthetic retained handle is not part of the
    # owner's validated collection, while its boundary paths remain real.
    from anymesher._material_region_binding import MaterialRegionBinding
    monkeypatch.setattr(MaterialRegionBinding, "_validate_boundary_owner",
                        lambda *_args: None)
    before_nodes = deepcopy(mesh.nodes)
    mesh.node_of_vertex = ({} if missing == "mapping"
                           else {retained_id: max(mesh.nodes) + 1})
    before_mapping = dict(mesh.node_of_vertex)

    def mutating_interior(*_args, **_kwargs):
        pytest.fail("interior must not run before retained-vertex preflight")

    monkeypatch.setattr(MaterialRegionBinding, "interior", mutating_interior)
    with pytest.raises(MeshError, match="retained vertex lacks an existing registered node"):
        binding.input_provenance(mesh, registry, loops, (), context, emitted)
    assert mesh.nodes == before_nodes
    assert mesh.node_of_vertex == before_mapping
