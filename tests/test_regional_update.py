from __future__ import annotations

import numpy as np
import pytest

from anymesher.errors import MeshError
from anymesher.mesh import Coupling, Mesh
from types import MappingProxyType

from anymesher.seeding import Seeding
from anymesher.regional_update import (
    CrossRevisionAuthority,
    OwnerBaselineReceipt,
    RegionalQualityLimits,
    assemble_regional_replacement,
    remap_mesh_ids,
)

LIMITS = RegionalQualityLimits(max_aspect_ratio=5.0, max_warp=0.05, min_scaled_jacobian=0.3)


# ------------------------------------------------ real generated mesh metadata


def _real_square(order: str):
    """Ordinary one-plate geometry meshed by the real mapped hybrid generator."""

    from anygeometry.model import GeometryModel

    from anymesher import generate_hybrid_mesh

    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    geometry.add_plate(vertices)
    mesh = generate_hybrid_mesh(geometry, target_size=0.25, strategy="mapped", order=order)
    return geometry, mesh


def _with_real_metadata(geometry, mesh, target_size=0.25):
    """Attach metadata produced by the owning classes (not hand-shaped dicts)."""

    from anymesher import (
        StructuredBlock, StructuredLayoutPlan, StructuredLayoutReport, StructuredFacePlan,
        StructuredMeshingOptions, regularity_metrics,
    )
    from anymesher.recovery import AutomaticMeshResult

    face_id = next(iter(geometry.faces))
    block = StructuredBlock(1, face_id, None, 1, "mapped", "plane", len(mesh.quads),
                            boundary_edges=tuple(sorted(geometry.edges)), working_face_id=face_id)
    plan = StructuredLayoutPlan(
        str(geometry.model_id), int(geometry.revision), target_size, StructuredMeshingOptions(),
        (StructuredFacePlan(face_id, "mapped", "ok", 4, 1, None, 1, "plane", (1,)),), blocks=(block,),
    )
    report = StructuredLayoutReport(
        plan, {face_id: (face_id,)}, {e: (e,) for e in geometry.edges},
        diagnostics=("mapped",), metrics=regularity_metrics(mesh, target_size=target_size),
        seed_solution={e: 4 for e in geometry.edges}, status="planned", blocks=(block,),
    )
    mesh.structural_preparation = {"structured_layout": report.to_dict()}
    mesh.hybrid_diagnostics["automation"] = AutomaticMeshResult(
        type("R", (), {"mesh": mesh})(), "mapped", "ready", ({"method": "mapped", "status": "selected"},)
    ).to_dict()
    return report


@pytest.mark.parametrize("order", ["linear", "quadratic"])
def test_real_generated_metadata_full_renumber_and_regional_discard(order):
    geometry, mesh = _real_square(order)
    _with_real_metadata(geometry, mesh)
    layout = mesh.structural_preparation["structured_layout"]
    assert isinstance(layout["metrics"]["edges_outside_balanced_size_band"], int)
    assert mesh.hybrid_diagnostics["automation"]["solver_admission"] == "ADMITTED"

    nm = {n: n + 1000 for n in mesh.nodes}
    em = {e: e + 500 for e in (*mesh.quads, *mesh.tris, *mesh.beams)}
    out = remap_mesh_ids(mesh, nm, em)
    assert set(out.nodes) == set(nm.values()) and set(out.quads) == {e + 500 for e in mesh.quads}
    assert out.structural_preparation["structured_layout"]["source_to_working_faces"] == (
        layout["source_to_working_faces"]
    )
    assert out.hybrid_diagnostics["automation"] == mesh.hybrid_diagnostics["automation"]

    # regional assembly: stale whole-mesh evidence must be discarded explicitly
    from anymesher import generate_hybrid_mesh
    replacement = generate_hybrid_mesh(geometry, target_size=0.25, strategy="mapped", order=order)
    _with_real_metadata(geometry, replacement)
    top = max(max(mesh.nodes), max(replacement.nodes))
    top_e = max(max(mesh.quads), max(replacement.quads))
    kwargs = dict(
        quality=RegionalQualityLimits(),
        node_ids={n: n + top for n in replacement.nodes},
        element_ids={e: e + top_e for e in replacement.quads},
        affected_element_ids=list(mesh.quads),
    )
    with pytest.raises(MeshError, match="discard_stale_diagnostics"):
        assemble_regional_replacement(mesh, replacement, **kwargs)
    result = assemble_regional_replacement(mesh, replacement, discard_stale_diagnostics=True, **kwargs)
    assert result.diagnostics_discarded
    assert "previous.hybrid_diagnostics.automation" in result.discarded_diagnostic_keys
    assert result.mesh.structural_preparation == {} and result.mesh.hybrid_diagnostics == {}


def test_automation_rejected_attempt_ids_are_translated_and_unknown_refused():
    geometry, mesh = _real_square("linear")
    eid = next(iter(mesh.quads))
    mesh.hybrid_diagnostics["automation"] = {
        "contract_id": "ANYMESHER_AUTOMATIC_RECOVERY_V1", "status": "inspection_only",
        "mesh_validity": "VALID", "selected_method": "mapped",
        "element_families": {"quads": len(mesh.quads), "triangles": 0, "beams": 0, "couplings": 0},
        "quality_warnings": [],
        "attempts": [{"method": "mapped", "status": "inspection_only", "problem_element_ids": [eid],
                      "quality_warnings": [], "reason": "x"}],
        "solver_admission": "BLOCKED",
    }
    nm = {n: n + 1000 for n in mesh.nodes}
    em = {e: e + 500 for e in mesh.quads}
    out = remap_mesh_ids(mesh, nm, em)
    assert out.hybrid_diagnostics["automation"]["attempts"][0]["problem_element_ids"] == [eid + 500]
    mesh.hybrid_diagnostics["automation"]["attempts"][0]["problem_element_ids"] = [10**6]
    with pytest.raises(MeshError):
        remap_mesh_ids(mesh, nm, em)
    mesh.hybrid_diagnostics["automation"]["unknown_ids"] = [1]
    with pytest.raises(MeshError):
        remap_mesh_ids(mesh, nm, em)


def xyz(*v):
    return np.array(v, dtype=float)


def rich_mesh() -> Mesh:
    m = Mesh(geometry_model_id="model-a", geometry_revision=3)
    m.nodes = {1: xyz(0, 0, 0), 2: xyz(1, 0, 0), 3: xyz(1, 1, 0), 4: xyz(0, 1, 0),
               5: xyz(0, 0, 1), 6: xyz(1, 0, 1)}
    m.quads = {1: (1, 2, 3, 4)}
    m.beams = {2: (5, 6)}
    m.couplings = {3: Coupling.node_to_node(5, 1, (0, 0, 1))}
    m.node_of_vertex = {10: 1}
    m.nodes_of_edge = {20: [1, 2]}
    m.offset_nodes_of_edge = {20: [5]}
    m.grid_of_face = {30: np.array([[1, 2], [4, 3]])}
    m.block_grids_of_face = {31: (np.array([[1, 2], [0, 3]]),)}
    m.elements_of_face = {30: [1]}
    m.elements_of_edge = {20: [2]}
    m.elements_of_sheet = {50: [1]}
    m.elements_of_member = {40: [2]}
    m.nodes_of_member = {40: [5, 6]}
    m.activity = {1: 1.0, 2: 0.5}
    m.thickness_of_face = {30: 0.01}
    m.declared_plate_junction_edges = ((1, 2),)
    return m


NM = {1: 101, 2: 102, 3: 103, 4: 104, 5: 105, 6: 106}
EM = {1: 11, 2: 12, 3: 13}


def test_remap_translates_everything_and_detaches():
    src = rich_mesh()
    out = remap_mesh_ids(src, dict(NM), dict(EM))
    assert out is not src
    assert set(out.nodes) == set(NM.values())
    assert out.quads == {11: (101, 102, 103, 104)} and out.beams == {12: (105, 106)}
    c = out.couplings[13]
    assert (c.beam_node, c.plate_nodes) == (105, (101,))
    assert out.node_of_vertex == {10: 101}
    assert out.nodes_of_edge == {20: [101, 102]} and out.offset_nodes_of_edge == {20: [105]}
    assert out.grid_of_face[30].tolist() == [[101, 102], [104, 103]]
    assert out.block_grids_of_face[31][0].tolist() == [[101, 102], [0, 103]]
    assert out.elements_of_face == {30: [11]} and out.elements_of_member == {40: [12]}
    assert out.nodes_of_member == {40: [105, 106]}
    assert out.activity == {11: 1.0, 12: 0.5}
    assert out.declared_plate_junction_edges == ((101, 102),)
    assert out.thickness_of_face == {30: 0.01}
    # source untouched, result detached
    assert set(src.nodes) == {1, 2, 3, 4, 5, 6}
    out.nodes[101][0] = 9.0
    assert src.nodes[1][0] == 0.0


def test_remap_identity_fast_path_is_detached_copy():
    src = rich_mesh()
    out = remap_mesh_ids(src, {n: n for n in src.nodes}, {1: 1, 2: 2, 3: 3})
    assert out is not src and set(out.nodes) == set(src.nodes)
    out.nodes[1][0] = 5.0
    out.grid_of_face[30][0, 0] = 77
    assert src.nodes[1][0] == 0.0 and src.grid_of_face[30][0, 0] == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda n, e: n.pop(6),                     # missing node
        lambda n, e: e.pop(3),                     # coupling id missing
        lambda n, e: n.update({99: 199}),          # unknown node
        lambda n, e: n.update({2: 101}),           # not injective
        lambda n, e: n.update({2: 0}),             # non positive
        lambda n, e: n.update({2: -4}),
        lambda n, e: n.update({2: 2**63}),         # beyond int64
        lambda n, e: n.update({2: True}),          # bool
        lambda n, e: n.update({2: 5.0}),           # float
        lambda n, e: e.update({3: 11}),            # shared element namespace
    ],
)
def test_remap_rejects_invalid_maps(mutate):
    n, e = dict(NM), dict(EM)
    mutate(n, e)
    with pytest.raises(MeshError):
        remap_mesh_ids(rich_mesh(), n, e)


def test_remap_rejects_non_dict_map():
    with pytest.raises(MeshError):
        remap_mesh_ids(rich_mesh(), list(NM.items()), dict(EM))  # type: ignore[arg-type]


def test_remap_rejects_dangling_and_duplicate_ids():
    m = rich_mesh()
    m.quads[1] = (1, 2, 3, 99)
    with pytest.raises(MeshError):
        remap_mesh_ids(m, dict(NM), dict(EM))
    m = rich_mesh()
    m.beams[1] = (5, 6)  # element id reused across families
    with pytest.raises(MeshError):
        remap_mesh_ids(m, dict(NM), {1: 11, 2: 12, 3: 13})


def test_remap_metadata_refusal_and_safe_retention():
    m = rich_mesh()
    m.hybrid_diagnostics = {"patch_node_ids": [1, 2]}
    with pytest.raises(MeshError):
        remap_mesh_ids(m, dict(NM), dict(EM))
    m.hybrid_diagnostics = {"stages": {3: "x"}}
    with pytest.raises(MeshError):
        remap_mesh_ids(m, dict(NM), dict(EM))
    m.hybrid_diagnostics = {"strategy": "mapped", "num_nodes": 6}  # no explicit schema
    with pytest.raises(MeshError):
        remap_mesh_ids(m, dict(NM), dict(EM))
    m.hybrid_diagnostics = {"preflight_count": 0, "certifiable": True}
    out = remap_mesh_ids(m, dict(NM), dict(EM))
    assert out.hybrid_diagnostics == {"preflight_count": 0, "certifiable": True}
    assert out.hybrid_diagnostics is not m.hybrid_diagnostics


def test_remap_rejects_subclass():
    class Sub(Mesh):
        pass

    with pytest.raises(MeshError):
        remap_mesh_ids(Sub(), {}, {})


# ------------------------------------------------------------------ assembly


def two_part_mesh() -> Mesh:
    """Part A: two quads and a triangle; part B (disconnected): two beams."""
    m = Mesh(geometry_model_id="model-a", geometry_revision=1)
    m.nodes = {1: xyz(0, 0, 0), 2: xyz(1, 0, 0), 3: xyz(2, 0, 0),
               4: xyz(0, 1, 0), 5: xyz(1, 1, 0), 6: xyz(2, 1, 0), 7: xyz(3, 0.5, 0),
               11: xyz(10, 0, 0), 12: xyz(11, 0, 0), 13: xyz(12, 0, 0)}
    m.quads = {1: (1, 2, 5, 4), 2: (2, 3, 6, 5)}
    m.tris = {5: (3, 7, 6)}
    m.beams = {3: (11, 12), 4: (12, 13)}
    m.nodes_of_member = {40: [11, 12, 13]}
    m.elements_of_member = {40: [3, 4]}
    m.elements_of_face = {30: [1, 2, 5]}
    m.node_of_vertex = {10: 1, 11: 13}
    return m


def beam_replacement() -> Mesh:
    r = Mesh(geometry_model_id="model-a", geometry_revision=1)
    r.nodes = {i: xyz(10 + 0.5 * (i - 1), 0, 0) for i in range(1, 6)}
    r.beams = {i: (i, i + 1) for i in range(1, 5)}
    r.nodes_of_member = {40: [1, 2, 3, 4, 5]}
    r.elements_of_member = {40: [1, 2, 3, 4]}
    r.node_of_vertex = {11: 5}
    return r


def test_disjoint_beam_part_replacement_keeps_shell_part_frozen():
    prev = two_part_mesh()
    prev_snapshot = {k: tuple(v) for k, v in prev.quads.items()}
    res = assemble_regional_replacement(
        prev, beam_replacement(), quality=LIMITS,
        node_ids={i: 20 + i for i in range(1, 6)},
        element_ids={i: 30 + i for i in range(1, 5)},
        affected_element_ids=[3, 4],
        # vertex 10 keeps node 1, vertex 11 must be rebuilt, but vertex 10 is not in the replacement
    )
    m = res.mesh
    assert res.interface_node_ids == () and res.retired_node_ids == (11, 12, 13)
    assert res.retired_element_ids == (3, 4) and res.retained_element_ids == (1, 2, 5)
    assert res.new_element_ids == (31, 32, 33, 34) and res.quality is None
    # outside frozen: same coords, connectivity and order
    assert list(m.quads.items()) == list(prev.quads.items()) == list(prev_snapshot.items())
    assert m.tris == prev.tris
    assert list(m.nodes)[:7] == [1, 2, 3, 4, 5, 6, 7]
    assert m.beams == {31: (21, 22), 32: (22, 23), 33: (23, 24), 34: (24, 25)}
    assert m.nodes_of_member == {40: [21, 22, 23, 24, 25]}
    assert m.elements_of_member == {40: [31, 32, 33, 34]}
    assert m.node_of_vertex == {10: 1, 11: 25}
    assert m.elements_of_face == {30: [1, 2, 5]}
    assert 11 not in m.nodes
    # the input was not mutated
    assert set(prev.beams) == {3, 4} and 11 in prev.nodes


def test_owner_map_selection_and_bad_ownership():
    prev = two_part_mesh()
    owners = {1: "A", 2: "A", 5: "A", 3: "B", 4: "B"}
    kw = dict(quality=LIMITS, node_ids={i: 20 + i for i in range(1, 6)},
              element_ids={i: 30 + i for i in range(1, 5)})
    res = assemble_regional_replacement(prev, beam_replacement(), owner_of_element=owners,
                                        source_owner="B", **kw)
    assert res.retired_element_ids == (3, 4)
    bad = dict(owners)
    bad.pop(5)
    with pytest.raises(MeshError):
        assemble_regional_replacement(prev, beam_replacement(), owner_of_element=bad, source_owner="B", **kw)
    with pytest.raises(MeshError):  # owner disagrees with explicit selection
        assemble_regional_replacement(prev, beam_replacement(), owner_of_element=owners,
                                      source_owner="B", affected_element_ids=[3], **kw)
    with pytest.raises(MeshError):  # no selection at all
        assemble_regional_replacement(prev, beam_replacement(), **kw)
    with pytest.raises(MeshError):  # unknown owner
        assemble_regional_replacement(prev, beam_replacement(), owner_of_element=owners,
                                      source_owner="Z", **kw)


def test_vertex_lost_without_rebuild_is_refused():
    r = beam_replacement()
    r.node_of_vertex = {}
    with pytest.raises(MeshError):
        assemble_regional_replacement(
            two_part_mesh(), r, quality=LIMITS, node_ids={i: 20 + i for i in range(1, 6)},
            element_ids={i: 30 + i for i in range(1, 5)}, affected_element_ids=[3, 4])


def shell_replacement() -> Mesh:
    """Quad 1 of the two-part mesh as two half-width quads; nodes 3 and 6 are interface."""
    r = Mesh(geometry_model_id="model-a", geometry_revision=1)
    r.nodes = {1: xyz(0, 0, 0), 2: xyz(0.5, 0, 0), 3: xyz(1, 0, 0),
               4: xyz(0, 1, 0), 5: xyz(0.5, 1, 0), 6: xyz(1, 1, 0)}
    r.quads = {1: (1, 2, 5, 4), 2: (2, 3, 6, 5)}
    r.elements_of_face = {30: [1, 2]}
    return r


SHELL_KW = dict(
    quality=LIMITS,
    node_ids={1: 7 + 100, 2: 8 + 100, 4: 9 + 100, 5: 10 + 100},
    element_ids={1: 41, 2: 42},
    affected_element_ids=[1],
    interface_node_ids={3: 2, 6: 5},
)


def test_connected_shell_replacement_with_mixed_families_outside():
    prev = prepared_shell_case()
    res = assemble_regional_replacement(
        prev, shell_replacement(), stations={2: (1, 0, 0), 5: (1, 1, 0)}, **SHELL_KW)
    m = res.mesh
    assert res.interface_node_ids == (2, 5) and res.retired_node_ids == (1, 4)
    assert res.stations_validated
    assert m.quads[2] == prev.quads[2] and m.tris == prev.tris and m.beams == prev.beams
    assert m.quads[41] == (107, 108, 110, 109) and m.quads[42] == (108, 2, 5, 110)
    assert 1 not in m.nodes and 4 not in m.nodes
    assert m.elements_of_face == {30: [2, 5, 41, 42]}
    assert res.quality is not None and res.quality.max_aspect_ratio == pytest.approx(2.0)
    assert m.node_of_vertex == {11: 13}


def test_connected_replacement_refuses_lost_vertex_metadata():
    # vertex 10 sits on retired node 1 and the replacement does not rebuild it
    with pytest.raises(MeshError):
        assemble_regional_replacement(two_part_mesh(), shell_replacement(), **SHELL_KW)
    r = shell_replacement()
    r.node_of_vertex = {10: 1}
    res = assemble_regional_replacement(two_part_mesh(), r, **SHELL_KW)
    assert res.mesh.node_of_vertex == {10: 107, 11: 13}


def prepared_shell_case():
    prev = two_part_mesh()
    prev.node_of_vertex = {11: 13}
    return prev


def test_shell_refusals():
    base = dict(SHELL_KW)

    def run(replacement=None, prev=None, **over):
        kw = {**base, **over}
        return assemble_regional_replacement(
            prev or prepared_shell_case(), replacement or shell_replacement(), **kw)

    run()  # sanity: the baseline succeeds
    with pytest.raises(MeshError):  # interface incomplete
        run(interface_node_ids={3: 2})
    with pytest.raises(MeshError):  # wrong interface node
        run(interface_node_ids={3: 2, 6: 6})
    with pytest.raises(MeshError):  # disjoint claim for a connected region
        run(interface_node_ids=None)
    r = shell_replacement()
    r.nodes[3] = xyz(1, 0, 0.01)
    with pytest.raises(MeshError):  # frozen interface coordinate moved
        run(replacement=r)
    with pytest.raises(MeshError):  # prescribed station disagrees
        run(stations={2: (1, 0, 0), 5: (1, 1, 0.5)})
    with pytest.raises(MeshError):  # reused permanent node id
        run(node_ids={1: 3, 2: 108, 4: 109, 5: 110})
    with pytest.raises(MeshError):  # reused permanent element id
        run(element_ids={1: 2, 2: 42})
    with pytest.raises(MeshError):  # interface renumbered through node_ids
        run(node_ids={1: 107, 2: 108, 4: 109, 5: 110, 3: 999})
    r = shell_replacement()
    r.geometry_revision = 2
    with pytest.raises(MeshError):  # geometry authority differs
        run(replacement=r)
    with pytest.raises(MeshError):  # explicit quality is mandatory and enforced
        run(quality=RegionalQualityLimits(1.5, 0.05, 0.3))
    with pytest.raises(MeshError):
        run(quality=None)


def test_nonconforming_seam_is_refused():
    r = shell_replacement()
    # split the interface edge 3-6 with a hanging node: previous side keeps a single edge
    r.nodes[7] = xyz(1, 0.5, 0)
    r.quads = {1: (1, 2, 5, 4), 2: (2, 3, 6, 5)}
    r.tris = {3: (3, 7, 6)}
    r.elements_of_face = {}
    kw = dict(SHELL_KW)
    kw["node_ids"] = {1: 107, 2: 108, 4: 109, 5: 110, 7: 111}
    kw["element_ids"] = {1: 41, 2: 42, 3: 43}
    with pytest.raises(MeshError):  # extra shell sharing the seam edge -> non-manifold
        assemble_regional_replacement(prepared_shell_case(), r, **kw)
    # replacement that drops the seam altogether
    r2 = shell_replacement()
    r2.quads = {1: (1, 2, 5, 4)}
    r2.elements_of_face = {}
    kw2 = dict(SHELL_KW, element_ids={1: 41})
    with pytest.raises(MeshError):
        assemble_regional_replacement(prepared_shell_case(), r2, **kw2)


def quadratic_pair() -> Mesh:
    m = Mesh(geometry_model_id="g", geometry_revision=1, order="quadratic")
    m.nodes = {1: xyz(0, 0, 0), 2: xyz(1, 0, 0), 3: xyz(1, 1, 0), 4: xyz(0, 1, 0),
               5: xyz(.5, 0, 0), 6: xyz(1, .5, 0), 7: xyz(.5, 1, 0), 8: xyz(0, .5, 0),
               9: xyz(2, 0, 0), 10: xyz(2, 1, 0), 11: xyz(1.5, 0, 0), 12: xyz(2, .5, 0),
               13: xyz(1.5, 1, 0)}
    m.quads = {1: (1, 2, 3, 4, 5, 6, 7, 8), 2: (2, 9, 10, 3, 11, 12, 13, 6)}
    return m


def quadratic_replacement() -> Mesh:
    r = Mesh(geometry_model_id="g", geometry_revision=1, order="quadratic")
    r.nodes = {1: xyz(0, 0, 0), 2: xyz(1, 0, 0), 3: xyz(1, 1, 0), 4: xyz(0, 1, 0),
               5: xyz(.5, 0, 0), 6: xyz(1, .5, 0), 7: xyz(.5, 1, 0), 8: xyz(0, .5, 0)}
    r.quads = {1: (1, 2, 3, 4, 5, 6, 7, 8)}
    return r


QKW = dict(
    quality=LIMITS,
    node_ids={1: 21, 4: 24, 5: 25, 7: 27, 8: 28},
    element_ids={1: 31},
    affected_element_ids=[1],
    interface_node_ids={2: 2, 3: 3, 6: 6},
)


def test_quadratic_interface_includes_midside_nodes():
    res = assemble_regional_replacement(quadratic_pair(), quadratic_replacement(), **QKW)
    assert res.interface_node_ids == (2, 3, 6)
    assert res.mesh.quads[31] == (21, 2, 3, 24, 25, 6, 27, 28)
    assert res.mesh.quads[2] == quadratic_pair().quads[2]


def test_quadratic_interface_refusals():
    with pytest.raises(MeshError):  # mid-side node omitted from the interface
        assemble_regional_replacement(
            quadratic_pair(), quadratic_replacement(), **{**QKW, "interface_node_ids": {2: 2, 3: 3}})
    r = quadratic_replacement()
    r.nodes[6] = xyz(1.1, .5, 0)
    with pytest.raises(MeshError):  # mid-side node moved
        assemble_regional_replacement(quadratic_pair(), r, **QKW)
    r = quadratic_replacement()
    r.order = "linear"
    r.quads = {1: (1, 2, 3, 4)}
    with pytest.raises(MeshError):  # order mismatch
        assemble_regional_replacement(quadratic_pair(), r, **QKW)


def test_diagnostics_refused_unless_explicitly_discarded():
    prev = two_part_mesh()
    prev.hybrid_diagnostics = {"certifiable": True}
    kw = dict(quality=LIMITS, node_ids={i: 20 + i for i in range(1, 6)},
              element_ids={i: 30 + i for i in range(1, 5)}, affected_element_ids=[3, 4])
    with pytest.raises(MeshError):
        assemble_regional_replacement(prev, beam_replacement(), **kw)
    res = assemble_regional_replacement(prev, beam_replacement(), discard_stale_diagnostics=True, **kw)
    assert res.diagnostics_discarded and res.mesh.hybrid_diagnostics == {}
    assert res.discarded_diagnostic_keys == ("previous.hybrid_diagnostics.certifiable",)
    assert prev.hybrid_diagnostics == {"certifiable": True}
    # unknown metadata is refused even when discarding was requested
    prev.hybrid_diagnostics = {"strategy": "mapped"}
    with pytest.raises(MeshError):
        assemble_regional_replacement(prev, beam_replacement(), discard_stale_diagnostics=True, **kw)


def test_replacement_with_unrebuildable_ordered_metadata_is_refused():
    prev = two_part_mesh()
    prev.nodes_of_edge = {60: [1, 2, 3]}   # edge spans retained and retired nodes
    r = shell_replacement()
    r.nodes_of_edge = {60: [1, 2, 3]}
    r.node_of_vertex = {10: 1}
    with pytest.raises(MeshError):
        assemble_regional_replacement(prev, r, **SHELL_KW)


# ------------------------------------------------- generator metadata / labels


def mapped_face_record():
    return {"requested_backend": "mapped", "selected_backend": "mapped",
            "actual_backend": "mapped", "fallback_reason": None, "phase_seconds": {}}


def test_native_surface_candidate_indices_and_serial_fallback_keep_their_namespaces():
    from anygeometry import GeometryModel
    from anymesher import generate_hybrid_mesh

    geometry = GeometryModel()
    face = geometry.add_plate(geometry.add_points(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))))
    geometry.add_sheet((face,))
    mesh = generate_hybrid_mesh(geometry, target_size=.4, strategy="native", native_backend="python",
                                recombine=False, qualified_s3=True)
    nm = {node: node + 100 for node in mesh.nodes}
    em = {element: element + 200 for element in mesh.tris}
    backend = mesh.hybrid_diagnostics["triangulation_backend_by_face"][face]
    quality = backend["quality_optimization"]["initial_quality"]
    # A face candidate can have different topology from the selected global
    # mesh. Even an identical integer does not acquire global element identity.
    quality["repair_element_ids"] = [min(em)]
    mesh.hybrid_diagnostics["parallel"] = {"used": False, "route": "serial", "reason": "one component"}
    numbered = remap_mesh_ids(mesh, nm, em)
    assert numbered.hybrid_diagnostics["triangulation_backend_by_face"][face] == backend
    assert set(numbered.structural_preparation["qualified_s3"]["element_ids"]) == set(em.values())
    assert numbered.hybrid_diagnostics["parallel"] == mesh.hybrid_diagnostics["parallel"]

    quality["unknown_element_ids"] = [min(em)]
    with pytest.raises(MeshError, match="unknown_element_ids"):
        remap_mesh_ids(mesh, nm, em)
    del quality["unknown_element_ids"]
    mesh.hybrid_diagnostics["parallel"].update(used=True, route="parallel")
    with pytest.raises(MeshError, match="mapped parallel plate joins only|cannot carry a certification"):
        remap_mesh_ids(mesh, nm, em)


def test_joined_plate_rejects_unknown_evidence_without_starting_a_pool():
    mesh = rich_mesh()
    mesh.hybrid_diagnostics = {
        "certifiable": False, "certification_mode": "interactive", "strategy_by_face": {30: "mapped"},
        "parallel": {"used": True, "route": "parallel", "components": 2, "separation": 0.5,
                     "merge_reasons": 0, "runtime": {"new_source_certificate": "unproven"}},
    }
    with pytest.raises(MeshError, match="new_source_certificate"):
        remap_mesh_ids(mesh, NM, EM)
    mesh.hybrid_diagnostics["certifiable"] = True
    with pytest.raises(MeshError, match="cannot carry a certification"):
        remap_mesh_ids(mesh, NM, EM)


def generator_diagnostics():
    return {
        "requested_target_size": 0.5,
        "strategy_by_face": {1: "mapped", 2: "mapped"},
        "triangulation_backend_by_face": {1: mapped_face_record(), 2: mapped_face_record()},
        "geometry_model_id": "model-a", "geometry_revision": 3,
        "certification_mode": "none", "certifiable": True, "preflight_count": 0,
        "structured_layout_status": "accepted", "structured_plan_hash": "abc",
        "structured_quality": {"accepted": True, "growth_violation_count": 0, "max_ratio": 1.2},
        "structural_preparation_hash": "def", "qualified_s3_preparation": True,
        "reused_prepared_working_copy": False, "phase_seconds": {"seeding": 0.01},
        "completed_phases": ["seeding"], "runtime": {"operation_counts": {"a": 3}},
    }


def test_remap_keeps_documented_generator_metadata_with_int_face_keys():
    m = rich_mesh()
    m.structural_preparation = {"schema": "anymesher.prep/1", "status": "ready"}
    m.hybrid_diagnostics = generator_diagnostics()
    m.seeding = Seeding(divisions={1: 2})
    out = remap_mesh_ids(m, MappingProxyType(dict(NM)), MappingProxyType(dict(EM)))
    assert out.hybrid_diagnostics == generator_diagnostics()
    assert out.structural_preparation == {"schema": "anymesher.prep/1", "status": "ready"}
    assert out.hybrid_diagnostics["strategy_by_face"] == {1: "mapped", 2: "mapped"}
    assert out.hybrid_diagnostics is not m.hybrid_diagnostics
    ident = remap_mesh_ids(m, MappingProxyType({n: n for n in NM}), MappingProxyType({1: 1, 2: 2, 3: 3}))
    assert ident.hybrid_diagnostics == generator_diagnostics() and ident is not m
    assert ident.seeding == m.seeding and ident.seeding is not m.seeding


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(mystery=1),                                        # unknown field
        lambda d: d["structured_quality"].update(worst_element=12),           # unknown integer
        lambda d: d["structured_quality"].update(worst=[1, 2]),               # sequence payload
        lambda d: d["triangulation_backend_by_face"][1].update(extra=3),      # not the mapped record
        lambda d: d["strategy_by_face"].update({3: 4}),                       # non-string strategy
        lambda d: d["runtime"].update(node_labels=[1]),
        lambda d: d.update(preflight_count=True),
        lambda d: d["runtime"]["operation_counts"].update(x=1.5),
    ],
)
def test_unknown_or_malformed_generator_metadata_is_refused(mutate):
    m = rich_mesh()
    m.hybrid_diagnostics = generator_diagnostics()
    mutate(m.hybrid_diagnostics)
    with pytest.raises(MeshError):
        remap_mesh_ids(m, dict(NM), dict(EM))
    with pytest.raises(MeshError):  # identity cannot launder it either
        remap_mesh_ids(m, {n: n for n in NM}, {1: 1, 2: 2, 3: 3})
    m = rich_mesh()
    m.structural_preparation = {"schema": "x", "status": "y", "nodes": [1]}
    with pytest.raises(MeshError):
        remap_mesh_ids(m, dict(NM), dict(EM))


def test_zero_temporary_labels_map_to_positive_permanent_ids():
    m = Mesh(geometry_model_id="g", geometry_revision=1)
    m.nodes = {0: xyz(0, 0, 0), 1: xyz(1, 0, 0), 2: xyz(1, 1, 0), 3: xyz(0, 1, 0)}
    m.quads = {0: (0, 1, 2, 3)}
    m.grid_of_face = {7: np.array([[0, 1], [3, 2]])}
    m.elements_of_face = {7: [0]}
    out = remap_mesh_ids(m, {0: 10, 1: 11, 2: 12, 3: 13}, {0: 20})
    assert out.quads == {20: (10, 11, 12, 13)}
    assert out.grid_of_face[7].tolist() == [[10, 11], [13, 12]]
    assert out.elements_of_face == {7: [20]}
    with pytest.raises(MeshError):  # 0 is never a permanent target
        remap_mesh_ids(m, {0: 0, 1: 11, 2: 12, 3: 13}, {0: 20})
    # quadratic grids use 0 as the empty-centre marker: node 0 would be ambiguous
    q = quadratic_replacement()
    q.nodes[0] = q.nodes.pop(1)
    q.quads = {1: (0, 2, 3, 4, 5, 6, 7, 8)}
    q.grid_of_face = {1: np.array([[0, 5], [0, 2]])}
    with pytest.raises(MeshError):
        remap_mesh_ids(q, {n: n + 100 for n in q.nodes}, {1: 5})


def test_assembly_accepts_zero_temporary_labels():
    r = beam_replacement()
    r.nodes = {i - 1: v for i, v in r.nodes.items()}
    r.beams = {i - 1: (a - 1, b - 1) for i, (a, b) in r.beams.items()}
    r.nodes_of_member = {40: [0, 1, 2, 3, 4]}
    r.elements_of_member = {40: [0, 1, 2, 3]}
    r.node_of_vertex = {11: 4}
    res = assemble_regional_replacement(
        two_part_mesh(), r, quality=LIMITS, node_ids={i: 21 + i for i in range(5)},
        element_ids={i: 31 + i for i in range(4)}, affected_element_ids=[3, 4])
    assert res.mesh.beams[31] == (21, 22) and res.mesh.node_of_vertex[11] == 25


def test_discarded_diagnostics_are_reported_not_silent():
    prev = two_part_mesh()
    prev.hybrid_diagnostics = generator_diagnostics()
    prev.structural_preparation = {"schema": "s", "status": "ok"}
    res = assemble_regional_replacement(
        prev, beam_replacement(), quality=LIMITS, node_ids={i: 20 + i for i in range(1, 6)},
        element_ids={i: 30 + i for i in range(1, 5)}, affected_element_ids=[3, 4],
        discard_stale_diagnostics=True)
    assert "previous.hybrid_diagnostics.structured_quality" in res.discarded_diagnostic_keys
    assert "previous.structural_preparation.status" in res.discarded_diagnostic_keys
    assert res.mesh.hybrid_diagnostics == {} and res.mesh.structural_preparation == {}


# ------------------------------------- production (hybrid / GUI) metadata shapes

import copy as _copy

from anymesher.regional_update import (
    GEOMETRY_BOUND_METADATA,
    MESH_ID_BEARING_METADATA,
    SPLICE_RECOMPUTE_METADATA,
)

_S3_SCHEMA = "anymesher.qualified-s3-production-preparation-v2"


def _s3_not_applicable():
    # shape of s3_production.prepare_qualified_s3_mesh(no triangles) + hybrid's source_* update
    return {
        "admission": {"elements": [], "qualified_junction_edges": [], "topology_violations": []},
        "authority_model": {"prepared_revision": 4, "scope": "PREPARED_GEOMETRY_ORIENTED_SHEET_FACE_USE",
                            "source_model_id": "model-a", "source_revision": 3},
        "contract_id": "ANYMESHER_QUALIFIED_S3_PRODUCTION_PREPARATION_V2",
        "element_ids": [],
        "formulation_id": "CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1",
        "legacy_fallback": "FORBIDDEN",
        "quality_contract_id": "q", "repair_contract_id": "r",
        "schema": _S3_SCHEMA, "status": "NOT_APPLICABLE_NO_TRIANGLES",
    }


def _quality_row(element_id):
    return {"edge_ratio": 1.2, "element_id": element_id, "maximum_angle_deg": 80.0,
            "minimum_angle_deg": 40.0, "minimum_scaled_jacobian": 0.7, "normalized_area": 0.9,
            "owner_normal_alignment": 1.0, "signed_area_ratio": 1.0, "violations": []}


def _policy():
    return {"maximum_angle_deg": 120.0, "maximum_edge_ratio": 4.0, "minimum_angle_deg": 15.0,
            "minimum_normalized_area": 0.1, "minimum_scaled_jacobian": 0.2}


def _s3_admitted(tri_id, nodes):
    a, b, c = nodes
    record = _s3_not_applicable()
    record.update(
        status="ADMITTED",
        element_ids=[tri_id],
        admission={"elements": [_quality_row(tri_id)], "qualified_junction_edges": [[a, b]],
                   "topology_violations": []},
        element_owner_normals={str(tri_id): [0.0, 0.0, 1.0]},
        element_owner_sources={str(tri_id): {"face_id": 30, "face_use_ids": [1], "sheet_ids": [50]}},
        nodal_normals={str(n): [0.0, 0.0, 1.0] for n in nodes},
        quality_policy=_policy(),
        quality_target={"met": False, "policy": _policy(), "shortfall_element_ids": [tri_id]},
        repair={"added_elements": 0, "added_nodes": 0, "edge_flip_attempts": 1, "edge_flips": 1,
                "refinement_attempts": 0, "refinement_splits": 0, "winding_repairs": 0,
                "attempts": [{"action": "flip", "detail": "d", "edge": [a, b], "element_ids": [tri_id],
                              "sequence": 0, "status": "APPLIED"}]},
    )
    return record


def _production_diagnostics(poor=(), pairs=()):
    d = generator_diagnostics()
    d["triangulation_backend_by_face"] = {
        30: {"requested_backend": "mapped", "selected_backend": "mapped", "actual_backend": "mapped",
             "fallback_reason": None, "structured_action": "mapped", "working_face_ids": [30]},  # no phase_seconds
    }
    d["complex_geometry"] = {
        "strategy_ladder": ["mapped_or_structured", "native"], "component_candidate_budget": 6,
        "source_face_count": 1, "final_face_count": 1, "declared_junction_edge_count": 0,
        "source_strategy_by_face": {"30": "mapped"},
    }
    d["qualified_s3_preparation"] = {
        "contract_id": "c", "element_count": 0, "formulation_id": "f",
        "legacy_fallback": "FORBIDDEN", "status": "NOT_APPLICABLE_NO_TRIANGLES",
    }
    d["structured_quality"] = {
        "accepted": True, "policy": {"minimum_angle": 15, "maximum_warpage": 0.5},
        "violation_counts": {"minimum_angle": 0}, "poor_element_ids": list(poor),
        "growth_violation_pairs": [list(p) for p in pairs], "growth_violation_count": len(pairs),
        "maximum_aspect_ratio": 1.1,
    }
    d["runtime"] = {"candidate_count": 1, "connectivity_application_count": 1,
                    "operation_counts": {"a": 3}, "phase_seconds": {"x": 0.1}}
    return d


def _closure():
    return {
        "model_id": "model-a", "source_revision": 3, "working_revision": 4,
        "options": {"automatic_face_connections": True, "automatic_member_connections": True,
                    "automatic_member_sheet_connections": True, "declare_missing_owners": True,
                    "maximum_candidate_pairs": None, "maximum_applications": 5,
                    "maximum_face_records": None, "maximum_edge_records": None},
        "source_to_working_faces": {"30": [30]}, "source_to_working_edges": {"20": [20, 21]},
        "temporary_sheet_ids": [], "temporary_member_ids": [], "declared_face_connection_edges": [20],
        "candidate_queries": 1, "applications": 0, "face_connections": 0, "member_connections": 0,
        "member_sheet_connections": 0, "diagnostics": [], "status": "applied", "preparation_hash": "h",
    }


def _production_preparation(s3):
    return {
        "status": "applied", "structural_closure": _closure(),
        "structured_layout": {"status": "applied", "plan": {"model_id": "model-a", "faces": [30, 31]},
                              "quality": None, "metrics": {"irregular_count": 0, "mean_ratio": 1.1}},
        "source_to_working_faces": {"30": [30]}, "source_to_working_edges": {"20": [20, 21]},
        "qualified_s3": s3,
    }


def test_gui_mapped_quad_only_output_survives_renumbering_exactly():
    m = rich_mesh()  # quads + beams + coupling, no triangles
    m.structural_preparation = _production_preparation(_s3_not_applicable())
    m.hybrid_diagnostics = _production_diagnostics(poor=[1], pairs=[(1, 2, 1.5)])
    src = _copy.deepcopy(m.hybrid_diagnostics)
    out = remap_mesh_ids(m, dict(NM), dict(EM))
    assert out.structural_preparation["qualified_s3"] == _s3_not_applicable()  # nothing to translate
    assert out.structural_preparation["structural_closure"] == _closure()
    assert out.structural_preparation["source_to_working_edges"] == {"20": [20, 21]}  # geometry ids intact
    q = out.hybrid_diagnostics["structured_quality"]
    assert q["poor_element_ids"] == [11] and q["growth_violation_pairs"] == [[11, 12, 1.5]]
    assert out.hybrid_diagnostics["strategy_by_face"] == {1: "mapped", 2: "mapped"}  # geometry keys intact
    assert m.hybrid_diagnostics == src and out.hybrid_diagnostics is not m.hybrid_diagnostics
    ident = remap_mesh_ids(m, {n: n for n in NM}, {1: 1, 2: 2, 3: 3})
    assert ident.hybrid_diagnostics["structured_quality"]["poor_element_ids"] == [1]


def _tri_mesh():
    m = rich_mesh()
    m.nodes[7] = xyz(1, 1, 1)
    m.tris = {4: (1, 2, 7)}
    m.elements_of_face = {30: [1, 4]}
    return m


def test_admitted_s3_record_translates_every_mesh_id_and_keeps_geometry_ids():
    m = _tri_mesh()
    m.structural_preparation = _production_preparation(_s3_admitted(4, (1, 2, 7)))
    nm, em = {**NM, 7: 107}, {**EM, 4: 14}
    out = remap_mesh_ids(m, nm, em)
    s3 = out.structural_preparation["qualified_s3"]
    assert s3["element_ids"] == [14] and s3["admission"]["elements"][0]["element_id"] == 14
    assert s3["admission"]["qualified_junction_edges"] == [[101, 102]]
    assert list(s3["element_owner_normals"]) == ["14"] and list(s3["nodal_normals"]) == ["101", "102", "107"]
    assert s3["element_owner_sources"]["14"] == {"face_id": 30, "face_use_ids": [1], "sheet_ids": [50]}
    assert s3["quality_target"]["shortfall_element_ids"] == [14]
    assert s3["repair"]["attempts"][0]["edge"] == [101, 102] and s3["repair"]["attempts"][0]["element_ids"] == [14]
    assert s3["authority_model"]["source_revision"] == 3  # a geometry revision, never remapped
    assert m.structural_preparation["qualified_s3"]["element_ids"] == [4]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(element_ids=[4, 99]),                       # stale: not the triangles
        lambda s: s["admission"]["elements"][0].update(element_id=1),  # a quad, not a triangle
        lambda s: s["nodal_normals"].update({"99": [0, 0, 1]}),        # node not in mesh
        lambda s: s["repair"]["attempts"][0].update(element_ids=[77]), # vanished element: fail closed
        lambda s: s.update(extra_ids=[1]),                             # unknown extension
        lambda s: s["repair"].update(moved_nodes=[1]),
        lambda s: s.update(status="ADMITTED_MAYBE"),
    ],
)
def test_admitted_s3_record_stale_or_unknown_is_refused(mutate):
    m = _tri_mesh()
    s3 = _s3_admitted(4, (1, 2, 7))
    mutate(s3)
    m.structural_preparation = {"status": "applied", "qualified_s3": s3}
    with pytest.raises(MeshError):
        remap_mesh_ids(m, {**NM, 7: 107}, {**EM, 4: 14})


def test_not_applicable_record_with_triangles_present_is_refused():
    m = _tri_mesh()
    m.structural_preparation = {"status": "applied", "qualified_s3": _s3_not_applicable()}
    with pytest.raises(MeshError):
        remap_mesh_ids(m, {**NM, 7: 107}, {**EM, 4: 14})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["structured_quality"].update(poor_element_ids=[99]),         # not an element
        lambda d: d["structured_quality"].update(growth_violation_pairs=[[1, 2]]),
        lambda d: d["structured_quality"].update(mystery_ids=[1]),
        lambda d: d["structured_quality"]["violation_counts"].update(x=-1),
        lambda d: d["triangulation_backend_by_face"][30].update(material_region={"source_faces": [1]}),
        lambda d: d["complex_geometry"].update(node_labels=[1]),
        lambda d: d.update(cylindrical_shared_boundary_final_repair=[{"nodes": [1]}]),  # unknown extension
        lambda d: d.update(route="quad-first", q5={"moved_node_ids": [1]}),              # quad-first: not yet
    ],
)
def test_unknown_id_bearing_production_extensions_stay_fail_closed(mutate):
    m = rich_mesh()
    m.hybrid_diagnostics = _production_diagnostics(poor=[1])
    mutate(m.hybrid_diagnostics)
    with pytest.raises(MeshError):
        remap_mesh_ids(m, dict(NM), dict(EM))
    with pytest.raises(MeshError):
        remap_mesh_ids(m, {n: n for n in NM}, {1: 1, 2: 2, 3: 3})


def test_unknown_structured_layout_and_preparation_fields_are_refused():
    for edit in (
        lambda p: p["structured_layout"].update(irregular_nodes=[1, 2]),
        lambda p: p["structured_layout"]["metrics"].update(nodes=[1]),
        lambda p: p["structured_layout"]["metrics"].update(worst_element=3),
        lambda p: p["structural_closure"].update(patch_node_ids=[1]),
        lambda p: p.update(patch_node_ids=[1]),
    ):
        m = rich_mesh()
        m.structural_preparation = _production_preparation(_s3_not_applicable())
        edit(m.structural_preparation)
        with pytest.raises(MeshError):
            remap_mesh_ids(m, dict(NM), dict(EM))


def test_splice_drops_production_evidence_it_cannot_translate_and_says_so():
    prev = two_part_mesh()
    prev.structural_preparation = _production_preparation(_s3_admitted(5, (3, 7, 6)))  # tri 5 is in part A
    prev.hybrid_diagnostics = _production_diagnostics(poor=[1], pairs=[(1, 2, 1.5)])
    kw = dict(quality=LIMITS, node_ids={i: 20 + i for i in range(1, 6)},
              element_ids={i: 30 + i for i in range(1, 5)}, affected_element_ids=[3, 4])
    with pytest.raises(MeshError):  # whole-mesh evidence is stale for a splice
        assemble_regional_replacement(prev, beam_replacement(), **kw)
    res = assemble_regional_replacement(prev, beam_replacement(), discard_stale_diagnostics=True, **kw)
    assert res.mesh.structural_preparation == {} and res.mesh.hybrid_diagnostics == {}
    assert "previous.structural_preparation.qualified_s3" in res.discarded_diagnostic_keys
    assert "previous.hybrid_diagnostics.structured_quality" in res.discarded_diagnostic_keys
    # unknown extensions are still refused when discarding was requested
    prev.hybrid_diagnostics["q5"] = {"moved_node_ids": [1]}
    with pytest.raises(MeshError):
        assemble_regional_replacement(prev, beam_replacement(), discard_stale_diagnostics=True, **kw)


def test_metadata_policy_constants_partition_renumber_and_splice():
    assert set(MESH_ID_BEARING_METADATA) <= set(SPLICE_RECOMPUTE_METADATA)
    assert not set(GEOMETRY_BOUND_METADATA) & set(SPLICE_RECOMPUTE_METADATA)


# --------------------------------------------------------------- cross revision

OWNERS = {1: "A", 2: "A", 5: "A", 3: "B", 4: "B"}


def receipts(model="model-a", revision=1, digests=None, verified=True):
    digests = digests or {"A": "a0", "B": "b0"}
    return {o: OwnerBaselineReceipt(o, model, revision, d, verified) for o, d in digests.items()}


def authority(**over):
    base = dict(
        old_receipts=receipts(),
        new_receipts=receipts(revision=2, digests={"A": "a0", "B": "b1"}),
        changed_owners=frozenset({"B"}),
        unchanged_geometry_keys={"vertex": frozenset({10}), "face": frozenset({30})},
    )
    base.update(over)
    return CrossRevisionAuthority(**base)


def cross_kw(**over):
    kw = dict(quality=LIMITS, node_ids={i: 20 + i for i in range(1, 6)},
              element_ids={i: 30 + i for i in range(1, 5)}, owner_of_element=OWNERS,
              cross_revision=authority())
    kw.update(over)
    return kw


def new_revision_replacement():
    r = beam_replacement()
    r.geometry_revision = 2
    return r


def test_cross_revision_success_retains_other_owners_and_adopts_new_revision():
    res = assemble_regional_replacement(two_part_mesh(), new_revision_replacement(), **cross_kw())
    assert res.cross_revision and res.retired_element_ids == (3, 4)
    assert res.mesh.geometry_revision == 2 and res.mesh.geometry_model_id == "model-a"
    assert res.mesh.quads == two_part_mesh().quads and res.mesh.node_of_vertex == {10: 1, 11: 25}
    assert res.mesh.elements_of_face == {30: [1, 2, 5]}


@pytest.mark.parametrize(
    "case",
    ["no_authority", "unverified", "wrong_old_revision", "wrong_new_revision", "missing_owner",
     "retained_owner_changed", "face_not_attested", "changed_owner_unknown", "no_owner_map"],
)
def test_cross_revision_refusals(case):
    kw = cross_kw()
    if case == "no_authority":
        kw.pop("cross_revision")
    elif case == "unverified":
        kw["cross_revision"] = authority(old_receipts=receipts(verified=False))
    elif case == "wrong_old_revision":
        kw["cross_revision"] = authority(old_receipts=receipts(revision=9))
    elif case == "wrong_new_revision":
        kw["cross_revision"] = authority(new_receipts=receipts(revision=3, digests={"A": "a0", "B": "b1"}))
    elif case == "missing_owner":
        kw["cross_revision"] = authority(old_receipts={"B": receipts()["B"]})
    elif case == "retained_owner_changed":
        kw["cross_revision"] = authority(new_receipts=receipts(revision=2, digests={"A": "a9", "B": "b1"}))
    elif case == "face_not_attested":
        kw["cross_revision"] = authority(unchanged_geometry_keys={"vertex": frozenset({10})})
    elif case == "changed_owner_unknown":
        kw["cross_revision"] = authority(changed_owners=frozenset({"Z"}))
    elif case == "no_owner_map":
        kw.pop("owner_of_element")
        kw["affected_element_ids"] = [3, 4]
    with pytest.raises(MeshError):
        assemble_regional_replacement(two_part_mesh(), new_revision_replacement(), **kw)


def test_revision_difference_needs_authority_even_with_explicit_selection():
    with pytest.raises(MeshError):
        assemble_regional_replacement(
            two_part_mesh(), new_revision_replacement(), quality=LIMITS,
            node_ids={i: 20 + i for i in range(1, 6)}, element_ids={i: 30 + i for i in range(1, 5)},
            affected_element_ids=[3, 4])


def test_seed_change_allowed_only_for_named_edges_and_others_retained():
    prev = two_part_mesh()
    prev.seeding = Seeding(divisions={1: 2, 2: 3}, classes={1: 0, 2: 1})
    r = beam_replacement()
    r.seeding = Seeding(divisions={1: 2, 2: 5}, classes={2: 0})
    kw = dict(quality=LIMITS, node_ids={i: 20 + i for i in range(1, 6)},
              element_ids={i: 30 + i for i in range(1, 5)}, affected_element_ids=[3, 4])
    res = assemble_regional_replacement(prev, r, seed_changed_edges=[2], **kw)
    assert res.mesh.seeding.divisions == {1: 2, 2: 5} and res.seeding_changed_edges == (2,)
    assert res.mesh.seeding.classes == {1: 0, 2: 2}
    assert prev.seeding.divisions == {1: 2, 2: 3}
    for bad in (None, [1]):
        with pytest.raises(MeshError):
            assemble_regional_replacement(prev, r, seed_changed_edges=bad, **kw)
    # via the cross-revision edge owners
    r2 = new_revision_replacement()
    r2.seeding = Seeding(divisions={1: 2, 2: 5})
    res = assemble_regional_replacement(
        prev, r2, **cross_kw(cross_revision=authority(edge_owner={1: "A", 2: "B"})))
    assert res.mesh.seeding.divisions == {1: 2, 2: 5}
    with pytest.raises(MeshError):  # edge 1 belongs to a retained owner
        assemble_regional_replacement(
            prev, r2, **cross_kw(cross_revision=authority(edge_owner={1: "B", 2: "A"})))


def test_fully_retired_geometry_keys_of_removed_owner_are_dropped():
    r = beam_replacement()
    r.nodes_of_member, r.elements_of_member, r.node_of_vertex = {}, {}, {}
    kw = dict(quality=LIMITS, node_ids={i: 20 + i for i in range(1, 6)},
              element_ids={i: 30 + i for i in range(1, 5)}, affected_element_ids=[3, 4])
    with pytest.raises(MeshError):  # undeclared loss stays refused
        assemble_regional_replacement(two_part_mesh(), r, **kw)
    res = assemble_regional_replacement(
        two_part_mesh(), r, retired_geometry_keys={"vertex": [11], "member": [40]}, **kw)
    assert res.mesh.node_of_vertex == {10: 1} and res.mesh.nodes_of_member == {}
    assert res.mesh.elements_of_member == {}
    assert res.dropped_geometry_keys == {"vertex": (11,), "member": (40,)}
    # a key that still has retained mesh cannot be declared retired
    with pytest.raises(MeshError):
        assemble_regional_replacement(
            two_part_mesh(), r, retired_geometry_keys={"vertex": [10, 11], "member": [40]}, **kw)
    with pytest.raises(MeshError):  # unknown key
        assemble_regional_replacement(
            two_part_mesh(), r, retired_geometry_keys={"vertex": [11, 99], "member": [40]}, **kw)


# ------------------------------------------------- pure birth / pure death of an owner


def state_receipts(revision, spec):
    """spec: owner -> (digest, state)."""
    return {o: OwnerBaselineReceipt(o, "model-a", revision, d, True, s) for o, (d, s) in spec.items()}


def death_authority(**over):
    base = dict(
        old_receipts=state_receipts(1, {"A": ("a0", "present"), "B": ("b0", "present")}),
        new_receipts=state_receipts(2, {"A": ("a0", "present"), "B": ("gone", "absent")}),
        changed_owners=frozenset(), removed_owners=frozenset({"B"}),
        replacement_element_owner={},
        unchanged_geometry_keys={"vertex": frozenset({10}), "face": frozenset({30})},
    )
    base.update(over)
    return CrossRevisionAuthority(**base)


def empty_replacement(revision=2):
    return Mesh(geometry_model_id="model-a", geometry_revision=revision)


DEATH_KW = dict(quality=LIMITS, node_ids={}, element_ids={}, owner_of_element=OWNERS,
                retired_geometry_keys={"vertex": [11], "member": [40]})


def test_pure_deletion_retires_only_the_removed_owner_with_an_empty_replacement():
    prev = two_part_mesh()
    res = assemble_regional_replacement(
        prev, empty_replacement(), cross_revision=death_authority(), **DEATH_KW)
    m = res.mesh
    assert res.retired_element_ids == (3, 4) and res.retired_node_ids == (11, 12, 13)
    assert res.new_element_ids == () and res.new_node_ids == ()
    assert m.quads == prev.quads and m.tris == prev.tris and not m.beams
    assert set(m.nodes) == {1, 2, 3, 4, 5, 6, 7}
    assert m.node_of_vertex == {10: 1} and m.elements_of_face == {30: [1, 2, 5]}
    assert m.geometry_revision == 2
    assert res.dropped_geometry_keys == {"vertex": (11,), "member": (40,)}


@pytest.mark.parametrize(
    "case",
    ["undeclared_loss", "unknown_retired_key", "retained_key_retired", "present_new_receipt",
     "no_absence_receipt_revision", "retired_but_attested"],
)
def test_pure_deletion_refusals(case):
    kw = dict(DEATH_KW)
    authority_ = death_authority()
    if case == "undeclared_loss":
        kw["retired_geometry_keys"] = {"vertex": [11]}
    elif case == "unknown_retired_key":
        kw["retired_geometry_keys"] = {"vertex": [11, 99], "member": [40]}
    elif case == "retained_key_retired":
        kw["retired_geometry_keys"] = {"vertex": [10, 11], "member": [40]}
    elif case == "present_new_receipt":
        authority_ = death_authority(
            new_receipts=state_receipts(2, {"A": ("a0", "present"), "B": ("b1", "present")}))
    elif case == "no_absence_receipt_revision":
        authority_ = death_authority(
            new_receipts=state_receipts(3, {"A": ("a0", "present"), "B": ("gone", "absent")}))
    elif case == "retired_but_attested":
        authority_ = death_authority(unchanged_geometry_keys={
            "vertex": frozenset({10, 11}), "face": frozenset({30})})
    with pytest.raises(MeshError):
        assemble_regional_replacement(
            two_part_mesh(), empty_replacement(), cross_revision=authority_, **kw)


def test_empty_replacement_is_refused_without_a_removed_owner():
    with pytest.raises(MeshError, match="no elements"):
        assemble_regional_replacement(
            two_part_mesh(), empty_replacement(1), quality=LIMITS, node_ids={}, element_ids={},
            affected_element_ids=[3, 4])
    # changed (regenerated) owners must come back with elements
    changed = CrossRevisionAuthority(
        old_receipts=receipts(), new_receipts=receipts(revision=2, digests={"A": "a0", "B": "b1"}),
        changed_owners=frozenset({"B"}),
        unchanged_geometry_keys={"vertex": frozenset({10}), "face": frozenset({30})})
    with pytest.raises(MeshError):
        assemble_regional_replacement(
            two_part_mesh(), empty_replacement(), quality=LIMITS, node_ids={}, element_ids={},
            owner_of_element=OWNERS, cross_revision=changed, retired_geometry_keys={"vertex": [11]})


def far_chain() -> Mesh:
    r = Mesh(geometry_model_id="model-a", geometry_revision=2)
    r.nodes = {1: xyz(0, 50, 0), 2: xyz(1, 50, 0), 3: xyz(2, 50, 0)}
    r.beams = {1: (1, 2), 2: (2, 3)}
    r.nodes_of_member = {41: [1, 2, 3]}
    r.elements_of_member = {41: [1, 2]}
    r.node_of_vertex = {12: 3}
    return r


def birth_authority(**over):
    base = dict(
        old_receipts=state_receipts(1, {"A": ("a0", "present"), "B": ("b0", "present"),
                                        "C": ("none", "absent")}),
        new_receipts=state_receipts(2, {"A": ("a0", "present"), "B": ("b0", "present"),
                                        "C": ("c0", "present")}),
        changed_owners=frozenset(), added_owners=frozenset({"C"}),
        replacement_element_owner={1: "C", 2: "C"},
        unchanged_geometry_keys={"vertex": frozenset({10, 11}), "face": frozenset({30}),
                                 "member": frozenset({40})},
    )
    base.update(over)
    return CrossRevisionAuthority(**base)


BIRTH_KW = dict(quality=LIMITS, node_ids={1: 101, 2: 102, 3: 103}, element_ids={1: 201, 2: 202},
                owner_of_element=OWNERS)


def test_pure_addition_retires_nothing_and_uses_fresh_numbers():
    prev = two_part_mesh()
    res = assemble_regional_replacement(prev, far_chain(), cross_revision=birth_authority(), **BIRTH_KW)
    m = res.mesh
    assert res.retired_element_ids == () and res.retired_node_ids == ()
    assert res.retained_element_ids == tuple(sorted(OWNERS)) and res.interface_node_ids == ()
    assert set(prev.nodes) <= set(m.nodes) and set(prev.beams) <= set(m.beams)
    assert m.beams[201] == (101, 102) and m.node_of_vertex == {10: 1, 11: 13, 12: 103}
    assert m.nodes_of_member == {40: [11, 12, 13], 41: [101, 102, 103]}
    assert res.dropped_geometry_keys == {}


@pytest.mark.parametrize(
    "case",
    ["owner_exists", "added_and_changed", "added_and_removed", "present_old_receipt",
     "absent_new_receipt", "no_owner_map", "reused_key", "coincident_node", "uncovered_owner"],
)
def test_pure_addition_refusals(case):
    kw = dict(BIRTH_KW)
    replacement = far_chain()
    if case == "owner_exists":      # an existing owner cannot be born again
        kw["cross_revision"] = birth_authority(
            added_owners=frozenset({"B"}), replacement_element_owner={1: "B", 2: "B"})
    elif case == "added_and_changed":
        with pytest.raises(MeshError):
            birth_authority(changed_owners=frozenset({"C"}))
        return
    elif case == "added_and_removed":
        with pytest.raises(MeshError):
            birth_authority(removed_owners=frozenset({"C"}))
        return
    elif case == "present_old_receipt":
        kw["cross_revision"] = birth_authority(old_receipts=state_receipts(
            1, {"A": ("a0", "present"), "B": ("b0", "present"), "C": ("c0", "present")}))
    elif case == "absent_new_receipt":
        kw["cross_revision"] = birth_authority(new_receipts=state_receipts(
            2, {"A": ("a0", "present"), "B": ("b0", "present"), "C": ("none", "absent")}))
    elif case == "no_owner_map":
        with pytest.raises(MeshError):
            birth_authority(replacement_element_owner=None)
        return
    elif case == "uncovered_owner":
        kw["cross_revision"] = birth_authority(replacement_element_owner={1: "C"})
    elif case == "reused_key":   # the new owner claims a member the old mesh already holds
        replacement.nodes_of_member = {40: [1, 2, 3]}
        replacement.elements_of_member = {40: [1, 2]}
    elif case == "coincident_node":  # sits exactly on retained node 11
        replacement.nodes[1] = xyz(10, 0, 0)
    kw.setdefault("cross_revision", birth_authority())
    with pytest.raises(MeshError):
        assemble_regional_replacement(two_part_mesh(), replacement, **kw)


def test_simultaneous_death_and_birth_uses_disjoint_keys_and_refuses_overlap():
    authority_ = CrossRevisionAuthority(
        old_receipts=state_receipts(1, {"A": ("a0", "present"), "B": ("b0", "present"),
                                        "C": ("none", "absent")}),
        new_receipts=state_receipts(2, {"A": ("a0", "present"), "B": ("gone", "absent"),
                                        "C": ("c0", "present")}),
        changed_owners=frozenset(), removed_owners=frozenset({"B"}), added_owners=frozenset({"C"}),
        replacement_element_owner={1: "C", 2: "C"},
        unchanged_geometry_keys={"vertex": frozenset({10}), "face": frozenset({30})},
    )
    kw = dict(BIRTH_KW, retired_geometry_keys={"vertex": [11], "member": [40]})
    res = assemble_regional_replacement(two_part_mesh(), far_chain(), cross_revision=authority_, **kw)
    assert res.retired_element_ids == (3, 4) and res.new_element_ids == (201, 202)
    assert set(res.mesh.nodes) == {1, 2, 3, 4, 5, 6, 7, 101, 102, 103}
    # the new owner takes over a key that the removed owner retires: lineage is ambiguous
    chain = far_chain()
    chain.nodes_of_member, chain.elements_of_member = {40: [1, 2, 3]}, {40: [1, 2]}
    with pytest.raises(MeshError):
        assemble_regional_replacement(two_part_mesh(), chain, cross_revision=authority_, **kw)
    # ...or lands exactly on the vacated nodes
    chain = far_chain()
    chain.nodes[1] = xyz(10, 0, 0)
    with pytest.raises(MeshError):
        assemble_regional_replacement(two_part_mesh(), chain, cross_revision=authority_,
                                      **dict(kw, node_ids={1: 11, 2: 102, 3: 103}))


def test_removed_owner_cannot_own_replacement_elements():
    with pytest.raises(MeshError):
        death_authority(replacement_element_owner={1: "B"})
