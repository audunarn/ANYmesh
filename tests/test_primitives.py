"""Primitives: panels, plates and beams meshed without a geometry model.

Node and element numbering is asserted explicitly rather than left implied.  A
consumer stores results per ID, so a renumbering that is mathematically
irrelevant still invalidates its baselines -- which makes the numbering a
contract and not an implementation detail.
"""

from __future__ import annotations

import numpy as np
import pytest

from anymesher import (
    PANEL_EDGE_IDS,
    PANEL_FACE_ID,
    STIFFENER_EDGE_ID_BASE,
    EntityRef,
    MeshError,
    PanelMeshConfig,
    StiffenedPanel,
    StiffenerCrossSection,
    beam_mesh,
    panel_edge_nodes,
    simple_panel_mesh,
    stiffened_panel_mesh,
)


def _panel(**overrides) -> StiffenedPanel:
    properties = {
        "length": 4.0,
        "width": 3.0,
        "plate_thickness": 0.012,
        "stiffener_type": "T-bar",
        "stiffener_spacing": 1.5,
        "stiffener_height": 0.30,
        "stiffener_web_thickness": 0.010,
        "stiffener_flange_width": 0.150,
        "stiffener_flange_thickness": 0.015,
        "num_stiffeners": 1,
    }
    properties.update(overrides)
    return StiffenedPanel(**properties)


def test_plate_node_and_element_numbering_is_a_contract() -> None:
    mesh = simple_panel_mesh(2.0, 1.0, 0.01, num_divisions_x=2, num_divisions_y=2)

    # Nodes from 1, in i-then-j order; elements from 1, same traversal.
    assert sorted(mesh.nodes) == list(range(1, 10))
    assert sorted(mesh.quads) == [1, 2, 3, 4]
    assert mesh.nodes[1] == pytest.approx([0.0, 0.0, 0.0])
    assert mesh.nodes[2] == pytest.approx([0.0, 0.5, 0.0])
    assert mesh.nodes[4] == pytest.approx([1.0, 0.0, 0.0])
    assert mesh.quads[1] == (1, 4, 5, 2)
    assert mesh.thickness_of_face[PANEL_FACE_ID] == pytest.approx(0.01)
    assert mesh.elements_of_face[PANEL_FACE_ID] == [1, 2, 3, 4]


def test_stiffener_ids_are_offset_so_they_cannot_collide_with_plating() -> None:
    mesh = stiffened_panel_mesh(
        _panel(num_stiffeners=2, stiffener_spacing=1.0),
        PanelMeshConfig(shell_num_divisions_x=4, shell_num_divisions_y=4, beam_num_divisions=2),
    )

    plate_nodes = [n for n in mesh.nodes if n < 10000]
    beam_nodes = sorted(n for n in mesh.nodes if n >= 10000)
    assert plate_nodes and beam_nodes
    # 10000 + stiffener * (divisions + 1) + station
    assert beam_nodes == [10000, 10001, 10002, 10003, 10004, 10005]
    # 20000 + stiffener * divisions + span
    assert sorted(mesh.beams) == [20000, 20001, 20002, 20003]
    assert min(mesh.couplings) == 30000
    assert not set(mesh.quads) & set(mesh.beams) & set(mesh.couplings)


def test_beam_nodes_stand_at_the_stiffener_height() -> None:
    panel = _panel(stiffener_height=0.30)
    mesh = stiffened_panel_mesh(panel, PanelMeshConfig(beam_num_divisions=2))

    for node_id in (n for n in mesh.nodes if n >= 10000):
        assert mesh.nodes[node_id][2] == pytest.approx(0.30)
    # ... and the plating stays in the z = 0 plane, so they are genuinely offset.
    for node_id in (n for n in mesh.nodes if n < 10000):
        assert mesh.nodes[node_id][2] == pytest.approx(0.0)


def test_couplings_interpolate_the_shell_under_every_beam_node() -> None:
    mesh = stiffened_panel_mesh(
        _panel(),
        # An odd division count across the width puts the stiffener inside an
        # element rather than on a mesh line, which is the case interpolation
        # exists for.
        PanelMeshConfig(shell_num_divisions_x=3, shell_num_divisions_y=3, beam_num_divisions=2),
    )

    beam_nodes = {n for n in mesh.nodes if n >= 10000}
    assert len(mesh.couplings) == len(beam_nodes)
    assert {c.beam_node for c in mesh.couplings.values()} == beam_nodes

    for coupling in mesh.couplings.values():
        assert not coupling.is_node_to_node
        assert len(coupling.plate_nodes) == 4
        assert sum(coupling.weights) == pytest.approx(1.0)
        # The eccentricity must run from the interpolated plate point to the beam
        # node, so reconstructing it reproduces the beam node exactly.
        interpolated = sum(
            weight * mesh.nodes[node]
            for weight, node in zip(coupling.weights, coupling.plate_nodes)
        )
        assert interpolated + np.asarray(coupling.eccentricity) == pytest.approx(
            mesh.nodes[coupling.beam_node]
        )


def test_aligned_mesh_puts_beam_nodes_on_plate_node_lines() -> None:
    mesh = stiffened_panel_mesh(
        _panel(num_stiffeners=2, stiffener_spacing=1.0),
        PanelMeshConfig(shell_num_divisions_x=4, shell_num_divisions_y=6, align_mesh_to_stiffeners=True),
    )

    transverse = sorted({float(mesh.nodes[n][1]) for n in mesh.nodes if n < 10000})
    for stiffener in (1.0, 2.0):
        assert min(abs(value - stiffener) for value in transverse) < 1.0e-9


def test_a_stiffener_outside_the_plating_is_refused() -> None:
    # Spacing wider than the panel puts the stiffener off the plate, so there is
    # nothing for its coupling to attach to.
    with pytest.raises(MeshError, match="beam-shell couplings"):
        stiffened_panel_mesh(_panel(num_stiffeners=1, stiffener_spacing=9.0))


def test_couplings_can_be_switched_off() -> None:
    mesh = stiffened_panel_mesh(_panel(), PanelMeshConfig(use_coupling_elements=False))

    assert mesh.beams
    assert mesh.couplings == {}


def test_panel_edges_are_addressable_as_entities() -> None:
    mesh = stiffened_panel_mesh(
        _panel(), PanelMeshConfig(shell_num_divisions_x=4, shell_num_divisions_y=3)
    )
    edges = panel_edge_nodes(mesh)

    assert set(edges) == {"x0", "xL", "y0", "yW", "all"}
    assert len(edges["x0"]) == 4  # ny + 1
    assert len(edges["y0"]) == 5  # nx + 1
    # The same nodes come back through the entity association, so a consumer
    # needs only one lookup path whether or not there is geometry behind it.
    assert mesh.nodes_on(EntityRef("edge", PANEL_EDGE_IDS["x0"])) == edges["x0"]
    assert mesh.nodes_on(EntityRef("face", PANEL_FACE_ID))

    for node_id in edges["x0"]:
        assert mesh.nodes[node_id][0] == pytest.approx(0.0)
    for node_id in edges["yW"]:
        assert mesh.nodes[node_id][1] == pytest.approx(3.0)
    # Sorted along the edge, not by node ID.
    assert edges["x0"] == sorted(edges["x0"], key=lambda n: mesh.nodes[n][1])


def test_stiffener_elements_are_grouped_per_stiffener() -> None:
    mesh = stiffened_panel_mesh(
        _panel(num_stiffeners=3, stiffener_spacing=0.75),
        PanelMeshConfig(beam_num_divisions=2),
    )

    for stiffener in range(3):
        edge_id = STIFFENER_EDGE_ID_BASE + stiffener
        assert len(mesh.elements_of_edge[edge_id]) == 2
        assert len(mesh.offset_nodes_of_edge[edge_id]) == 3
        # A restraint on a stiffener line must reach its offset nodes.
        assert set(mesh.offset_nodes_of_edge[edge_id]) <= set(
            mesh.constrained_nodes_on(EntityRef("edge", edge_id))
        )


def test_eight_node_shells_add_mid_side_nodes_without_changing_element_counts() -> None:
    linear = simple_panel_mesh(2.0, 1.0, 0.01, 3, 2, use_8node_elements=False)
    quadratic = simple_panel_mesh(2.0, 1.0, 0.01, 3, 2, use_8node_elements=True)

    assert len(linear.quads) == len(quadratic.quads) == 6
    assert quadratic.num_nodes > linear.num_nodes
    assert all(len(nodes) == 4 for nodes in linear.quads.values())
    assert all(len(nodes) == 8 for nodes in quadratic.quads.values())
    assert quadratic.is_quadratic and not linear.is_quadratic
    # Corners come first, so corner lookups do not care about the order.
    assert quadratic.corners_of(1) == linear.quads[1][:4]


def test_beam_mesh_records_its_ends_as_vertices() -> None:
    mesh = beam_mesh(3.0, num_divisions=3)

    assert sorted(mesh.nodes) == [1, 2, 3, 4]
    assert sorted(mesh.beams) == [1, 2, 3]
    assert mesh.beams[1] == (1, 2)
    assert mesh.nodes[4] == pytest.approx([3.0, 0.0, 0.0])
    assert mesh.node_of_vertex == {1: 1, 2: 4}
    assert mesh.nodes_on(EntityRef("vertex", 2)) == [4]


def test_degenerate_division_counts_are_clamped_rather_than_dividing_by_zero() -> None:
    mesh = simple_panel_mesh(1.0, 1.0, 0.01, num_divisions_x=0, num_divisions_y=-3)

    assert len(mesh.quads) == 1
    assert beam_mesh(1.0, num_divisions=0).beams == {1: (1, 2)}


def test_effective_spacing_divides_the_width_when_unset() -> None:
    assert _panel(stiffener_spacing=0.0, num_stiffeners=3).effective_stiffener_spacing == pytest.approx(0.75)
    assert _panel(stiffener_spacing=1.25).effective_stiffener_spacing == pytest.approx(1.25)


@pytest.mark.parametrize("kind", ["T-bar", "L-bulb", "Angle", "Flatbar", "unknown-profile"])
def test_every_profile_family_gives_a_positive_definite_section(kind: str) -> None:
    section = StiffenerCrossSection.from_geometry(kind, 0.30, 0.010, 0.150, 0.015)

    assert section.area > 0.0
    assert section.Iy > 0.0 and section.Iz > 0.0
    assert section.J > 0.0
    assert section.torsion_modulus > 0.0
    assert section.c_y > 0.0 or kind == "Flatbar"
    assert section.c_z > 0.0
    assert section.as_dict()["orientation"] == (0.0, 0.0, 1.0)


def test_tbar_section_matches_hand_calculation() -> None:
    hw, tw, b, tf = 0.30, 0.010, 0.150, 0.015
    section = StiffenerCrossSection.from_geometry("T-bar", hw, tw, b, tf)

    assert section.area == pytest.approx(hw * tw + b * tf)
    # Open thin-walled torsion, J = sum(l t^3) / 3.
    assert section.J == pytest.approx((hw * tw**3 + b * tf**3) / 3.0)
    assert section.torsion_modulus == pytest.approx(section.J / max(tw, tf))

    web_area, flange_area = hw * tw, b * tf
    centroid = (web_area * hw / 2.0 + flange_area * (hw + tf / 2.0)) / (web_area + flange_area)
    expected_Iy = (
        tw * hw**3 / 12.0
        + web_area * (hw / 2.0 - centroid) ** 2
        + b * tf**3 / 12.0
        + flange_area * (hw + tf / 2.0 - centroid) ** 2
    )
    assert section.Iy == pytest.approx(expected_Iy)


def test_section_from_panel_uses_the_panel_dimensions() -> None:
    panel = _panel()
    assert StiffenerCrossSection.from_panel(panel) == StiffenerCrossSection.from_geometry(
        panel.stiffener_type,
        panel.stiffener_height,
        panel.stiffener_web_thickness,
        panel.stiffener_flange_width,
        panel.stiffener_flange_thickness,
    )
