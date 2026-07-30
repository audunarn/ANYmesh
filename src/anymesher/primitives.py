"""Meshes for shapes that need no geometry model.

A rectangular stiffened panel, a flat plate, a straight beam.  These are the
cases where building a BRep and seeding it would be more machinery than the
problem needs, and where the same shape gets meshed thousands of times in a
parameter study.

What they produce is the same neutral :class:`~anymesher.mesh.Mesh` the mapped
mesher produces, with the same association fields filled in against **synthetic**
entity IDs -- there is no geometry model behind them, but a consumer resolving
"the nodes on the x=0 edge" uses one code path either way.

Node and element numbering is a contract, not an implementation detail.  A
consumer records results per node and element ID, so a renumbering that is
mathematically irrelevant still invalidates stored baselines:

* shell nodes from 1, in ``i`` (along the length) then ``j`` (across) order;
  for 8-node shells all corner nodes first, then the mid-side nodes along the
  length, then those across it,
* shell elements from 1, same traversal,
* beam nodes from 10000, ``10000 + stiffener * (divisions + 1) + station``,
* beam elements from 20000, ``20000 + stiffener * divisions + span``,
* couplings from 30000.

These offsets are why a stiffener's beam nodes never collide with plate nodes,
and they are asserted by test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .coupling import build_structured_shell_grid, locate_shell_element_at_xy
from .errors import MeshError
from .mesh import Coupling, Mesh

__all__ = [
    "PANEL_EDGE_IDS",
    "PANEL_FACE_ID",
    "STIFFENER_EDGE_ID_BASE",
    "PanelMeshConfig",
    "StiffenedPanel",
    "StiffenerCrossSection",
    "beam_mesh",
    "panel_edge_nodes",
    "simple_panel_mesh",
    "stiffened_panel_mesh",
]

# Synthetic entity IDs.  A primitive has no geometry model, but its mesh still
# carries an association so downstream code needs only one lookup path.
PANEL_FACE_ID = 1
PANEL_EDGE_IDS: Mapping[str, int] = {"x0": 1, "xL": 2, "y0": 3, "yW": 4}
STIFFENER_EDGE_ID_BASE = 10

_SHELL_NODE_BASE = 1
_SHELL_ELEMENT_BASE = 1
_BEAM_NODE_BASE = 10000
_BEAM_ELEMENT_BASE = 20000
_COUPLING_BASE = 30000


@dataclass
class PanelMeshConfig:
    """How finely to mesh a panel, and how to attach its stiffeners.

    Materials and support conditions are deliberately absent.  A material name is
    not a meshing decision, and interpreting a support description such as
    ``"Integrated"`` is structural convention rather than mesh topology -- both
    belong to the consumer that builds the analysis model.
    """

    shell_num_divisions_x: int = 4
    shell_num_divisions_y: int = 4
    beam_num_divisions: int = 1

    use_coupling_elements: bool = True
    tolerance: float = 1.0e-6

    use_8node_shells: bool = False
    align_mesh_to_stiffeners: bool = False


@dataclass
class StiffenedPanel:
    """Geometry of a rectangular stiffened panel.

    Stresses, pressures, materials and support conditions are not here: they
    describe the analysis, not the shape being meshed.
    """

    length: float = 0.0
    width: float = 0.0

    plate_thickness: float = 0.0

    stiffener_type: str = "T-bar"
    stiffener_spacing: float = 0.0
    stiffener_height: float = 0.0
    stiffener_web_thickness: float = 0.0
    stiffener_flange_width: float = 0.0
    stiffener_flange_thickness: float = 0.0
    num_stiffeners: int = 1

    @property
    def effective_stiffener_spacing(self) -> float:
        """Spacing as meshed, filling in an even division when unset."""

        count = max(int(self.num_stiffeners), 1)
        if self.stiffener_spacing > 0.0:
            return float(self.stiffener_spacing)
        return float(self.width) / (count + 1)


@dataclass
class StiffenerCrossSection:
    """Cross-section properties for a line stiffener.

    Pure geometry: areas and second moments of the rectangles a rolled profile is
    idealized as.  What a solver does with them is its own business, which is why
    this lives with the mesher that knows the profile dimensions.
    """

    area: float
    Iy: float
    Iz: float
    J: float
    shear_factor_y: float = 5.0 / 6.0
    shear_factor_z: float = 5.0 / 6.0
    c_y: float = 0.0
    c_z: float = 0.0
    torsion_modulus: float = 0.0

    @staticmethod
    def _composite_rectangles(
        rectangles: List[Tuple[float, float, float, float]],
    ) -> Tuple[float, float, float, float, float]:
        """
        Return A, Iy, Iz, c_y, c_z for rectangles described by
        y, z, width_y, height_z, with c_y/c_z the extreme fiber distances
        from the centroid.
        """
        areas = np.asarray([width * height for _y, _z, width, height in rectangles], dtype=float)
        total_area = max(float(np.sum(areas)), 1.0e-30)
        y_centroid = float(
            np.sum([area * rect[0] for area, rect in zip(areas, rectangles)]) / total_area
        )
        z_centroid = float(
            np.sum([area * rect[1] for area, rect in zip(areas, rectangles)]) / total_area
        )

        Iy = 0.0
        Iz = 0.0
        c_y = 0.0
        c_z = 0.0
        for area, (y, z, width, height) in zip(areas, rectangles):
            Iy += width * height**3 / 12.0 + area * (z - z_centroid) ** 2
            Iz += height * width**3 / 12.0 + area * (y - y_centroid) ** 2
            c_y = max(c_y, abs(y - y_centroid) + width / 2.0)
            c_z = max(c_z, abs(z - z_centroid) + height / 2.0)
        return total_area, Iy, Iz, c_y, c_z

    @classmethod
    def from_geometry(
        cls, stiffener_type: str, hw: float, tw: float, b: float, tf: float
    ) -> "StiffenerCrossSection":
        # Open thin-walled torsion: J = sum(l*t^3)/3 and tau_max = T*t_max/J,
        # so the torsional section modulus is Wt = J / t_max.
        if stiffener_type == "T-bar":
            A, Iy, Iz, c_y, c_z = cls._composite_rectangles(
                [
                    (0.0, hw / 2.0, tw, hw),
                    (0.0, hw + tf / 2.0, b, tf),
                ]
            )
            J = (hw * tw**3 + b * tf**3) / 3.0
            t_max = max(tw, tf)
        elif stiffener_type in ("L-bulb", "Angle"):
            A, Iy, Iz, c_y, c_z = cls._composite_rectangles(
                [
                    (tw / 2.0, hw / 2.0, tw, hw),
                    (b / 2.0, hw + tf / 2.0, b, tf),
                ]
            )
            J = (hw * tw**3 + b * tf**3) / 3.0
            t_max = max(tw, tf)
        elif stiffener_type == "Flatbar":
            A, Iy, Iz, c_y, c_z = cls._composite_rectangles([(0.0, 0.0, b, tf)])
            J = b * tf**3 / 3.0
            t_max = min(b, tf)
        else:
            A, Iy, Iz, c_y, c_z = cls._composite_rectangles([(0.0, hw / 2.0, tw, hw)])
            J = hw * tw**3 / 3.0
            t_max = tw
        torsion_modulus = J / max(t_max, 1.0e-30)
        return cls(area=A, Iy=Iy, Iz=Iz, J=J, c_y=c_y, c_z=c_z, torsion_modulus=torsion_modulus)

    @classmethod
    def from_panel(cls, panel: StiffenedPanel) -> "StiffenerCrossSection":
        """The section of a panel's stiffeners."""

        return cls.from_geometry(
            panel.stiffener_type,
            panel.stiffener_height,
            panel.stiffener_web_thickness,
            panel.stiffener_flange_width,
            panel.stiffener_flange_thickness,
        )

    def as_dict(self) -> Dict[str, float]:
        """Section properties as a mapping, with the local-z web direction.

        A panel lies in the global ``z = 0`` plane with stiffener webs pointing in
        ``+Z``, which pins the beam local frame so ``Iy`` and ``Iz`` keep the
        meaning they have here.
        """

        return {
            "area": self.area,
            "Iy": self.Iy,
            "Iz": self.Iz,
            "J": self.J,
            "shear_factor_y": self.shear_factor_y,
            "shear_factor_z": self.shear_factor_z,
            "c_y": self.c_y,
            "c_z": self.c_z,
            "torsion_modulus": self.torsion_modulus,
            "orientation": (0.0, 0.0, 1.0),
        }


def _safe_divisions(value: int) -> int:
    return max(int(value), 1)


def _transverse_grid(panel: StiffenedPanel, config: PanelMeshConfig, ny: int) -> Tuple[List[float], int]:
    """Node positions across the panel, and the division count they imply.

    With ``align_mesh_to_stiffeners`` the grid is forced to contain a line at
    every stiffener, and the remaining divisions are shared out between the
    resulting strips in proportion to their width.  Without it the spacing is
    uniform and a stiffener generally lands inside an element -- which the
    interpolated coupling handles, so alignment is an option rather than a
    requirement.
    """

    W = panel.width
    if not config.align_mesh_to_stiffeners:
        return [j * W / ny for j in range(ny + 1)], ny

    num_stiffeners = max(int(panel.num_stiffeners), 1)
    spacing = panel.effective_stiffener_spacing
    y_stiffeners = [(s + 1) * spacing for s in range(num_stiffeners)]
    key_ys = [0.0] + y_stiffeners + [W]
    n_segments = len(key_ys) - 1

    if ny <= n_segments:
        segment_divs = [1] * n_segments
        ny = n_segments
    else:
        segment_divs = [1] * n_segments
        remaining = ny - n_segments
        widths = np.array([key_ys[k + 1] - key_ys[k] for k in range(n_segments)], dtype=float)
        shares = widths / np.sum(widths) * remaining
        floored = np.floor(shares).astype(int)
        segment_divs = [divs + f for divs, f in zip(segment_divs, floored)]
        remaining -= np.sum(floored)
        fractional = shares - floored
        for idx in np.argsort(fractional)[::-1][:remaining]:
            segment_divs[idx] += 1
        ny = sum(segment_divs)

    y_grid: List[float] = []
    for k in range(n_segments):
        y0_seg = key_ys[k]
        y1_seg = key_ys[k + 1]
        divs = segment_divs[k]
        for d in range(divs):
            y_grid.append(y0_seg + d * (y1_seg - y0_seg) / divs)
    y_grid.append(W)
    return y_grid, ny


def _shell_mesh(
    panel: StiffenedPanel,
    config: PanelMeshConfig,
) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[int, Tuple[List[int], float]]]:
    """Generate the 4-node or 8-node quadrilateral mesh for the plating."""

    nodes: Dict[int, Tuple[float, float, float]] = {}
    elements: Dict[int, Tuple[List[int], float]] = {}
    nx = _safe_divisions(config.shell_num_divisions_x)
    ny = _safe_divisions(config.shell_num_divisions_y)
    L = panel.length
    t = panel.plate_thickness
    y_grid, ny = _transverse_grid(panel, config, ny)

    if config.use_8node_shells:
        node_id = _SHELL_NODE_BASE
        corner_nodes: Dict[Tuple[int, int], int] = {}
        for i in range(nx + 1):
            for j in range(ny + 1):
                corner_nodes[(i, j)] = node_id
                nodes[node_id] = (i * L / nx, y_grid[j], 0.0)
                node_id += 1
        h_mid_nodes: Dict[Tuple[int, int], int] = {}
        for i in range(nx):
            for j in range(ny + 1):
                h_mid_nodes[(i, j)] = node_id
                nodes[node_id] = ((i + 0.5) * L / nx, y_grid[j], 0.0)
                node_id += 1
        v_mid_nodes: Dict[Tuple[int, int], int] = {}
        for i in range(nx + 1):
            for j in range(ny):
                v_mid_nodes[(i, j)] = node_id
                nodes[node_id] = (i * L / nx, 0.5 * (y_grid[j] + y_grid[j + 1]), 0.0)
                node_id += 1
        elem_id = _SHELL_ELEMENT_BASE
        for i in range(nx):
            for j in range(ny):
                elements[elem_id] = (
                    [
                        corner_nodes[(i, j)],
                        corner_nodes[(i + 1, j)],
                        corner_nodes[(i + 1, j + 1)],
                        corner_nodes[(i, j + 1)],
                        h_mid_nodes[(i, j)],
                        v_mid_nodes[(i + 1, j)],
                        h_mid_nodes[(i, j + 1)],
                        v_mid_nodes[(i, j)],
                    ],
                    t,
                )
                elem_id += 1
    else:
        node_id = _SHELL_NODE_BASE
        node_grid: Dict[Tuple[int, int], int] = {}
        for i in range(nx + 1):
            for j in range(ny + 1):
                node_grid[(i, j)] = node_id
                nodes[node_id] = (i * L / nx, y_grid[j], 0.0)
                node_id += 1
        elem_id = _SHELL_ELEMENT_BASE
        for i in range(nx):
            for j in range(ny):
                elements[elem_id] = (
                    [
                        node_grid[(i, j)],
                        node_grid[(i + 1, j)],
                        node_grid[(i + 1, j + 1)],
                        node_grid[(i, j + 1)],
                    ],
                    t,
                )
                elem_id += 1
    return nodes, elements


def _beam_nodes_and_elements(
    panel: StiffenedPanel,
    config: PanelMeshConfig,
) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[int, Tuple[int, List[int]]]]:
    """Separate beam nodes and elements for the longitudinal stiffeners.

    The beam nodes stand at the stiffener height rather than in the plate, so the
    stiffener's neutral axis is where the profile puts it.  A stiffener sharing
    plate nodes would be a materially different structure.
    """

    nodes: Dict[int, Tuple[float, float, float]] = {}
    elements: Dict[int, Tuple[int, List[int]]] = {}
    n_div = _safe_divisions(config.beam_num_divisions)
    num_stiffeners = max(int(panel.num_stiffeners), 1)
    spacing = panel.effective_stiffener_spacing

    for s in range(num_stiffeners):
        y_pos = (s + 1) * spacing
        for i in range(n_div + 1):
            node_id = _BEAM_NODE_BASE + s * (n_div + 1) + i
            nodes[node_id] = (i * panel.length / n_div, y_pos, panel.stiffener_height)
        for i in range(n_div):
            n1 = _BEAM_NODE_BASE + s * (n_div + 1) + i
            elem_id = _BEAM_ELEMENT_BASE + s * n_div + i
            elements[elem_id] = (s, [n1, n1 + 1])
    return nodes, elements


def _edge_node_sets(
    panel: StiffenedPanel,
    nodes: Mapping[int, Sequence[float]],
    tolerance: float,
) -> Dict[str, List[int]]:
    """All plate nodes lying on the four rectangular panel edges."""

    L = panel.length
    W = panel.width
    tol = max(float(tolerance), 1.0e-9, 1.0e-8 * max(abs(L), abs(W), 1.0))
    edge_nodes: Dict[str, List[int]] = {"x0": [], "xL": [], "y0": [], "yW": []}
    for node_id, coords in nodes.items():
        x, y, _z = coords
        if abs(x) <= tol:
            edge_nodes["x0"].append(node_id)
        if abs(x - L) <= tol:
            edge_nodes["xL"].append(node_id)
        if abs(y) <= tol:
            edge_nodes["y0"].append(node_id)
        if abs(y - W) <= tol:
            edge_nodes["yW"].append(node_id)
    edge_nodes["x0"].sort(key=lambda nid: nodes[nid][1])
    edge_nodes["xL"].sort(key=lambda nid: nodes[nid][1])
    edge_nodes["y0"].sort(key=lambda nid: nodes[nid][0])
    edge_nodes["yW"].sort(key=lambda nid: nodes[nid][0])
    edge_nodes["all"] = sorted(
        set(edge_nodes["x0"] + edge_nodes["xL"] + edge_nodes["y0"] + edge_nodes["yW"])
    )
    return edge_nodes


def panel_edge_nodes(mesh: Mesh) -> Dict[str, List[int]]:
    """Read the four panel edges back out of a primitive mesh.

    Keys are ``"x0"``, ``"xL"``, ``"y0"``, ``"yW"`` and ``"all"``.  Interpreting
    them -- deciding that "Integrated" means all four edges held in three
    translations -- is the consumer's job.
    """

    edges = {
        name: list(mesh.nodes_of_edge.get(edge_id, ()))
        for name, edge_id in PANEL_EDGE_IDS.items()
    }
    edges["all"] = sorted({node for nodes in edges.values() for node in nodes})
    return edges


def _new_mesh(order: str) -> Mesh:
    return Mesh(order=order)


def _install_plating(
    mesh: Mesh,
    panel: StiffenedPanel,
    shell_nodes: Mapping[int, Tuple[float, float, float]],
    shell_elements: Mapping[int, Tuple[List[int], float]],
    tolerance: float,
) -> None:
    for node_id, coords in shell_nodes.items():
        mesh.nodes[node_id] = np.asarray(coords, dtype=float)
    for elem_id, (node_ids, _thickness) in shell_elements.items():
        mesh.quads[elem_id] = tuple(int(node) for node in node_ids)
    mesh.elements_of_face[PANEL_FACE_ID] = sorted(shell_elements)
    mesh.thickness_of_face[PANEL_FACE_ID] = float(panel.plate_thickness)

    edges = _edge_node_sets(panel, shell_nodes, tolerance)
    for name, edge_id in PANEL_EDGE_IDS.items():
        mesh.nodes_of_edge[edge_id] = list(edges[name])


def stiffened_panel_mesh(
    panel: StiffenedPanel, config: Optional[PanelMeshConfig] = None
) -> Mesh:
    """Mesh a rectangular stiffened panel.

    The plating is a structured quadrilateral grid; each stiffener is a line of
    beam elements at the stiffener height, coupled back to the plating through
    interpolated coupling records.
    """

    config = config or PanelMeshConfig()
    order = "quadratic" if config.use_8node_shells else "linear"
    mesh = _new_mesh(order)

    shell_nodes, shell_elements = _shell_mesh(panel, config)
    beam_nodes, beam_elements = _beam_nodes_and_elements(panel, config)

    _install_plating(mesh, panel, shell_nodes, shell_elements, config.tolerance)

    for node_id, coords in beam_nodes.items():
        mesh.nodes[node_id] = np.asarray(coords, dtype=float)
    per_stiffener: Dict[int, List[int]] = {}
    for elem_id, (stiffener, node_ids) in beam_elements.items():
        mesh.beams[elem_id] = tuple(int(node) for node in node_ids)
        per_stiffener.setdefault(stiffener, []).append(elem_id)
    for stiffener, element_ids in per_stiffener.items():
        edge_id = STIFFENER_EDGE_ID_BASE + stiffener
        mesh.elements_of_edge[edge_id] = sorted(element_ids)
        mesh.offset_nodes_of_edge[edge_id] = sorted(
            {node for eid in element_ids for node in mesh.beams[eid]}
        )

    if config.use_coupling_elements:
        grid = build_structured_shell_grid(shell_nodes, shell_elements, config.tolerance)
        elem_id = _COUPLING_BASE
        for beam_node_id in sorted(beam_nodes):
            beam_coords = np.asarray(beam_nodes[beam_node_id], dtype=float)
            located = locate_shell_element_at_xy(
                beam_coords[0],
                beam_coords[1],
                shell_nodes,
                shell_elements,
                config.tolerance,
                grid,
            )
            if located is None:
                continue
            shell_node_ids, weights, shell_point = located
            mesh.couplings[elem_id] = Coupling(
                beam_node=int(beam_node_id),
                plate_nodes=tuple(int(node) for node in shell_node_ids),
                weights=tuple(float(weight) for weight in weights),
                eccentricity=tuple(float(value) for value in beam_coords - shell_point),
            )
            elem_id += 1
        if len(mesh.couplings) != len(beam_nodes):
            # A beam node that landed outside the plating means the stiffener
            # positions and the panel dimensions disagree.  Meshing on regardless
            # would produce a stiffener attached to nothing.
            raise MeshError(
                f"only generated {len(mesh.couplings)} beam-shell couplings for "
                f"{len(beam_nodes)} beam nodes. Check stiffener positions and panel dimensions."
            )

    return mesh


def simple_panel_mesh(
    length: float,
    width: float,
    thickness: float,
    num_divisions_x: int = 4,
    num_divisions_y: int = 4,
    use_8node_elements: bool = False,
) -> Mesh:
    """Mesh a flat rectangular plate with no stiffeners."""

    config = PanelMeshConfig(
        shell_num_divisions_x=num_divisions_x,
        shell_num_divisions_y=num_divisions_y,
        use_8node_shells=use_8node_elements,
    )
    panel = StiffenedPanel(length=length, width=width, plate_thickness=thickness)
    shell_nodes, shell_elements = _shell_mesh(panel, config)
    mesh = _new_mesh("quadratic" if use_8node_elements else "linear")
    _install_plating(mesh, panel, shell_nodes, shell_elements, config.tolerance)
    return mesh


def beam_mesh(length: float, num_divisions: int = 10) -> Mesh:
    """Mesh a straight beam along the global x axis.

    The two ends are recorded as vertices 1 and 2, so a consumer can restrain an
    end without searching for it by coordinate.
    """

    n_div = _safe_divisions(num_divisions)
    mesh = _new_mesh("linear")
    for i in range(n_div + 1):
        mesh.nodes[i + 1] = np.array([i * length / n_div, 0.0, 0.0], dtype=float)
    for i in range(n_div):
        mesh.beams[i + 1] = (i + 1, i + 2)
    mesh.node_of_vertex[1] = 1
    mesh.node_of_vertex[2] = n_div + 1
    mesh.nodes_of_edge[1] = list(range(1, n_div + 2))
    mesh.elements_of_edge[1] = list(range(1, n_div + 1))
    return mesh
