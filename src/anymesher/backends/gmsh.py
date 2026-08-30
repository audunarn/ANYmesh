"""The gmsh backend.

Unstructured meshing through gmsh, for regions the mapped mesher will not take.
It returns the same neutral :class:`~anymesher.mesh.Mesh`, with the same
association fields, so results stay addressable by geometry entity either way.

Two differences from the mapped mesher are real and are not papered over:

* **No structured grid.** ``grid_of_face`` stays empty, because there is no
  ``(i, j)`` indexing behind an unstructured mesh.  Anything that needs one has
  to use the mapped backend.
* **Triangles happen.** Recombination is asked for, and gmsh delivers quads where
  it can, but an odd number of boundary divisions leaves triangles behind.  They
  land in ``Mesh.tris`` rather than being dropped or refused.

Planar faces only, for now.  Reconstructing a non-planar surface from a boundary
loop means choosing a surface that the boundary does not determine, and the
mapped mesher already meshes those exactly -- so this refuses with a message
pointing there instead of guessing.

gmsh is process-global state, not an object, so every entry point here acquires
it, finalizes it in a ``finally``, and never leaves it initialized on the way out.
"""

from __future__ import annotations

import contextlib
import ctypes
import sys
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import gmsh

from anygeometry.curves import Arc, Spline, Straight
from anygeometry.model import GeometryModel

from ..errors import MeshError
from ..mesh import Mesh

__all__ = ["generate_mesh"]

# gmsh element type codes for the shells this package represents.
_TRI3, _QUAD4, _TRI6, _QUAD8 = 2, 3, 9, 16
_LINE2, _LINE3 = 1, 8

_SHELL_TYPES = {_TRI3: "tris", _QUAD4: "quads", _TRI6: "tris", _QUAD8: "quads"}

# How far out of its own best-fit plane a face may sit, relative to its size.
_PLANARITY_RTOL = 1.0e-6


_NativeEnvironmentSnapshot = Optional[Tuple[bool, str]]
_ERROR_ENVVAR_NOT_FOUND = 203

if sys.platform == "win32":
    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _GET_ENVIRONMENT_VARIABLE = _KERNEL32.GetEnvironmentVariableW
    _GET_ENVIRONMENT_VARIABLE.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
    )
    _GET_ENVIRONMENT_VARIABLE.restype = ctypes.c_uint32
    _SET_ENVIRONMENT_VARIABLE = _KERNEL32.SetEnvironmentVariableW
    _SET_ENVIRONMENT_VARIABLE.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
    )
    _SET_ENVIRONMENT_VARIABLE.restype = ctypes.c_int
else:
    _GET_ENVIRONMENT_VARIABLE = None
    _SET_ENVIRONMENT_VARIABLE = None


def _snapshot_native_environment_variable(
    name: str,
) -> _NativeEnvironmentSnapshot:
    """Read the real Windows process environment without consulting ``os.environ``.

    Gmsh 4.14 on Windows changes the native process ``PATH`` during
    ``initialize()`` without updating Python's cached environment mapping.
    The Win32 API is therefore the only authoritative source for the value we
    must restore.  Other platforms need no boundary repair.
    """

    if _GET_ENVIRONMENT_VARIABLE is None:
        return None
    while True:
        ctypes.set_last_error(0)
        required = int(_GET_ENVIRONMENT_VARIABLE(name, None, 0))
        if required == 0:
            error = ctypes.get_last_error()
            if error == _ERROR_ENVVAR_NOT_FOUND:
                return False, ""
            if error:
                raise ctypes.WinError(error)
            return True, ""
        buffer = ctypes.create_unicode_buffer(required)
        ctypes.set_last_error(0)
        copied = int(_GET_ENVIRONMENT_VARIABLE(name, buffer, required))
        if copied == 0:
            error = ctypes.get_last_error()
            if error == _ERROR_ENVVAR_NOT_FOUND:
                return False, ""
            if error:
                raise ctypes.WinError(error)
            return True, ""
        if copied < required:
            return True, buffer.value
        # The value grew between the size query and the read.  Retry rather
        # than accepting a truncated process-global environment value.


def _restore_native_environment_variable(
    name: str,
    snapshot: _NativeEnvironmentSnapshot,
) -> None:
    """Restore an exact native environment snapshot without touching ``os.environ``."""

    if _SET_ENVIRONMENT_VARIABLE is None or snapshot is None:
        return
    existed, value = snapshot
    ctypes.set_last_error(0)
    if not _SET_ENVIRONMENT_VARIABLE(name, value if existed else None):
        raise ctypes.WinError(ctypes.get_last_error())


@contextlib.contextmanager
def _gmsh_session(verbose: bool = False) -> Iterator[None]:
    """Initialize gmsh for the duration of a block, and always finalize."""

    native_path = _snapshot_native_environment_variable("PATH")
    initialized = False
    try:
        try:
            gmsh.initialize()
            initialized = True
        finally:
            # Restore immediately: callers may launch other tools while a
            # process-global Gmsh session is active.
            _restore_native_environment_variable("PATH", native_path)
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        yield
    finally:
        if initialized:
            try:
                gmsh.finalize()
            finally:
                _restore_native_environment_variable("PATH", native_path)
        else:
            # Initialization exceptions can corrupt the native environment as
            # well.  Repeat restoration in the outer cleanup boundary.
            _restore_native_environment_variable("PATH", native_path)


def _face_planarity_error(geometry: GeometryModel, face_id: int) -> Optional[str]:
    """Return a complaint when a face is too far from planar, else ``None``."""

    points: List[np.ndarray] = []
    face = geometry.faces[face_id]
    for boundary in (face.loop, *getattr(face, "holes", ())):
        for item in boundary:
            points.extend(geometry.sample_edge(item.edge, np.linspace(0.0, 1.0, 5)))
    cloud = np.asarray(points, dtype=float)
    centred = cloud - cloud.mean(axis=0)
    extent = float(np.max(np.linalg.norm(centred, axis=1)))
    if extent <= 0.0:
        return f"face {face_id} is degenerate"
    # The smallest singular value is the out-of-plane spread.
    deviation = float(np.linalg.svd(centred, compute_uv=False)[2])
    if deviation > _PLANARITY_RTOL * extent:
        return (
            f"face {face_id} is not planar (out-of-plane deviation "
            f"{deviation:.3e} over extent {extent:.3e}). The gmsh backend meshes "
            "planar faces; mesh this model with backend='mapped', which meshes "
            "curved patches exactly."
        )
    return None


def _build_geometry(
    geometry: GeometryModel, face_ids: Sequence[int], beam_edge_ids: Sequence[int]
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """Recreate the model in gmsh, returning our-ID to gmsh-tag maps."""

    needed_vertices: List[int] = []
    needed_edges: List[int] = []
    for face_id in face_ids:
        face = geometry.faces[face_id]
        for boundary in (face.loop, *getattr(face, "holes", ())):
            for item in boundary:
                needed_edges.append(item.edge)
    needed_edges.extend(beam_edge_ids)
    needed_edges = list(dict.fromkeys(needed_edges))
    for edge_id in needed_edges:
        edge = geometry.edges[edge_id]
        needed_vertices.extend((edge.start, edge.end))
    needed_vertices = list(dict.fromkeys(needed_vertices))

    point_tags: Dict[int, int] = {}
    for vertex_id in needed_vertices:
        x, y, z = (float(value) for value in geometry.vertex_position(vertex_id))
        point_tags[vertex_id] = gmsh.model.geo.addPoint(x, y, z)

    curve_tags: Dict[int, int] = {}
    for edge_id in needed_edges:
        edge = geometry.edges[edge_id]
        curve = edge.curve
        if isinstance(curve, Straight):
            curve_tags[edge_id] = gmsh.model.geo.addLine(
                point_tags[edge.start], point_tags[edge.end]
            )
        elif isinstance(curve, Arc):
            # An Arc stores only its via vertex; the circle is resolved by the
            # geometry model, which is also where sweeps beyond 180 degrees are
            # supported.  gmsh's built-in kernel cannot express one as a single
            # circle arc, so that case is refused rather than approximated.
            frame = geometry.arc_frame(edge_id)
            if abs(float(frame.sweep)) >= np.pi - 1.0e-9:
                raise MeshError(
                    f"line {edge_id} is an arc sweeping "
                    f"{np.degrees(abs(float(frame.sweep))):.1f} degrees, and gmsh's "
                    "built-in kernel only takes arcs under 180. Split it into shorter "
                    "arcs, or mesh this model with backend='mapped'."
                )
            centre_tag = gmsh.model.geo.addPoint(
                *(float(value) for value in np.asarray(frame.center, dtype=float))
            )
            curve_tags[edge_id] = gmsh.model.geo.addCircleArc(
                point_tags[edge.start], centre_tag, point_tags[edge.end]
            )
        elif isinstance(curve, Spline):
            # ANYgeometry's lightweight spline is a Bezier curve, so this is an
            # exact reconstruction rather than a sampled polyline approximation.
            # Control vertices remain geometry-only; mesh association stays on
            # the owning edge.
            control_tags = []
            for vertex in curve.control_vertices:
                position = geometry.vertex_position(vertex)
                control_tags.append(
                    gmsh.model.geo.addPoint(*(float(value) for value in position))
                )
            curve_tags[edge_id] = gmsh.model.geo.addBezier(
                [point_tags[edge.start], *control_tags, point_tags[edge.end]]
            )
        else:
            raise MeshError(
                f"line {edge_id} has curve type {type(curve).__name__}, which the "
                "gmsh backend cannot rebuild; mesh this model with backend='mapped'"
            )

    surface_tags: Dict[int, int] = {}
    for face_id in face_ids:
        complaint = _face_planarity_error(geometry, face_id)
        if complaint is not None:
            raise MeshError(complaint)
        face = geometry.faces[face_id]
        loops = []
        for boundary in (face.loop, *getattr(face, "holes", ())):
            oriented = [
                curve_tags[item.edge] if item.forward else -curve_tags[item.edge]
                for item in boundary
            ]
            loops.append(gmsh.model.geo.addCurveLoop(oriented))
        surface_tags[face_id] = gmsh.model.geo.addPlaneSurface(loops)

    gmsh.model.geo.synchronize()
    return point_tags, curve_tags, surface_tags


def _read_nodes() -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    """Read connected 1D/2D nodes, renumbered densely in Gmsh tag order.

    Gmsh also meshes construction points used to define circle centres and
    Bezier controls as isolated 0D nodes. They are not part of the neutral mesh:
    importing them would create disconnected solver degrees of freedom.
    """

    tags, coords, _parametric = gmsh.model.mesh.getNodes()
    positions = np.asarray(coords, dtype=float).reshape(-1, 3)
    connected_tags: set[int] = set()
    for dimension in (1, 2):
        _types, _element_tags, node_tags = gmsh.model.mesh.getElements(dimension)
        for block in node_tags:
            connected_tags.update(int(tag) for tag in block)
    order = sorted(
        (index for index, tag in enumerate(tags) if int(tag) in connected_tags),
        key=lambda index: int(tags[index]),
    )
    nodes: Dict[int, np.ndarray] = {}
    remap: Dict[int, int] = {}
    for new_id, index in enumerate(order, start=1):
        remap[int(tags[index])] = new_id
        nodes[new_id] = positions[index].copy()
    return nodes, remap


def _ordered_edge_nodes(curve_tag: int, remap: Mapping[int, int]) -> List[int]:
    """Nodes along one curve, ordered from its start to its end."""

    tags, _coords, parametric = gmsh.model.mesh.getNodes(
        1, curve_tag, includeBoundary=True, returnParametricCoord=True
    )
    if len(tags) == 0:
        return []
    values = np.asarray(parametric, dtype=float)
    order = np.argsort(values)
    return [remap[int(tags[index])] for index in order if int(tags[index]) in remap]


def generate_mesh(
    geometry: GeometryModel,
    *,
    target_size: float,
    face_ids: Optional[Iterable[int]] = None,
    beam_edges: Iterable[int] = (),
    order: str = "linear",
    algorithm: int = 8,
    recombine: bool = True,
    verbose: bool = False,
) -> Mesh:
    """Mesh a geometry model with gmsh.

    ``algorithm`` is gmsh's 2D meshing algorithm; 8 is Frontal-Delaunay for
    quads, which is what gives recombination the most to work with.
    ``recombine`` asks for quadrilaterals; without it the result is all
    triangles, which is occasionally what a comparison wants.

    Eccentric beam offsets are not supported here.  The mapped mesher builds them
    from plate normals it computed itself, and inventing a normal for an
    unstructured patch is a different calculation rather than the same one --
    mesh a stiffened model with ``backend='mapped'``.
    """

    if order not in ("linear", "quadratic"):
        raise MeshError(f"unknown element order {order!r}; expected 'linear' or 'quadratic'")
    if float(target_size) <= 0.0:
        raise MeshError("target_size must be positive")

    faces = list(geometry.faces) if face_ids is None else list(dict.fromkeys(int(f) for f in face_ids))
    for face_id in faces:
        if face_id not in geometry.faces:
            raise MeshError(f"no face {face_id}")
    beam_edge_ids = list(dict.fromkeys(int(e) for e in beam_edges))
    for edge_id in beam_edge_ids:
        if edge_id not in geometry.edges:
            raise MeshError(f"no edge {edge_id}")
    if not faces and not beam_edge_ids:
        raise MeshError("nothing to mesh: no faces and no beam edges")

    with _gmsh_session(verbose):
        gmsh.model.add("anymesher")
        point_tags, curve_tags, surface_tags = _build_geometry(geometry, faces, beam_edge_ids)

        gmsh.option.setNumber("Mesh.MeshSizeMin", float(target_size))
        gmsh.option.setNumber("Mesh.MeshSizeMax", float(target_size))
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.Algorithm", int(algorithm))
        gmsh.option.setNumber("Mesh.RecombineAll", 1 if recombine else 0)
        gmsh.option.setNumber("Mesh.ElementOrder", 2 if order == "quadratic" else 1)
        # Serendipity rather than Lagrange: an 8-node quad and a 6-node
        # triangle, matching what the mapped mesher and the solvers use.
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)

        gmsh.model.mesh.generate(2 if faces else 1)

        mesh = Mesh(order=order)
        mesh.nodes, remap = _read_nodes()

        next_element = 0
        for vertex_id, tag in point_tags.items():
            tags, _coords, _parametric = gmsh.model.mesh.getNodes(0, tag)
            for node_tag in tags:
                if int(node_tag) in remap:
                    mesh.node_of_vertex[vertex_id] = remap[int(node_tag)]

        for edge_id, tag in curve_tags.items():
            sequence = _ordered_edge_nodes(tag, remap)
            if sequence:
                mesh.nodes_of_edge[edge_id] = sequence

        for face_id, tag in surface_tags.items():
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(2, tag)
            face_elements: List[int] = []
            for element_type, tags_for_type, nodes_for_type in zip(
                element_types, element_tags, node_tags
            ):
                bucket = _SHELL_TYPES.get(int(element_type))
                if bucket is None:
                    raise MeshError(
                        f"gmsh produced element type {int(element_type)} on face {face_id}, "
                        "which is not a 3/6-node triangle or 4/8-node quadrilateral"
                    )
                stride = len(nodes_for_type) // len(tags_for_type)
                for index in range(len(tags_for_type)):
                    next_element += 1
                    connectivity = tuple(
                        remap[int(node)]
                        for node in nodes_for_type[index * stride : (index + 1) * stride]
                    )
                    getattr(mesh, bucket)[next_element] = connectivity
                    face_elements.append(next_element)
            mesh.elements_of_face[face_id] = sorted(face_elements)

        for edge_id in beam_edge_ids:
            sequence = mesh.nodes_of_edge.get(edge_id, [])
            step = 2 if order == "quadratic" else 1
            element_ids: List[int] = []
            for start in range(0, len(sequence) - step, step):
                next_element += 1
                mesh.beams[next_element] = tuple(sequence[start : start + step + 1])
                element_ids.append(next_element)
            mesh.elements_of_edge[edge_id] = element_ids

    if not mesh.quads and not mesh.tris and not mesh.beams:
        raise MeshError("gmsh produced no elements")
    return mesh
