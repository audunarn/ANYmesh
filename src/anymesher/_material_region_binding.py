"""Discretization adapters for geometry-owner qualified material regions."""
from dataclasses import dataclass, replace
from fractions import Fraction
from numbers import Integral
import numpy as np

from .errors import MeshError


@dataclass(frozen=True)
class BoundaryIntervalReceipt:
    """Source identity of one owner boundary occurrence, never split permission."""
    binding: object
    loop_index: int
    path_index: int
    interval_index: int
    source_edge: int
    nodes: tuple[int, int]
    parameters: tuple[Fraction, Fraction]
    uv: tuple[tuple[float, float], tuple[float, float]]
    xyz: tuple[tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class MaterialRegionBinding:
    geometry: object
    collection: object
    region: object
    authored_face: int

    @property
    def face_ids(self):
        return frozenset(int(face.id) for face in self.region.faces)

    @property
    def representative(self):
        return min(self.face_ids)

    def validate(self, cancellation_check=None):
        from anygeometry import validate_material_surface_regions_binding
        validate_material_surface_regions_binding(
            self.geometry, self.collection, cancellation_check=cancellation_check)

    def path_chain(self, path, mesh, registry):
        """Reuse source-edge station identities in the owner's path direction."""
        edge_id = path.source_edge
        if edge_id is None or edge_id not in mesh.nodes_of_edge:
            raise MeshError("material region path has no registered source edge")
        entries = registry.entries(edge_id)
        by_node = {int(entry.node_id): float(entry.key.parameter)
                   for entry in entries if entry.node_id is not None}
        nodes = list(mesh.nodes_of_edge[edge_id])
        if any(node not in by_node for node in nodes):
            raise MeshError("material region source station has no registry receipt")
        nodes.sort(key=by_node.__getitem__)
        ends = np.asarray(path.curve.evaluate(np.asarray((0., 1.))), dtype=float)
        source_ends = self.geometry.sample_edge(edge_id, np.asarray((0., 1.)))
        tolerance = self.region.world_tolerance
        forward = np.allclose(ends, source_ends, rtol=0., atol=tolerance)
        reverse = np.allclose(ends, source_ends[::-1], rtol=0., atol=tolerance)
        if forward == reverse:
            raise MeshError("material region path/source interval is ambiguous")
        if reverse:
            nodes.reverse()
        parameters = np.asarray([by_node[node] for node in nodes])
        path_parameters = parameters if forward else 1. - parameters
        # The owner domain supplies periodic branch conventions for each exact path.
        uv = np.asarray([self.region.domain.uv(path.curve, float(t))
                         for t in path_parameters], dtype=float)
        expected = np.asarray(path.curve.evaluate(path_parameters), dtype=float)
        registered = np.asarray([mesh.nodes[node] for node in nodes], dtype=float)
        if not np.allclose(expected, registered, rtol=0., atol=tolerance):
            raise MeshError("material region path does not bind its registered stations")
        if uv.shape != (len(nodes), 2) or not np.isfinite(uv).all():
            raise MeshError("material region returned invalid station parameters")
        return nodes, uv, parameters

    def _validate_boundary_source_owner(self, mesh, registry):
        # Coincident coordinates and reusable local IDs cannot transfer receipts
        # between geometry owners, including independently loaded copies.
        view = registry.view
        if view.source is not self.geometry:
            raise MeshError('material boundary registry belongs to a different geometry owner')
        view.assert_current(self.geometry)
        mesh_owner = getattr(mesh, 'geometry_model_id', None)
        if mesh_owner is not None and str(mesh_owner) != str(self.geometry.model_id):
            raise MeshError('material boundary mesh belongs to a different geometry owner')

    def _boundary_source_chain(self, path, mesh, registry):
        self._validate_boundary_source_owner(mesh, registry)
        nodes, uv, parameters = self.path_chain(path, mesh, registry)
        entries = registry.entries(path.source_edge)
        selected = set(nodes)
        seen = set()
        for entry in entries:
            if entry.node_id not in selected:
                raise MeshError('material boundary chain omitted a registered source station')
            if entry.node_id in seen:
                raise MeshError('material boundary node has multiple source stations')
            seen.add(entry.node_id)
        source_points = np.asarray(self.geometry.sample_edge(path.source_edge, parameters), dtype=float)
        registered = np.asarray([mesh.nodes[node] for node in nodes], dtype=float)
        if (source_points.shape != registered.shape or not np.isfinite(source_points).all()
                or not np.allclose(source_points, registered, rtol=0., atol=self.region.world_tolerance)):
            raise MeshError('material boundary registry does not bind source parameters')
        return nodes, uv, parameters

    def _validate_boundary_owner(self, cancellation_check):
        if not any(region is self.region for region in self.collection.regions):
            raise MeshError('material boundary region is not its validated owner collection')
        self.validate(cancellation_check)

    def boundary_intervals(self, mesh, registry, cancellation_check=None):
        """Retain exact registered stations even on protected curved paths.

        Eligibility is deliberately absent. A caller must separately qualify its
        owner, curve-error, resource and whole-component publication contracts.
        """
        self._validate_boundary_owner(cancellation_check)
        result = []
        for loop_index, paths in enumerate(self.region.boundaries):
            for path_index, path in enumerate(paths):
                if cancellation_check is not None:
                    cancellation_check('material boundary source receipts')
                nodes, uv, parameters = self._boundary_source_chain(path, mesh, registry)
                if (len(nodes) < 2 or len(set(nodes)) != len(nodes)
                        or any(isinstance(node, (bool, np.bool_))
                               or not isinstance(node, Integral) or node < 1 for node in nodes)):
                    raise MeshError('material boundary has ambiguous station identities')
                differences = np.diff(parameters)
                if not (np.all(differences > 0.) or np.all(differences < 0.)):
                    raise MeshError('material boundary has ambiguous source intervals')
                for index in range(len(nodes)-1):
                    pair = tuple(nodes[index:index+2])
                    result.append(BoundaryIntervalReceipt(
                        self, loop_index, path_index, index, int(path.source_edge), pair,
                        tuple(Fraction(float(t)) for t in parameters[index:index+2]),
                        tuple(tuple(map(float, row)) for row in uv[index:index+2]),
                        tuple(tuple(map(float, mesh.nodes[node])) for node in pair)))
        self._validate_boundary_owner(cancellation_check)
        # No callback follows these final source/chain checks. Cache by owner
        # occurrence so a long seeded boundary is checked once, not per interval.
        self._validate_boundary_source_owner(mesh, registry)
        chains = {}
        for receipt in result:
            occurrence = (receipt.loop_index, receipt.path_index)
            if occurrence not in chains:
                path = self.region.boundaries[receipt.loop_index][receipt.path_index]
                chains[occurrence] = self._boundary_source_chain(path, mesh, registry)
            self._assert_boundary_receipt_current(receipt, mesh, chains[occurrence])
        return tuple(result)

    @staticmethod
    def _assert_boundary_receipt_current(receipt, mesh, chain):
        nodes, uv, parameters = chain
        index = receipt.interval_index
        current_nodes = tuple(nodes[index:index+2])
        current_parameters = tuple(Fraction(float(t)) for t in parameters[index:index+2])
        if (current_nodes != receipt.nodes or current_parameters != receipt.parameters
                or tuple(tuple(map(float, row)) for row in uv[index:index+2]) != receipt.uv
                or tuple(tuple(map(float, mesh.nodes[node])) for node in current_nodes) != receipt.xyz):
            raise MeshError('material boundary station receipt is no longer current')

    def boundary_station(self, receipt, mesh, registry, parameter, cancellation_check=None):
        """Evaluate a still-bound exact source station in its owner occurrence.

        This read-only query grants no refinement permission and registers no
        station. Nonrepresentable rational parameters fail instead of silently
        changing their shared identity to a nearby binary64 value.
        """
        if not isinstance(receipt, BoundaryIntervalReceipt) or receipt.binding is not self:
            raise MeshError('material boundary station requires its source receipt')
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
               or value < 0 for value in
               (receipt.loop_index, receipt.path_index, receipt.interval_index)):
            raise MeshError('material boundary source occurrence is unavailable')
        if not isinstance(parameter, Fraction):
            raise MeshError('material boundary station requires an exact rational parameter')
        lower, upper = sorted(receipt.parameters)
        if not lower < parameter < upper:
            raise MeshError('material boundary station must be inside its original interval')
        value = float(parameter)
        if not np.isfinite(value) or Fraction(value) != parameter:
            raise MeshError('material boundary station is not exactly representable')
        self._validate_boundary_owner(cancellation_check)
        # Reconstruct only the declared occurrence; no coordinate identity search.
        try:
            path = self.region.boundaries[receipt.loop_index][receipt.path_index]
        except (IndexError, TypeError):
            raise MeshError('material boundary source occurrence is unavailable') from None
        if (receipt.loop_index < 0 or receipt.path_index < 0 or receipt.interval_index < 0
                or path.source_edge != receipt.source_edge):
            raise MeshError('material boundary source occurrence changed')
        self._assert_boundary_receipt_current(
            receipt, mesh, self._boundary_source_chain(path, mesh, registry))
        path_parameter = value if receipt.parameters[0] < receipt.parameters[1] else 1.-value
        row = np.asarray(self.region.domain.uv(path.curve, path_parameter), dtype=float)
        point = np.asarray(self.geometry.sample_edge(receipt.source_edge, np.asarray((value,))), dtype=float)
        expected = np.asarray(path.curve.evaluate(path_parameter), dtype=float)
        if (row.shape != (2,) or point.shape != (1, 3) or expected.shape != (3,)
                or not np.isfinite(row).all() or not np.isfinite(point).all()
                or not np.isfinite(expected).all()
                or not np.allclose(point[0], expected, rtol=0., atol=self.region.world_tolerance)):
            raise MeshError('material boundary station lost its exact owner correspondence')
        from anygeometry import evaluate_material_surface_region
        owner_point = evaluate_material_surface_region(
            self.geometry, self.collection, self.representative, row[None, :],
            require_material=True, cancellation_check=cancellation_check)
        if not np.allclose(owner_point[0], point[0], rtol=0., atol=self.region.world_tolerance):
            raise MeshError('material boundary station does not bind its owner support')
        self._validate_boundary_owner(cancellation_check)
        self._assert_boundary_receipt_current(
            receipt, mesh, self._boundary_source_chain(path, mesh, registry))
        return row.copy(), point[0].copy()

    def loops(self, mesh, registry, splittable):
        result = []
        for paths in self.region.boundaries:
            nodes, rows, specifications = [], [], []
            previous = None
            for path in paths:
                chain, uv, parameters = self.path_chain(path, mesh, registry)
                if previous is not None and previous != chain[0]:
                    raise MeshError("material region boundary lost source vertex identity")
                nodes.extend(chain[:-1]); rows.extend(uv[:-1])
                specifications.extend(
                    (int(path.source_edge), float(min(a, b)), float(max(a, b)))
                    if int(path.source_edge) in splittable else None
                    for a, b in zip(parameters[:-1], parameters[1:]))
                previous = chain[-1]
            if not nodes or previous != nodes[0] or len(set(nodes)) != len(nodes):
                raise MeshError("material region has an ambiguous station cycle")
            result.append((tuple(nodes), np.asarray(rows), tuple(specifications)))
        return tuple(result)

    def input_provenance(self, mesh, registry, loops, pinned_ids, eligibility_context,
                         emitted_inputs,
                         cancellation_check=None):
        """Bind pre-native input identities without granting split permission."""
        from ._material_input_provenance import make_material_input_provenance

        return make_material_input_provenance(
            self, mesh, registry, loops, pinned_ids, eligibility_context,
            emitted_inputs,
            cancellation_check)

    def interior(self, mesh, registry, *, boundary_rows=None):
        segments, pinned = [], {}
        if boundary_rows is None:
            boundary_rows = {node: row for nodes, uv, _ in self.loops(mesh, registry, ())
                             for node, row in zip(nodes, uv)}
        canonical = {int(node): np.asarray(row, dtype=float).copy()
                     for node, row in boundary_rows.items()}
        boundary_nodes = set(canonical)

        def receipt(node, row):
            # Canonicalize only by an already known registry node identity.
            # Keep the submitted boundary/first interior row bit-for-bit.
            previous = canonical.setdefault(int(node), np.asarray(row, dtype=float).copy())
            if not np.allclose(previous, row, rtol=0., atol=2e-14):
                raise MeshError("material constraint has ambiguous chart correspondence")
            return previous

        for path in self.region.interior_constraints:
            nodes, uv, _ = self.path_chain(path, mesh, registry)
            uv = np.asarray([receipt(node, row) for node, row in zip(nodes, uv)])
            segments.extend(np.asarray((a, b)) for a, b in zip(uv[:-1], uv[1:]))
            for node, row in zip(nodes, uv):
                if node not in boundary_nodes:
                    pinned[node] = row
        for handle in self.region.retained_vertices:
            node = mesh.node_of_vertex.get(handle.id)
            if node is None:
                node = max(mesh.nodes, default=0) + 1
                mesh.node_of_vertex[handle.id] = node
                mesh.nodes[node] = np.asarray(self.geometry.vertex_position(handle.id), dtype=float)
            if node not in boundary_nodes:
                if node in canonical:
                    # A physical source path already supplied this vertex's
                    # owner-qualified branch coordinate. Reprojection is both
                    # redundant and may choose a different inverse branch.
                    pinned[node] = canonical[node]
                else:
                    row = self.region.support.local_uv_many([mesh.nodes[node]])[0]
                    pinned[node] = receipt(node, row)
        ids = tuple(sorted(pinned))
        return tuple(segments), ids, np.asarray([pinned[node] for node in ids]).reshape(-1, 2)

    def validate_constraints(self, core, mesh, registry, row_to_node, cancellation_check=None):
        """Require each protected source-station interval in active connectivity."""
        node_to_row = {node: row for row, node in row_to_node.items()}
        incidence = {}
        active_rows = set()
        for connectivity, activity in ((core.triangle_connectivity, core.triangle_active),
                                       (core.quad_connectivity, core.quad_active)):
            for index, cell in enumerate(connectivity[activity]):
                if cancellation_check is not None and index % 256 == 0:
                    cancellation_check('material region constraint incidence')
                active_rows.update(map(int, cell))
                for first, second in zip(cell, np.roll(cell, -1)):
                    key = tuple(sorted((int(first), int(second))))
                    incidence[key] = incidence.get(key, 0) + 1
        for path in self.region.interior_constraints:
            if cancellation_check is not None:
                cancellation_check('material region constraint stations')
            nodes, _, _ = self.path_chain(path, mesh, registry)
            for first, second in zip(nodes[:-1], nodes[1:]):
                if first not in node_to_row or second not in node_to_row:
                    raise MeshError('material constraint lost a registered source station')
                key = tuple(sorted((node_to_row[first], node_to_row[second])))
                if incidence.get(key) != 2:
                    raise MeshError('material constraint station interval was lost or split without owner identity')
        for handle in self.region.retained_vertices:
            if cancellation_check is not None:
                cancellation_check('material region retained vertices')
            node = mesh.node_of_vertex.get(handle.id)
            if node not in node_to_row or node_to_row[node] not in active_rows:
                raise MeshError('material region retained vertex is absent from active connectivity')

    def certify_published(self, mesh, registry, settings, cancellation_check=None):
        """Check the final detached region after other faces complete shared splits."""
        from .core import MeshCore
        from ._analytic_metric_chart import AnalyticMetricChart
        self.validate(cancellation_check)
        elements = mesh.elements_of_face.get(self.representative, ())
        triangles, quads, used = [], [], set()
        for index, element in enumerate(elements):
            if cancellation_check is not None and index % 256 == 0:
                cancellation_check('material region published cells')
            if element in mesh.tris:
                cell = mesh.tris[element]
                triangles.append(cell)
            elif element in mesh.quads:
                cell = mesh.quads[element]
                quads.append(cell)
            else:
                raise MeshError('material region publication refers to a missing cell')
            used.update(cell)
        nodes = sorted(used)
        local = {node: row for row, node in enumerate(nodes)}
        core = MeshCore(np.asarray([mesh.nodes[node] for node in nodes]),
                        np.asarray([[local[node] for node in cell] for cell in triangles], dtype=int).reshape(-1, 3),
                        np.asarray([[local[node] for node in cell] for cell in quads], dtype=int).reshape(-1, 4))
        self.validate_constraints(core, mesh, registry, {row: node for node, row in local.items()}, cancellation_check)
        chart = AnalyticMetricChart(self.geometry, self.representative, cancellation_check,
                                    region_binding=self)
        return chart.certify_physical_core(core, settings)


def prepare_material_regions(geometry, native_faces, source_to_final_faces, *,
                             order, native_options, supplied_seeding=False,
                             edge_overrides=(), cancellation_check=None):
    """Select unions only when publication has one exact authored owner."""
    if order != "linear" or native_options.point_placement != "frontal_delaunay" or supplied_seeding:
        return {}
    try:
        from anygeometry import ExtrudedSurface, query_material_surface_regions, MaterialSurfaceRegions
    except ImportError:
        return {}
    selected = tuple(face for face in native_faces
                     if isinstance(geometry.faces[face].surface, ExtrudedSurface)
                     and geometry.faces[face].parameterization is None)
    if not selected:
        return {}
    preimages = {}
    for authored, descendants in source_to_final_faces.items():
        for face in descendants:
            preimages.setdefault(int(face), set()).add(int(authored))
    collection = query_material_surface_regions(
        geometry, selected, expected_revision=geometry.revision,
        cancellation_check=cancellation_check)
    bindings = {}
    for region in collection.regions:
        faces = frozenset(int(handle.id) for handle in region.faces)
        if len(faces) < 2 or not region.cancelled_seams or not faces.issubset(selected):
            continue
        owners = {frozenset(preimages.get(face, ())) for face in faces}
        if len(owners) != 1 or len(next(iter(owners))) != 1:
            continue
        if any(handle.id in edge_overrides for handle in region.cancelled_seams):
            continue
        if region.source_attachments:
            raise MeshError("coalesced material region attachments require a qualified consumer")
        scoped = MaterialSurfaceRegions(replace(collection.source, charts=region.sources), (region,))
        binding = MaterialRegionBinding(geometry, scoped, region, next(iter(next(iter(owners)))))
        binding.validate(cancellation_check)
        for face in faces:
            bindings[face] = binding
    return bindings


def registered_core_rows(core, receipts):
    """Map unchanged submitted chart rows, without discovering geometry identity."""
    lookup = {}
    for row, point in enumerate(core.node_coordinates[:, :2]):
        lookup.setdefault(tuple(map(float, point)), []).append(row)
    result = {}
    node_rows = {}
    active_rows = set(map(int, core.triangle_connectivity[core.triangle_active].ravel()))
    active_rows.update(map(int, core.quad_connectivity[core.quad_active].ravel()))
    for point, node in receipts:
        matches = lookup.get(tuple(map(float, point)), ())
        if len(matches) != 1:
            raise MeshError("material region registered input was moved, lost or merged")
        row = matches[0]
        if row not in active_rows:
            raise MeshError('material region registered input is absent from active connectivity')
        previous = result.setdefault(row, int(node))
        if previous != int(node):
            raise MeshError("material region registered inputs have conflicting source identities")
        previous_row = node_rows.setdefault(int(node), row)
        if previous_row != row:
            raise MeshError("material region source station has ambiguous chart rows")
    return result
