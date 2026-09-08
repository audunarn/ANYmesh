"""Detached T6 staging and late promotion, not a public activation gate.

The owner/chart coordinator must supply certified eligible straight edges and
owner evaluators. Returned candidates still require physical quality, coverage,
freshness and whole-component qualification before publication.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType

import numpy as np

from .errors import MeshError
from .mesh import Mesh


def _edge(a, b):
    return (a, b) if a < b else (b, a)


def _point(value):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise MeshError("quadratic staging requires finite physical coordinates")
    return result


def _edges(corners):
    return tuple(_edge(a, corners[(i + 1) % len(corners)]) for i, a in enumerate(corners))


@dataclass(frozen=True)
class QuadraticPromotion:
    mesh: Mesh
    # Newly evaluated native stations, to be registered only at component commit.
    boundary_stations: tuple


class QuadraticComponentStage:
    """Freeze T6 identity/ownership; expose only a disposable component mesh."""

    def __init__(self, source, face_ids, station_entries, *, eligible_edges=()):
        faces = tuple(sorted(set(face_ids)))
        if source.order != "quadratic" or not faces:
            raise MeshError("quadratic component staging requires a nonempty T6 component")
        cells = {}
        owners = {}
        midsides = {}
        original_boundaries = {}
        for face in faces:
            ids = tuple(source.elements_of_face.get(face, ()))
            if not ids:
                raise MeshError("quadratic component face has no seed elements")
            incidence = Counter()
            for element in ids:
                nodes = source.tris.get(element)
                if nodes is None or len(nodes) != 6 or len(set(nodes)) != 6 or element in owners:
                    raise MeshError("quadratic component seeds need uniquely owned T6 cells")
                cells[element] = tuple(nodes[:3])
                owners[element] = face
                for pair, midpoint in zip(_edges(nodes[:3]), nodes[3:]):
                    if pair in midsides and midsides[pair] != midpoint:
                        raise MeshError("quadratic shared edge has inconsistent midside identity")
                    midsides[pair] = midpoint
                    incidence[pair] += 1
            if any(count > 2 for count in incidence.values()):
                raise MeshError("quadratic seed has non-manifold face incidence")
            original_boundaries[face] = frozenset(pair for pair, count in incidence.items() if count == 1)
        for face, ids in source.elements_of_face.items():
            if face not in faces and set(ids).intersection(owners):
                raise MeshError("quadratic seed element has external face ownership")
        used = {n for element in owners for n in source.tris[element]}
        raw = {n: _point(source.nodes[n]).tobytes() for n in used}
        entries = {}
        for entry in station_entries:
            if entry.node_id is None:
                continue
            key = (entry.key.edge_id, entry.node_id)
            value = (Fraction.from_float(entry.key.parameter), _point(entry.point).tobytes())
            if key in entries and entries[key] != value:
                raise MeshError("boundary node has conflicting native station identity")
            entries[key] = value
        loops = {}
        stations = {}
        intervals = {}
        interval_edges = {}
        protected = set()
        for edge_id, sequence in sorted(source.nodes_of_edge.items()):
            if not set(sequence).intersection(used):
                continue
            if len(sequence) < 3 or len(sequence) % 2 != 1 or not set(sequence).issubset(used):
                raise MeshError("component boundary needs complete quadratic edge seeds")
            sequence = tuple(sequence)
            if len(set(sequence)) != len(sequence):
                raise MeshError("ambiguous repeated boundary seed identity")
            for node in sequence:
                station, coordinate = entries.get((edge_id, node), (None, None))
                if station is None or coordinate != raw[node]:
                    raise MeshError("quadratic boundary station is missing or byte-inconsistent")
                stations[(edge_id, node)] = station
            parameters = [stations[(edge_id, n)] for n in sequence]
            if not (all(a < b for a, b in zip(parameters, parameters[1:])) or
                    all(a > b for a, b in zip(parameters, parameters[1:]))):
                raise MeshError("quadratic boundary native parameters are not strictly ordered")
            loops[edge_id] = sequence
            protected.update(sequence)
            for i in range(0, len(sequence) - 2, 2):
                a, m, b = sequence[i:i + 3]
                pair = _edge(a, b)
                if midsides.get(pair) != m or pair in interval_edges:
                    raise MeshError("quadratic boundary interval has ambiguous topology ownership")
                interval_edges[pair] = edge_id
                lo, hi = sorted((stations[(edge_id, a)], stations[(edge_id, b)]))
                intervals[(edge_id, lo, hi)] = (stations[(edge_id, m)], m)
        eligible = frozenset(eligible_edges)
        if not eligible.issubset(loops):
            raise MeshError("eligible split edge lacks component boundary inventory")
        forbidden = {n for nodes in source.beams.values() for n in nodes}
        forbidden.update(source.node_of_vertex.values())
        for coupling in source.couplings.values():
            forbidden.update((coupling.beam_node, *coupling.plate_nodes))
        forbidden.update(n for nodes in source.nodes_of_member.values() for n in nodes)
        external = {n for e, nodes in source.shells.items() if e not in owners for n in nodes}
        junction_nodes = {n for pair in source.declared_plate_junction_edges for n in pair}
        for edge_id in eligible:
            # Mixed components, beams, role-constrained seeds and junctions
            # cannot acquire a new corner role through this staging path.
            if set(loops[edge_id][1::2]).intersection(forbidden | external | junction_nodes):
                raise MeshError("eligible quadratic split has an external or protected midside role")
            if set(loops[edge_id]).intersection(external | junction_nodes):
                raise MeshError("eligible quadratic edge is not wholly component-owned")
        self.faces = faces
        self._binding = (source.geometry_model_id, source.geometry_revision)
        self._raw = MappingProxyType(raw)
        self._midsides = MappingProxyType(midsides)
        self._loops = MappingProxyType(loops)
        self._stations = MappingProxyType(stations)
        self._intervals = MappingProxyType(intervals)
        self._interval_edges = MappingProxyType(interval_edges)
        self._boundaries = MappingProxyType(original_boundaries)
        self._protected = frozenset(protected)
        self._eligible = eligible
        self._proposal_cache = {}
        self._proposable_intervals = set(intervals)
        self._child_stations = {}
        self._generated_raw = {}
        self._authorized_splits = []
        self._first_new_node = max(source.nodes, default=0) + 1
        self.mesh = Mesh(
            geometry_model_id=source.geometry_model_id, geometry_revision=source.geometry_revision,
            nodes={n: np.frombuffer(value, dtype=np.float64).copy() for n, value in raw.items()},
            tris=cells, order="linear",
            elements_of_face={face: sorted(e for e, owner in owners.items() if owner == face) for face in faces},
            nodes_of_edge={edge: list(sequence[::2]) for edge, sequence in loops.items()},
            node_of_vertex={v: n for v, n in source.node_of_vertex.items() if n in used},
        )

    def station(self, edge_id, node_id):
        key = (edge_id, node_id)
        if key in self._stations:
            return self._stations[key]
        return self._child_stations[key]

    def proposed_point(self, edge_id, station):
        for (edge, lower, upper), (t, raw, node) in self._proposal_cache.items():
            if edge == edge_id and t == station:
                return np.frombuffer(raw, dtype=np.float64).copy()
        raise MeshError("quadratic split has no owner-evaluated proposal")

    def record_split(self, edge_id, first, last, node_id, station):
        """Retain a successfully staged split, not a speculative proposal."""
        lower, upper = sorted((self.station(edge_id, first), self.station(edge_id, last)))
        proposal = self._proposal_cache.get((edge_id, lower, upper))
        if proposal is None or proposal[0] != station:
            raise MeshError("quadratic split does not match its owner proposal")
        _, raw, reserved = proposal
        if reserved is not None and reserved != node_id:
            raise MeshError("quadratic split replaced a reserved node identity")
        key = (edge_id, node_id)
        if key in self._stations:
            if self._stations[key] != station or self._raw[node_id] != raw:
                raise MeshError("quadratic split conflicts with original station identity")
        else:
            if node_id in self._raw or node_id in self._generated_raw or key in self._child_stations:
                raise MeshError("quadratic child split reused an occupied node identity")
            self._child_stations[key] = station
            self._generated_raw[node_id] = raw
        self._authorized_splits.append((edge_id, first, node_id, last))

    def split_proposal(self, chart_point_for_node, *, evaluate_edge=None):
        """Bind exact saved native-t and ID; owner projection is supplied once."""
        def propose(edge_id, lower, upper):
            if edge_id not in self._eligible:
                return None
            key = (edge_id, min(lower, upper), max(lower, upper))
            if key not in self._proposable_intervals:
                return None
            entry = self._proposal_cache.get(key)
            if entry is None:
                original = self._intervals.get(key)
                if original is not None:
                    station, node_id = original
                    raw = self._raw[node_id]
                elif evaluate_edge is not None:
                    # Round the new native parameter once. Both incident charts
                    # consume this exact binary64 station and the same owner point.
                    native_t = float((lower + upper) / 2)
                    station = Fraction.from_float(native_t)
                    if not min(lower, upper) < station < max(lower, upper):
                        return None
                    raw = _point(evaluate_edge(edge_id, native_t)).tobytes()
                    node_id = None
                else:
                    return None
                entry = (station, raw, node_id)
                self._proposal_cache[key] = entry
                self._proposable_intervals.update(((edge_id, key[1], station), (edge_id, station, key[2])))
            station, raw, node_id = entry
            physical = np.frombuffer(raw, dtype=np.float64).copy()
            return station, chart_point_for_node(edge_id, node_id, physical), node_id
        return propose

    def promote(self, candidate, *, evaluate_edge, evaluate_interior, cancellation_check=None):
        """Return detached T6/Q8 connectivity; do not publish or weaken quality."""
        def checkpoint(phase):
            if cancellation_check is not None:
                cancellation_check(phase)

        checkpoint("quadratic component promotion start")
        if (candidate.geometry_model_id, candidate.geometry_revision) != self._binding:
            raise MeshError("quadratic promotion has a stale or foreign component binding")
        if set(candidate.elements_of_face) != set(self.faces) or candidate.order != "linear":
            raise MeshError("quadratic promotion requires the complete linear component")
        for node in self._protected:
            if node not in candidate.nodes or _point(candidate.nodes[node]).tobytes() != self._raw[node]:
                raise MeshError("quadratic staging changed a protected node")
        for node, raw in self._generated_raw.items():
            if node not in candidate.nodes or _point(candidate.nodes[node]).tobytes() != raw:
                raise MeshError("quadratic staging changed an owner-evaluated shared node")
        boundary_pairs = {}
        replacements = {}
        for edge_id, original in self._loops.items():
            corners = tuple(candidate.nodes_of_edge.get(edge_id, ()))
            expected = []
            for i in range(0, len(original) - 2, 2):
                a, m, b = original[i:i + 3]
                chain = [a, b]
                lower, upper = sorted((self.station(edge_id, a), self.station(edge_id, b)))
                for edge, first, mid, last in self._authorized_splits:
                    if edge != edge_id or not lower < self.station(edge, mid) < upper:
                        continue
                    if first not in chain or last not in chain or abs(chain.index(first) - chain.index(last)) != 1:
                        raise MeshError("quadratic split receipts do not form a conformal interval tree")
                    chain.insert(min(chain.index(first), chain.index(last)) + 1, mid)
                if m in corners and m not in chain:
                    if len(chain) != 2:
                        raise MeshError("quadratic boundary omitted its original midside split")
                    chain.insert(1, m)
                if len(chain) > 2 and edge_id not in self._eligible:
                    raise MeshError("quadratic staging split an ineligible protected interval")
                expected.extend(chain[:-1])
                replacements[_edge(a, b)] = tuple(_edge(x, y) for x, y in zip(chain, chain[1:]))
                for x, y in zip(chain, chain[1:]):
                    boundary_pairs[_edge(x, y)] = edge_id
            expected.append(original[-1])
            if corners != tuple(expected):
                raise MeshError("quadratic staging changed boundary ordering or seed connectivity")
        owners = {}
        pair_owners = {}
        for face in self.faces:
            incidence = Counter()
            for element in candidate.elements_of_face[face]:
                if element in owners:
                    raise MeshError("quadratic candidate element has ambiguous ownership")
                owners[element] = face
                corners = candidate.tris.get(element, candidate.quads.get(element))
                count = 3 if element in candidate.tris else 4
                if corners is None or len(corners) != count or len(set(corners)) != count:
                    raise MeshError("quadratic promotion requires T3/Q4 corner connectivity")
                for pair in _edges(corners):
                    incidence[pair] += 1
                    pair_owners.setdefault(pair, set()).add(face)
            expected = {child for pair in self._boundaries[face] for child in replacements.get(pair, (pair,))}
            if any(n > 2 for n in incidence.values()) or {p for p, n in incidence.items() if n == 1} != expected:
                raise MeshError("quadratic candidate changed face boundary or incidence")
        if set(owners) != set(candidate.tris) | set(candidate.quads):
            raise MeshError("quadratic candidate contains unowned elements")
        result = deepcopy(candidate)
        next_node = max(self._first_new_node, max(result.nodes, default=0) + 1)
        midsides = {}
        additions = [(edge, float(station), node, self._generated_raw[node])
                     for (edge, node), station in sorted(self._child_stations.items())]
        corner_ids = {n for nodes in candidate.shells.values() for n in nodes}
        for pair in sorted(pair_owners):
            checkpoint("quadratic component midside evaluation")
            a, b = pair
            old_mid = self._midsides.get(pair)
            unchanged = all(n in self._raw and _point(candidate.nodes[n]).tobytes() == self._raw[n] for n in pair)
            if old_mid is not None and unchanged:
                if old_mid in corner_ids:
                    raise MeshError("quadratic midside has conflicting unsplit corner role")
                midsides[pair] = old_mid
                result.nodes[old_mid] = np.frombuffer(self._raw[old_mid], dtype=np.float64).copy()
                continue
            edge_id = boundary_pairs.get(pair)
            if edge_id is not None:
                lower, upper = sorted((self.station(edge_id, a), self.station(edge_id, b)))
                native_t = float((lower + upper) / 2)
                if not lower < Fraction.from_float(native_t) < upper:
                    raise MeshError("quadratic child midpoint has no representable interior station")
                coordinate = _point(evaluate_edge(edge_id, native_t))
                additions.append((edge_id, native_t, next_node, coordinate.tobytes()))
            else:
                faces = pair_owners[pair]
                if len(faces) != 1:
                    raise MeshError("shared quadratic midpoint lacks geometry-owned edge identity")
                coordinate = _point(evaluate_interior(
                    next(iter(faces)), _point(candidate.nodes[a]), _point(candidate.nodes[b])
                ))
            midsides[pair] = next_node
            result.nodes[next_node] = coordinate
            next_node += 1
        result.tris = {e: tuple(nodes) + tuple(midsides[p] for p in _edges(nodes)) for e, nodes in candidate.tris.items()}
        result.quads = {e: tuple(nodes) + tuple(midsides[p] for p in _edges(nodes)) for e, nodes in candidate.quads.items()}
        result.nodes_of_edge = {}
        for edge_id, corners in candidate.nodes_of_edge.items():
            sequence = []
            for a, b in zip(corners, corners[1:]):
                sequence.extend((a, midsides[_edge(a, b)]))
            sequence.append(corners[-1])
            result.nodes_of_edge[edge_id] = sequence
        result.order = "quadratic"
        checkpoint("quadratic component promotion complete")
        return QuadraticPromotion(result, tuple(additions))
