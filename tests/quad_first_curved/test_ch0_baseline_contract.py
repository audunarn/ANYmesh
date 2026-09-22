"""CH0 baseline contract tests: curved and higher-order quad scope.

These nine tests are **frozen contract assertions**. They pass before any CH1
or CH2 implementation work and must remain passing unless a specific frozen
decision is explicitly re-opened. They verify:

- The Q8 node ordering (corners then four cyclic midsides) and shape-function
  evaluation at each node location.
- Partition-of-unity on the closed reference square.
- Promotion from linear (Q4/T3) topology to quadratic (Q8/T6) without
  disturbing the source nodes.
- Skeleton-only quality: higher-order midside coordinates do not affect
  ``triangle_quality`` or ``quad_quality``.
- The public symbol surface resolves through documented paths.
- ``QuadPublicUnsupported`` guards still fire for order="quadratic" and
  planar=False (neither is a public route at CH0).
- The ``route_quad_first`` dispatch: ``None`` is the legacy sentinel; an
  explicit ``QuadMeshingOptions`` routes quad-first.
- ``ELEMENT_ORDERS`` is exactly ('linear', 'quadratic'): Q9 and beyond are
  deferred and not named by any public token.

Each test is self-contained and requires **no** worker binary and **no**
compiled native extension.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test 1 — Node ordering: corners then cyclic midsides
# ---------------------------------------------------------------------------

def test_node_ordering_is_corners_then_cyclic_midsides() -> None:
    """Q8 = 4 corners (in-plane winding) + 4 cyclic midsides.

    shape_functions_8node returns 1.0 at the k-th node reference position and
    0.0 at every other position, for all 8 nodes. Node 0 is the corner at
    (-1,-1) (in-plane winding, index 3 in the standard Q8 convention that
    couples.py:45 follows), and nodes 4-7 are the midsides in the cyclic
    edge order (edge 01, edge 12, edge 23, edge 30).
    """

    from anymesher.coupling import shape_functions_8node

    # (xi, eta) for each node: corners 0-3 in winding order, midsides 4-7
    node_positions = [
        (-1.0, -1.0),  # corner 0
        (1.0,  -1.0),  # corner 1
        (1.0,   1.0),  # corner 2
        (-1.0,  1.0),  # corner 3
        (0.0,  -1.0),  # midside edge 01 (bottom)
        (1.0,   0.0),  # midside edge 12 (right)
        (0.0,   1.0),  # midside edge 23 (top)
        (-1.0,  0.0),  # midside edge 30 (left)
    ]
    tol = 1e-12
    for k, (xi, eta) in enumerate(node_positions):
        values = np.asarray(shape_functions_8node(xi, eta), dtype=np.float64)
        assert values.shape[0] == 8, (
            f"shape_functions_8node returned {values.shape[0]} values, expected 8"
        )
        assert abs(float(values[k]) - 1.0) <= tol, (
            f"node {k} at ({xi},{eta}): value[{k}]={values[k]}, expected 1.0"
        )
        for j in range(8):
            if j != k:
                assert abs(float(values[j])) <= tol, (
                    f"node {k} at ({xi},{eta}): value[{j}]={values[j]}, expected 0.0"
                )


# ---------------------------------------------------------------------------
# Test 2 — Promotion: linear topology maps to quadratic
# ---------------------------------------------------------------------------

def test_promotion_maps_linear_topology_to_quadratic() -> None:
    """insert_midside_nodes promotes a single Q4 to Q8, is idempotent, and
    preserves the four corner node ids as the first four columns.
    """

    from anymesher.core import MeshCore
    from anymesher.surface_mesh import insert_midside_nodes

    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    linear_mesh = MeshCore(node_coordinates=pts, quad_connectivity=np.array([[0, 1, 2, 3]]))

    promoted = insert_midside_nodes(linear_mesh)

    # Width went from 4 to 8
    assert promoted.quad_connectivity.shape[1] == 8
    # is_quadratic flag is True
    assert promoted.is_quadratic
    # Source node count went from 4 to exactly 4 + 4 (one per edge)
    assert promoted.node_coordinates.shape[0] == 8
    # First four node ids are unchanged
    assert list(promoted.quad_connectivity[0, :4]) == [0, 1, 2, 3]
    # idempotent: second promotion is a no-op
    promoted2 = insert_midside_nodes(promoted)
    assert promoted2.node_coordinates.shape[0] == 8
    assert promoted2.quad_connectivity.shape[1] == 8


# ---------------------------------------------------------------------------
# Test 3 — Partition of unity for Q8 shape functions
# ---------------------------------------------------------------------------

def test_shape_functions_form_a_partition_of_unity() -> None:
    """sum_{k=0}^{7} N_k(xi, eta) == 1 for all (xi, eta) in [-1, 1]^2."""

    from anymesher.coupling import shape_functions_8node

    xs = np.linspace(-1.0, 1.0, 13)
    ys = np.linspace(-1.0, 1.0, 13)
    tol = 1e-12
    for xi in xs:
        for eta in ys:
            values = np.asarray(shape_functions_8node(float(xi), float(eta)), dtype=np.float64)
            total = float(np.sum(values))
            assert abs(total - 1.0) <= tol, (
                f"partition-of-unity violated at ({xi},{eta}): sum={total}"
            )


# ---------------------------------------------------------------------------
# Test 4 — Quality is skeleton-only
# ---------------------------------------------------------------------------

def test_quality_mechanism_is_skeleton_only() -> None:
    """Moving T6 midsides must not change triangle_quality.
    Moving Q8 midsides must not change quad_quality.
    """

    from anymesher.quality_v2 import quad_quality, triangle_quality

    # T6 case: two elements share the first-3 corners but differ in last-3
    t6_points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, -0.1, 0.0],   # midside ab
        [0.6,  0.7, 0.2],   # midside bc
        [-0.1, 0.6, 0.3],   # midside ca
    ])
    t6_a = np.array([[0, 1, 2, 3, 4, 5]])
    t6_b = np.array([[0, 1, 2, 4, 5, 3]])  # permuted higher-order nodes
    result_a = triangle_quality(t6_points, t6_a)
    result_b = triangle_quality(t6_points, t6_b)
    assert np.allclose(result_a.area, result_b.area), "moving T6 midsides changed area"
    assert np.allclose(result_a.aspect_ratio, result_b.aspect_ratio), "moving T6 midsides changed aspect"
    assert np.allclose(result_a.minimum_angle, result_b.minimum_angle), "moving T6 midsides changed min angle"

    # Q8 case: same
    q8_points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5,  -0.1, 0.0],   # midside edge 01
        [1.1,  0.5,  0.2],   # midside edge 12
        [-0.1, 0.6,  0.3],   # midside edge 23
        [0.4,  0.5, -0.2],   # midside edge 30
    ])
    q8_a = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    q8_b = np.array([[0, 1, 2, 3, 7, 6, 5, 4]])  # permuted higher-order nodes
    qa = quad_quality(q8_points, q8_a)
    qb = quad_quality(q8_points, q8_b)
    assert np.allclose(qa.area, qb.area), "moving Q8 midsides changed area"
    assert np.allclose(qa.warpage, qb.warpage), "moving Q8 midsides changed warpage"


# ---------------------------------------------------------------------------
# Test 5 — Public symbols resolve through documented import paths
# ---------------------------------------------------------------------------

def test_public_symbols_resolve_at_documented_paths() -> None:
    """Every symbol in the curved/HO reuse matrix resolves via its documented
    path. This prevents a later tranche from importing a private name instead.
    """

    import anymesher
    import anymesher.core
    import anymesher.quality_v2
    import anymesher.quad.public_integration
    import anymesher.surface_mesh

    # Top-level re-exports
    assert callable(anymesher.shape_functions_8node)
    assert callable(anymesher.shape_functions_4node)
    assert isinstance(anymesher.Mesh, type)
    assert callable(anymesher.generate_hybrid_mesh_result)
    assert isinstance(anymesher.QuadMeshingOptions, type)
    assert tuple(anymesher.ELEMENT_ORDERS) == ("linear", "quadratic")

    # Submodule paths
    assert isinstance(anymesher.core.MeshCore, type)
    assert callable(anymesher.core.corner_edges)
    assert callable(anymesher.quality_v2.triangle_quality)
    assert callable(anymesher.quality_v2.quad_quality)
    assert callable(anymesher.quality_v2.quad_candidate_quality)
    assert issubclass(anymesher.quad.public_integration.QuadPublicUnsupported, Exception)
    assert issubclass(anymesher.quad.public_integration.QuadCapabilityMissing, Exception)
    assert callable(anymesher.quad.public_integration.coerce_public_quad_options)
    assert callable(anymesher.quad.public_integration.route_quad_first)
    assert callable(anymesher.surface_mesh.insert_midside_nodes)


# ---------------------------------------------------------------------------
# Test 6 — QuadPublicUnsupported guards still fire for HO/curved requests
# ---------------------------------------------------------------------------

def test_quad_first_public_route_rejects_nonlinear_nonplanar() -> None:
    """order="quadratic" and planar=False must each raise
    QuadPublicUnsupported. order="linear"+planar=True must pass.
    """

    from anymesher.quad.options import QuadMeshingOptions
    from anymesher.quad.public_integration import (
        QuadPublicUnsupported,
        coerce_public_quad_options,
    )

    options = QuadMeshingOptions()

    with pytest.raises(QuadPublicUnsupported):
        coerce_public_quad_options(options, order="quadratic", planar=True)

    with pytest.raises(QuadPublicUnsupported):
        coerce_public_quad_options(options, order="linear", planar=False)

    # The accepted pair: no error, returns the same instance
    result = coerce_public_quad_options(options, order="linear", planar=True)
    assert result is options


# ---------------------------------------------------------------------------
# Test 7 — route_quad_first: None sentinel + ELEMENT_ORDERS contract
# ---------------------------------------------------------------------------

def test_linear_planar_route_is_the_qualified_public_contract() -> None:
    """Freeze dispatch and execute a real small linear planar quad-first mesh."""

    import anymesher
    from anygeometry.model import GeometryModel
    from anymesher.hybrid import generate_hybrid_mesh_result
    from anymesher.quad.options import QuadMeshingOptions
    from anymesher.quad.public_integration import route_quad_first

    options = QuadMeshingOptions(max_local_optimizations=0)

    assert route_quad_first(None) == (None, None)
    dispatched, report = route_quad_first(
        options, order="linear", planar=True, require_workers=False
    )
    assert dispatched is options
    assert report is not None
    assert tuple(anymesher.ELEMENT_ORDERS) == ("linear", "quadratic")

    geometry = GeometryModel()
    vertices = geometry.add_points(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )
    face = geometry.add_plate(vertices)
    before = (
        geometry.model_id,
        geometry.revision,
        tuple(sorted(geometry.vertices)),
        tuple(sorted(geometry.edges)),
        tuple(sorted(geometry.faces)),
        tuple(
            (vertex, tuple(float(x) for x in geometry.vertex_position(vertex)))
            for vertex in sorted(geometry.vertices)
        ),
    )

    result = generate_hybrid_mesh_result(
        geometry,
        target_size=0.5,
        face_ids=(face,),
        quad_options=options,
    )
    mesh = result.mesh
    diagnostics = mesh.hybrid_diagnostics
    assert diagnostics["route"] == "quad-first"
    assert mesh.quads
    assert diagnostics["q4"]["status"] in {
        "APPLIED",
        "NO_ELIGIBLE",
        "UNAVAILABLE_SKIPPED",
    }
    assert diagnostics["q5"]["status"] == "DISABLED"

    after = (
        geometry.model_id,
        geometry.revision,
        tuple(sorted(geometry.vertices)),
        tuple(sorted(geometry.edges)),
        tuple(sorted(geometry.faces)),
        tuple(
            (vertex, tuple(float(x) for x in geometry.vertex_position(vertex)))
            for vertex in sorted(geometry.vertices)
        ),
    )
    assert after == before


# ---------------------------------------------------------------------------
# Test 8 — Source geometry is unchanged by promotion
# ---------------------------------------------------------------------------

def test_source_geometry_is_unchanged_by_promotion() -> None:
    """Promotion adds midside nodes; it must not move or delete any source
    node, and the promoted quad must preserve the original four corner ids
    in the first four columns.
    """

    from anymesher.core import MeshCore
    from anymesher.surface_mesh import insert_midside_nodes

    original_pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    linear = MeshCore(node_coordinates=original_pts, quad_connectivity=np.array([[0, 1, 2, 3]]))
    promoted = insert_midside_nodes(linear)

    # Node count: original 4 + 4 midsides = 8
    assert promoted.node_coordinates.shape[0] == 8

    # Every original source node coordinate is preserved (first 4 rows)
    for i in range(4):
        assert np.allclose(promoted.node_coordinates[i], original_pts[i]), (
            f"source node {i} was moved: {promoted.node_coordinates[i]} != {original_pts[i]}"
        )

    # Corner ids in promoted element: first four unchanged
    assert list(promoted.quad_connectivity[0, :4]) == [0, 1, 2, 3]

    # No midside node coincidentally occupies a source node id
    corner_ids = set(int(x) for x in promoted.quad_connectivity[0, :4])
    midside_ids = set(int(x) for x in promoted.quad_connectivity[0, 4:])
    assert corner_ids.isdisjoint(midside_ids), (
        f"midside ids {midside_ids} overlap corner ids {corner_ids}"
    )


# ---------------------------------------------------------------------------
# Test 9 — Q9 and higher orders are deferred
# ---------------------------------------------------------------------------

def test_q9_and_higher_orders_are_deferred() -> None:
    """ELEMENT_ORDERS is exactly ('linear', 'quadratic') — no cubic, order-9,
    or curved token. The design doc explicitly names Q9 as deferred.
    """

    import anymesher
    import pathlib

    # No hidden extra order tokens beyond linear and quadratic
    assert tuple(anymesher.ELEMENT_ORDERS) == ("linear", "quadratic")

    # The design document exists and names Q9 / higher orders as deferred
    docs_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "docs"
    design = docs_dir / "CURVED_QUAD_DESIGN.md"
    assert design.is_file(), f"missing design doc: {design}"
    text = design.read_text(encoding="utf-8")
    assert "Q9" in text, "design doc must name Q9"
    assert "deferred" in text.lower(), "design doc must state Q9 is deferred"
