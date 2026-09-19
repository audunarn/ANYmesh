# Quad-first design (Q0 freeze)

Status: **Q0 — contracts and reuse freeze.** This document records the Q0
design decisions and the exact invariants the later milestones must preserve.
It is not a runbook; the milestone scope lives in `QUAD_FIRST_FULL_PROGRAMME.md`.

## 1. What "quad-first" means here

Quad-first is **quad-driven construction over a valid background CDT**.

It is explicitly *not*:

- triangle-free construction (no attempt to build a mesh with zero triangles);
- renamed greedy recombination (no relabeling of an existing triangle mesh as quads);
- a new general-purpose parameterization solver (no global surface parameterization,
  no boundary-driven remeshing framework).

The construction pipeline (target architecture, frozen at Q0) is:

```
structural preparation (existing, unchanged)
        │
        ▼
immutable quad input contract (new)
        │
        ▼
one valid CDT seed (single O(N) pass; NOT per-front)
        │
        ▼
planar cross-field guidance z = (cos 4θ, sin 4θ) + boundary tangent anchors
        │
        ▼
mixed local working state: residual T3 + accepted Q4, local incidence, front halfedges
        │
        ▼
front edge → bounded proposals (existing side/top edge → recovery → insertion)
                    → stage local cavity/Q4/T3 → validate → optionally optimize
                    → commit atomically OR roll back
        │
        ▼
explicit all-rejected / no-progress path → transition template / qualified closure /
                                            typed unsupported / one bounded fallback
        │
        ▼
independent global validation, S3 admission, component publication
```

## 2. Hard invariants (Q0 frozen)

Preserved from the existing codebase and the programme brief; **not** renegotiable
in Q1-Q7:

1. **ANYgeometry** remains authoritative for geometry and topology. We never
   re-derive node/edge/face identity from scratch.
2. **Exact shared node IDs** and **exact parameter stations** are preserved across
   components and through quad-front edits.
3. **Reversed shared-edge use** is preserved (component-level ownership, not
   global graph direction).
4. **Sheet ownership, material/thickness partitions, component transactions**
   are preserved.
5. **No coordinate welding.** No silent input repair. Degenerate overlaps are a
   typed rejection, not a merge.
6. **The `NativeMeshingOptions` schema is unchanged.**
   We add a *separate* `QuadMeshingOptions` (see §3). `native_v2.py` and its
   schema string `anymesher.native-meshing-options/1` are not touched.
7. **The `_native` extension stays C++17 and ABI-stable.** The quad-first route
   lives in a separate, scoped native module (Q6) that links to `_native` read-only
   and never changes its exported symbols.
8. **Determinism.** Ordering, tie-breaking, and transaction commit order are
   deterministic. The `QuadMeshingOptions.canonical_digest` is a stable SHA-256
   over the canonical JSON of the options object so callers can pin behaviour.
9. **Cancellation.** A cancellation callback can be invoked at front iteration,
   metric pullback, and optimization-step granularity (the latter two mirror the
   existing `native_v2` cancellation discipline).
10. **Typed failure and rollback.** A front edge that fails all candidates must
    either commit a transition template, close a qualified S3 patch, or return a
    *typed* unsupported-rejection. There is **no hidden fallback counted as
    quad-first success**.
11. **No helper triangles beneath active quads.** Residual T3 elements must be
    independently qualified S3 closures; they cannot be a "glue" beneath an
    accepted Q4.

## 3. Public API (Q0 frozen)

```python
# anymesher.quad.options
QUAD_MESHING_OPTIONS_SCHEMA = "anymesher.quad-meshing-options/1"

@dataclass(frozen=True)
class QuadMeshingOptions:
    seed_mode: str = "cdt"                       # Q0: only "cdt"
    orientation: str = "cross_4theta"            # or "boundary_tangent"
    quality_model: str = "shape"                 # or "shape_jacobian"
    line_search: str = "safeguarded"             # or "off" (requires max_local_optimizations=0)
    max_front_iterations: int = 100_000          # positive int
    max_local_optimizations: int = 64            # non-negative int
    cancellation_interval: int = 256             # positive int

    @property
    def canonical_digest(self) -> str: ...       # SHA-256 of canonical JSON
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, raw) -> "QuadMeshingOptions": ...   # strict field set + schema tag
    @classmethod
    def coerce(cls, value) -> "QuadMeshingOptions | None": ...
```

- `coerce(None)` returns `None` → *legacy path*.
- `coerce(QuadMeshingOptions)` returns the instance (frozen).
- `coerce(dict)` parses strictly and returns an instance.
- `coerce(other-type)` raises `MeshError`.

The strict `from_dict` mirrors `NativeMeshingOptions.from_dict` (exact field-set
+ schema-tag), so downstream serialization is symmetric.

### Entry-point plan (Q6; not added at Q0)

`quad_options: QuadMeshingOptions | None = None` on the appropriate existing
hybrid/surface entry points. When `None`, every existing call site executes the
byte-identical current path. When set, the quad-first route runs the
construction pipeline above.

### Capability errors (Q6)

- Missing/native quad library → `MeshError` typed as "capability missing" for
  the explicit request (not a silent fallback).
- Broken-present library / invariant failure / cancellation → same as today:
  a **real** failure, not an absence.

## 4. Resource guards (Q0 frozen)

| field                   | frozen default | bound meaning                                              |
| ----------------------- | -------------- | ---------------------------------------------------------- |
| `max_front_iterations`  | 100 000        | total front-edge processing steps before typed failure       |
| `max_local_optimizations` | 64           | local TinyAD patch-optimization iterations per accepted Q4   |
| `cancellation_interval` | 256            | cancel-check granularity (front steps / optim. iterations)   |

These are the only *tunable* numbers the user may set at Q0. The other fields
(`seed_mode`, `orientation`, `quality_model`, `line_search`) are closed enums
frozen at Q0 — adding new values in a later milestone bumps the schema tag
(`…/2`), not the field set.

## 5. Third-party reuse (Q0 freeze)

Authoritative pins are recorded in `third_party/quad/manifest.json`; per-license
text is reproduced in `third_party/quad/ATTRIBUTION.md`.

| id           | pinned commit | license                    | role at Q0                                        | Q0 vendor state |
| ------------ | ------------- | -------------------------- | ------------------------------------------------- | --------------- |
| `eigen`      | 3147391 (3.4.0) | MPL-2.0 (enforced `EIGEN_MPL2_ONLY`) | Dense linear algebra backbone | header-only 337 files |
| `tinyad`     | 4b48d1a       | MIT                        | reverse/forward AD for bounded local Q4 optims    | header-only 25 files |
| `libsatsuma` | 4e96979       | MIT                        | MCF/bipartite reduction for count systems         | headers + LICENSE; **no .cc** (we use LEMON's MCF directly, verified by smoke) |
| `libtimekeeper` | f4f4866    | MIT                        | wall-clock + CPU reporting in solver loops        | header-only 8 files |
| `q-morph`    | bb79875       | MIT                        | **reference only** at Q0 (not compiled into smoke) | LICENSE only |
| `lemon`      | 3c6aa54       | Boost-1.0                  | `NetworkSimplex` + `ListDigraph` + `Arc/NodeMap`  | headers + bits/ + concepts/ + generated `config.h`/`export.h` |

### What Q0 specifically does NOT do

- Do not build a full Q-Morph front. The Q0 vendoring is the *attribution +
  reference contract* for Q1 (port) only.
- Do not enable LEMON's LP/MIP solver set (Soplex / Clp / GMP-LPK) — the
  generated `config.h` has them disabled.
- Do not enable libSatsuma's Gurobi-backed solvers (`Lib/.../Solvers/*Gurobi.hh`
  are not included in our header subset).
- Do not pull in QuadWild, VCGLib, Qt, OpenVolumeMesh, a Java VM, or a
  commercial solver. Banned by policy.
- No install-time network fetch. `third_party/quad/` is committed to the
  worktree; the build does not touch the network.

## 6. Build (Q0 scoped native skeleton)

`third_party/quad/smoke/` contains the two tiny smoke tests that exercise the
**exact** abstractions the production path will use:

| smoke file             | exercised API                                                  | result |
| ---------------------- | -------------------------------------------------------------- | ------ |
| `tinyad_smoke.cc`      | `TinyAD::scalar_function<1,double>`, `x_from_data`, `eval_with_gradient` | **PASS** (f=13.0, g=(2.0,-12.0)) |
| `lemon_mcf_smoke.cc`   | `LEMON::ListDigraph`, `NetworkSimplex`, `ArcMap`/`NodeMap`, `totalCost<int>` | **PASS** (4-node CDT-reachable MCF, cost=6) |

Both compile under MSVC 14.50.35717 / Windows SDK 10.0.26100.0 / C++17, with
`-DEIGEN_MPL2_ONLY` enforced (see `third_party/quad/smoke/build.bat`). No
`_native` ABI is altered at Q0; the `_native` module is not even linked from
these two smokes.

```
third_party/quad/
├── ATTRIBUTION.md
├── manifest.json
├── licenses/
│   └── LICENSES.md                 # index → per-component LICENSE + manifest + ATTRIBUTION
└── smoke/
    ├── build.bat
    ├── build_lemon.bat
    ├── lemon_mcf_smoke.cc
    └── tinyad_smoke.cc
```

## 7. Scope of Q0 (and what is *not* done at Q0)

Done in Q0 (this milestone):

- [x] Pin, hash, vendor the six donor components.
- [x] `manifest.json` + `ATTRIBUTION.md` + `licenses/LICENSES.md`.
- [x] `QuadMeshingOptions` strict schema (`src/anymesher/quad/options.py`).
- [x] Smoke tests (TinyAD AD + LEMON MCF), both PASS under `EIGEN_MPL2_ONLY`.
- [x] `WORK_STATUS.md` (this evidence log), plus focused tests in `tests/quad_first/`.

Explicitly **not** Q0 (deferred to Q1+):

- Front state, primitive ports, recovery, insertion, transition templates.
- TinyAD local optimization on real Q4 patches.
- libSatsuma MCF count-system solver integration (Q4).
- Full cross-field guidance and low-confidence diagnostics (Q3).
- S3 qualification + component publication (Q6).
- Any change to the existing `_native` ABI or the legacy `NativeMeshingOptions`.
