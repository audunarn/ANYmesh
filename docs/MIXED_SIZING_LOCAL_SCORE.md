# Mixed-sizing local score (slice b)

`_physical_mixed_sizing._score_local(base, base_quality, candidate, settings, stats)` returns exactly
what `_score(candidate, settings)` returns, but computes cell quality only for the active rows that are
new, newly active or whose connectivity differs from the base mesh (`_recompute_rows`); all other rows
reuse the base values (`_spliced`). Validity is certified by `quality_v2._certified_linear_validity`
with the spliced metrics (`_quad_metrics`). Anything unusual (changed node prefix, shape change,
uncertified validity) falls back to the full `_score`. Counters live in an explicit `_ScoreStats`
object passed by the caller (`score_stats=`); there is no module state.

Measured (`mixed_growth_debt_44.npz`, 10,798 cells, `insertion_remaining=2`, Python quality path):
origin/main 94.2 s -> 8.4 s, identical output hash.

## Admission contract (honest scope)

`surface_mesh._physical_pairing_services()` pins the code, defaults and identity of each listed
entry-point function and the identity of the declared module objects. A replaced entry point or
declared object makes `repair_mixed_sizing` ineligible. It is not a defence against monkeypatching
of anything else. Outside the contract: builtins and their shadowing by module globals, numpy
functions (`np.cross`, `np.dot`, ...), class internals (methods, `__init__` defaults,
`__post_init__`), and native flags (`native_cpp.COMPILED_QUALITY_PIPELINE_AVAILABLE`). The same
limits already held for `triangle_quality` and the other pinned helpers.
