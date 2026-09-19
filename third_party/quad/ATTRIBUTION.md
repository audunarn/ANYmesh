# Quad-first third-party attribution (Q0 freeze)

This directory vendors, at pinned commits, the third-party abstractions that the
quad-first meshing programme may reuse. Every vendored source keeps its upstream
license text co-located under its own `vendor/<id>/` tree. See `manifest.json`
for the authoritative pins: repo, pinned commit, archive SHA-256, per-tree
SHA-256, and file counts.

## Eigen — MPL-2.0 pin (the programme's required choice)

ANYmesher itself is MPL-2.0. Eigen 3.4.0 is tri-licensed (MPL-2.0 /
LGPL-3.0-or-later / BSD-3-Clause). For compatibility with ANYmesher's MPL-2.0
license the programme explicitly selects **MPL-2.0** and pins the
`EIGEN_MPL2_ONLY` build flag:

- `EIGEN_MPL2_ONLY` is *enforcement*, not an opt-in. When defined, any Eigen
  header that is not MPL-2.0-compatible raises `#error Including non-MPL2 code
  in EIGEN_MPL2_ONLY mode` (`Eigen/src/Core/util/NonMPL2.h`).
- It is therefore **mandatory** in any compilation of the vendored headers in
  this tree; the flag is recorded in `manifest.json` → `build.mandatory_compile_flags`.
- The pinned Eigen source is commit `3147391` (release 3.4.0). The archive
  SHA-256 and the per-tree SHA-256 of `vendor/eigen/` (337 files) are recorded
  in `manifest.json`.

> Eigen copyright (full MPL-2.0 text is in the upstream 3.4.0 distribution and
> must be reproduced verbatim in any downstream bundle; the vendored
> header-only subset here is code, not a license file, so the MPL-2.0 text is
> kept in the record via this attribution + `manifest.json` pin).
>
> Copyright (C) 2008-2022 Gael Guennebaud <gael.guennebaud@inria.fr> and the
> Eigen contributors.
>
> This Source Code Form is subject to the terms of the Mozilla Public License,
> v. 2.0. If a copy of the MPL was not distributed with this file, You can
> obtain one at https://mozilla.org/MPL/2.0/.

## Permissive components (MIT)

- **TinyAD** — MIT. Copyright (c) 2022 Patrick Schmidt. Full text in
  `vendor/tinyad/LICENSE`.
- **libSatsuma** — MIT. Copyright (c) 2023 Martin Heistermann. Full text in
  `vendor/libsatsuma/LICENSE`.
- **libTimekeeper** — MIT. Copyright (c) 2022-2023 Patrick Schmidt. Full text
  in `vendor/libtimekeeper/LICENSE`.
- **Q-Morph** — MIT. Reference-only at Q0 (not compiled into the smoke path).
  Full text in `vendor/q-morph/LICENSE`.

MIT license text (reproduced for the record; identical in all four files):

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Boost-licensed component

- **LEMON** — Boost-1.0. Full text in `vendor/lemon/LICENSE`. Boost-1.0 is a
  permissive license (with an explicit, generous patent licence) that is
  compatible with ANYmesher's MPL-2.0 policy.

## Exclusions (per `QUAD_FIRST_FULL_PROGRAMME.md` policy)

- **QuadWild** — out of programme scope.
- **Gurobi** (commercial) — banned from the runtime path.
- **Blossom-V** — not the selected matching kernel.
- **Soplex / Clp / GMP-LPK** — LP solver suite; disabled in LEMON `config.h`.
- **Java runtime** — banned dependency by policy.
