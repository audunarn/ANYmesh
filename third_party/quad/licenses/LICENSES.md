# Quad-first licence and attribution index

Each vendored component ships its own co-located `LICENSE` file under
`../vendor/<id>/`. The authoritative record of pins, archive SHA-256s,
per-tree SHA-256s, licences, and build flags lives in
[`../manifest.json`](../manifest.json). A human-readable summary with the
MPL-2.0 text (for Eigen) and the full MIT text (reproduced once for the record)
lives in [`../ATTRIBUTION.md`](../ATTRIBUTION.md).

| id            | pinned commit             | licence                 | verbatim text                    |
| ------------- | ------------------------- | ----------------------- | -------------------------------- |
| `eigen`       | `3147391` (3.4.0)         | MPL-2.0 (enforced `EIGEN_MPL2_ONLY`) | `../ATTRIBUTION.md` (Eigen section) |
| `tinyad`      | `4b48d1a1a588874556a692a3abbdecd0db4c23e1` | MIT | `../vendor/tinyad/LICENSE`     |
| `libsatsuma`  | `4e96979ecb11bbfe8d9c05e8f8be1ecb992ca5fd` | MIT | `../vendor/libsatsuma/LICENSE` |
| `libtimekeeper` | `f4f486648faac7c740535678b45b387078e26918` | MIT | `../vendor/libtimekeeper/LICENSE` |
| `q-morph`     | `bb798752b998489497f4f38b5989c42d13ac2bc0` | MIT | `../vendor/q-morph/LICENSE`    |
| `lemon`       | `3c6aa54c62a9b524bb9502872fa172776c1f8244` | Boost-1.0 | `../vendor/lemon/LICENSE`     |

Patches: **none** at Q0. No component was patched.
