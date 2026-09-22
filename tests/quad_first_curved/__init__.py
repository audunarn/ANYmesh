"""CH0 baseline-contract tests for curved and higher-order quad meshing.

These tests freeze the contract that later tranches (CH1 promotion, CH2 curved
geometry) must honour. They are documentation-as-code: they pass now, require no
worker binary and no compiled native module, and must keep passing unless the
specific freeze is explicitly re-opened.

See docs/CURVED_QUAD_DESIGN.md and docs/CURVED_QUAD_ACCEPTANCE.md.
"""
