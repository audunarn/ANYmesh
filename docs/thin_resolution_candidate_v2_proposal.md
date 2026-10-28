# Candidate v2 — explicit five-target-spacing input policy

State: **draft; no native execution authority**. The first study remains NO_GO
for its rotated offset case at h/thickness0.25. Its two-call limit is exhausted;
unused process time cannot be spent on more meshing.

Proposed design delta: support a narrower thin sampling input envelope with
explicit requested h<=thickness/5, preserving the actual output requirement
max axial cell span/thickness<=0.5 and condition<=32. This adds an input margin;
it is not a guaranteed TetWild edge bound and never automatically changes a
caller's sizing or accepts failed output. A later production contract must
reject requests outside the qualified sampling envelope explicitly.

Fresh qualification would reuse both immutable source geometry snapshots and
their authoritative face/region identities, changing only the explicit sizing
request from0.0025m to0.002m. Freeze the delta, source/input/backend/runtime
identities and all original tolerances before execution. This also tests that
refinement preserves geometry and groups. One call per orientation, two maximum,
no retries/automatic sizing loop. Keep100000cells/150000coefficients,120s native,
480s complete case and a separate1200s supervised budget including preparation.
Neither the former two-call registration nor its remaining time is reused.

Each case must pass production validity/coverage/ownership, unchanged volume/
area/normal/physical-delta limits, strict sampling and conditioning, and all
unchanged P1/P2 finite-case checks. Any failure leaves the revised envelope
unqualified. Larger performance/accuracy claims remain separate. The proposal
is ready for an explicit next-candidate execution decision; no such new run was
started as part of closing the failed first study.
