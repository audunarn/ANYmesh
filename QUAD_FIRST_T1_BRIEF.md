# ANYmesher Quad-First T1 / M1 Execution Brief

Technical lead: ChatGPT Astra chat mode. Implementation lead: OpenCode. Primary worker: `ollama/qwen3.8:27b`.
Baseline: `2ccef378c3efb4ba3a9957b9ab896ac8fc454b9d`; branch/worktree: `opencode/quad-first-v1`.

Goal: deliver a real opt-in planar advancing quad front, not scaffolding, mapped delegation, or simple triangle pairing.
Scope: Q0 dependency/source audit + necessary Q1 mixed Q4/T3 state + smallest complete Q2 front vertical slice. Q3-Q7 are later unless explicitly authorized.

Hard invariants: preserve existing mapped/native defaults and schemas; ANYgeometry owns geometry/topology; exact shared node IDs and parameter stations; no coordinate welding; preserve S3 admission, cancellation, deterministic behavior, and component-level transactional publication. Keep existing C++17 `_native` ABI untouched.

Reuse: selectively port/adapt MIT Q-Morph donor logic (`KarlLevik/qmorph`, pinned and attributed). Audit/pin MIT libSatsuma and TinyAD plus their dependency licenses. No production Java, Gurobi, Blossom V, QuadWild, installation-time downloads, or broad ABI rewrite. Exercise bounded libSatsuma/TinyAD smoke calls only if feasible; do not claim later production integration.

Implementation: reuse one valid background CDT; maintain accepted Q4 + residual T3 topology; implement a front operation that genuinely recovers/inserts edges; support atomic rollback; bound candidate/recovery/no-progress loops. Produce complete small hole-plate and concave-plate examples with exact protected boundaries, real front work, and valid residual T3 closure.

Verification: focused tests + relevant existing regressions; add cavity, invalid-input, all-rejected, parity, ownership, cancellation, rollback, and native-boundary tests; exercise a genuine recovery/insertion case; perform fresh-context diff/contract review and fix material findings before completion.

Resources: routine fixtures only, approximately 1k-10k active elements. No 100k needed for T1. Never run 500k/workstation/equivalent renamed/aggregate workloads. Do not run the slow 10k hole diagnostic.

Git: preserve unrelated work; commit coherent accepted changes only on this task branch; do not push, PR, update main, tag, release, publish, reset, force-push, or destructively clean without explicit authorization.

Operate autonomously through ordinary implementation/test/debug/review cycles. Escalate only genuine architectural ambiguity, missing essential information after bounded investigation, destructive action, ownership risk, or unresolved acceptance blocker. Keep `reports/quad_first/<task-id>/WORK_STATUS.md` current.

Handoff: report milestone status, commits, changed files, actual donor/library use, architecture choices, exact test commands/results, review findings/fixes, small-case timings/counts, limitations, uncommitted changes, and process inventory. Do not claim full-plan completion.