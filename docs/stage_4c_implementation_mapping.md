# Stage 4C — Implementation Mapping (Reconnaissance Findings, Frozen)

**Status:** Frozen at commit (this document's own commit) on `claude/integrated-solution-analysis-atdfe6`. **No logging code has been written for this stage.** This document records the implementation-reconnaissance findings and the resulting amendments to the frozen Stage 4C schema; it is the boundary between what the live production system actually exposes and what the logger subsequently implements.
**Builds on:** `docs/stage_4c_prospective_validation_schema_design.md` (the frozen Decision/Execution/Outcome Log schema). Nothing in that document's methodology is invalidated by this mapping — the validation hierarchy, the append-only three-log architecture, and the Tier A/Tier B boundary all stand. What changes here is strictly the implementation-level correspondence between the schema's fields and the real codebase.
**Next step:** a logger implementation plan, informed by this mapping. No additional analytical stage is needed between this document and that plan.

## 1. Objective

The Stage 4C schema was designed against an idealized "decision event" abstraction. Before writing any logging code, the live engine and pipeline code (`extrafloat/engine/`, `run_credit_risk_pipeline.py`, `pd_model/`, `data/agent_profile_snapshot_query.sql`, `data/loan_state_query_updated_materialized.txt`) were read directly to establish what that abstraction actually corresponds to in production. Four real mismatches were found. Each is resolved below as a frozen amendment to the implementation mapping — not a reopening of the Stage 4C schema's methodology, and not something to leave for the logger's author to improvise.

## 2. Conclusion 1 — Logging Grain: `agent x scoring_run`, Not a Per-Agent Decision Event

**Finding.** `run_credit_risk_pipeline.py` is a batch, agent-grain scorer: Stage 6 (`run_extrafloat_limit_engine`) is called once per pipeline invocation against the full snapshot, producing one output row per `msisdn`, with a single `run_id` (`run_credit_risk_pipeline.py:535`, `_dt.datetime.utcnow().isoformat() + "Z"`) shared by every agent in that run. There is no per-agent, per-event timestamp or identifier anywhere in the live path — "decision_timestamp" as originally specified would have fabricated the appearance of an individual, agent-triggered credit-request event at a specific moment, when what actually happened is a scheduled batch refresh of every agent's limit at once.

**Frozen amendment.** The Decision Log's grain is redefined as **one row per `(agent_msisdn, scoring_run)`**, not a fictional per-agent decision event:

- `decision_id` (Section 4.1 of the frozen schema) is retained as a column name for backward compatibility with that document, but its *semantics* become a **shadow scoring-record id**: `shadow_decision_id = f(run_id, agent_msisdn)` for a deterministic `f` (e.g. a stable hash), with a uniqueness constraint on `(run_id, agent_msisdn)`.
- The hash is a convenience key only. `run_id` and `agent_msisdn` are **also** logged as separate, independent fields — the hash is never the sole record of provenance, and nothing may rely on inverting it.
- `decision_timestamp` becomes **`scoring_run_timestamp`**: the batch run's own timestamp, identical for every agent scored in that run. It must not be presented or read as though it were an individual, per-agent decision moment.
- The natural execution relationship this grain implies: `S_{i,t} -> {L_{i,t,1}, L_{i,t,2}, ..., L_{i,t,n}}` — a scoring state `S_{i,t}` (one row: this agent, this run) governs zero or more subsequent loan executions, until the next scoring run `S_{i,t+1}` supersedes it. This matches the business architecture already established in Analysis 3/Stage 4A: a slow-moving state followed by potentially several loan episodes, not a one-for-one event pairing.

This is a terminology/grain amendment, not a change to what Tier A validates or how the three-log architecture works — Section 3 of the frozen schema's append-only, immutable-Decision-Log discipline is unaffected.

## 3. Conclusion 2 — Execution Attribution: 0..N Disbursements, Temporally Attributed to the Latest Prior Scoring State

**Finding.** `disbursement_fid` / `disbursement_ts` (`data/loan_state_query_updated_materialized.txt:93,99,130`) are genuine, independent loan-grain facts, entirely decoupled from any scoring run. Because the engine re-scores the whole agent population periodically rather than per-request, a single scoring state can legitimately govern many disbursements before the next run replaces it — the Execution Log's 0..N cardinality (already in the frozen schema) is not just schema-level permissiveness; it is the real production shape.

**Frozen amendment.** Because the grain is now `agent x scoring_run`, attributing a given execution to "the" governing scoring state requires an explicit, deterministic temporal rule rather than a direct foreign key chosen at execution time:

> For agent *i* and execution *e* with timestamp `t(e)`, let `S_{i,1}, S_{i,2}, ...` be agent *i*'s scoring states in chronological order of `scoring_run_timestamp`. Execution *e* is attributed to the unique `S_{i,j}` such that `t(S_{i,j}) <= t(e) < t(S_{i,j+1})` (or `j` is agent *i*'s latest scoring state at the time of `e`, if no later state exists yet).

This is the prospective counterpart of the same historical attribution discipline already established for Phase 2.1/2.2 PIT reconstruction (Stage 4A.1) — "latest state strictly before the event, never a later one."

- The link is **logged at write time, not re-derived later**: `execution.shadow_decision_id` is written when the execution record is created, resolved via the rule above at that moment.
- Hard-fail invariant: `execution_ts >= scoring_run_timestamp` of the referenced `shadow_decision_id` (mirrors the frozen schema's existing `disbursement_timestamp >= decision_timestamp` invariant, restated in the new grain's terms).
- Hard-fail invariant (new): no scoring state for the same agent with a `scoring_run_timestamp` strictly between the referenced state's and `execution_ts` may exist — i.e. the attributed state really is the *latest* one governing at execution time, not merely *a* valid one.

This strengthens, rather than weakens, the frozen schema's Section 6 invariant set.

## 4. Conclusion 3 — Capacity-Input Gate I0: PASS (Static-Code), Operational Freshness Not Yet Confirmed

**The gate.** Before logging any `capacity_shadow_k0_05`-type value, it must be established that the exact frozen Float/Commission definitions underlying the `k=0.05` selection (Stage 4B) can be generated live, strictly point-in-time, with identical units, and deterministically — not reinvented inside the logger. Re-deriving the formula independently inside a logging writer would risk validating a *different*, unversioned `Capacity(F_new)`, not the `Capacity(F)` actually selected at `k=0.05`.

**Findings, each with file:line evidence:**

1. **Commission** is a raw, materialized column: `data/agent_profile_snapshot_query.sql:288`, `COALESCE(p.commission, 0) AS commission`.
2. **`float_activity_value_1m`** is not raw — it is `cash_in_value_1m.fillna(0) + payment_value_1m.fillna(0)`, and this formula has exactly one definition in the entire codebase: `scripts/build_loan_episode_capacity_dataset.py:191-192`. Both raw inputs are materialized by the same query: `cash_in_value_1m` (`agent_profile_snapshot_query.sql:208`), `payment_value_1m` (`agent_profile_snapshot_query.sql:229`).
3. **This is the same query** that produces the live `--transaction-file` consumed by `run_credit_risk_pipeline.py:307` (`df_agent = pd.read_csv(transaction_file)`) — confirmed by that query's own header, which documents it as the shared source for both PD Phase 2.1 and the ExtraFloat engine's transaction-capacity input. `df_agent` is the **raw** read, before `load_transaction_capacity_features()` reshapes it into `features_df`; `cash_in_value_1m`, `payment_value_1m`, and `commission` are present on `df_agent` untouched on every live run, even though the engine's own feature loader (`extrafloat/io/extrafloat_data_loaders.py`) never extracts them downstream for its own (different) capacity formula.
4. **PIT at `scoring_run_timestamp`.** The 1-month window anchors to `dt_m1 = MAX(tbl_dt) - 1 month` (`agent_profile_snapshot_query.sql:71,97`) — a batch-relative reference, computed off whichever file is loaded. **This is the exact same formula shape Stage 4A flagged as a leakage risk for historical reconstruction** (an old loan asking "the latest date in the file" silently gets today's date instead of its own). It is explicitly **not fixed** here, and must not be described that way. What differs is the *setting* it is being asked to answer: for live scoring, the file being queried at run time genuinely is the current extract, so `MAX(tbl_dt)` corresponds to "now," not a wrong answer dressed as a right one. **Historical reconstruction and prospective execution have different PIT properties under the identical formula** — the formula was never the problem in the historical case; using it against a file that does not contain the row's own true "now" was.
5. **Units/scaling** are identical to the `k=0.05` selection by construction: this is the literal query definition Stage 4B's research scripts already read from, not a parallel pipeline that happens to share column names.
6. **Deterministic reproduction**: `fillna(0) + fillna(0)` over two already-materialized columns is pure arithmetic, no stochastic step.

**Verdict, stated precisely:**

> **Static-code Gate I0: PASS.** The checked-in query and pipeline code establish that the exact frozen Float/Commission definitions used to select `k=0.05` can be mechanically reconstructed from `df_agent` at live scoring time, point-in-time by construction, with identical units, deterministically.
> **Operational freshness: NOT YET CONFIRMED.** This repository cannot see whether the production ETL job that produces the live `--transaction-file` extract actually re-runs `agent_profile_snapshot_query.sql` freshly before each scoring cycle, versus serving a stale or cached extract. That is an operational fact about infrastructure outside this codebase, not a code-level finding — it must be confirmed with whoever owns the live scoring job before Gate I0 can be treated as fully closed for deployment purposes. Until confirmed, treat capacity-input freshness as a deployment prerequisite, not an established fact.

**Implementation consequence.** The shadow logger must source Float/Commission from the raw `df_agent` snapshot DataFrame inside `run_credit_risk_pipeline.py`, never from `features_df` (the engine's already-transformed feature set, which does not carry these columns through) and never by independently re-deriving the formula from a different input path.

**Availability-semantics nuance (preserved from the frozen schema, not weakened by this finding).** Gate I0's `fillna(0) + fillna(0)` determinism describes the **arithmetic**, not the **availability bookkeeping**. The frozen schema (Section 4.2) deliberately separates `MISSING` from `NONPOSITIVE` for exactly this reason: zero-filling before that distinction is recorded would silently erase whether a value was genuinely absent (`MISSING`) versus present-but-non-positive (`NONPOSITIVE`). Implementation must therefore:
1. First inspect the raw, pre-fill values of `cash_in_value_1m`, `payment_value_1m`, and `commission` on `df_agent` to set `float_availability_status` / `commission_availability_status` (per the frozen schema's three-orthogonal-status design).
2. Only *then* apply the frozen zero-fill arithmetic to compute `float_activity_value_1m` for the capacity formula itself.
Reversing this order — zero-filling first, deriving availability status from the already-filled value — would make `MISSING` and a genuine `0` indistinguishable, defeating the reason the schema separated them.

## 5. Conclusion 4 — Risk-Signal Naming: Rename the Stage 4C Field, Never the Production Field

**Finding.** The live engine already outputs a column literally named `risk_score` (`docs/engine_output_data_dictionary.md`, "Risk Score and Tier Details"), confirmed in code at `extrafloat/engine/extrafloat_limit_engine_caps.py:353-358`: on the `cal_pd` path, `risk_score = 1 - cal_pd` (clipped to `[min_score, max_score]`) — **high = safe**, the opposite orientation from a PD-like quantity, and a different number from Stage 4C's proposed new field (route-specific risk signal, **high = risky**, same direction as `cal_pd`).

**Frozen amendment.** The Stage 4C field is renamed to avoid the collision; production's existing field keeps its existing name and meaning unconditionally:

- `risk_score` / `risk_score_type` (as specified in the frozen schema's Section 4.4) are renamed to **`challenger_risk_signal`** / **`challenger_risk_signal_type`**, with the same enum values (`CALIBRATED_CHAMPION_PD` / `THIN_SCORECARD_PD_LIKE`) and the same nullability rules as previously specified for `risk_score` / `risk_score_type`.
- Production's `risk_score` (`= 1 - cal_pd` on the live path) is retained under its existing name, optionally logged for audit alongside `cal_pd` and `challenger_risk_signal`.
- **New hard-fail invariant for thick rows**: `risk_score_live + cal_pd = 1` within numeric tolerance — a direct, free consistency check that falls out of the two quantities' known relationship, catching a drift in either value's computation.
- The rename resolves the collision without touching production code or terminology — "the shadow schema adapts to established production semantics," not the reverse.

## 6. Conclusion 5 — `execution_status`: Removed

**Finding.** The only status-like field found anywhere in the loan data model is `loan_status` (`data/loan_state_query_updated_materialized.txt`, values including `NEVER_BORROWED`, `OPEN`, `ANOMALY_OPEN`, `SETTLED`, `CLOSED`, `OVERPAID`) — a **post-disbursement lifecycle** state (what happened to a loan that was successfully created), not a **payment-attempt outcome** (whether a disbursement itself completed, failed, or was reversed). No evidence exists anywhere in this repository's data model of a disbursement that failed, was cancelled, or was reversed at the point of attempted execution.

**Frozen amendment.** `execution_status` is **removed** from the Execution Log as specified in the frozen schema (Section 5). The existence of an Execution Log row already carries exactly the semantics needed: a loan/disbursement was observed in the source. `loan_status` must not be substituted in its place — it answers a different question (loan lifecycle/outcome) and belongs, if used at all, to Outcome Log construction, never to the Execution Log. If a future upstream system exposes genuine disbursement-attempt/reversal events, that would justify introducing a separate execution-event concept at that time — not now, and not as a speculative placeholder.

## 7. Summary Table

| Finding | Resolution | Analytical impact on frozen Stage 4C methodology |
|---|---|---|
| Batch, not per-event, scoring | Decision Log grain -> `agent x scoring_run`; `decision_id`/`decision_timestamp` semantics become `shadow_decision_id`/`scoring_run_timestamp` | None — terminology/grain fix only |
| Capacity inputs absent from the live engine's own feature set | Gate I0: PASS (static-code); source from raw `df_agent`, never re-derive independently; availability status computed pre-fill | Material implementation prerequisite, now resolved; operational freshness remains a deployment-time confirmation, not a code finding |
| `risk_score` naming collision with production | Rename Stage 4C field to `challenger_risk_signal` / `challenger_risk_signal_type`; add a `risk_score_live + cal_pd = 1` invariant | None — naming fix plus a free new invariant |
| No execution-attempt status concept in the data | Remove `execution_status` from the Execution Log | None — removes a speculative field |
| 0..N executions per scoring state confirmed real | Explicit temporal attribution rule (`t(S_j) <= t(e) < t(S_{j+1})`), logged at write time as `execution.shadow_decision_id`, with two new hard-fail invariants | Strengthens the frozen schema's existing invariant set |

## 8. What This Document Does Not Do

No logging code has been written. No change has been made to any production file (`pd_model/`, `extrafloat/engine/`, any SQL). The frozen Stage 4C schema's validation hierarchy, three-log architecture, and Tier A/Tier B boundary are unchanged — this document amends only the implementation-level correspondence between that schema's fields and the real codebase, per the five conclusions above. The next step is a logger implementation plan informed by this mapping, followed by synthetic/invariant verification before any live shadow deployment — no additional analytical stage is needed between this document and that plan.
