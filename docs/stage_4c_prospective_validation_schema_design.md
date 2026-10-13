# Stage 4C — Prospective Validation Schema Design

**Status:** Design frozen at commit (this document's own commit) on `claude/integrated-solution-analysis-atdfe6`. **No code has been written for this stage.** This document specifies the logging architecture and schema only; implementing it against the live engine is explicitly the next, separate step.
**Builds on:** `docs/stage_4b_capacity_k_selection_findings.md` (freezes `k_shadow=0.05`, `k_sensitivity=0.04/0.06`, and the `Capacity(F) -> M_C3(PD) -> L_full -> product policy` architecture this stage logs prospectively).
**Revision note:** this revision incorporates ten review corrections to the originally-committed design (Outcome Log re-keying, thin-file risk-score semantics, PIT/formula/source provenance, and five smaller schema refinements). Per explicit instruction, scope was held to exactly these ten points — this is the schema freeze, not a reopening of the broader design.

## 1. Objective

Stage 4B froze the capacity-scale decision entirely from retrospective, consequence-based analysis. The explicit next step recorded there was prospective validation, not further retrospective tuning. This document establishes, before any logging code is written, (a) what prospective shadow logging can and cannot validate, and (b) the exact schema for the data that will let us answer the questions it *can* answer.

## 2. The Validation Hierarchy

The central methodological risk at this stage is treating an *observational* association between the challenger's output and subsequent outcomes as evidence that *acting on* the challenger's output would be safe — when no agent has actually received the challenger's recommended exposure. An agent's observed repayment behavior only ever reflects the exposure they *actually* received (their current-policy limit), never the counterfactual `L_full`. This is the same endogeneity concern that has governed this entire workstream (never fitting `Capacity(F)` to historical `assigned_limit`), arriving at the logging stage.

| Question | Shadow logging | Matured outcomes | Controlled exposure |
|---|---|---|---|
| Pipeline reproducible/stable? | Yes | - | - |
| Fundamentals/capacity stable? | Yes | - | - |
| Production coverage acceptable? | Yes | - | - |
| C3 mechanically correct (implementation verification)? | Yes | - | - |
| Live PD prospectively discriminates risk (`PD_t -> Y_{t+30}`)? | Yes | Required | - |
| Challenger/current disagreement characterized? | Yes | - | - |
| Outcomes differ across disagreement groups? | Yes (descriptive) | Required | - |
| Challenger caused better/worse outcomes? | No | No | Required |
| Higher challenger exposure is safe? | No | No | Required |
| Challenger improves portfolio economics? | No | No | Required |

**Tier A** (this document's scope): implementation verification, stability, coverage, prospective PD discrimination, disagreement characterization, and decision-provenance reproducibility. All directly observable from passive logging; none requires changing anyone's exposure.

**Tier B** (explicitly out of scope, a later and separate undertaking): causal validation of the challenger's recommended exposure itself. Requires genuine controlled exposure variation (a pilot) with real downside-risk monitoring, since it means intentionally extending some agents more credit than current policy would. Not designed here.

**Governing constraint on Tier A:** `k_shadow=0.05` and the C3 policy are frozen inputs to this stage, not outputs. Tier A answers "how does the frozen challenger behave prospectively," never "what k would look best after seeing the next month's outcomes." The wide, explicitly-labeled `_shadow_k0_05` / `_sensitivity_k0_04` / `_sensitivity_k0_06` column naming below (never a generic `k` column) is the schema-level mechanism enforcing this.

## 3. Architecture: Three Linked, Append-Only Logs

```
Decision Log  --(1)-->  Execution Log  --(0..N)-->  Outcome Log
   (1 row)                (0..N rows)                (0..N rows per execution)
```

- **Decision Log**: exactly what was known and computed at the moment of evaluation — business-state inputs, the frozen challenger's capacity/risk/recommendation outputs, current policy's decision, their characterized disagreement, and full artifact provenance. Written once per decision event, **never updated**.
- **Execution Log**: what credit was actually extended, if any. Zero or more rows per decision (a decision may be declined and never executed; some architectures may support multiple executions against one decision).
- **Outcome Log**: what subsequently happened, observed under the **actual** exposure. **Re-keyed in this revision**: joined to the Execution Log by `execution_id`, never directly to the Decision Log. Because the Execution Log explicitly allows 0..N executions per decision, a `decision_id`-only key on the Outcome Log would be ambiguous about which of several executions produced a given outcome row. The lineage chain is therefore three hops, each independently verifiable: `outcome.execution_id -> execution.decision_id -> decision.decision_id`. `decision_id` and `disbursement_fid` are still carried on the Outcome Log as denormalized convenience fields (so a reader need not always join through Execution), but they are redundant lookups, not keys — each must equal the referenced execution row's own `decision_id` / `disbursement_fid` (Section 6, invariant 7). The Decision-to-Execution-to-Outcome cardinality is not strictly 1:1:1: a single execution can accumulate more than one Outcome Log row over time if `outcome_definition_version` changes (Section 5).

This separation is what makes "why did this agent get this recommendation on this date, and which specific disbursement produced this outcome" permanently reconstructable, keeps point-in-time integrity (what was known at decision time vs. what became observable later), and lets the outcome-label definition itself evolve without rewriting history.

## 4. Decision Log

One row per live decision/evaluation event — **every evaluation, not only disbursements** (see `shadow_population_label` below for the fallback if the live system cannot expose non-disbursed evaluations).

### 4.1 Identity

| Field | Type | Null? | Notes |
|---|---|---|---|
| `decision_id` | string (UUID) | NOT NULL, PK | Generated at evaluation time. |
| `agent_msisdn` | string | NOT NULL | |
| `decision_timestamp` | timestamp (UTC) | NOT NULL | |
| `current_decision_status` | enum: `APPROVED` / `DECLINED` / `ZERO_LIMIT_ELIGIBLE` / `OTHER` | NOT NULL | Disambiguates `l_current=0` between declined, eligible-but-zero, and data-missing. |
| `decision_reason_code` | string | NULLABLE | Only if the current engine exposes one; never inferred. |
| `shadow_population_label` | enum: `FULL_EVALUATION_POPULATION` / `DISBURSEMENT_CONDITIONED_SHADOW_POPULATION` | NOT NULL | A deployment-level capability fact, logged per row defensively. The pipeline must not write `FULL_EVALUATION_POPULATION` unless the live system is confirmed, at deployment time, able to expose non-disbursed evaluations. Coverage/disagreement claims must carry whichever label applies — never presented as the full live-eligible population if the label says otherwise. |

### 4.2 Business-state inputs

| Field | Type | Null? | Notes |
|---|---|---|---|
| `fundamentals_snapshot_date` | date | NULLABLE | NULL iff no eligible prior snapshot exists. **Hard PIT invariant (Section 6, #9):** when NOT NULL, `fundamentals_snapshot_date < date(decision_timestamp)` — strict, same guard discipline used throughout this workstream (e.g. `attach_pretrade_fundamentals`'s `allow_exact_matches=False`). |
| `fundamentals_snapshot_id` | string | NULLABLE | Only if the source mart exposes a stable id; never fabricated. |
| `fundamentals_snapshot_available_at` | timestamp (UTC) | NULLABLE | New in this revision. Only populated if the source mart/pipeline exposes when the snapshot itself became available for consumption (distinct from the snapshot's own `fundamentals_snapshot_date`, which is a business-as-of date, not a publication time). When present, invariant `fundamentals_snapshot_available_at <= decision_timestamp` (Section 6, #9) — a snapshot the decision could not yet have consumed cannot have been used to make it. Never fabricated or back-filled when the source does not expose it. |
| `fundamentals_snapshot_age_days` | integer | NULLABLE | NULL iff `fundamentals_snapshot_date` is NULL. Derived/secondary only — `fundamentals_snapshot_date` and the hard PIT invariant above are the load-bearing facts; this field exists so snapshot-sharing across decisions (the Analysis-3 cadence-gate finding) stays visible rather than hidden, not as its own guard. |
| `float_activity_value_1m` | numeric | NULLABLE | |
| `commission` | numeric | NULLABLE | |
| `float_availability_status` | enum: `OK` / `MISSING` / `NONPOSITIVE` | NOT NULL | New in this revision, replacing the single `capacity_not_estimable_reason` enum. Distinguishes "the field was never populated" (`MISSING`) from "it was populated but <= 0" (`NONPOSITIVE`) — two different data-quality signatures the old combined enum could not separate. |
| `commission_availability_status` | enum: `OK` / `MISSING` / `NONPOSITIVE` | NOT NULL | Same distinction, independently, for `commission`. |
| `snapshot_availability_status` | enum: `OK` / `UNAVAILABLE` | NOT NULL | `UNAVAILABLE` iff `fundamentals_snapshot_date` is NULL (no eligible prior snapshot). Orthogonal to the two fundamentals-level statuses above — a snapshot can be available with a non-positive fundamental, or unavailable entirely. |
| `capacity_estimable` | boolean | NOT NULL | Single derived gate, kept for convenience and backward-compatible readability. **Derivation (Section 6, #11):** `TRUE` iff all three of `float_availability_status`, `commission_availability_status`, `snapshot_availability_status` equal `OK`; `FALSE` otherwise. Never set independently of the three status fields. |

### 4.3 Capacity output (wide; governance mechanism, not merely storage)

| Field | Type | Null? | Notes |
|---|---|---|---|
| `capacity_shadow_k0_05` | numeric | NULLABLE | NULL iff `capacity_estimable=FALSE`. The frozen shadow decision. |
| `capacity_sensitivity_k0_04` | numeric | NULLABLE | Downside/reference sensitivity. Same nullability rule. |
| `capacity_sensitivity_k0_06` | numeric | NULLABLE | Upside sensitivity. Same nullability rule. |

*Invariants:* all three NULL together or all three NOT NULL together; when NOT NULL, `capacity_sensitivity_k0_04 < capacity_shadow_k0_05 < capacity_sensitivity_k0_06` with ratios exactly `0.04/0.05` and `0.06/0.05` (within floating tolerance) — any deviation indicates a config error, hard-fail (Section 6, #2). **New in this revision (Section 6, #10):** `capacity_shadow_k0_05` must independently reproduce from the logged `float_activity_value_1m` / `commission` via `k=0.05 * (sqrt((1+Float)*(1+Commission)) - 1)` within numeric tolerance, and likewise for the two sensitivity columns at their own `k`. The ratio check above only catches the three columns disagreeing with *each other*; it cannot catch a shared, systematic error (e.g. a stale weights artifact) that moves all three together while preserving their ratios. This formula-reproducibility check closes that gap.

### 4.4 Risk / C3 decision-time values

| Field | Type | Null? | Notes |
|---|---|---|---|
| `risk_route` | enum: `THICK_CHAMPION_SCORED` / `THIN_FILE_SCORECARD` | NOT NULL | Deliberately new names, distinct from the retrospective `ROUTE_THICK`/`ROUTE_THIN` constants, which encode a PIT-reproducibility concept that does not apply to live scoring. Describes *how* the agent was scored. |
| `raw_model_score` | numeric | NULLABLE | NOT NULL iff `risk_route='THICK_CHAMPION_SCORED'`; NULL iff `THIN_FILE_SCORECARD`. Retained specifically so a future calibration change can be distinguished from a model change. |
| `risk_score` | numeric | NOT NULL | New in this revision. The route-specific final risk signal actually used downstream — on the thick route, this equals `cal_pd`; on the thin route, this equals the scorecard's thin-file value. Always populated regardless of route, so "what risk signal did this decision use" never requires branching on `risk_route` to find a non-null column. |
| `risk_score_type` | enum: `CALIBRATED_CHAMPION_PD` / `THIN_SCORECARD_PD_LIKE` | NOT NULL | New in this revision. Tags `risk_score`'s semantics so it is never implicitly treated as comparable across routes in a future "PD discrimination" analysis — a calibrated champion PD and a thin-file PD-like scorecard value are not the same quantity, even though both currently feed the same downstream multiplier logic. |
| `cal_pd` | numeric | **NULLABLE** | **Changed in this revision** — previously NOT NULL on both routes. Populated **only** when `risk_score_type='CALIBRATED_CHAMPION_PD'`; NULL for `THIN_SCORECARD_PD_LIKE`. The prior design populated `cal_pd` for the thin-file scorecard value too, which silently treated a non-calibrated, non-comparable score as if it were the calibrated champion PD. `cal_pd` now means exactly one thing: the isotonic-calibrated thick-route PD, never a stand-in for the thin-file value. |
| `shadow_multiplier_base` | numeric | NULLABLE | Computed from `cal_pd` specifically (never the generic `risk_score`), consistent with Open Item 1 (Section 7)'s default-closed stance: `THIN_SCORECARD_PD_LIKE` rows have no `cal_pd` and therefore no `shadow_multiplier_*` until that item is explicitly resolved. |
| `shadow_multiplier_conservative` | numeric | NULLABLE | |
| `l_full_shadow_k0_05_base` / `_conservative` | numeric | NULLABLE | `= capacity_shadow_k0_05 * shadow_multiplier_*`. |
| `l_full_sensitivity_k0_04_base` / `_conservative` | numeric | NULLABLE | |
| `l_full_sensitivity_k0_06_base` / `_conservative` | numeric | NULLABLE | |

### 4.5 Current-policy decision

| Field | Type | Null? | Notes |
|---|---|---|---|
| `l_current_capacity` | numeric | NULLABLE | Current engine's own pre-risk capacity component, only if architecturally separable. |
| `current_risk_multiplier` | numeric | NULLABLE | Current engine's own tier/risk multiplier, if exposed. Lets a later read distinguish `Capacity_new(F)` vs. `Capacity_current` from `M_C3` vs. `M_tier`, without making either comparison causal. |
| `l_current` | numeric | NULLABLE | NOT NULL whenever `current_decision_status IN ('APPROVED','ZERO_LIMIT_ELIGIBLE')`; NULL permitted for `DECLINED`/`OTHER`. |
| `current_policy_version` | string | NOT NULL | |

(Unchanged from the prior revision — the user confirmed this split is fine as-is.)

### 4.6 Challenger disagreement (derived; descriptive monitoring only)

| Field | Type | Null? | Notes |
|---|---|---|---|
| `disagreement_abs_ugx_shadow_base` / `_conservative` | numeric | NULLABLE | `l_full_shadow_k0_05_* - l_current`. |
| `disagreement_ratio_shadow_base` / `_conservative` | numeric | NULLABLE | Guarded: explicitly NULL (never Inf) when `l_current=0`. |

**Removed in this revision:** `disagreement_direction_shadow_base` / `_conservative`. The Decision Log is immutable and append-only; a directional classification (`CHALLENGER_HIGHER` / `CHALLENGER_LOWER` / `CONCORDANT_WITHIN_TOLERANCE` / `NOT_COMPARABLE`) depends on a tolerance rule that is explicitly not yet chosen (Section 7) and, more fundamentally, is an analytical judgment rather than a raw fact — writing it into an immutable table would freeze today's tolerance choice into every future read of this data, even after the rule is revisited. The two raw quantities above (`disagreement_abs_ugx_*`, `disagreement_ratio_*`) remain in the Decision Log as immutable facts; any directional classification must be computed in a separate analytical layer, always paired with an explicit `disagreement_rule_version` identifying which tolerance rule produced it, and never persisted bare back into this table.

### 4.7 Artifact provenance

| Field | Type | Null? | Notes |
|---|---|---|---|
| `capacity_weights_artifact_version` | string | NOT NULL | Logged unconditionally, even on `capacity_estimable=FALSE` rows. |
| `capacity_function_version` | string | NOT NULL | New in this revision. Versions the `Capacity(F)` formula/implementation itself (the `k * (sqrt((1+Float)*(1+Commission)) - 1)` transformation), distinct from `capacity_weights_artifact_version` — the weights artifact alone does not uniquely identify the full transformation if the functional form itself is ever revised (e.g. a different aggregation than the geometric mean). |
| `fundamentals_source_version` | string | NOT NULL | New in this revision. Versions the upstream transaction-mart/pipeline export that produced `float_activity_value_1m` / `commission`, distinct from the capacity formula's own version — input-source lineage (did the mart's extraction logic change) is a different failure mode from formula lineage (did the capacity transformation change), and conflating them would make a future provenance audit unable to tell which one moved. |
| `k_shadow_value` | numeric | NOT NULL, CHECK `=0.05` | Literal, not inferred — hard-fails the pipeline on config drift. |
| `k_sensitivity_low_value` | numeric | NOT NULL, CHECK `=0.04` | |
| `k_sensitivity_high_value` | numeric | NOT NULL, CHECK `=0.06` | |
| `champion_model_version` | string | NOT NULL | Single source of truth; not duplicated in 4.4. |
| `pd_calibration_artifact_version` | string | NULLABLE | NULL legitimately when `risk_route='THIN_FILE_SCORECARD'`. |
| `c3_shadow_artifact_version` | string | NULLABLE | NULL iff both `shadow_multiplier_*` are NULL. |
| `c3_policy_version` | string | NULLABLE | Same nullability. |

## 5. Execution Log and Outcome Log

**Execution Log** (append-only; 0..N rows per `decision_id`):

| Field | Type | Null? | Notes |
|---|---|---|---|
| `execution_id` | string (UUID) | NOT NULL, PK | Now the key the Outcome Log joins against (Section 3). |
| `decision_id` | string | NOT NULL, FK -> Decision Log | |
| `disbursement_fid` | string | NOT NULL, UNIQUE | The real production loan identifier. |
| `disbursement_timestamp` | timestamp | NOT NULL | Must be >= `decision_timestamp` (hard-fail if violated). |
| `actual_disbursed_amount` | numeric | NOT NULL | |
| `execution_status` | enum (exact values TBD against the real payment architecture, e.g. `COMPLETED` / `REVERSED` / `FAILED` / `CANCELLED`) | NULLABLE | New in this revision. Only added, and only populated, if the source payment/disbursement system actually exposes an execution-outcome concept of this kind — never fabricated if it doesn't. Left NULLABLE and with its enum values explicitly marked TBD because the real architecture's concept (if any) has not yet been confirmed; implementation must confirm the real values before this column is made NOT NULL or its enum finalized. |

*Soft invariant (flag, not hard-fail — operational override rules are not fully known at design time):* `actual_disbursed_amount` materially exceeding `l_current` is logged as a data-quality flag.

**Outcome Log** (append-only; **re-keyed in this revision** — joined by `execution_id`, not `decision_id`):

| Field | Type | Null? | Notes |
|---|---|---|---|
| `outcome_id` | string (UUID) | NOT NULL, PK | |
| `execution_id` | string | NOT NULL, FK -> Execution Log | **Primary join key, changed from `decision_id` in this revision.** Because Execution Log allows 0..N rows per decision, this is the only key that unambiguously identifies which specific disbursement produced this outcome. |
| `decision_id` | string | NOT NULL | **Denormalized lineage field, not a key.** Must equal the `decision_id` of the Execution Log row identified by `execution_id` (Section 6, #7) — carried here purely so a reader can filter/join to the Decision Log without an intermediate hop, never as an independent source of truth. |
| `disbursement_fid` | string | NOT NULL | **Denormalized lineage field, not a key.** Must equal the `disbursement_fid` of the referenced Execution Log row (Section 6, #7), same rationale as `decision_id` above. |
| `outcome_window_start` | date | NOT NULL | Anchored to the **Execution** record's `disbursement_timestamp`, not `decision_timestamp` -- this is what makes the eventual read `Y \| L_actual`, never `Y \| L_current_recommendation`. |
| `outcome_window_end` | date | NOT NULL | |
| `bad_state_3dpd_30d` | boolean | NULLABLE | NULL iff `label_eligible_30d=FALSE` (censored). |
| `label_eligible_30d` | boolean | NOT NULL | |
| `label_eligibility_reason_30d` | string | NOT NULL | |
| `outcome_definition_version` | string | NOT NULL | Versions the 3DPD/30d definition itself, so a future definition change never silently breaks reproducibility of an old decision/outcome pair. A single `execution_id` may therefore accumulate more than one Outcome Log row over time if this version changes (Section 3) — never a reason to update or delete the earlier row. |

*Invariant:* an Outcome Log row may exist only for an `execution_id` that exists in the Execution Log — there is no `Y` to observe for a declined or never-executed decision, and the lineage back to the Decision Log is always via that execution, never direct.

## 6. Cross-Table Invariants (pipeline hard-fail conditions)

1. `decision_id` unique; Decision Log rows never updated post-insert.
2. Capacity triple (4.3): all-null or all-not-null together; exact k-ratio check.
3. `raw_model_score` nullability tied to `risk_route` exactly as specified (4.4); `cal_pd` nullability tied to `risk_score_type` exactly as specified (4.4) — NOT NULL iff `risk_score_type='CALIBRATED_CHAMPION_PD'`; `risk_score` itself is NOT NULL unconditionally.
4. `l_full_*` NOT NULL iff both its capacity and its multiplier are NOT NULL for that (k, scenario) pair.
5. `k_shadow_value` / `k_sensitivity_low_value` / `k_sensitivity_high_value` literal CHECK constraints against 0.05 / 0.04 / 0.06 -- the concrete mechanism preventing Tier A from becoming a re-tuning exercise.
6. Execution Log: `disbursement_timestamp >= decision_timestamp`; `decision_id` must exist in Decision Log.
7. Outcome Log: `execution_id` must exist in Execution Log before an outcome row can be written; the Outcome Log's own `decision_id` and `disbursement_fid` must exactly match the referenced Execution Log row's `decision_id` / `disbursement_fid` (hard-fail on mismatch — these are denormalized lineage fields, not independent data); `outcome_window_start` derived from the Execution record, never the Decision record.
8. `shadow_population_label` set from a pre-deployment capability confirmation, never inferred per-row from observed behavior.
9. **New in this revision — point-in-time (PIT) guard:** `fundamentals_snapshot_date < date(decision_timestamp)` whenever `fundamentals_snapshot_date` is NOT NULL (strict); `fundamentals_snapshot_available_at <= decision_timestamp` whenever `fundamentals_snapshot_available_at` is NOT NULL.
10. **New in this revision — formula-reproducibility guard:** `capacity_shadow_k0_05`, `capacity_sensitivity_k0_04`, and `capacity_sensitivity_k0_06` must each independently reproduce, within numeric tolerance, from the row's own `float_activity_value_1m` / `commission` via the capacity formula at that column's respective `k`. Catches a systematic error shared across all three columns, which the ratio-only check (#2) structurally cannot detect.
11. **New in this revision — capacity-estimability derivation guard:** `capacity_estimable = TRUE` iff `float_availability_status='OK' AND commission_availability_status='OK' AND snapshot_availability_status='OK'`; `FALSE` otherwise. Never set independently of the three status fields.
12. `shadow_multiplier_*` / `l_full_*` NULL whenever `cal_pd` is NULL (i.e., whenever `risk_score_type='THIN_SCORECARD_PD_LIKE'`), per Open Item 1's default-closed stance (Section 7) — never computed from the generic `risk_score`.

## 7. Explicitly Open Items (deferred, not resolved by this design)

1. **C3-on-thin-file.** The frozen C3 artifact (Stage 4B-R4) was built and verified only against thick-file, model-reconstructed `cal_pd`. Whether to apply an analogous multiplier to the thin-file `risk_score` (`risk_score_type='THIN_SCORECARD_PD_LIKE'`) in live shadow mode is unresolved. Default stance if undecided at implementation time: `cal_pd`, `shadow_multiplier_*`, and every downstream `l_full_*` stay NULL for `THIN_SCORECARD_PD_LIKE` rows -- thin-file gets no challenger recommendation in Tier A until this is explicitly decided. (Unchanged from the prior revision; the `cal_pd`/`risk_score` rename in Section 4.4 makes the boundary of this default more precise but does not resolve it.)
2. **Disagreement-direction classification rule.** The prior revision deferred only the `CONCORDANT_WITHIN_TOLERANCE` threshold inside an immutable `disagreement_direction_*` column. This revision removes that column from the Decision Log entirely (Section 4.6) — the open item is now broader: the tolerance rule, and the versioned, non-immutable analytical layer that will apply it (producing a classification paired with a `disagreement_rule_version`, per Section 4.6), remain fully unspecified and deliberately deferred, per explicit instruction not to choose them as part of schema design.

Neither item blocks freezing this schema. Both must be resolved before the logging pipeline is implemented against the live engine.

## 8. What This Document Does Not Do

No code has been written. No logging pipeline has been built or hooked into the live engine. No decision on Tier B (controlled exposure / pilot design) has been made — that remains a separate, later, and considerably heavier undertaking. This document specifies the schema and the Tier A/Tier B boundary only.
