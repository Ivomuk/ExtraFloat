# Stage 4C — Prospective Validation Schema Design

**Status:** Design frozen at commit (this document's own commit) on `claude/integrated-solution-analysis-atdfe6`. **No code has been written for this stage.** This document specifies the logging architecture and schema only; implementing it against the live engine is explicitly the next, separate step.
**Builds on:** `docs/stage_4b_capacity_k_selection_findings.md` (freezes `k_shadow=0.05`, `k_sensitivity=0.04/0.06`, and the `Capacity(F) -> M_C3(PD) -> L_full -> product policy` architecture this stage logs prospectively).

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
Decision Log  ->  Execution Log  ->  Outcome Log
```

- **Decision Log**: exactly what was known and computed at the moment of evaluation — business-state inputs, the frozen challenger's capacity/risk/recommendation outputs, current policy's decision, their characterized disagreement, and full artifact provenance. Written once per decision event, **never updated**.
- **Execution Log**: what credit was actually extended, if any. Zero or more rows per decision (a decision may be declined and never executed; some architectures may support multiple executions against one decision).
- **Outcome Log**: what subsequently happened, observed under the **actual** exposure (via the Execution Log), not the decision's recommendation. Joined to the Decision Log by `decision_id`, never written back into it.

This separation is what makes "why did this agent get this recommendation on this date" permanently reconstructable, keeps point-in-time integrity (what was known at decision time vs. what became observable later), and lets the outcome-label definition itself evolve without rewriting history.

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
| `fundamentals_snapshot_date` | date | NULLABLE | NULL iff no eligible prior snapshot exists. |
| `fundamentals_snapshot_id` | string | NULLABLE | Only if the source mart exposes a stable id; never fabricated. |
| `fundamentals_snapshot_age_days` | integer | NULLABLE | NULL iff `fundamentals_snapshot_date` is NULL. Logged on every row so snapshot-sharing across decisions (the Analysis-3 cadence-gate finding) stays visible rather than hidden. |
| `float_activity_value_1m` | numeric | NULLABLE | |
| `commission` | numeric | NULLABLE | |
| `capacity_estimable` | boolean | NOT NULL | |
| `capacity_not_estimable_reason` | enum: `NONE` / `FLOAT_MISSING_OR_NONPOSITIVE` / `COMMISSION_MISSING_OR_NONPOSITIVE` / `BOTH_MISSING_OR_NONPOSITIVE` / `SNAPSHOT_UNAVAILABLE` | NOT NULL | `NONE` iff `capacity_estimable=TRUE`. |

### 4.3 Capacity output (wide; governance mechanism, not merely storage)

| Field | Type | Null? | Notes |
|---|---|---|---|
| `capacity_shadow_k0_05` | numeric | NULLABLE | NULL iff `capacity_estimable=FALSE`. The frozen shadow decision. |
| `capacity_sensitivity_k0_04` | numeric | NULLABLE | Downside/reference sensitivity. Same nullability rule. |
| `capacity_sensitivity_k0_06` | numeric | NULLABLE | Upside sensitivity. Same nullability rule. |

*Invariant:* all three NULL together or all three NOT NULL together; when NOT NULL, `capacity_sensitivity_k0_04 < capacity_shadow_k0_05 < capacity_sensitivity_k0_06` with ratios exactly `0.04/0.05` and `0.06/0.05` (within floating tolerance) — any deviation indicates a config error, hard-fail.

### 4.4 Risk / C3 decision-time values

| Field | Type | Null? | Notes |
|---|---|---|---|
| `risk_route` | enum: `THICK_CHAMPION_SCORED` / `THIN_FILE_SCORECARD` | NOT NULL | Deliberately new names, distinct from the retrospective `ROUTE_THICK`/`ROUTE_THIN` constants, which encode a PIT-reproducibility concept that does not apply to live scoring. |
| `raw_model_score` | numeric | NULLABLE | NOT NULL iff `risk_route='THICK_CHAMPION_SCORED'`; NULL iff `THIN_FILE_SCORECARD`. Retained specifically so a future calibration change can be distinguished from a model change. |
| `cal_pd` | numeric | NOT NULL | Populated on both routes (thick via isotonic calibration on `raw_model_score`; thin via the scorecard's `never_loan_pd_like`, optionally LR-overridden), matching the real `score_new_agents` routing. |
| `shadow_multiplier_base` | numeric | NULLABLE | **Open item — see Section 6.** |
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

### 4.6 Challenger disagreement (derived; descriptive monitoring only)

| Field | Type | Null? | Notes |
|---|---|---|---|
| `disagreement_abs_ugx_shadow_base` / `_conservative` | numeric | NULLABLE | `l_full_shadow_k0_05_* - l_current`. |
| `disagreement_ratio_shadow_base` / `_conservative` | numeric | NULLABLE | Guarded: explicitly NULL (never Inf) when `l_current=0`. |
| `disagreement_direction_shadow_base` / `_conservative` | enum: `CHALLENGER_HIGHER` / `CHALLENGER_LOWER` / `CONCORDANT_WITHIN_TOLERANCE` / `NOT_COMPARABLE` | NULLABLE | `NOT_COMPARABLE` when either side is NULL. **The tolerance rule is deliberately unspecified — see Section 6.** |

### 4.7 Artifact provenance

| Field | Type | Null? | Notes |
|---|---|---|---|
| `capacity_weights_artifact_version` | string | NOT NULL | Logged unconditionally, even on `capacity_estimable=FALSE` rows. |
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
| `execution_id` | string (UUID) | NOT NULL, PK | |
| `decision_id` | string | NOT NULL, FK -> Decision Log | |
| `disbursement_fid` | string | NOT NULL, UNIQUE | The real production loan identifier. |
| `disbursement_timestamp` | timestamp | NOT NULL | Must be >= `decision_timestamp` (hard-fail if violated). |
| `actual_disbursed_amount` | numeric | NOT NULL | |

*Soft invariant (flag, not hard-fail — operational override rules are not fully known at design time):* `actual_disbursed_amount` materially exceeding `l_current` is logged as a data-quality flag.

**Outcome Log** (append-only; joined by `decision_id`, never updates the Decision Log):

| Field | Type | Null? | Notes |
|---|---|---|---|
| `outcome_id` | string (UUID) | NOT NULL, PK | |
| `decision_id` | string | NOT NULL, FK -> Decision Log | |
| `outcome_window_start` | date | NOT NULL | Anchored to the **Execution** record's `disbursement_timestamp`, not `decision_timestamp` -- this is what makes the eventual read `Y \| L_actual`, never `Y \| L_current_recommendation`. |
| `outcome_window_end` | date | NOT NULL | |
| `bad_state_3dpd_30d` | boolean | NULLABLE | NULL iff `label_eligible_30d=FALSE` (censored). |
| `label_eligible_30d` | boolean | NOT NULL | |
| `label_eligibility_reason_30d` | string | NOT NULL | |
| `outcome_definition_version` | string | NOT NULL | Versions the 3DPD/30d definition itself, so a future definition change never silently breaks reproducibility of an old decision/outcome pair. |

*Invariant:* an Outcome Log row may exist only for a `decision_id` with >=1 Execution Log row — there is no `Y` to observe for a declined or never-executed decision.

## 6. Cross-Table Invariants (pipeline hard-fail conditions)

1. `decision_id` unique; Decision Log rows never updated post-insert.
2. Capacity triple (4.3): all-null or all-not-null together; exact k-ratio check.
3. `raw_model_score` nullability tied to `risk_route` exactly as specified (4.4).
4. `l_full_*` NOT NULL iff both its capacity and its multiplier are NOT NULL for that (k, scenario) pair.
5. `k_shadow_value` / `k_sensitivity_low_value` / `k_sensitivity_high_value` literal CHECK constraints against 0.05 / 0.04 / 0.06 -- the concrete mechanism preventing Tier A from becoming a re-tuning exercise.
6. Execution Log: `disbursement_timestamp >= decision_timestamp`; `decision_id` must exist in Decision Log.
7. Outcome Log: `decision_id` must have >=1 Execution Log row before an outcome row can be written; `outcome_window_start` derived from the Execution record, never the Decision record.
8. `shadow_population_label` set from a pre-deployment capability confirmation, never inferred per-row from observed behavior.

## 7. Explicitly Open Items (deferred, not resolved by this design)

1. **C3-on-thin-file.** The frozen C3 artifact (Stage 4B-R4) was built and verified only against thick-file, model-reconstructed `cal_pd`. Whether to apply it to thin-file `cal_pd` (the scorecard value) in live shadow mode is unresolved. Default stance if undecided at implementation time: `shadow_multiplier_*` and every downstream `l_full_*` stay NULL for `THIN_FILE_SCORECARD` rows -- thin-file gets no challenger recommendation in Tier A until this is explicitly decided.
2. **Concordance tolerance rule.** `disagreement_direction_*`'s `CONCORDANT_WITHIN_TOLERANCE` threshold is deliberately left unspecified, per explicit instruction not to choose it as part of schema design.

Neither item blocks freezing this schema. Both must be resolved before the logging pipeline is implemented against the live engine.

## 8. What This Document Does Not Do

No code has been written. No logging pipeline has been built or hooked into the live engine. No decision on Tier B (controlled exposure / pilot design) has been made — that remains a separate, later, and considerably heavier undertaking. This document specifies the schema and the Tier A/Tier B boundary only.
