# Stage 4C — Decision Log Logger: Implementation Plan

**Status:** Plan frozen at commit (this document's own commit) on `claude/integrated-solution-analysis-atdfe6`, after one review round that required four corrections before freeze (Float-missingness semantics, thin-route risk-signal construction, run-level completeness tracking, storage-sink gating) — see Section 8 for the resolution of each. **No logging code has been written.** This document specifies the module structure, exact insertion point, function-level design, invariant enforcement, and verification plan for the Decision Log writer only. Writing and verifying the code against synthetic fixtures is the explicit next step after this plan is accepted; no real-data run happens until asked for, same standing instruction as every other script in this workstream.
**Builds on:** `docs/stage_4c_prospective_validation_schema_design.md` (frozen schema, as amended by `docs/stage_4c_implementation_mapping.md`'s five conclusions — `shadow_decision_id`/`scoring_run_timestamp` grain, `challenger_risk_signal`/`challenger_risk_signal_type` naming, Gate I0's `df_agent`-sourced capacity inputs, no `execution_status`, explicit temporal execution attribution).

## 1. Scope

**In scope:** the **Decision Log writer** — computing and persisting one Decision Log row per `(agent_msisdn, run_id)` scoring event, hooked into `run_credit_risk_pipeline.py`'s existing batch run.

**Explicitly out of scope, deferred to a separate later reconnaissance + plan:** Execution Log and Outcome Log population. Both depend on upstream systems this reconnaissance has not mapped — the live disbursement event stream (where/how an execution record would actually be created and in what system) and the outcome-maturation job (what triggers writing a matured `bad_state_3dpd_30d` row, and from what source). Writing that code now would mean inventing integration points rather than using verified ones — exactly the mistake the Stage 4C-I0 gate exercise was designed to prevent. The temporal attribution rule (`docs/stage_4c_implementation_mapping.md` Section 3) is already specified and will not need to change when that follow-up work starts; it simply has nothing to attach to yet.

This mirrors Stage 4A.1's own narrow scoping discipline: solve the piece that is actually reconnoitered, not the whole architecture at once.

## 2. Module Structure

Two new packages, kept deliberately separate because they answer different questions:

```
extrafloat/
  capacity/
    __init__.py
    capacity_function.py       -- the single, canonical Capacity(F) formula + weights loader
  shadow_logging/
    __init__.py
    stage_4c_decision_logger.py -- Decision Log row construction + invariant enforcement
```

**Decision 1: APPROVED — promote the formula.** This is refactoring/provenance hardening, not a model change; the formula itself (`C_k(F,C) = k * [sqrt((1+F)(1+C)) - 1]`) is untouched.

**Design decision (1 of 2, now resolved): promoting the Capacity(F) formula into `extrafloat/`.**
`Capacity(F)` currently exists only inside `scripts/derive_capacity_function_and_backtest_frontier.py` (`compute_capacity_ugx`), per the one-way convention that `scripts/` depends on `extrafloat/`, never the reverse. For the live logger to consume the *exact same, frozen* formula — never an independently reinvented copy, per Gate I0's own requirement in the implementation mapping — one of two things must happen: either the logger restates the formula a second time (risking drift between the two copies, which is precisely what the frozen `capacity_function_version` provenance field and the formula-reproducibility invariant exist to catch but not prevent), or the formula is promoted into `extrafloat/capacity/capacity_function.py` as the single canonical source, with `scripts/derive_capacity_function_and_backtest_frontier.py` updated to import from it instead of defining its own copy.

**Recommendation:** promote it. This is the one case in this workstream where code moves *into* `extrafloat/` rather than being restated outward from it, and it is the right direction for exactly the reason the formula-reproducibility invariant was added for: a single source of truth is strictly better than a check that can only detect drift after the fact. The promoted module changes nothing about the formula itself (same `k * (sqrt((1+Float)*(1+Commission)) - 1)`, same weights-artifact loading), only its location. `scripts/derive_capacity_function_and_backtest_frontier.py`'s existing scratchpad verification suite must be re-run unchanged against the new import to confirm the move introduced no behavioral change before anything is committed.

If this is not the preferred direction, the alternative (restate a second frozen copy inside `extrafloat/shadow_logging/`, accepting the drift risk the formula-reproducibility invariant is specifically there to catch) is a straightforward substitution in Section 3 below — flagging this now rather than deciding it silently.

## 3. Function-Level Design

### 3.1 `extrafloat/capacity/capacity_function.py`

- `load_capacity_weights(artifacts_path)` — reads the frozen weights artifact (`capacity_score_weights.json`, from `derive_capacity_score_weights.py`), returns `(weights_dict, version_string)`. Restated/promoted from the existing script's own loader.
- `compute_capacity_ugx(float_activity_value_1m, commission, weights, k)` — the frozen formula, vectorized, restated/promoted verbatim from `scripts/derive_capacity_function_and_backtest_frontier.py`'s `compute_capacity_ugx`. No behavioral change from the existing, already-verified implementation.

### 3.2 `extrafloat/shadow_logging/stage_4c_decision_logger.py`

All functions are pure/read-only: they accept DataFrames already in memory during a live `run_credit_risk_pipeline.py` run and return new DataFrames or columns. None mutate an input DataFrame in place, and none ever write to any existing engine output column — the same hard discipline `extrafloat_shadow_risk_multiplier.py` already follows.

- `extract_raw_capacity_inputs(df_agent_raw)` — reads `cash_in_value_1m`, `payment_value_1m`, `commission` from the **raw** `df_agent` DataFrame (per Gate I0's finding: these columns exist on `df_agent`, not on `features_df`).

  **Correction from review, before anything was coded:** the first draft of this function determined `float_availability_status` from the combined pre-fill null/sign state of `cash_in_value_1m` and `payment_value_1m` individually — e.g. `cash_in_value_1m=1,000,000, payment_value_1m=NULL` would have been flagged `MISSING`. That is wrong: the frozen Stage 4B formula is `Float = fillna0(cash_in_value_1m) + fillna0(payment_value_1m)`, under which that exact row produces a perfectly valid `Float = 1,000,000` — one missing *component* does not mean Float itself is missing. Flagging it `MISSING` would have made `capacity_estimable=False` and nulled out `Capacity(F)` for rows the frozen `k=0.05` selection would have scored validly, silently introducing a new capacity-eligibility rule never part of that selection. Corrected design, keeping component data quality and capacity estimability as two separate questions:
  1. Record `cash_in_availability_status` and `payment_availability_status` (`OK`/`MISSING`/`NONPOSITIVE`) independently, from each raw column's own pre-fill null/sign state — pure data-quality/audit information, logged but **not** consumed by `capacity_estimable`.
  2. Apply the frozen `Float = cash_in_value_1m.fillna(0) + payment_value_1m.fillna(0)` arithmetic, unconditionally, exactly as the `k=0.05` selection defines it.
  3. Derive `float_availability_status` **from the resulting `Float` value, not from the pre-fill components**: `OK` if `Float > 0`, `NONPOSITIVE` if `Float <= 0` (this is the only case that can produce the ambiguous "both components null, fills to exactly 0" row, and it is correctly treated the same as a genuine, measured zero — consistent with how the frozen formula already treats it). `MISSING` is reserved for the structurally different case where the agent has no row in `df_agent` at all (so `cash_in_value_1m`/`payment_value_1m` are not merely null-valued but absent), never for a present row with one or both components null.
  `commission_availability_status` needs no equivalent correction — `commission` is a single raw pass-through column, not a two-component sum, so pre-fill and post-fill status coincide for it.
- `compute_snapshot_availability(df_agent_raw, scoring_run_timestamp)` — derives `snapshot_availability_status` (`OK`/`UNAVAILABLE`) and `fundamentals_snapshot_date` from `df_agent_raw`'s own `snapshot_dt`/`tbl_dt` column, and enforces the hard PIT invariant `fundamentals_snapshot_date < date(scoring_run_timestamp)` at construction time (raises, does not silently drop the row) — never NULLs it away.
- `compute_capacity_estimable(float_status, commission_status, snapshot_status)` — the pure derivation `capacity_estimable = (all three == OK)`. No independent logic path; this function exists only so the derivation rule has one place, matching the frozen schema's invariant #11.
- `compute_capacity_challenger_outputs(float_activity_value_1m, commission, capacity_estimable, weights, weights_version)` — calls `capacity_function.compute_capacity_ugx` at `k=0.05/0.04/0.06`, masks all three to NaN where `capacity_estimable=False`, and immediately self-checks the formula-reproducibility invariant (schema invariant #10) and the capacity-triple ratio invariant (#2) on its own output before returning — a construction-time self-check, not a separate downstream audit step that could be skipped.
- `attach_challenger_risk_signal(result_df)` — reads `cal_pd`/`thin_file_flag` directly off `result_df` (both already joined onto `features_df` in the pipeline's own Stage 5, before Stage 6 calls the engine — see Section 4's insertion-point correction for why this column needs the same `keep_intermediate=True`-before-trim treatment as `capacity_cap`/`policy_multiplier`). No separate `pd_scored` input is needed.

  **Correction from review, before anything was coded:** the first draft said `challenger_risk_signal = cal_pd` in both cases *while also* saying the output schema's `cal_pd` field stays nullable, populated only on the thick route — those two statements are inconsistent (if the output's `cal_pd` is NULL for thin, `challenger_risk_signal` cannot simply equal "`cal_pd`" for thin). Pinning down the actual mechanics from the reconnaissance (`pd_model/modeling/inference.py`): the live engine's own internal `cal_pd` column already pools both routes under one name — for thin-file agents it already holds `never_loan_pd_like` (optionally LR-calibrated), promoted into the shared `cal_pd` column by the existing `score_new_agents` logic, before this module ever sees it. So no new extraction from `pd_model` internals is needed; the correction is purely about which *output* field keeps that pooled value:
  ```
  THICK (thin_file_flag == 0):
      challenger_risk_signal       = result_df["cal_pd"]   # the engine's pooled column, thick value
      challenger_risk_signal_type  = CALIBRATED_CHAMPION_PD
      output cal_pd                = result_df["cal_pd"]   # same value, kept

  THIN (thin_file_flag == 1):
      challenger_risk_signal       = result_df["cal_pd"]   # the engine's pooled column, thin value (never_loan_pd_like)
      challenger_risk_signal_type  = THIN_SCORECARD_PD_LIKE
      output cal_pd                = NULL                  # explicitly nulled out here, never copied through
  ```
  `challenger_risk_signal` is always read from the engine's existing pooled `cal_pd` column (correct for both routes, since that pooling already happened upstream); the output schema's own `cal_pd` field is a *separate* write that is explicitly nulled for the thin route rather than mirroring the engine column — this is exactly why `challenger_risk_signal` was introduced in the first place, and `shadow_multiplier_*`/`l_full_*` stay NULL on thin rows per the frozen default-closed C3-on-thin-file stance, unaffected by this fix.
- `attach_current_policy_snapshot(result_df)` — pulls `l_current = assigned_limit`, `current_risk_multiplier = policy_multiplier`, `l_current_capacity = capacity_cap` (the engine's own six-component figure — explicitly NOT a stand-in for `Capacity(F)`, logged purely for side-by-side audit as the frozen schema's Section 4.5 already intends), `current_policy_version` (a config/version constant, sourced from engine config versioning if one exists, else a literal placed here once and bumped manually on any `DEFAULT_CAP_CONFIG` change — this sourcing detail is noted as an implementation-time check against whatever versioning the engine config already has, not assumed).

  **Correction caught while drafting this plan, before anything was committed:** `policy_multiplier` and `capacity_cap` are *intermediate* engine columns. `FINAL_OUTPUT_COLUMNS` (`run_extrafloat_limit_engine.py:64-84`) does **not** include either — `_trim_output_columns` drops them whenever `keep_intermediate=False`, and `run_credit_risk_pipeline`'s own `keep_intermediate` parameter defaults to `False` (`run_credit_risk_pipeline.py:226`), which is passed straight through to the engine call (`run_credit_risk_pipeline.py:536-542`). So under the pipeline's default invocation, `result_df` genuinely does not carry these columns, and `attach_current_policy_snapshot` as first drafted would silently read NaN-filled/missing columns rather than real values. See Section 4 for the corrected insertion-point design this forces.
- `compute_disagreement(l_full_shadow_k0_05, l_current)` — `disagreement_abs_ugx_*` and `disagreement_ratio_shadow_*` only (guarded NULL, never Inf, at `l_current=0`), per the frozen schema's Section 4.6 as already amended (no `disagreement_direction_*` — that classification lives outside this module entirely, in a later analytical layer, never computed here).
- `build_shadow_decision_id(run_id, agent_msisdn)` — the deterministic hash `f(run_id, agent_msisdn)`, with `run_id` and `agent_msisdn` always carried as separate columns alongside it (per the implementation mapping's explicit "never rely on the hash for provenance" instruction). **The real protection is a business/storage-level `UNIQUE(run_id, agent_msisdn)` constraint on those two source columns, not the hash** — `shadow_decision_id` is a convenient surrogate key derived from that already-unique pair, not an independent source of uniqueness. The hash-collision check in the verification plan (Section 5) is cheap insurance, not the load-bearing guarantee.
- `build_decision_log_rows(df_agent_raw, result_df, run_id, scoring_run_timestamp, capacity_weights, capacity_weights_version, artifact_versions)` — the single orchestration entry point, taking `result_df` with its intermediate columns intact (per Section 4's corrected insertion point) rather than a separate `pd_scored` input. Joins the two inputs on `msisdn`/`agent_msisdn` (reusing the same normalization discipline `run_credit_risk_pipeline.py` already applies — `_norm_msisdn`, restated here rather than imported, since `run_credit_risk_pipeline.py` is a top-level script, not an `extrafloat/` module, so the existing one-way convention still points the same direction), calls every function above in the correct order, assembles the full Decision Log schema (Sections 4.1-4.7 of the frozen schema, with the Section 4.3/4.4 renames from the implementation mapping), and returns one row per agent present in that run's `result_df`.
- `verify_decision_log_invariants(decision_log_df)` — enforces every Section 6 invariant that applies at Decision-Log-construction time (1, 2, 3, 4, 5, 8, 9, 10, 11 — the renumbered set from the frozen schema; 6 and 7 are Execution/Outcome Log invariants, out of scope here per Section 1 above). Raises on any violation — **never silently drops or NULLs a violating row**, since a construction-time invariant violation indicates a logger bug, not a legitimate missing-data case (which the availability-status fields already handle correctly).

### 3.3 Run-level completeness/failure manifest (added from review)

**Gap identified in review:** the `try/except` around the whole Decision Log block (Section 4) correctly ensures a construction or write failure never breaks live credit decisioning — but, as specified in the first draft, it also leaves no trace that a given run's shadow logging was incomplete or failed. A `logger.exception()` call protects production; it does not protect later analysis from silently treating a partial day's shadow population as the complete one. A construction/invariant failure must not fail production, but it must make that run's shadow data visibly unusable, not silently partial.

**Addition:** a small, separate, per-run operational record — **not** a new per-agent Decision Log field, and not gated by the same invariants as the Decision Log itself (it must still be written even when Decision Log construction fails entirely):

| Field | Notes |
|---|---|
| `run_id` | Same `run_id` as the scoring run. |
| `shadow_logging_status` | `SUCCESS` / `FAILED` / `PARTIAL`. |
| `expected_agent_count` | `len(result_df)` for this run — the population the Decision Log should have covered. |
| `logged_agent_count` | Rows actually constructed and passed invariant verification. |
| `failure_stage` / `error_code` | Which step failed (e.g. `extract_raw_capacity_inputs`, `verify_decision_log_invariants`, `write_decision_log`) and a short error code, when `shadow_logging_status != SUCCESS`. |

`expected_agent_count != logged_agent_count` is itself a `PARTIAL` result, not silently rounded up to `SUCCESS`. This manifest is written unconditionally (its own try/except, logging but never raising, so a manifest-write failure cannot cascade into failing the pipeline either) — it is the thing a later audit checks first before trusting any given run's Decision Log rows at all.

### 3.4 Storage sink — module development vs. deployment gating (corrected from review)

**Not resolved by this plan; flagged explicitly rather than guessed.** The Decision Log is append-only and, given the batch/agent-grain volume established in the implementation mapping (one row per agent per scoring run, potentially the full agent population on every run), will accumulate substantial volume over time. This repository has no visibility into what storage the production deployment actually has available (a warehouse table alongside `momo_loan_book_tracker`, a partitioned file store, something else) — the same category of operational-infrastructure question Gate I0 already flagged as outside this codebase's visibility.

**Correction from review:** the first draft proposed leaving `write_decision_log(df, sink)` as a `NotImplementedError` stub *inside* the production code path that the Section 4 `try/except` wraps. That is the wrong shape — a bare exception swallowed by a belt-and-braces `try/except` in a live pipeline would make the system *appear* successfully deployed while silently failing every shadow write, exactly the failure mode the new manifest in Section 3.3 exists to catch, but catching it after the fact is worse than not creating the condition at all. Corrected design, distinguishing module development from production wiring explicitly:
- The pure `build_decision_log_rows`/`verify_decision_log_invariants` DataFrame layer can be written, verified, and reviewed with **no sink at all** — nothing in Sections 3.1-3.2 depends on a sink existing.
- `write_decision_log(df, sink)` itself is written against whatever sink interface is eventually confirmed, but the Section 4 pipeline integration is gated behind an explicit, off-by-default feature flag (e.g. a `stage_4c_logging.enabled` config key, default `False`) — when disabled, the whole Decision Log block (construction, invariants, write, manifest) simply does not run, rather than running and failing into a caught exception every time.
- Enabling Stage 4C logging in any real deployment requires a valid, confirmed sink configuration to be set at the same time the flag is turned on — "the sink is not chosen yet" and "the feature is live" can never both be true simultaneously.

## 4. Insertion Point (corrected)

Two inputs the Decision Log needs live in two different scopes, and no single existing call site has both:
- `df_agent` (raw mart read, for `cash_in_value_1m`/`payment_value_1m`/`commission` per Gate I0) exists only in `run_credit_risk_pipeline.py` — `run_extrafloat_limit_engine.py` never sees it.
- `capacity_cap`/`policy_multiplier`/`combined_cap` (needed for `attach_current_policy_snapshot`) are intermediate engine columns that exist on `df` *inside* `run_extrafloat_limit_engine`, before `_trim_output_columns` runs — but are not reliably present on `result_df` back in `run_credit_risk_pipeline.py` under the pipeline's default `keep_intermediate=False` (Section 3.2 correction above). `cal_pd`/`thin_file_flag`, by contrast, ARE already present going into `run_extrafloat_limit_engine` (joined onto `features_df` in Stage 5, before Stage 6 calls the engine), so `attach_challenger_risk_signal` does not have this problem.

**Resolution:** keep all Decision Log construction logic in `run_credit_risk_pipeline.py` — the one place `df_agent` is in scope — rather than splitting it across both files or touching `run_extrafloat_limit_engine.py`'s internals. This is done by having `run_credit_risk_pipeline.py` always request `keep_intermediate=True` from its *internal* call to `run_extrafloat_limit_engine` (independent of the pipeline's own `keep_intermediate` parameter, which governs only what the pipeline ultimately returns to its caller), building the Decision Log from that fully-intermediate `result_df` plus `df_agent`, and only afterward trimming `result_df` down to `FINAL_OUTPUT_COLUMNS`-equivalent if the pipeline's own caller requested `keep_intermediate=False`. Concretely, inserted immediately after Stage 8 (`result_df["score_source"]`, `result_df["scored_at"] = run_id`, i.e. after line 595 in the current file), **before** any final trimming of the pipeline's own return value:

```python
# Stage 6 call (modified): always keep intermediate columns internally,
# regardless of this pipeline function's own keep_intermediate argument --
# the Decision Log needs capacity_cap/policy_multiplier/combined_cap
# before they would otherwise be trimmed away.
result_df = run_extrafloat_limit_engine(
    features_df, config=engine_config, keep_intermediate=True,
    shadow_artifacts_dir=artifacts_dir, run_id=run_id,
)
...
# Stage 8 (existing) ...
result_df["score_source"] = ...
result_df["scored_at"] = run_id

# Stage 9 (new): Stage 4C Decision Log -- read-only, additive, never
# affects the pipeline's own return value either way.
try:
    decision_log_df = build_decision_log_rows(
        df_agent_raw=df_agent, result_df=result_df,
        run_id=run_id, scoring_run_timestamp=run_id, ...
    )
    verify_decision_log_invariants(decision_log_df)
    write_decision_log(decision_log_df, sink=...)  # pending Section 3.4; block gated off by a feature flag until sink is confirmed
    write_run_manifest(run_id, decision_log_df, expected_agent_count=len(result_df))  # Section 3.3, written unconditionally
except Exception:
    logger.exception("Stage 4C Decision Log write failed for run_id=%s -- live pipeline output unaffected", run_id)

# Existing behavior preserved: trim back down to FINAL_OUTPUT_COLUMNS-
# equivalent columns if THIS pipeline call's own keep_intermediate was
# False, now that the Decision Log has already captured what it needed.
if not keep_intermediate:
    result_df = _trim_to_final_output_columns(result_df)
```

`build_decision_log_rows` no longer takes a separate `pd_scored` argument — `cal_pd`/`thin_file_flag` are read directly off `result_df` (confirmed present there regardless of trimming, since `cal_pd` is already in `FINAL_OUTPUT_COLUMNS`; `thin_file_flag` needs the same `keep_intermediate=True`-before-trim treatment as `capacity_cap`/`policy_multiplier`, so it is covered by the same fix). This is wrapped in the identical belt-and-braces `try/except` pattern already proven for the shadow risk multiplier (`run_extrafloat_limit_engine.py:207-220`): a bug inside Decision Log construction must never fail the live pipeline run, and must never fail silently — logged loudly via `logger.exception`, with the pipeline's own return value completely unaffected either way, and identical to what a caller would have received with `keep_intermediate` requested at its original value.

This keeps `run_extrafloat_limit_engine.py` itself completely untouched — the only production file this plan modifies is the top-level `run_credit_risk_pipeline.py` orchestrator, never the lower-level engine module.

## 5. Verification Plan (scratchpad, not committed — same discipline as every prior script)

1. **Formula promotion regression**: re-run `derive_capacity_function_and_backtest_frontier.py`'s existing scratchpad suite unchanged against the new `extrafloat/capacity/capacity_function.py` import, confirming zero behavioral change from the move.
2. **Availability-status semantics** (corrected): a fixture with a genuinely-missing `cash_in_value_1m` (NaN) vs. a present-but-zero `payment_value_1m`, asserting `cash_in_availability_status`/`payment_availability_status` correctly distinguish `MISSING` from `NONPOSITIVE` at the *component* level, pre-fill — while `float_availability_status` is derived only from the resulting post-fill `Float` value's own sign (`OK` if `>0`, `NONPOSITIVE` if `<=0`), never from which component was null. A fixture where both components are null (`Float` fills to exactly `0`) must assert `float_availability_status=NONPOSITIVE`, not `MISSING` — `MISSING` is reserved for the agent having no row in `df_agent` at all.
3. **Capacity-triple invariants**: both the ratio check (#2) and the independent formula-reproducibility check (#10) on known-good and deliberately-poisoned fixtures (e.g. a stale weights artifact silently shifted), confirming each check fires independently — the ratio check must not mask a shared systematic error, and vice versa.
4. **`capacity_estimable` derivation** (#11): all eight combinations of the three status fields, confirming the boolean gate matches exactly.
5. **PIT invariant** (#9): a fixture with `fundamentals_snapshot_date >= scoring_run_timestamp` (poisoned), confirming `verify_decision_log_invariants` raises rather than silently NULLing the row.
6. **`challenger_risk_signal`/`challenger_risk_signal_type` routing**: thin vs. thick fixtures, confirming `cal_pd` nullability matches the route exactly (#3) and that `challenger_risk_signal` never silently falls back to a wrong-route value.
7. **`risk_score_live + cal_pd = 1` invariant**: a thick-route fixture engineered to violate it by a known epsilon, confirming the check fires at exactly the intended tolerance boundary, not looser or tighter.
8. **`shadow_decision_id` determinism/uniqueness**: same `(run_id, agent_msisdn)` pair hashed twice must match exactly; two different agents in the same run must never collide (a real, not just theoretical, collision check across a large synthetic population).
9. **Disagreement guard**: `l_current=0` fixture, confirming `disagreement_ratio_*` is NULL, never Inf or a divide-by-zero warning.
10. **End-to-end**: a full synthetic `(df_agent, result_df)` pair — `result_df` built with intermediate columns intact, matching the corrected Section 4 insertion point — covering every branch above in one run, asserting `build_decision_log_rows` produces exactly the frozen schema's column set (with the two renames applied) and that `verify_decision_log_invariants` passes cleanly on it. **Hard release gate, not just a nice-to-have**: a companion check asserts DataFrame equivalence — identical columns, identical column order, identical dtypes where contractually relevant, identical values, identical row order — between `run_credit_risk_pipeline`'s output at `keep_intermediate=False` with the new Stage 9 block present vs. with it absent (flag off). This gate must pass before the insertion-point change can be committed; it is the only thing standing between this change and every existing caller's output contract.
11. **Forbidden-columns regression**: confirm the logger never reads `assigned_limit`'s own upstream *target* equivalents (there is no historical-assignment-as-target risk here, since this is prospective, but the check is cheap insurance against an accidental copy-paste from the retrospective scripts) and never writes to any existing `result_df` column.
12. **Partial Float-component missingness** (new, from review): two fixtures — `cash_in_value_1m` populated / `payment_value_1m` NULL, and the reverse — asserting the frozen `Float` value is still computed correctly in both (not NaN-propagated), `float_availability_status` reflects the *resulting* `Float`'s own sign (not a component null), `capacity_estimable` is not wrongly forced to `False`, and `cash_in_availability_status`/`payment_availability_status` correctly record which specific component was null, separately from the Float/estimability result.
13. **Shadow-logging atomicity/completeness** (new, from review): a fixture where one agent's row deliberately fails construction (e.g. a poisoned snapshot date) partway through an otherwise-valid batch. Asserts the failure does not result in a `SUCCESS`-labeled manifest with a silently reduced `logged_agent_count` masquerading as complete — either the write is all-or-nothing for the batch (no partial persistence), or the run manifest (Section 3.3) records `PARTIAL`/`FAILED` with `logged_agent_count < expected_agent_count`, and no code path produces `SUCCESS` when they differ.

Bugs found during this verification are fixed before anything is committed, per the same standing discipline as every prior script in this workstream. No real-data run happens until explicitly asked for.

## 6. Execution Order

1. Formula promotion (Section 2) is approved; proceed directly. Storage sink (Section 3.4) remains open — build and verify the pure DataFrame layer regardless, and leave the pipeline integration behind the off-by-default feature flag until a sink is confirmed.
2. Promote `capacity_function.py`; re-run its existing verification suite unchanged; commit only if it passes with zero behavioral change.
3. Write `stage_4c_decision_logger.py` (including the corrected availability-status logic, the corrected `challenger_risk_signal` construction, and the run-level manifest from Section 3.3); verify against the Section 5 fixtures; fix any bugs found; commit.
4. Wire the insertion point into `run_credit_risk_pipeline.py` (Section 4), behind the Section 3.4 feature flag; re-run the pipeline's own existing test suite (`tests/pipeline/test_credit_risk_pipeline.py`) to confirm the live output is identical — same columns, same column order, same dtypes, same values, same row order — with the new block present (flag on, in test mode) vs. absent (flag off); commit.
5. Report back. Do not run any of this against real data without being asked — same standing instruction as every other script in this workstream.

## 7. What This Plan Does Not Do

No code has been written. No production file has been modified. Execution Log and Outcome Log logging are not designed here (Section 1). The storage sink is not chosen here (Section 3.4) — the feature stays behind an off-by-default flag until one is confirmed. Live shadow deployment does not happen until after the verification plan in Section 5 passes and the storage sink is confirmed.

## 8. Review Round 1 — Resolutions Applied

A review of the first draft of this plan required four corrections before freeze, plus confirmation of the two previously-open decisions. All are now incorporated above; this section records the verdict for traceability:

1. **Float-missingness semantics (Section 3.2, `extract_raw_capacity_inputs`)** — fixed. `float_availability_status` is now derived from the frozen post-fill `Float` value, never from individual pre-fill component nulls; component-level data quality is tracked separately (`cash_in_availability_status`/`payment_availability_status`) and does not feed `capacity_estimable`. The first draft would have introduced a new capacity-eligibility rule never part of the `k=0.05` selection.
2. **Thin-route `challenger_risk_signal` construction (Section 3.2, `attach_challenger_risk_signal`)** — fixed. Both routes read `challenger_risk_signal` from the engine's existing pooled `cal_pd` column (already correct per-route, per `pd_model/modeling/inference.py`'s existing promotion logic); the output schema's own `cal_pd` field is a separate write, explicitly nulled for the thin route rather than mirroring the pooled column. The first draft's two statements about `cal_pd` were internally inconsistent.
3. **Run-level completeness/failure tracking (Section 3.3, new)** — added. A `try/except` protects production but does not by itself prevent silently analyzing an incomplete run's shadow population later; the new per-run manifest (`shadow_logging_status`/`expected_agent_count`/`logged_agent_count`/`failure_stage`) closes that gap without adding a new per-agent field.
4. **Storage sink as a deployment gate, not a production stub (Section 3.4)** — corrected. A `NotImplementedError` inside the `try/except`-wrapped production path was the wrong shape (it could make the system appear deployed while silently failing every write); the pipeline integration is now gated behind an explicit, off-by-default feature flag instead, so "sink unconfirmed" and "feature live" can never both be true.
5. **Formula promotion (Section 2)** — approved as originally proposed, no change.
6. **Insertion-point architecture (Section 4: internal `keep_intermediate=True`, build, then restore the caller's original output contract)** — approved as originally proposed. The equivalence test in Section 5, item 10 is elevated from a verification nicety to a **hard release gate**: DataFrame equivalence (identical columns, column order, dtypes, values, and row order) between the old and new code paths at `keep_intermediate=False`, stated in those precise terms rather than as "byte-identical," which would imply a literal serialization comparison this plan does not intend.
7. **`shadow_decision_id` (Section 3.2, `build_shadow_decision_id`)** — reframed, no functional change: the deterministic hash is a convenience surrogate key; the load-bearing protection is a `UNIQUE(run_id, agent_msisdn)` constraint on the two source columns.
8. **Verification plan (Section 5)** — two tests added: partial Float-component-missingness (one of `cash_in_value_1m`/`payment_value_1m` null, the other populated, confirming `Float` is still computed and `capacity_estimable` is not wrongly set to `False`), and shadow-logging atomicity/completeness (a construction failure partway through a batch must never leave a partial set of rows silently treated as a complete run — either the sink write is all-or-nothing for the batch, or the run manifest records `PARTIAL`/`FAILED`, never `SUCCESS`).

No further conceptual design cycle is expected after these corrections; the next step is synthetic implementation and verification against Section 5's fixture set, still with no real-data run.
