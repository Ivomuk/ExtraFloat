-- ============================================================================
-- loan_summary_query_validation_queries.sql
-- ============================================================================
-- Validation suite for data/loan_summary_query.txt, mirroring the approach
-- established in data/borrower_history_validation_queries.sql: a canonical
-- GATE 0 source layer, structural checks against the real output, and
-- targeted diagnostics for the specific risk areas this query's design
-- carries. Read that file's own header for the general philosophy this
-- one inherits; only points specific to loan_summary_query.txt are
-- repeated here.
--
-- UNLIKE borrower_history.txt, this file has NO checkpoint markers and no
-- stage-count concerns observed at design time (see loan_summary_query.txt's
-- own EXECUTION comment) -- it runs as a single statement, so GATE 1 below
-- can wrap the query directly in one CREATE VIEW, no build script needed.
-- If a live run ever does hit the warehouse's stage-count ceiling, the
-- documented split point is materializing loan_cure as a checkpoint table
-- first -- revisit this file's GATE 1 if that becomes necessary.
--
-- PARAMETERS -- substitute before running:
--   :validation_schema  -- schema to create GATE 0 views / output view in
--   :snapshot_dt         -- e.g. 20260731 -- MUST match the snapshot_dt
--                           literal baked into loan_summary_query.txt's own
--                           snapshots CTE for this run (this file's GATE 0
--                           views take it as a bind param; the production
--                           query itself hardcodes it -- keep them in sync)
--   :as_of_load_ts        -- e.g. TIMESTAMP '2026-08-20 00:00:00.000' --
--                           same reproducibility rationale, must match
-- ============================================================================
--
-- ============================================================================
-- ACCEPTANCE THRESHOLDS (proposed defaults -- ratify with the team before
-- using any of these as a go/no-go gate; consistent with borrower_history_
-- validation_queries.sql's own caveat that these are starting points)
-- ============================================================================
--   Final-output grain violations         (GATE 1a: rows with msisdn not
--                                           unique, null, or empty)          0
--   Final-output range violations         (GATE 1b: dates after snapshot_dt,
--                                           negative volumes/values,
--                                           non-monotonic 1M<=3M<=6M)        0
--   Column contract (12 required)         (GATE 1c)                         all present
--   Monthly-bucket reconciliation gap     (Section A: |SUM(M1..M6) - 6M
--                                           aggregate| / 6M aggregate)       CONTEXT,
--                                           NOT A GATE -- M1-M6 are fixed
--                                           30-day buckets (180 days total)
--                                           while 3M/6M use calendar-month
--                                           arithmetic (28-31 day months) --
--                                           some gap is structurally
--                                           expected, not a defect. Report
--                                           the size; investigate only if
--                                           it's far larger than the
--                                           day-count vs month-count
--                                           difference could explain.
--   Duplicate repayment_fid exposure      (Section B: same repayment_uid,
--                                           multiple repayment_fid rows,
--                                           landing in the 180-day window)   CONTEXT,
--                                           same finding as borrower_
--                                           history.txt's B2b investigation
--                                           -- report the volume inflation,
--                                           re-check against data/
--                                           repayment_uid_rebuild_
--                                           verification.sql's confirmed fix
--   Penalty count sanity                  (Section C: every loan's penalty
--                                           count in {0, 1, 2}, never > 2)   0 violations
--   Cure-timing sanity                    (Section D: principal_cure_ts
--                                           never precedes disbursement_ts)  0 violations
-- ============================================================================


-- ============================================================================
-- GATE 0 -- Canonical source layer (create once, reuse everywhere below)
-- ============================================================================
-- Mirrors loan_summary_query.txt's own disb_raw -> disb_dedup -> disb_windows,
-- repay_raw -> repay_dedup chain exactly (same casts, same filters, same
-- dedup ORDER BY, same as_of_load_ts freeze) -- if you change loan_summary_
-- query.txt's dedup/window logic, update these views to match.

CREATE OR REPLACE VIEW :validation_schema.vw_ls_disb_dedup AS
SELECT disbursement_fid, phonenumber, disbursement_ts, disbursed_amount
FROM (
SELECT d.*,
ROW_NUMBER() OVER (
PARTITION BY disbursement_fid
ORDER BY inserted_ts DESC
) rn
FROM (
SELECT
disbursement_fid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber,
try_cast(disbursement_ts AS timestamp) AS disbursement_ts,
cast(disbursement_amount_ugx AS double) AS disbursed_amount,
inserted_ts
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND try_cast(disbursement_ts AS timestamp) IS NOT NULL
AND disbursement_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date_key <= :snapshot_dt
AND date(try_cast(disbursement_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
AND inserted_ts <= :as_of_load_ts
) d
)
WHERE rn = 1;

CREATE OR REPLACE VIEW :validation_schema.vw_ls_disb_windows AS
SELECT
disbursement_fid,
phonenumber,
disbursement_ts,
disbursed_amount,
LEAD(disbursement_ts) OVER (
PARTITION BY phonenumber ORDER BY disbursement_ts, disbursement_fid
) AS next_disbursement_ts
FROM :validation_schema.vw_ls_disb_dedup;

CREATE OR REPLACE VIEW :validation_schema.vw_ls_repay_dedup AS
SELECT repayment_fid, repayment_uid, phonenumber, repayment_ts, repayment_amount
FROM (
SELECT r.*,
ROW_NUMBER() OVER (
PARTITION BY repayment_fid
ORDER BY inserted_ts DESC
) rn
FROM (
SELECT
repayment_fid,
repayment_uid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber,
try_cast(repayment_ts AS timestamp) AS repayment_ts,
cast(repayment_amount_ugx AS double) AS repayment_amount,
inserted_ts
FROM analytics.momo_loan_book_tracker_repayments_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND try_cast(repayment_ts AS timestamp) IS NOT NULL
AND repayment_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date_key <= :snapshot_dt
AND date(try_cast(repayment_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
AND inserted_ts <= :as_of_load_ts
) r
)
WHERE rn = 1;

CREATE OR REPLACE VIEW :validation_schema.vw_ls_base AS
SELECT DISTINCT
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS msisdn
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND try_cast(disbursement_ts AS timestamp) IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date_key <= :snapshot_dt
AND date(try_cast(disbursement_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
AND inserted_ts <= :as_of_load_ts;

-- The production query itself, as a view -- gives a real compile check
-- (CREATE VIEW fails if it doesn't parse/analyze) and a queryable target
-- for GATE 1. ACTION REQUIRED: do NOT hand-paste data/loan_summary_
-- query.txt here -- manual assembly risks validating a different revision
-- than what's actually checked in. Instead run:
--   python scripts/build_vw_ls_output.py <validation_schema> > vw_ls_output.sql
-- and execute the generated vw_ls_output.sql. Unlike borrower_history.txt's
-- build_vw_bh_output.py, this produces a SINGLE statement (loan_summary_
-- query.txt has no checkpoint markers and runs as one statement -- see the
-- header of this file), so there's nothing to split: just
--   CREATE OR REPLACE VIEW :validation_schema.vw_ls_output AS
--   <data/loan_summary_query.txt's query body, verbatim>
--   ;
-- The generated file also stamps the git commit SHA of data/loan_summary_
-- query.txt as a comment, and echoes the exact :snapshot_dt/:as_of_load_ts
-- literals baked into that file -- use those SAME values everywhere else in
-- this validation file (GATE 0's views above), or GATE 0 and vw_ls_output
-- silently validate two different cutoffs with no error raised. Record the
-- SHA in GATE 6. The statement below is a structural placeholder only,
-- showing what the generated file's shape looks like -- it is NOT meant to
-- be run as written.
-- CREATE OR REPLACE VIEW :validation_schema.vw_ls_output AS
-- <generated from data/loan_summary_query.txt by scripts/build_vw_ls_output.py>
-- ;


-- ============================================================================
-- GATE 1 -- Final-output checks (against vw_ls_output)
-- ============================================================================

-- 1a. Grain: exactly one row per msisdn, no null/empty identifiers.
SELECT
COUNT(*) AS total_rows,
COUNT(DISTINCT msisdn) AS distinct_msisdns,
COUNT(*) - COUNT(DISTINCT msisdn) AS grain_violation_count,
SUM(CASE WHEN msisdn IS NULL OR msisdn = '' THEN 1 ELSE 0 END) AS null_or_empty_msisdn
FROM :validation_schema.vw_ls_output;
-- RESULT:
-- INTERPRETATION: grain_violation_count and null_or_empty_msisdn must both
-- be 0 -- prepare_loan_summary_recent_features() assumes one row per
-- agent (base is built from SELECT DISTINCT msisdn, and features is
-- grouped by msisdn+snapshot_dt with a single snapshot_dt, so a violation
-- here means the base/features join fanned out unexpectedly).

-- 1b. Range and monotonicity checks. 1M/3M/6M and M1-M6 windows are
-- structurally nested (see Section A below for the exact relationship),
-- so vol/val/penalty counts should be monotonically non-decreasing from
-- 1M to 3M to 6M for every agent -- a violation means the window
-- boundaries in win/t_by_txn have drifted apart from each other.
SELECT
SUM(CASE WHEN Last_disbursement_date > date_parse(cast(snapshot_dt AS varchar), '%Y%m%d') THEN 1 ELSE 0 END) AS bad_last_disbursement_date,
SUM(CASE WHEN Last_repayment_date > date_parse(cast(snapshot_dt AS varchar), '%Y%m%d') THEN 1 ELSE 0 END) AS bad_last_repayment_date,
SUM(CASE WHEN disbursement_vol_1M < 0 OR repayment_vol_1M < 0 OR penalties_1M < 0 THEN 1 ELSE 0 END) AS bad_negative_counts,
SUM(CASE WHEN disbursement_val_1M < 0 OR repayment_val_1M < 0 THEN 1 ELSE 0 END) AS bad_negative_values,
SUM(CASE WHEN disbursement_vol_1M > disbursement_vol_3M OR disbursement_vol_3M > disbursement_vol_6M THEN 1 ELSE 0 END) AS bad_disbursement_vol_monotonicity,
SUM(CASE WHEN repayment_vol_1M > repayment_vol_3M OR repayment_vol_3M > repayment_vol_6M THEN 1 ELSE 0 END) AS bad_repayment_vol_monotonicity,
SUM(CASE WHEN penalties_1M > penalties_3M OR penalties_3M > penalties_6M THEN 1 ELSE 0 END) AS bad_penalties_monotonicity,
SUM(CASE WHEN disbursement_val_1M > disbursement_val_3M OR disbursement_val_3M > disbursement_val_6M THEN 1 ELSE 0 END) AS bad_disbursement_val_monotonicity,
SUM(CASE WHEN repayment_val_1M > repayment_val_3M OR repayment_val_3M > repayment_val_6M THEN 1 ELSE 0 END) AS bad_repayment_val_monotonicity
FROM :validation_schema.vw_ls_output;
-- RESULT:
-- INTERPRETATION: every bad_* column must be 0. Dates after snapshot_dt or
-- negative counts/values point at a real bug in the window filters or
-- casts; a monotonicity violation means the 1M/3M/6M nesting assumption
-- (each window is a superset of the shorter one) has been broken somewhere
-- in win/t_by_txn's boundary definitions.

-- 1c. Column contract: confirm every column
-- prepare_loan_summary_recent_features() hard-requires is present. Run
-- this via the query engine's information_schema / DESCRIBE equivalent
-- against vw_ls_output, or simply SELECT each column by name -- if any is
-- missing the query below fails to parse, which is itself the signal.
SELECT
msisdn, snapshot_dt, Last_disbursement_date, Last_repayment_date,
disbursement_vol_1M, disbursement_val_1M, repayment_vol_1M, repayment_val_1M,
penalties_1M, disbursement_val_3M, repayment_val_3M, penalties_3M
FROM :validation_schema.vw_ls_output
LIMIT 1;
-- RESULT:
-- INTERPRETATION: success (a row or an empty result with no error) confirms
-- all 12 columns LOAN_SUMMARY_REQUIRED_COLUMNS needs are present and
-- correctly named (case-insensitively -- extrafloat_data_loaders.py
-- lowercases every column on load, confirmed against extrafloat_data_
-- loaders.py:168, so the mixed-case names in loan_summary_query.txt's own
-- SELECT are not a bug). A parse error naming a missing column is the
-- failure signal here, not a row count.


-- ============================================================================
-- Section A -- Monthly-bucket reconciliation (context, not a gate)
-- ============================================================================
-- M1-M6 are fixed 30-day buckets covering exactly 180 days
-- (dt_d30/d60/.../d180); 3M/6M are calendar-month arithmetic
-- (date_add('month', -3/-6, snapshot_dt)), which is 89-92 / 181-184 days
-- depending on which months are spanned. SUM(M1..M3) vs 3M, and
-- SUM(M1..M6) vs 6M, should be CLOSE but not necessarily identical --
-- this quantifies the gap so it can be told apart from a real bug.
SELECT
SUM(disbursement_vol_M1 + disbursement_vol_M2 + disbursement_vol_M3) AS sum_disb_vol_m1_m3,
SUM(disbursement_vol_3M) AS disb_vol_3M,
SUM(disbursement_vol_M1 + disbursement_vol_M2 + disbursement_vol_M3 + disbursement_vol_M4 + disbursement_vol_M5 + disbursement_vol_M6) AS sum_disb_vol_m1_m6,
SUM(disbursement_vol_6M) AS disb_vol_6M,
SUM(Repayment_vol_M1 + Repayment_vol_M2 + Repayment_vol_M3) AS sum_repay_vol_m1_m3,
SUM(repayment_vol_3M) AS repay_vol_3M,
SUM(Repayment_vol_M1 + Repayment_vol_M2 + Repayment_vol_M3 + Repayment_vol_M4 + Repayment_vol_M5 + Repayment_vol_M6) AS sum_repay_vol_m1_m6,
SUM(repayment_vol_6M) AS repay_vol_6M
FROM :validation_schema.vw_ls_output;
-- RESULT:
-- INTERPRETATION: compute |sum_*_m1_m3 - *_3M| / *_3M and the m1_m6/6M
-- equivalent yourself from these two columns each (deliberately not
-- precomputed as a ratio, same reasoning as borrower_history_validation_
-- queries.sql's B2b comment -- dividing aggregates inside the SELECT can
-- hide which raw numbers produced it). A gap on the order of a few percent
-- (roughly matching how much 89-92/181-184 actual days differs from the
-- 90/180 the M-buckets assume) is expected and not a defect. A gap far
-- larger than that -- e.g. 3M or 6M badly UNDER-counting relative to the
-- M-bucket sum -- would point at a real boundary bug in win's dt_m3/dt_m6
-- calculation instead.


-- ============================================================================
-- Section B -- Duplicate repayment_fid exposure (context, not a gate)
-- ============================================================================
-- repayment_val_1M/3M/6M and repayment_vol_* pull directly from
-- vw_ls_repay_dedup (deduped by repayment_fid only) -- NOT through the
-- phonenumber-window attribution heuristic at all, since these are simple
-- "did a repayment happen in this window" facts, not per-loan
-- reconciliation. That means they're exposed to the SAME duplicate-
-- repayment_fid-under-one-repayment_uid batch-tagging issue documented at
-- length in borrower_history_validation_queries.sql's B2b investigation
-- (data/repayment_uid_dedup_test.sql and siblings), independent of
-- attribution -- a real repayment posted multiple times under different
-- repayment_fid values (same repayment_uid) inflates these features
-- directly. penalties_1M/3M/6M carry a SECOND, different exposure: they DO
-- go through repay_attributed/loan_cure's phonenumber-window heuristic, so
-- they're also exposed to loan_uid-merge-style misattribution, not just
-- duplicate posting.
WITH windowed_repay AS (
SELECT repayment_fid, repayment_uid, repayment_amount
FROM :validation_schema.vw_ls_repay_dedup
WHERE repayment_ts > date_add('day', -180, date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d'))
AND repayment_ts <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
)
SELECT
COUNT(*) AS total_repayment_rows_in_window,
approx_distinct(COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR))) AS distinct_repayment_uids_approx,
COUNT(*) - approx_distinct(COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR))) AS duplicate_rows_by_uid_approx,
ROUND(100.0 * (COUNT(*) - approx_distinct(COALESCE(CAST(repayment_uid AS VARCHAR), CAST(repayment_fid AS VARCHAR))))
/ NULLIF(COUNT(*), 0), 2) AS pct_rows_are_uid_duplicates_approx,
SUM(repayment_amount) AS total_repayment_amount_in_window
FROM windowed_repay;
-- RESULT:
-- INTERPRETATION: compare pct_rows_are_uid_duplicates_approx against the
-- pre-rebuild 40.6% and post-rebuild (Jan-Mar) 4.1% figures from data/
-- repayment_uid_dedup_test.sql / repayment_uid_rebuild_verification.sql --
-- this tells you directly how much repayment_val/vol_1M/3M/6M are
-- currently inflated by duplicate postings for THIS query's specific
-- 180-day trailing window, not just the full historical table.


-- ============================================================================
-- Section C -- Penalty count sanity
-- ============================================================================
-- penalties_1M/3M/6M are counts of synthetic 24h/48h penalty EVENTS per
-- loan, at most 2 per loan (one at +24h, one at +48h) by construction of
-- penalty_events -- so for any single agent, penalties_6M (the widest
-- window) should never exceed 2 x (number of loans disbursed in the
-- trailing 180 days). This is a looser check than an exact bound (an
-- agent with many loans could legitimately have a high penalty count) but
-- catches a broken penalty_events join (e.g. a fan-out multiplying
-- penalty rows) immediately.
SELECT
o.msisdn,
o.penalties_6M,
COUNT(DISTINCT d.disbursement_fid) * 2 AS max_possible_penalties_6M
FROM :validation_schema.vw_ls_output o
JOIN :validation_schema.vw_ls_disb_dedup d ON d.phonenumber = o.msisdn
WHERE d.disbursement_ts > date_add('day', -180, date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d'))
AND d.disbursement_ts <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
GROUP BY o.msisdn, o.penalties_6M
HAVING o.penalties_6M > COUNT(DISTINCT d.disbursement_fid) * 2;
-- RESULT:
-- INTERPRETATION: must return ZERO rows. Any row here is an agent whose
-- penalty count exceeds what's structurally possible (at most 2 synthetic
-- penalty events per loan disbursed in the window) -- a fan-out bug in
-- penalty_events or its join into t_pull, not a business-logic edge case.


-- ============================================================================
-- Section D -- Cure-timing sanity
-- ============================================================================
-- principal_cure_ts (used only internally by loan_cure/penalty_events, not
-- exposed in the final output) must never precede its own loan's
-- disbursement_ts -- a violation would mean the cumulative-repaid window
-- function in loan_cure is drawing from the wrong partition or the
-- forward-fill attribution (attribution_filled) is leaking a different
-- loan's disbursement_fid onto a repayment that predates this loan
-- entirely. Rebuild loan_cure's logic inline here since it isn't exposed
-- as its own view.
WITH disb_windows AS (
SELECT * FROM :validation_schema.vw_ls_disb_windows
),
attribution_timeline AS (
SELECT phonenumber, disbursement_ts AS event_ts, disbursed_amount AS raw_amount, 0 AS is_repayment, disbursement_fid
FROM disb_windows
UNION ALL
SELECT phonenumber, repayment_ts AS event_ts, repayment_amount AS raw_amount, 1 AS is_repayment, NULL AS disbursement_fid
FROM :validation_schema.vw_ls_repay_dedup
),
attribution_filled AS (
SELECT phonenumber, event_ts, raw_amount, is_repayment,
LAST_VALUE(disbursement_fid) IGNORE NULLS OVER (
PARTITION BY phonenumber ORDER BY event_ts, is_repayment, disbursement_fid
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
) AS disbursement_fid
FROM attribution_timeline
),
repay_attributed AS (
SELECT disbursement_fid, phonenumber, event_ts AS repayment_ts, ABS(raw_amount) AS repayment_amount
FROM attribution_filled
WHERE is_repayment = 1 AND disbursement_fid IS NOT NULL
),
loan_cure AS (
SELECT
w.disbursement_fid, w.disbursement_ts, w.disbursed_amount,
MIN(CASE WHEN cum.cumulative_repaid >= w.disbursed_amount THEN cum.repayment_ts END) AS principal_cure_ts
FROM disb_windows w
LEFT JOIN (
SELECT disbursement_fid, repayment_ts,
SUM(repayment_amount) OVER (PARTITION BY disbursement_fid ORDER BY repayment_ts) AS cumulative_repaid
FROM repay_attributed
) cum ON cum.disbursement_fid = w.disbursement_fid
GROUP BY w.disbursement_fid, w.disbursement_ts, w.disbursed_amount
)
SELECT COUNT(*) AS n_cure_before_disbursement
FROM loan_cure
WHERE principal_cure_ts IS NOT NULL AND principal_cure_ts < disbursement_ts;
-- RESULT:
-- INTERPRETATION: must be 0. Any nonzero count means a repayment is being
-- attributed to a loan disbursed AFTER that repayment happened --
-- impossible under correct forward-fill semantics, and worth tracing by
-- hand (same method as borrower_history.txt's B2b investigation) if it
-- ever occurs.


-- ============================================================================
-- GATE 6 -- Record approval evidence
-- ============================================================================
-- Fill this in every time this file is actually run. An unfilled template
-- is not evidence, per the same point made in borrower_history_
-- validation_queries.sql's own GATE 6.
--
--   Execution date:                    ____________________
--   loan_summary_query.txt git SHA:    ____________________ (from scripts/build_vw_ls_output.py's
--                                       output comment -- proves which revision was actually validated)
--   snapshot_dt used:                  ____________________
--   as_of_load_ts used:                ____________________
--   Query engine/version:              ____________________
--   GATE 1a grain violations:          ____________________
--   GATE 1b range/monotonicity violations: ________________
--   GATE 1c column contract:           ____________________ (pass/fail, missing columns if any)
--   Section A monthly-bucket gap (3M / 6M): ________________ (context, not pass/fail)
--   Section B duplicate repayment_fid exposure: ___________ (context, compare against B2b's documented figures)
--   Section C penalty count sanity:    ____________________ (must be 0 rows)
--   Section D cure-timing sanity:      ____________________ (must be 0)
--   Threshold decision:                ____________________ (pass / fail / conditional, and why)
--   Approved by:                       ____________________


-- ============================================================================
-- TEARDOWN -- drop the GATE 0 views once validation is complete, if
-- :validation_schema is a shared scratch database other work might collide
-- with names in.
-- ============================================================================
-- DROP VIEW IF EXISTS :validation_schema.vw_ls_output;
-- DROP VIEW IF EXISTS :validation_schema.vw_ls_base;
-- DROP VIEW IF EXISTS :validation_schema.vw_ls_repay_dedup;
-- DROP VIEW IF EXISTS :validation_schema.vw_ls_disb_windows;
-- DROP VIEW IF EXISTS :validation_schema.vw_ls_disb_dedup;
