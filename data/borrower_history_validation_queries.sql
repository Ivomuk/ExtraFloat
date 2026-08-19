-- ============================================================================
-- borrower_history_validation_queries.sql
-- ============================================================================
-- Diagnostics for the borrower_history.txt rewrite (xtrafloat_daily_trans ->
-- momo_loan_book_tracker_* tables). None of these have been run -- this repo
-- has no live warehouse connection, so every query here is prepared, not
-- executed. Run each section in Athena/Trino, read the interpretation note,
-- and treat the rewrite as unverified until this file's findings are filled
-- in (see the "RESULT:" placeholders).
--
-- Set these to match the borrower_history.txt run being validated:
--   :snapshot_dt      -- e.g. 20260531
--   :as_of_load_ts    -- e.g. TIMESTAMP '2026-06-01 00:00:00.000'
-- ============================================================================


-- ============================================================================
-- SECTION A -- Repayment amount semantics: gross or principal-only?
-- ============================================================================
-- Blocks cure-timing feature validity if wrong (see review). Uses ONLY
-- loan_state_daily's own authoritative totals -- independent of the
-- repay_attributed heuristic in borrower_history.txt, so this is a clean
-- check even if attribution turns out to be imperfect.
--
-- Logic: for CLOSED loans with a non-zero interest_and_penalty_ugx, compare
-- lifetime_repaid_ugx against (a) principal alone and (b) principal + charge.
-- Whichever ratio clusters near 1.0 tells you what repayment_amount_ugx
-- actually represents.
WITH closed_charged_loans AS (
SELECT
disbursement_fid,
lifetime_disbursed_ugx,
lifetime_repaid_ugx,
interest_and_penalty_ugx,
lifetime_repaid_ugx / NULLIF(lifetime_disbursed_ugx, 0) AS repaid_over_principal,
lifetime_repaid_ugx / NULLIF(lifetime_disbursed_ugx + interest_and_penalty_ugx, 0) AS repaid_over_gross
FROM (
SELECT lsd.*,
ROW_NUMBER() OVER (
PARTITION BY disbursement_fid
ORDER BY date_key DESC, inserted_ts DESC
) rn
FROM analytics.momo_loan_book_tracker_loan_state_daily lsd
WHERE loan_status = 'CLOSED'
AND interest_and_penalty_ugx > 0
AND date_key <= :snapshot_dt
)
WHERE rn = 1
)
SELECT
COUNT(*) AS n_closed_charged_loans,
approx_percentile(repaid_over_principal, 0.5) AS median_repaid_over_principal,
approx_percentile(repaid_over_gross, 0.5) AS median_repaid_over_gross,
SUM(CASE WHEN ABS(repaid_over_principal - 1.0) <= 0.02 THEN 1 ELSE 0 END) AS n_matches_principal_within_2pct,
SUM(CASE WHEN ABS(repaid_over_gross - 1.0) <= 0.02 THEN 1 ELSE 0 END) AS n_matches_gross_within_2pct
FROM closed_charged_loans;
-- RESULT:
-- INTERPRETATION: if n_matches_principal_within_2pct >> n_matches_gross_within_2pct,
-- repayment_amount_ugx is principal-only and borrower_history.txt's cure-timing
-- logic (comparing cumulative repayments to disbursed_amount) is correct as
-- written. If the reverse, repayments are gross-of-charges and principal_cure_ts
-- will read early; the fix is to floor the cure comparison against
-- (disbursed_amount only, excluding any charge component) -- confirm which
-- component of repayment_amount_ugx is charge vs principal before changing
-- the query, since the fix depends on the answer.


-- ============================================================================
-- SECTION B -- Repayment attribution quality (repay_attributed heuristic)
-- ============================================================================
-- Reproduces the borrower_history.txt attribution join, then checks it
-- against loan_state_daily's authoritative per-loan repaid total.
WITH disb AS (
SELECT
disbursement_fid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber,
try_cast(disbursement_ts AS timestamp) AS disbursement_ts,
cast(disbursement_amount_ugx AS double) AS disbursed_amount
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE try_cast(disbursement_ts AS timestamp) IS NOT NULL
AND disbursement_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date(try_cast(disbursement_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
AND inserted_ts <= :as_of_load_ts
),
disb_windows AS (
SELECT
disbursement_fid,
phonenumber,
disbursement_ts,
disbursed_amount,
LEAD(disbursement_ts) OVER (
PARTITION BY phonenumber ORDER BY disbursement_ts, disbursement_fid
) AS next_disbursement_ts
FROM disb
),
repay AS (
SELECT
repayment_fid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber,
try_cast(repayment_ts AS timestamp) AS repayment_ts,
cast(repayment_amount_ugx AS double) AS repayment_amount
FROM analytics.momo_loan_book_tracker_repayments_daily
WHERE try_cast(repayment_ts AS timestamp) IS NOT NULL
AND repayment_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date(try_cast(repayment_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
AND inserted_ts <= :as_of_load_ts
),
attributed AS (
SELECT
w.disbursement_fid,
r.repayment_fid,
r.repayment_ts,
r.repayment_amount,
w.disbursement_ts,
w.next_disbursement_ts,
-- flag repayments landing within 1 hour of the *next* disbursement --
-- ambiguous cases where the attribution boundary is a close call
CASE WHEN w.next_disbursement_ts IS NOT NULL
AND date_diff('second', r.repayment_ts, w.next_disbursement_ts) <= 3600
THEN 1 ELSE 0 END AS near_boundary_flag
FROM repay r
JOIN disb_windows w
ON r.phonenumber = w.phonenumber
AND r.repayment_ts >= w.disbursement_ts
AND (w.next_disbursement_ts IS NULL OR r.repayment_ts < w.next_disbursement_ts)
)

-- B1. Coverage: how many repayments got dropped for lack of a matching disbursement?
SELECT
(SELECT COUNT(*) FROM repay) AS total_repayments,
(SELECT COUNT(*) FROM attributed) AS attributed_repayments,
(SELECT COUNT(*) FROM repay) - (SELECT COUNT(*) FROM attributed) AS dropped_repayments,
(SELECT COUNT(*) FROM attributed WHERE near_boundary_flag = 1) AS near_next_disbursement_boundary;
-- RESULT:
-- INTERPRETATION: dropped_repayments should be small and explainable (e.g.
-- repayments for loans disbursed before the source table's coverage starts).
-- A large dropped count means the attribution logic is silently discarding
-- real cash flow. near_next_disbursement_boundary flags repayments that could
-- plausibly belong to either the current or the next loan -- inspect these by
-- hand; a high count here is the clearest sign the sequential-loans
-- assumption is being stressed (e.g. rapid top-up/reborrow behavior).

-- B2. Reconciliation against loan_state_daily's authoritative repaid total.
WITH per_loan_attributed AS (
SELECT disbursement_fid, SUM(repayment_amount) AS attributed_repaid
FROM attributed
GROUP BY disbursement_fid
),
loan_state_latest AS (
SELECT disbursement_fid, lifetime_repaid_ugx
FROM (
SELECT lsd.*,
ROW_NUMBER() OVER (
PARTITION BY disbursement_fid
ORDER BY date_key DESC, inserted_ts DESC
) rn
FROM analytics.momo_loan_book_tracker_loan_state_daily lsd
WHERE date_key <= :snapshot_dt
)
WHERE rn = 1
)
SELECT
COUNT(*) AS n_loans_compared,
SUM(CASE WHEN ABS(COALESCE(a.attributed_repaid, 0) - COALESCE(ls.lifetime_repaid_ugx, 0)) <= 1 THEN 1 ELSE 0 END) AS n_exact_match,
SUM(CASE WHEN ABS(COALESCE(a.attributed_repaid, 0) - COALESCE(ls.lifetime_repaid_ugx, 0)) > 1 THEN 1 ELSE 0 END) AS n_mismatch,
approx_percentile(ABS(COALESCE(a.attributed_repaid, 0) - COALESCE(ls.lifetime_repaid_ugx, 0)), 0.5) AS median_abs_diff_ugx,
approx_percentile(ABS(COALESCE(a.attributed_repaid, 0) - COALESCE(ls.lifetime_repaid_ugx, 0)), 0.95) AS p95_abs_diff_ugx
FROM loan_state_latest ls
LEFT JOIN per_loan_attributed a ON a.disbursement_fid = ls.disbursement_fid;
-- RESULT:
-- INTERPRETATION: n_mismatch should be low and p95_abs_diff_ugx small relative
-- to typical loan size. A high mismatch rate means the time-window heuristic
-- is materially misattributing repayments across loans -- do not treat the
-- rewrite as equivalent to the original until this reconciles.


-- ============================================================================
-- SECTION C -- ANOMALY_OPEN exclusion impact
-- ============================================================================
SELECT
COUNT(*) AS total_loans,
SUM(CASE WHEN loan_status = 'ANOMALY_OPEN' OR is_anomaly_open = true THEN 1 ELSE 0 END) AS anomaly_open_loans,
COUNT(DISTINCT customer_msisdn) AS total_borrowers,
COUNT(DISTINCT CASE WHEN loan_status = 'ANOMALY_OPEN' OR is_anomaly_open = true THEN customer_msisdn END) AS borrowers_with_an_anomaly_loan
FROM (
SELECT lsd.*,
ROW_NUMBER() OVER (
PARTITION BY disbursement_fid
ORDER BY date_key DESC, inserted_ts DESC
) rn
FROM analytics.momo_loan_book_tracker_loan_state_daily lsd
WHERE date_key <= :snapshot_dt
)
WHERE rn = 1;
-- RESULT:
-- INTERPRETATION: report anomaly_open_loans / total_loans and
-- borrowers_with_an_anomaly_loan / total_borrowers as the population-shift
-- cost of the exclusion decision -- record this figure alongside the
-- decision, since it changes every affected borrower's prior-loan windows,
-- streaks, and lifetime rates, not just the anomalous loan itself.


-- ============================================================================
-- SECTION D -- has_pre_window_history impact
-- ============================================================================
SELECT
COUNT(*) AS total_borrowers,
SUM(CASE WHEN flagged THEN 1 ELSE 0 END) AS borrowers_with_pre_window_history,
SUM(CASE WHEN flagged THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) AS pct_left_censored
FROM (
SELECT
customer_msisdn,
bool_or(has_pre_window_history) AS flagged
FROM analytics.momo_loan_book_tracker_loan_state_daily
WHERE date_key <= :snapshot_dt
GROUP BY customer_msisdn
);
-- RESULT:
-- INTERPRETATION: pct_left_censored is the share of borrowers whose
-- total_loans/first_loan_ts/lifetime rates from borrower_history.txt should
-- be treated as incomplete, not zero-history. Feed this into model
-- monitoring; if material, consider segmenting or excluding these borrowers
-- from lifetime-rate-dependent policy decisions until resolved.


-- ============================================================================
-- SECTION E -- Candidate old->new loan identity bridge
-- ============================================================================
-- disbursement_external_id / repayment_external_id may tie back to the old
-- transaction log's transactionid/requestid. Existence check only -- if these
-- match, they're a far better reconciliation key than the msisdn+timestamp
-- proxy used elsewhere in this file.
SELECT
COUNT(*) AS new_disbursements_with_external_id,
COUNT(DISTINCT d.disbursement_external_id) AS distinct_external_ids,
SUM(CASE WHEN t.requestid IS NOT NULL THEN 1 ELSE 0 END) AS matched_to_old_requestid
FROM analytics.momo_loan_book_tracker_disbursements_daily d
LEFT JOIN devdata.xtrafloat_daily_trans t
ON cast(t.requestid AS varchar) = cast(d.disbursement_external_id AS varchar)
WHERE d.disbursement_external_id IS NOT NULL
AND date(try_cast(d.disbursement_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d');
-- RESULT:
-- INTERPRETATION: if matched_to_old_requestid is a high fraction of
-- new_disbursements_with_external_id, disbursement_external_id is a real
-- bridge to the old requestid space -- use it (not msisdn+timestamp
-- proximity) for any old-vs-new loan-level comparison, including a tighter
-- version of Section B's reconciliation. If near-zero, external_id is not the
-- bridge and old-vs-new comparison must stay at the aggregate/population
-- level (Section C/D style), not loan-by-loan.
