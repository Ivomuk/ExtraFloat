-- ============================================================================
-- loan_state_closed_loan_guard_gap_impact_test.sql -- ad-hoc diagnostic, not
-- part of the committed validation suite. Quantifies the real-world impact
-- of the closed-loan-detection guard gap fixed in
-- data/loan_state_query_updated_materialized.txt's prior_loan_state_
-- candidates CTE (the same-day "loan 1 already closed" admission branch
-- checked closure_date IS NOT NULL alone, missing the
-- loan_status IN ('SETTLED','CLOSED','OVERPAID') guard used everywhere
-- else in that file -- see that file's lines 1193-1218 after the fix).
--
-- Approximates that CTE's same-day-loan-1 scenario directly against raw
-- tables (loan_uid/loan_seq/tmp_target_loans aren't available outside that
-- file's own statement pipeline): for every customer with 2+ disbursements
-- on the same calendar day, pulls the EARLIER same-day loan's ("loan 1")
-- snapshot row from that exact date_key, and classifies whether the OLD
-- (buggy) condition vs the NEW (fixed) condition would have admitted it as
-- a valid prior loan for the LATER same-day loan ("loan 2").
--
-- Self-contained (raw tables), bounded to date_key <= 20260609.
-- ============================================================================

WITH disb_dedup AS (
SELECT disbursement_fid, phonenumber, disbursement_ts
FROM (
SELECT d.*,
ROW_NUMBER() OVER (PARTITION BY disbursement_fid ORDER BY inserted_ts DESC) rn
FROM (
SELECT
disbursement_fid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber,
try_cast(disbursement_ts AS timestamp) AS disbursement_ts,
inserted_ts
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND try_cast(disbursement_ts AS timestamp) IS NOT NULL
AND disbursement_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date_key <= 20260609
) d
)
WHERE rn = 1
),
same_day_disb AS (
-- Rank same-customer, same-calendar-day disbursements by time. Loan 1 =
-- rank 1 (earlier), loan 2 = rank 2+ (later) -- the exact scenario the
-- fixed branch is about: does loan 1's own same-day state row count as a
-- legitimate "prior loan" for loan 2.
SELECT
disbursement_fid,
phonenumber,
disbursement_ts,
CAST(date_format(disbursement_ts, '%Y%m%d') AS BIGINT) AS loan_date_key,
ROW_NUMBER() OVER (
PARTITION BY phonenumber, CAST(date_format(disbursement_ts, '%Y%m%d') AS BIGINT)
ORDER BY disbursement_ts, disbursement_fid
) AS same_day_rank,
COUNT(*) OVER (
PARTITION BY phonenumber, CAST(date_format(disbursement_ts, '%Y%m%d') AS BIGINT)
) AS same_day_disb_count
FROM disb_dedup
),
loan1_candidates AS (
SELECT disbursement_fid AS loan1_fid, phonenumber, loan_date_key
FROM same_day_disb
WHERE same_day_rank = 1
AND same_day_disb_count > 1
),
loan1_same_day_state AS (
-- Loan 1's own snapshot row from the exact same date_key as the
-- same-day disbursement cluster -- mirrors
-- "ls.current_loan_start_date = d.loan_date AND ls.state_date <= d.loan_date"
-- collapsed to the same-day case, since this table is date_key-grained.
SELECT disbursement_fid, loan_status, closure_date, is_anomaly_open
FROM (
SELECT lsd.*,
ROW_NUMBER() OVER (PARTITION BY disbursement_fid, date_key ORDER BY inserted_ts DESC) rn
FROM analytics.momo_loan_book_tracker_loan_state_daily lsd
WHERE ova = 'XTRAFLOAT-AGENT'
AND disbursement_fid IS NOT NULL
AND date_key <= 20260609
)
WHERE rn = 1
),
joined AS (
SELECT
c.loan1_fid,
c.phonenumber,
c.loan_date_key,
s.loan_status,
s.closure_date,
s.is_anomaly_open,
CASE WHEN s.closure_date IS NOT NULL
AND CAST(date_format(s.closure_date, '%Y%m%d') AS BIGINT) <= c.loan_date_key
THEN 1 ELSE 0 END AS old_code_would_admit,
CASE WHEN s.closure_date IS NOT NULL
AND CAST(date_format(s.closure_date, '%Y%m%d') AS BIGINT) <= c.loan_date_key
AND s.loan_status IN ('SETTLED', 'CLOSED', 'OVERPAID')
THEN 1 ELSE 0 END AS new_code_admits
FROM loan1_candidates c
JOIN loan1_same_day_state s ON s.disbursement_fid = c.loan1_fid
)
SELECT
COUNT(*) AS n_same_day_loan1_candidates,
SUM(old_code_would_admit) AS n_old_code_admitted,
SUM(new_code_admits) AS n_new_code_admits,
SUM(CASE WHEN old_code_would_admit = 1 AND new_code_admits = 0 THEN 1 ELSE 0 END) AS n_affected_by_fix,
SUM(CASE WHEN old_code_would_admit = 1 AND new_code_admits = 0
AND (loan_status = 'ANOMALY_OPEN' OR is_anomaly_open = true) THEN 1 ELSE 0 END) AS n_affected_and_anomaly_open,
SUM(CASE WHEN old_code_would_admit = 1 AND new_code_admits = 0
AND loan_status NOT IN ('ANOMALY_OPEN') AND NOT COALESCE(is_anomaly_open, false) THEN 1 ELSE 0 END) AS n_affected_other_non_terminal_status
FROM joined;
-- RESULT:
-- INTERPRETATION: n_affected_by_fix is the count of same-day "loan 1"
-- candidates that the OLD (buggy) condition would have wrongly admitted as
-- a closed prior loan, which the NEW (fixed) condition correctly excludes
-- -- i.e. the exact population this fix changes. n_affected_and_anomaly_open
-- isolates how much of that is specifically the failure mode this session
-- has been investigating all along (a stale closure_date on a genuinely
-- ANOMALY_OPEN row) versus n_affected_other_non_terminal_status (some other
-- non-terminal loan_status, e.g. OPEN, also carrying a stale closure_date).
-- A small n_affected_by_fix relative to n_same_day_loan1_candidates means
-- this was a real but narrow-blast-radius gap, consistent with it only
-- affecting prior-loan-history training features (historical_anomaly_
-- open_loan_rate and similar), not the risk-cap haircut signal already
-- confirmed clean.
