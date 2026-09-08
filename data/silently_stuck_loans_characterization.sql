-- ============================================================================
-- silently_stuck_loans_characterization.sql -- ad-hoc diagnostic, not part
-- of the committed validation suite. anomaly_open_aging_only_test.sql
-- found 6,734 of 63,130 loans (customers with no second loan ever taken,
-- aged 7+ days) never resolve to CLOSED/SETTLED and never get flagged
-- ANOMALY_OPEN either (which requires a second loan by construction, per
-- that same test's 0-of-63,130 result) -- these loans are untracked by
-- ANY existing status mechanism. This characterizes that population
-- directly: what loan_status do they actually carry, how much UGX is
-- outstanding across them, how old are they, and is this a persistent
-- baseline rate or something concentrated in specific recent months.
--
-- Self-contained (raw tables), bounded to date_key <= 20260609.
-- ============================================================================

WITH disb_dedup AS (
SELECT disbursement_fid, phonenumber, disbursement_ts, disbursed_amount
FROM (
SELECT d.*,
ROW_NUMBER() OVER (PARTITION BY disbursement_fid ORDER BY inserted_ts DESC) rn
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
AND date_key <= 20260609
) d
)
WHERE rn = 1
),
disb_windows AS (
SELECT
disbursement_fid, phonenumber, disbursement_ts, disbursed_amount,
LEAD(disbursement_fid) OVER (PARTITION BY phonenumber ORDER BY disbursement_ts, disbursement_fid) AS next_disbursement_fid
FROM disb_dedup
),
loan_state_history AS (
SELECT disbursement_fid, date_key, loan_status, is_anomaly_open, days_aging
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
loan_summary AS (
SELECT
disbursement_fid,
MAX(CASE WHEN loan_status = 'ANOMALY_OPEN' OR is_anomaly_open = true THEN 1 ELSE 0 END) AS ever_anomaly_open,
MAX(CASE WHEN loan_status IN ('CLOSED', 'SETTLED') THEN 1 ELSE 0 END) AS ever_resolved
FROM loan_state_history
GROUP BY disbursement_fid
),
loan_state_latest AS (
SELECT disbursement_fid, loan_status, aging_bucket, days_aging, lifetime_repaid_ugx
FROM (
SELECT lsld.*,
ROW_NUMBER() OVER (PARTITION BY disbursement_fid ORDER BY date_key DESC) rn2
FROM (
SELECT *
FROM (
SELECT lsd.*,
ROW_NUMBER() OVER (PARTITION BY disbursement_fid, date_key ORDER BY inserted_ts DESC) rn
FROM analytics.momo_loan_book_tracker_loan_state_daily lsd
WHERE ova = 'XTRAFLOAT-AGENT'
AND disbursement_fid IS NOT NULL
AND date_key <= 20260609
)
WHERE rn = 1
) lsld
)
WHERE rn2 = 1
),
stuck_loans AS (
SELECT
w.disbursement_fid,
w.disbursement_ts,
w.disbursed_amount,
ls.loan_status AS latest_loan_status,
ls.aging_bucket,
ls.days_aging,
ls.lifetime_repaid_ugx,
(w.disbursed_amount - ls.lifetime_repaid_ugx) AS outstanding_ugx
FROM disb_windows w
JOIN loan_summary lsu ON lsu.disbursement_fid = w.disbursement_fid
JOIN loan_state_latest ls ON ls.disbursement_fid = w.disbursement_fid
WHERE w.next_disbursement_fid IS NULL
AND w.disbursement_ts <= TIMESTAMP '2026-06-02 00:00:00.000'
AND lsu.ever_anomaly_open = 0
AND lsu.ever_resolved = 0
)
-- Step 1: characterize by current loan_status and repayment activity.
SELECT
latest_loan_status,
COUNT(*) AS n_loans,
SUM(disbursed_amount) AS total_disbursed_ugx,
SUM(lifetime_repaid_ugx) AS total_repaid_ugx,
SUM(outstanding_ugx) AS total_outstanding_ugx,
COUNT_IF(lifetime_repaid_ugx = 0) AS n_zero_repayment_at_all,
ROUND(100.0 * COUNT_IF(lifetime_repaid_ugx = 0) / NULLIF(COUNT(*), 0), 2) AS pct_zero_repayment_at_all,
APPROX_PERCENTILE(days_aging, 0.5) AS median_days_aging,
MAX(days_aging) AS max_days_aging
FROM stuck_loans
GROUP BY latest_loan_status
ORDER BY n_loans DESC;
-- RESULT (Step 1):
-- INTERPRETATION: total_outstanding_ugx is the real financial exposure
-- number for this population -- unlike ANOMALY_OPEN loans (which mostly
-- self-resolve within days per anomaly_open_raw_timeline_examples.sql),
-- these have NO subsequent loan and NO resolution, so there is no
-- mechanism pulling them toward closure at all. pct_zero_repayment_at_all
-- close to 100% would mean these are fully abandoned from day one, not
-- loans making slow partial progress that simply haven't crossed a
-- closure threshold yet -- a meaningfully different (more concerning)
-- risk profile than a slow-but-progressing loan. Check what
-- latest_loan_status actually says -- if it is uniformly "OPEN", these
-- loans are indistinguishable in the raw status field from a loan that
-- was disbursed yesterday, even ones aged 100+ days -- a real reporting
-- gap regardless of the underlying repayment behavior.

-- Step 2: is this a persistent baseline rate or concentrated recently.
-- Rebuilds the same population from scratch as its own statement (Trino/
-- Athena does not carry CTEs across separate statements), and computes
-- the rate directly against ALL single-loan customers per month (not just
-- the stuck subset), so nothing needs cross-referencing against a
-- different file's population definition.
WITH disb_dedup AS (
SELECT disbursement_fid, phonenumber, disbursement_ts, disbursed_amount
FROM (
SELECT d.*,
ROW_NUMBER() OVER (PARTITION BY disbursement_fid ORDER BY inserted_ts DESC) rn
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
AND date_key <= 20260609
) d
)
WHERE rn = 1
),
disb_windows AS (
SELECT
disbursement_fid, phonenumber, disbursement_ts, disbursed_amount,
LEAD(disbursement_fid) OVER (PARTITION BY phonenumber ORDER BY disbursement_ts, disbursement_fid) AS next_disbursement_fid
FROM disb_dedup
),
loan_state_history AS (
SELECT disbursement_fid, date_key, loan_status, is_anomaly_open, lifetime_repaid_ugx
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
loan_summary AS (
SELECT
disbursement_fid,
MAX(CASE WHEN loan_status = 'ANOMALY_OPEN' OR is_anomaly_open = true THEN 1 ELSE 0 END) AS ever_anomaly_open,
MAX(CASE WHEN loan_status IN ('CLOSED', 'SETTLED') THEN 1 ELSE 0 END) AS ever_resolved,
MAX(lifetime_repaid_ugx) AS max_lifetime_repaid_ugx
FROM loan_state_history
GROUP BY disbursement_fid
),
eligible AS (
SELECT w.disbursement_fid, w.disbursement_ts, w.disbursed_amount, lsu.ever_anomaly_open, lsu.ever_resolved,
lsu.max_lifetime_repaid_ugx
FROM disb_windows w
JOIN loan_summary lsu ON lsu.disbursement_fid = w.disbursement_fid
WHERE w.next_disbursement_fid IS NULL
AND w.disbursement_ts <= TIMESTAMP '2026-06-02 00:00:00.000'
)
SELECT
date_trunc('month', disbursement_ts) AS disb_month,
COUNT(*) AS n_eligible_single_loan_customers,
COUNT_IF(ever_anomaly_open = 0 AND ever_resolved = 0) AS n_stuck_loans,
ROUND(100.0 * COUNT_IF(ever_anomaly_open = 0 AND ever_resolved = 0) / NULLIF(COUNT(*), 0), 2) AS pct_stuck,
SUM(CASE WHEN ever_anomaly_open = 0 AND ever_resolved = 0 THEN disbursed_amount - max_lifetime_repaid_ugx ELSE 0 END) AS total_outstanding_ugx
FROM eligible
GROUP BY date_trunc('month', disbursement_ts)
ORDER BY disb_month;
-- RESULT (Step 2):
-- INTERPRETATION: pct_stuck flat across months means this is a
-- persistent baseline behavior (a fixed fraction of single-loan customers
-- always end up here, regardless of when they borrowed). pct_stuck
-- climbing toward the most recent months means it is a newer, emerging
-- problem worth escalating with the same urgency as the ANOMALY_OPEN
-- trend -- though also check whether the most recent 1-2 months' loans
-- have simply not had enough elapsed time to resolve yet (the
-- disbursement_ts <= 2026-06-02 filter already guards against loans
-- younger than ~7 days, but a loan disbursed June 1st has had far less
-- time to resolve than one from January, independent of any real trend).
