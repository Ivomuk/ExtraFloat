-- ============================================================================
-- silently_stuck_loans_raw_examples.sql -- ad-hoc diagnostic, not part of
-- the committed validation suite. silently_stuck_loans_characterization.sql
-- established the aggregate shape of this population (6,734 loans: no
-- second loan ever disbursed, aged 7+ days, never resolved to CLOSED/
-- SETTLED, and therefore structurally incapable of ever being flagged
-- ANOMALY_OPEN either -- see anomaly_open_aging_only_test.sql). This pulls
-- actual raw rows for that population -- concrete, lookup-able
-- disbursement_fid/phonenumber examples -- so the table owner can pull
-- these up directly in their own systems, not just see an aggregate
-- percentage.
--
-- Two samples: (1) the most extreme cases by days_aging, the most
-- persuasive evidence that "OPEN" is indistinguishable from a day-1 loan
-- even at 100+ days; (2) a broader sample across the full population (not
-- just the extremes), since silently_stuck_loans_characterization.sql
-- found 94.8% of this population IS making some repayment progress -- the
-- extremes alone would misleadingly read as "abandoned debt," so a
-- representative sample matters for an accurate picture.
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
SELECT disbursement_fid, loan_status, is_anomaly_open
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
w.phonenumber,
w.disbursement_ts,
w.disbursed_amount,
ls.loan_status AS latest_loan_status,
ls.aging_bucket,
ls.days_aging,
ls.lifetime_repaid_ugx,
(w.disbursed_amount - ls.lifetime_repaid_ugx) AS outstanding_ugx,
ROUND(100.0 * ls.lifetime_repaid_ugx / NULLIF(w.disbursed_amount, 0), 1) AS pct_repaid
FROM disb_windows w
JOIN loan_summary lsu ON lsu.disbursement_fid = w.disbursement_fid
JOIN loan_state_latest ls ON ls.disbursement_fid = w.disbursement_fid
WHERE w.next_disbursement_fid IS NULL
AND w.disbursement_ts <= TIMESTAMP '2026-06-02 00:00:00.000'
AND lsu.ever_anomaly_open = 0
AND lsu.ever_resolved = 0
)
-- Sample 1: the most extreme cases -- oldest loans, still latest_loan_status
-- = 'OPEN', to show the taxonomy gap at its starkest (a 150+ day old loan
-- reads identically to a day-1 loan in the raw status field).
SELECT 'MOST_AGED' AS sample, disbursement_fid, phonenumber, disbursement_ts,
disbursed_amount, latest_loan_status, aging_bucket, days_aging,
lifetime_repaid_ugx, outstanding_ugx, pct_repaid
FROM stuck_loans
ORDER BY days_aging DESC
LIMIT 15;
-- RESULT (Sample 1):

-- Sample 2: a representative cross-section (not just the extremes) --
-- confirms most of this population is making SOME repayment progress, not
-- simply abandoned, which the "MOST_AGED" sample alone could misrepresent.
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
SELECT disbursement_fid, loan_status, is_anomaly_open
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
w.phonenumber,
w.disbursement_ts,
w.disbursed_amount,
ls.loan_status AS latest_loan_status,
ls.aging_bucket,
ls.days_aging,
ls.lifetime_repaid_ugx,
(w.disbursed_amount - ls.lifetime_repaid_ugx) AS outstanding_ugx,
ROUND(100.0 * ls.lifetime_repaid_ugx / NULLIF(w.disbursed_amount, 0), 1) AS pct_repaid
FROM disb_windows w
JOIN loan_summary lsu ON lsu.disbursement_fid = w.disbursement_fid
JOIN loan_state_latest ls ON ls.disbursement_fid = w.disbursement_fid
WHERE w.next_disbursement_fid IS NULL
AND w.disbursement_ts <= TIMESTAMP '2026-06-02 00:00:00.000'
AND lsu.ever_anomaly_open = 0
AND lsu.ever_resolved = 0
)
SELECT 'CROSS_SECTION' AS sample, disbursement_fid, phonenumber, disbursement_ts,
disbursed_amount, latest_loan_status, aging_bucket, days_aging,
lifetime_repaid_ugx, outstanding_ugx, pct_repaid
FROM stuck_loans
ORDER BY MOD(disbursement_fid, 97), days_aging DESC
LIMIT 20;
-- RESULT (Sample 2):
-- INTERPRETATION: hand both samples to the table owner alongside the
-- aggregate figures already documented (6,734 loans, 774,650,716 UGX
-- outstanding, 94.8% with some repayment activity, median aging 53 days).
-- Sample 1 makes the taxonomy gap concrete (specific disbursement_fids they
-- can look up showing loan_status='OPEN' at 100+ days aged); Sample 2
-- guards against the extreme sample being read as "these are all dead
-- loans" -- most rows here should show pct_repaid > 0 and often
-- substantial, illustrating these are genuinely slow-paying borrowers with
-- no status to reflect that, not abandoned debt.
