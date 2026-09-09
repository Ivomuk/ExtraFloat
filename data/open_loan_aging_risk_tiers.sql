-- ============================================================================
-- open_loan_aging_risk_tiers.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. Business context from discussions: OPEN
-- loans should be risk-tiered by days_aging -- 1-7 days is normal
-- (expected to cure, covered by the existing 24h/48h penalty events),
-- 7-30 days suggests the agent is having business challenges, 30+ days is
-- serious. This is exactly the missing signal behind the silently-stuck-
-- loans taxonomy gap (silently_stuck_loans_characterization.sql,
-- silently_stuck_loans_raw_examples.sql): loan_status='OPEN' alone can't
-- distinguish a day-1 loan from a day-150 loan. This (1) checks whether
-- aging_bucket (an existing raw column, always passed through as-is
-- everywhere in this project but never actually inspected for its values)
-- already encodes something like this tiering, and (2) quantifies the
-- CURRENT open-loan book against the exact proposed scheme -- both for
-- ALL open loans (broader than the single-loan-customer "silently stuck"
-- population already characterized) and for that specific subset, so the
-- two analyses can be compared side by side.
--
-- Self-contained (raw tables), bounded to date_key <= 20260609.
-- ============================================================================

-- Step 1: does aging_bucket already implement (or approximate) this
-- tiering? If so, the "gap" may just be that nothing downstream surfaces
-- it -- a much smaller fix than adding new derived logic.
WITH loan_state_latest AS (
SELECT disbursement_fid, loan_status, aging_bucket, days_aging
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
)
SELECT
loan_status,
aging_bucket,
COUNT(*) AS n_loans,
MIN(days_aging) AS min_days_aging,
MAX(days_aging) AS max_days_aging,
APPROX_PERCENTILE(days_aging, 0.5) AS median_days_aging
FROM loan_state_latest
WHERE loan_status = 'OPEN'
GROUP BY loan_status, aging_bucket
ORDER BY min_days_aging;
-- RESULT (Step 1):
-- INTERPRETATION: if each aging_bucket value maps to a tight, non-
-- overlapping days_aging range that lines up with 1-7/7-30/30+ (or
-- something close), the source system already carries this signal and it
-- just needs to be surfaced downstream. If aging_bucket's ranges don't
-- line up, or every OPEN loan shares one aging_bucket value regardless of
-- days_aging, the tiering needs to be derived fresh from days_aging
-- directly (Step 2 below already does this, independent of aging_bucket,
-- so the fix works either way).

-- Step 2: quantify the CURRENT open-loan book against the exact proposed
-- scheme, using days_aging directly (authoritative regardless of what
-- aging_bucket turns out to encode). Includes ANOMALY_OPEN as a separate
-- row for comparison -- it has its own handling already, but is still
-- technically "open" in the sense of unresolved.
WITH loan_state_latest AS (
SELECT disbursement_fid, loan_status, days_aging
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
disb_dedup AS (
SELECT disbursement_fid, disbursed_amount, lifetime_repaid_ugx
FROM (
SELECT d.disbursement_fid,
cast(d.disbursement_amount_ugx AS double) AS disbursed_amount,
ls.lifetime_repaid_ugx,
ROW_NUMBER() OVER (PARTITION BY d.disbursement_fid ORDER BY d.inserted_ts DESC) rn
FROM analytics.momo_loan_book_tracker_disbursements_daily d
JOIN (
SELECT disbursement_fid, lifetime_repaid_ugx
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
) ls ON ls.disbursement_fid = d.disbursement_fid
WHERE d.ova = 'XTRAFLOAT-AGENT'
AND d.disbursement_fid IS NOT NULL
AND d.customer_msisdn IS NOT NULL
AND d.date_key <= 20260609
)
WHERE rn = 1
),
tiered AS (
SELECT
lsl.loan_status,
lsl.days_aging,
d.disbursed_amount,
d.lifetime_repaid_ugx,
(d.disbursed_amount - d.lifetime_repaid_ugx) AS outstanding_ugx,
CASE
WHEN lsl.days_aging BETWEEN 1 AND 7 THEN '1-7 days (normal)'
WHEN lsl.days_aging BETWEEN 8 AND 30 THEN '7-30 days (business challenges)'
WHEN lsl.days_aging > 30 THEN '30+ days (serious)'
ELSE '0 days / same-day'
END AS risk_tier
FROM loan_state_latest lsl
JOIN disb_dedup d ON d.disbursement_fid = lsl.disbursement_fid
WHERE lsl.loan_status IN ('OPEN', 'ANOMALY_OPEN')
)
SELECT
loan_status,
risk_tier,
COUNT(*) AS n_loans,
SUM(outstanding_ugx) AS total_outstanding_ugx,
APPROX_PERCENTILE(days_aging, 0.5) AS median_days_aging,
MAX(days_aging) AS max_days_aging
FROM tiered
GROUP BY loan_status, risk_tier
ORDER BY loan_status,
CASE risk_tier
WHEN '0 days / same-day' THEN 0
WHEN '1-7 days (normal)' THEN 1
WHEN '7-30 days (business challenges)' THEN 2
WHEN '30+ days (serious)' THEN 3
END;
-- RESULT (Step 2):
-- INTERPRETATION: this is the number to bring back to the business --
-- concrete counts and UGX exposure per proposed tier, for the WHOLE open
-- book (not just the 6,734-loan silently-stuck subset already
-- characterized, which by construction only covers single-loan customers
-- aged 7+ days with no second loan -- a strict subset of "7-30 days" +
-- "30+ days" here that also excludes anyone who ever took a second loan).
-- The "30+ days (serious)" row's total_outstanding_ugx is the headline
-- exposure figure for that category. Compare ANOMALY_OPEN's tier
-- distribution against OPEN's -- if ANOMALY_OPEN loans cluster almost
-- entirely in "1-7 days," that's consistent with this session's earlier
-- finding that ANOMALY_OPEN loans typically resolve within 1-6 days
-- (anomaly_open_raw_timeline_examples.sql); if a meaningful share sits in
-- 7-30/30+ too, that is itself a further quantification of how many
-- ANOMALY_OPEN loans are ALSO stuck long-term, not just briefly
-- overlapping with a second loan.
