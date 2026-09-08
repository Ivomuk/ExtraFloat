-- ============================================================================
-- anomaly_open_raw_timeline_examples.sql -- ad-hoc diagnostic, not part of
-- the committed validation suite. anomaly_open_balance_transfer_test.sql's
-- aggregate result (97.2% of ANOMALY_OPEN loans show zero inflation in the
-- next loan) says the documented "balance transferred" mechanism doesn't
-- hold on average -- but an aggregate can hide what's actually happening
-- day to day. This pulls RAW, unaggregated rows instead: every loan for 5
-- example customers who had a recent ANOMALY_OPEN case, and for each of
-- those loans, its FULL daily loan_state_daily history (every date_key,
-- not collapsed to the latest snapshot) -- so the actual sequence of
-- events is visible directly, not summarized away.
--
-- Unlike every other diagnostic this session, this one deliberately does
-- NOT collapse to one row per disbursement_fid -- seeing the day-by-day
-- evolution of loan_status/lifetime_disbursed_ugx/lifetime_repaid_ugx is
-- the whole point (a loan could pass through ANOMALY_OPEN and later
-- resolve to something else, which every prior "latest snapshot only"
-- query would have missed entirely).
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
disbursement_fid, phonenumber, disbursement_ts,
LEAD(disbursement_fid) OVER (PARTITION BY phonenumber ORDER BY disbursement_ts, disbursement_fid) AS next_disbursement_fid
FROM disb_dedup
),
loan_state_latest AS (
SELECT disbursement_fid, loan_status, is_anomaly_open
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
candidates AS (
SELECT w.phonenumber, w.disbursement_ts
FROM disb_windows w
JOIN loan_state_latest ls ON ls.disbursement_fid = w.disbursement_fid
WHERE (ls.loan_status = 'ANOMALY_OPEN' OR ls.is_anomaly_open = true)
AND w.next_disbursement_fid IS NOT NULL
AND w.disbursement_ts >= DATE '2026-04-01'
ORDER BY w.disbursement_ts DESC
LIMIT 5
),
sample_phonenumbers AS (SELECT DISTINCT phonenumber FROM candidates),
sample_loans AS (
SELECT d.disbursement_fid, d.phonenumber, d.disbursement_ts, d.disbursed_amount
FROM disb_dedup d
WHERE d.phonenumber IN (SELECT phonenumber FROM sample_phonenumbers)
),
raw_daily_state AS (
SELECT disbursement_fid, date_key, loan_status, is_anomaly_open,
lifetime_disbursed_ugx, lifetime_repaid_ugx, aging_bucket, days_aging, is_active_loan
FROM (
SELECT lsd.*,
ROW_NUMBER() OVER (PARTITION BY disbursement_fid, date_key ORDER BY inserted_ts DESC) rn
FROM analytics.momo_loan_book_tracker_loan_state_daily lsd
WHERE ova = 'XTRAFLOAT-AGENT'
AND disbursement_fid IN (SELECT disbursement_fid FROM sample_loans)
AND date_key <= 20260609
)
WHERE rn = 1
)
SELECT
sl.phonenumber,
sl.disbursement_fid,
sl.disbursement_ts,
sl.disbursed_amount,
rds.date_key,
rds.loan_status,
rds.is_anomaly_open,
rds.lifetime_disbursed_ugx,
rds.lifetime_repaid_ugx,
rds.aging_bucket,
rds.days_aging,
rds.is_active_loan
FROM sample_loans sl
JOIN raw_daily_state rds ON rds.disbursement_fid = sl.disbursement_fid
ORDER BY sl.phonenumber, sl.disbursement_ts, rds.date_key;
-- RESULT:
-- INTERPRETATION: read this grouped by phonenumber, then by
-- disbursement_fid (each block of date_key rows is one loan's daily
-- history), in disbursement_ts order -- literally the customer's loan
-- timeline. For each phonenumber, look at:
--   1. Does an earlier loan's block show loan_status flip TO
--      ANOMALY_OPEN on some date_key, and does that date roughly line up
--      with the next loan's own disbursement_ts? Confirms the flag is
--      genuinely tied to the rollover event, not something unrelated.
--   2. Does the earlier loan's lifetime_repaid_ugx stop changing (flatten)
--      once ANOMALY_OPEN is set -- i.e. does the system just stop
--      tracking further repayment against it -- or does it keep moving
--      after the flag appears?
--   3. Does the LATER loan's lifetime_disbursed_ugx, across its own daily
--      history, ever change from its own disbursed_amount to something
--      larger -- even a few days after its own disbursement -- which the
--      "latest snapshot only" aggregate test could have caught, but only
--      if the transfer isn't reflected until a later date_key than the
--      one already-loaded data provides.
-- If nothing in the raw rows looks like a transfer either, this confirms
-- (with literal evidence, not just an aggregate rate) that the old loan's
-- balance is simply abandoned/untracked once ANOMALY_OPEN is set, not
-- carried forward anywhere in this table.
