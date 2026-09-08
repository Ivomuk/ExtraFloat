-- ============================================================================
-- anomaly_open_rollover_examples.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. loan_concurrency_by_month.sql found a real,
-- accelerating ANOMALY_OPEN rate (0.3% Jan -> 10.6% Jun). This pulls actual
-- rows for a sample of recent (April 2026 onward) rollover cases to (a)
-- present as concrete evidence rather than just an aggregate rate, and
-- (b) confirm whether the loan_status documentation's claim -- "agent took
-- a NEW loan before settling the current one; the outstanding balance was
-- transferred to the new loan" -- is actually what the data shows now,
-- post-rebuild, rather than assuming the old documentation still applies.
--
-- Mechanism under test: a rollover should show up as multiple
-- disbursement_fid rows sharing ONE loan_uid (loan_state_daily generates
-- loan_uid; it is NOT the same as disbursement_fid, per loan_state_query_
-- updated.txt's own header), with the earlier disbursement_fid flagged
-- ANOMALY_OPEN, and lifetime_disbursed_ugx/lifetime_repaid_ugx recorded
-- at the shared loan_uid level (identical across every disbursement_fid
-- in the group) rather than reflecting each individual disbursement on
-- its own -- direct evidence the balance was transferred/tracked jointly,
-- not that money went missing.
--
-- Sampled to loan_uid groups with >=2 distinct disbursement_fid AND at
-- least one ANOMALY_OPEN-flagged row, where the group's latest
-- disbursement is on or after 2026-04-01 -- keeps the sample relevant to
-- the period actually under investigation. Self-contained (raw tables,
-- no GATE 0 dependency), bounded to date_key <= 20260609.
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
loan_state_dedup AS (
SELECT disbursement_fid, loan_uid, loan_status, is_anomaly_open,
lifetime_disbursed_ugx, lifetime_repaid_ugx, current_loan_start_date, date_key
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
rollover_candidates AS (
SELECT ls.loan_uid
FROM loan_state_dedup ls
JOIN disb_dedup d ON d.disbursement_fid = ls.disbursement_fid
WHERE ls.loan_uid IS NOT NULL
GROUP BY ls.loan_uid
HAVING COUNT(DISTINCT ls.disbursement_fid) >= 2
AND MAX(CASE WHEN ls.loan_status = 'ANOMALY_OPEN' OR ls.is_anomaly_open = true THEN 1 ELSE 0 END) = 1
AND MAX(d.disbursement_ts) >= DATE '2026-04-01'
),
sampled AS (
SELECT loan_uid FROM rollover_candidates
ORDER BY loan_uid
LIMIT 15
)
SELECT
ls.loan_uid,
ls.disbursement_fid,
d.phonenumber,
d.disbursement_ts,
d.disbursed_amount,
ls.loan_status,
ls.is_anomaly_open,
ls.lifetime_disbursed_ugx,
ls.lifetime_repaid_ugx,
ls.current_loan_start_date,
ls.date_key
FROM loan_state_dedup ls
JOIN disb_dedup d ON d.disbursement_fid = ls.disbursement_fid
WHERE ls.loan_uid IN (SELECT loan_uid FROM sampled)
ORDER BY ls.loan_uid, d.disbursement_ts;
-- RESULT:
-- INTERPRETATION: group the output by loan_uid (rows sharing a loan_uid
-- are one rollover chain). For each group, check:
--   1. Same phonenumber across every row in the group (confirms it's the
--      same agent, not a loan_uid collision across customers).
--   2. The earlier disbursement_ts row(s) show loan_status='ANOMALY_OPEN'
--      / is_anomaly_open=true, and a LATER disbursement_ts exists for the
--      same loan_uid -- i.e. a new loan really was taken before the
--      earlier one's flag would suggest it settled.
--   3. lifetime_disbursed_ugx and lifetime_repaid_ugx are IDENTICAL across
--      every disbursement_fid row in the group (not each row showing only
--      its own individual disbursed_amount) -- this is the direct
--      evidence that the balance is tracked/transferred at the loan_uid
--      level, confirming the documented mechanism still holds. If instead
--      each row shows its own distinct lifetime_disbursed_ugx unrelated to
--      the others, the "balance transferred" claim does NOT hold as
--      currently documented and needs correcting, not just re-confirming.
-- If rollover_candidates returns zero groups (LIMIT clause has nothing to
-- select), broaden by removing the 2026-04-01 filter, or check that
-- loan_uid is actually populated (non-null) in this warehouse load --
-- some earlier investigations found the field's population rate varies.
