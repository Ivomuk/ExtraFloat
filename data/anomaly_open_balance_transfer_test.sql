-- ============================================================================
-- anomaly_open_balance_transfer_test.sql -- ad-hoc diagnostic, not part of
-- the committed validation suite. anomaly_rollover_diagnose_zero_rows.sql
-- found loan_uid is always 1:1 with disbursement_fid in this data
-- (n_loan_uids_with_multiple_fids = 0) -- so "the outstanding balance was
-- transferred to the new loan" can NOT be evidenced via a shared loan_uid,
-- contrary to what anomaly_open_rollover_examples.sql assumed. This tests
-- the same claim a different way: for each ANOMALY_OPEN loan, find its
-- NEXT loan for the same phonenumber (by timing, same LEAD()-based
-- approach as disb_windows elsewhere in this project -- no loan_uid
-- involved at all), and check whether that next loan's
-- lifetime_disbursed_ugx exceeds its own disbursed_amount -- and, if so,
-- whether the excess matches the anomaly loan's own outstanding balance.
-- That would be direct, ID-independent evidence of a balance transfer.
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
LEAD(disbursement_fid) OVER (PARTITION BY phonenumber ORDER BY disbursement_ts, disbursement_fid) AS next_disbursement_fid,
LEAD(disbursed_amount) OVER (PARTITION BY phonenumber ORDER BY disbursement_ts, disbursement_fid) AS next_disbursed_amount
FROM disb_dedup
),
loan_state_dedup AS (
SELECT disbursement_fid, loan_status, is_anomaly_open, lifetime_disbursed_ugx, lifetime_repaid_ugx
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
anomaly_with_next AS (
SELECT
w.disbursement_fid AS anomaly_disbursement_fid,
w.disbursed_amount AS anomaly_disbursed_amount,
ls.lifetime_repaid_ugx AS anomaly_lifetime_repaid_ugx,
(w.disbursed_amount - ls.lifetime_repaid_ugx) AS anomaly_outstanding_ugx,
w.next_disbursement_fid,
w.next_disbursed_amount,
ls_next.lifetime_disbursed_ugx AS next_lifetime_disbursed_ugx,
(ls_next.lifetime_disbursed_ugx - w.next_disbursed_amount) AS next_loan_inflation_ugx
FROM disb_windows w
JOIN loan_state_dedup ls ON ls.disbursement_fid = w.disbursement_fid
LEFT JOIN loan_state_dedup ls_next ON ls_next.disbursement_fid = w.next_disbursement_fid
WHERE ls.loan_status = 'ANOMALY_OPEN' OR ls.is_anomaly_open = true
)
SELECT
COUNT(*) AS n_anomaly_loans_total,
COUNT_IF(next_disbursement_fid IS NULL) AS n_no_next_loan_yet,
COUNT_IF(next_disbursement_fid IS NOT NULL AND next_lifetime_disbursed_ugx IS NULL) AS n_next_loan_missing_state_row,
COUNT_IF(next_disbursement_fid IS NOT NULL AND next_lifetime_disbursed_ugx IS NOT NULL) AS n_testable,
COUNT_IF(next_loan_inflation_ugx > 1) AS n_next_loan_inflated,
ROUND(100.0 * COUNT_IF(next_loan_inflation_ugx > 1)
/ NULLIF(COUNT_IF(next_disbursement_fid IS NOT NULL AND next_lifetime_disbursed_ugx IS NOT NULL), 0), 2) AS pct_next_loan_inflated,
COUNT_IF(ABS(next_loan_inflation_ugx - anomaly_outstanding_ugx) <= 1) AS n_inflation_matches_outstanding_exactly,
ROUND(100.0 * COUNT_IF(ABS(next_loan_inflation_ugx - anomaly_outstanding_ugx) <= 1)
/ NULLIF(COUNT_IF(next_disbursement_fid IS NOT NULL AND next_lifetime_disbursed_ugx IS NOT NULL), 0), 2) AS pct_inflation_matches_outstanding_exactly,
APPROX_PERCENTILE(next_loan_inflation_ugx, 0.5) AS median_next_loan_inflation_ugx,
APPROX_PERCENTILE(anomaly_outstanding_ugx, 0.5) AS median_anomaly_outstanding_ugx
FROM anomaly_with_next;
-- RESULT:
-- INTERPRETATION:
--   n_no_next_loan_yet             -- ANOMALY_OPEN loans with no
--                                     subsequent disbursement observed at
--                                     all yet (can't test these -- either
--                                     the "new loan" hasn't loaded, or the
--                                     flag doesn't actually require one).
--   pct_next_loan_inflated near 0% -- the next loan's lifetime_disbursed_
--                                     ugx never exceeds its own
--                                     disbursed_amount -- NO evidence of
--                                     balance transfer via this mechanism
--                                     either; the documented claim does
--                                     not hold as tested, full stop.
--   pct_next_loan_inflated high AND
--   pct_inflation_matches_outstanding_exactly high -- strong, direct
--                                     evidence the balance really is
--                                     rolled onto the next loan, sized to
--                                     match what was still owed -- exactly
--                                     what the documentation claims.
--   pct_next_loan_inflated high BUT pct_inflation_matches_outstanding_
--   exactly low -- something IS being added to the next loan, but not in
--                  an amount that cleanly matches the old loan's
--                  outstanding balance -- worth a closer look at a few
--                  individual examples before concluding either way.
-- Compare median_next_loan_inflation_ugx against median_anomaly_
-- outstanding_ugx directly regardless of the exact-match rate -- if they
-- are similar in MAGNITUDE even without matching row-by-row, that is still
-- meaningful evidence, just noisier than an exact per-loan match.
