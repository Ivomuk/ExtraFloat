-- ============================================================================
-- clean_loan_mismatch_sample.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. Every prior B2b hypothesis (loan_uid-aware
-- windowing, gross vs net, ever-anomalous exclusion) has been tested on
-- the whole population and given only a small improvement -- meaning most
-- of the mismatch survives even among loans with NO merge and NO anomaly
-- history at all. This finds concrete, genuinely CLEAN mismatched loans
-- (SETTLED status, single disbursement_event_count, single-loan_uid) to
-- trace by hand, the same way the anomaly cases were traced.
-- ============================================================================

-- Step 1: sample of clean (SETTLED, never-merged, never-anomalous)
-- mismatched loans, a mix of over- and under-attributed.
WITH clean_loan_state AS (
SELECT disbursement_fid, phonenumber, loan_uid, loan_status,
lifetime_disbursed_ugx, lifetime_repaid_ugx, lifetime_gross_repaid_ugx,
disbursement_event_count, repayment_event_count
FROM (
SELECT lsld.*,
ROW_NUMBER() OVER (
PARTITION BY disbursement_fid
ORDER BY date_key DESC
) rn2
FROM (
SELECT *
FROM (
SELECT lsd.*,
ROW_NUMBER() OVER (
PARTITION BY disbursement_fid, date_key
ORDER BY inserted_ts DESC
) rn
FROM analytics.momo_loan_book_tracker_loan_state_daily lsd
WHERE ova = 'XTRAFLOAT-AGENT'
AND disbursement_fid IS NOT NULL
AND date_key <= 20260731
AND inserted_ts <= TIMESTAMP '2026-08-20 00:00:00.000'
)
WHERE rn = 1
) lsld
)
WHERE rn2 = 1
AND loan_status = 'SETTLED'
AND disbursement_event_count = 1
),
attributed AS (
SELECT w.disbursement_fid, r.repayment_amount
FROM :validation_schema.vw_bh_repay_dedup r
JOIN :validation_schema.vw_bh_disb_windows w
ON r.phonenumber = w.phonenumber
AND r.repayment_ts >= w.disbursement_ts
AND (w.next_disbursement_ts IS NULL OR r.repayment_ts < w.next_disbursement_ts)
WHERE w.disbursement_fid IN (SELECT disbursement_fid FROM clean_loan_state)
),
per_loan_attributed AS (
SELECT
w.disbursement_fid,
w.phonenumber,
w.disbursement_ts,
w.next_disbursement_ts,
w.disbursed_amount,
COALESCE(SUM(ABS(a.repayment_amount)), 0) AS attributed_repaid_abs
FROM :validation_schema.vw_bh_disb_windows w
LEFT JOIN attributed a ON a.disbursement_fid = w.disbursement_fid
WHERE w.disbursement_fid IN (SELECT disbursement_fid FROM clean_loan_state)
GROUP BY w.disbursement_fid, w.phonenumber, w.disbursement_ts, w.next_disbursement_ts, w.disbursed_amount
),
mismatched_clean AS (
SELECT
pla.*,
cls.lifetime_repaid_ugx,
cls.lifetime_gross_repaid_ugx,
cls.repayment_event_count,
(pla.attributed_repaid_abs - cls.lifetime_repaid_ugx) AS diff_ugx
FROM per_loan_attributed pla
JOIN clean_loan_state cls ON cls.disbursement_fid = pla.disbursement_fid
WHERE ABS(pla.attributed_repaid_abs - cls.lifetime_repaid_ugx) > 1
)
SELECT disbursement_fid, phonenumber, disbursement_ts, next_disbursement_ts,
disbursed_amount, attributed_repaid_abs, lifetime_repaid_ugx,
lifetime_gross_repaid_ugx, repayment_event_count, diff_ugx
FROM (
(SELECT * FROM mismatched_clean WHERE diff_ugx > 0 ORDER BY diff_ugx DESC LIMIT 5)
UNION ALL
(SELECT * FROM mismatched_clean WHERE diff_ugx < 0 ORDER BY diff_ugx ASC LIMIT 5)
);
-- RESULT:
-- EYEBALL: pick 2-3 of these (note disbursement_fid, phonenumber,
-- disbursement_ts, next_disbursement_ts) to pull raw disbursement +
-- repayment rows for directly, same as the 256774/256772 traces --
-- WHERE ova='XTRAFLOAT-AGENT' AND customer_msisdn normalizes to this
-- phonenumber AND event_ts BETWEEN disbursement_ts AND next_
-- disbursement_ts (or a few days past disbursement_ts if next_
-- disbursement_ts is NULL). repayment_event_count tells you how many
-- repayment rows loan_state_daily itself thinks contributed to this loan
-- -- compare that count against how many raw repayment rows you actually
-- find in the window to see if it's a coverage/window problem or something
-- else (e.g. one single repayment amount that just doesn't equal the
-- principal for a reason not yet identified).
