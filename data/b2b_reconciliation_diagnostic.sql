-- ============================================================================
-- b2b_reconciliation_diagnostic.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. B2b's reconciliation rate is confirmed real
-- and stable (68.6% exact-match count, 74.3% value, p95 error 31.9% of
-- median principal -- unchanged across two separate live runs). This digs
-- into WHY: whether mismatches concentrate among multi-loan (rapid-
-- reborrow) borrowers, where the time-window attribution heuristic has to
-- make a judgment call, vs. single-loan borrowers where attribution is
-- unambiguous.
-- ============================================================================

-- Step 1: mismatch rate split by single-loan vs multi-loan borrowers. If
-- the rapid-reborrow heuristic is the culprit, single-loan borrowers should
-- show a MUCH higher exact-match rate than multi-loan borrowers.
WITH attributed AS (
SELECT w.disbursement_fid, r.repayment_amount
FROM :validation_schema.vw_bh_repay_dedup r
JOIN :validation_schema.vw_bh_disb_windows w
ON r.phonenumber = w.phonenumber
AND r.repayment_ts >= w.disbursement_ts
AND (w.next_disbursement_ts IS NULL OR r.repayment_ts < w.next_disbursement_ts)
WHERE w.disbursement_fid IN (SELECT disbursement_fid FROM :validation_schema.vw_bh_surviving_loans)
),
per_loan_attributed AS (
SELECT
w.disbursement_fid,
w.phonenumber,
w.disbursed_amount,
COALESCE(SUM(ABS(a.repayment_amount)), 0) AS attributed_repaid_abs
FROM :validation_schema.vw_bh_surviving_loans w
LEFT JOIN attributed a ON a.disbursement_fid = w.disbursement_fid
GROUP BY w.disbursement_fid, w.phonenumber, w.disbursed_amount
),
loan_counts AS (
SELECT phonenumber, COUNT(*) AS n_loans
FROM :validation_schema.vw_bh_disb_dedup
GROUP BY phonenumber
),
joined AS (
SELECT
pla.disbursement_fid,
pla.attributed_repaid_abs,
lsl.lifetime_repaid_ugx,
lc.n_loans
FROM per_loan_attributed pla
JOIN :validation_schema.vw_bh_loan_state_snapshot lsl ON lsl.disbursement_fid = pla.disbursement_fid
JOIN loan_counts lc ON lc.phonenumber = pla.phonenumber
)
SELECT
CASE WHEN n_loans = 1 THEN 'single_loan_borrower' ELSE 'multi_loan_borrower' END AS borrower_type,
COUNT(*) AS n_matched_loans,
SUM(CASE WHEN ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_exact_match,
ROUND(100.0 * SUM(CASE WHEN ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_exact_match,
approx_percentile(ABS(attributed_repaid_abs - lifetime_repaid_ugx), 0.5) AS median_abs_diff_ugx
FROM joined
GROUP BY CASE WHEN n_loans = 1 THEN 'single_loan_borrower' ELSE 'multi_loan_borrower' END;
-- RESULT:

-- Step 2: pull 10 actual mismatched loans (a mix of over- and
-- under-attributed) with their disbursement/attribution/state details, plus
-- how many OTHER loans that same phonenumber has (a quick proxy for
-- rapid-reborrow exposure without a full join).
WITH attributed AS (
SELECT w.disbursement_fid, r.repayment_amount
FROM :validation_schema.vw_bh_repay_dedup r
JOIN :validation_schema.vw_bh_disb_windows w
ON r.phonenumber = w.phonenumber
AND r.repayment_ts >= w.disbursement_ts
AND (w.next_disbursement_ts IS NULL OR r.repayment_ts < w.next_disbursement_ts)
WHERE w.disbursement_fid IN (SELECT disbursement_fid FROM :validation_schema.vw_bh_surviving_loans)
),
per_loan_attributed AS (
SELECT
w.disbursement_fid,
w.phonenumber,
w.disbursement_ts,
w.next_disbursement_ts,
w.disbursed_amount,
COALESCE(SUM(ABS(a.repayment_amount)), 0) AS attributed_repaid_abs
FROM :validation_schema.vw_bh_surviving_loans w
LEFT JOIN attributed a ON a.disbursement_fid = w.disbursement_fid
GROUP BY w.disbursement_fid, w.phonenumber, w.disbursement_ts, w.next_disbursement_ts, w.disbursed_amount
),
mismatched AS (
SELECT
pla.*,
lsl.lifetime_repaid_ugx,
lsl.lifetime_gross_repaid_ugx,
(pla.attributed_repaid_abs - lsl.lifetime_repaid_ugx) AS diff_ugx
FROM per_loan_attributed pla
JOIN :validation_schema.vw_bh_loan_state_snapshot lsl ON lsl.disbursement_fid = pla.disbursement_fid
WHERE ABS(pla.attributed_repaid_abs - lsl.lifetime_repaid_ugx) > 1
)
SELECT disbursement_fid, phonenumber, disbursement_ts, next_disbursement_ts,
disbursed_amount, attributed_repaid_abs, lifetime_repaid_ugx,
lifetime_gross_repaid_ugx, diff_ugx
FROM (
(SELECT * FROM mismatched WHERE diff_ugx > 0 ORDER BY diff_ugx DESC LIMIT 5)
UNION ALL
(SELECT * FROM mismatched WHERE diff_ugx < 0 ORDER BY diff_ugx ASC LIMIT 5)
);
-- RESULT:
-- EYEBALL: note the disbursement_fid/phonenumber/disbursement_ts/
-- next_disbursement_ts for a couple of these -- pick 2-3 to trace with
-- Step 3 below (substitute :sample_fid and :sample_phonenumber manually).

-- Step 3: raw trace for ONE sampled mismatched loan -- every attributed
-- repayment plus the window boundaries, so you can see by eye whether
-- attribution looks right. Substitute :sample_fid / :sample_phonenumber /
-- :window_start / :window_end from a Step 2 row before running.
SELECT r.repayment_fid, r.customer_msisdn, r.repayment_ts, r.repayment_amount_ugx, r.inserted_ts
FROM analytics.momo_loan_book_tracker_repayments_daily r
WHERE r.ova = 'XTRAFLOAT-AGENT'
AND regexp_replace(trim(cast(r.customer_msisdn AS varchar)), '[^0-9]', '') = ':sample_phonenumber'
AND r.repayment_ts >= TIMESTAMP ':window_start'
AND (r.repayment_ts < TIMESTAMP ':window_end' OR ':window_end' = 'NULL')
ORDER BY r.repayment_ts;
-- RESULT:
-- EYEBALL: sum repayment_amount_ugx by hand and compare against Step 2's
-- attributed_repaid_abs for this loan (should match, confirming the
-- attribution window itself is correct) and against lifetime_repaid_ugx
-- (the actual gap to explain). If the hand-sum matches attributed_repaid_
-- abs but NOT lifetime_repaid_ugx, the gap is NOT an attribution-window
-- problem -- it's something about what loan_state_daily's lifetime_
-- repaid_ugx itself represents (a derived/allocated figure, not a raw
-- repayment sum) that the attribution heuristic can't reproduce by
-- summing atomic repayments, no matter how correctly it draws the window.
