-- ============================================================================
-- repayment_full_schema_check.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. clean_loan_raw_trace.sql found 10 repayment
-- rows for 256772, all amount=750,000, all within 2026-04-01
-- 17:00:22-17:00:36, which duplicate_repayment_burst_test.sql treated as
-- likely duplicate/retry postings of ONE real repayment. But the user's own
-- pull of 256772's loan_state_daily history shows a SEPARATE loan_uid
-- (disbursement_fid 38194819923, loan_seq 19, disbursed 2026-01-30) stuck in
-- ANOMALY_OPEN for months, unresolved in every daily snapshot through at
-- least 2026-03-09 -- i.e. a real backlog of unresolved loans behind this
-- agent's normal daily cycle. That raises a different explanation: the 10
-- repayment rows could be 10 GENUINELY DIFFERENT loan_uids' repayments
-- (a batch catch-up clearing several backlogged 750,000 loans at once),
-- not 10 duplicate copies of the same one -- our phonenumber+time-window
-- heuristic can't tell those two scenarios apart, but the raw repayments
-- table might carry a direct loan reference we've never actually selected
-- (borrower_history.txt / vw_bh_repay_dedup only ever pull repayment_fid,
-- customer_msisdn, repayment_ts, repayment_amount_ugx). This pulls every
-- column (SELECT *) for exactly the 10 suspicious repayment_fids to check
-- for a disbursement_fid/loan_uid column, and to see if any other column
-- actually differs between the 10 rows.
-- ============================================================================

SELECT *
FROM analytics.momo_loan_book_tracker_repayments_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND repayment_fid IN (
39638225888, 39638228609, 39638229613, 39638229690, 39638229557,
39638230069, 39638230268, 39638230591, 39638230307, 39638231217
)
ORDER BY repayment_ts;
-- RESULT:
-- EYEBALL: (1) is there a disbursement_fid/loan_uid column? If so, do all
-- 10 rows share the SAME value (true duplicates of one loan) or 10
-- DIFFERENT values (10 distinct loans settled in one batch -- meaning we
-- should join repayments to loans directly by that key instead of by
-- time-window heuristic at all)? (2) even without such a column, do ANY
-- other fields differ between the 10 rows (a hidden loan reference, a
-- different partner/product code, a different original transaction id)?
-- Identical rows in every column except repayment_fid and the sub-second
-- timestamp would support genuine duplication; any other differing column
-- supports 10 distinct real repayments.
