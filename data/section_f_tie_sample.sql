-- ============================================================================
-- section_f_tie_sample.sql -- ad-hoc diagnostic, not part of the committed
-- validation suite. Pulls a handful of repayment_fid values that Section F
-- flagged as having multiple rows tied at the same max(inserted_ts), then
-- shows every raw row for those fids so we can see WHY they're tied:
--   - identical rows (true duplicate load -> dedup choosing arbitrarily
--     between byte-identical copies is harmless)
--   - different amounts/timestamps under one shared fid (repayment_fid is
--     not a 1:1 transaction key -- dedup is silently dropping real distinct
--     repayment line items)
-- snapshot_dt/as_of_load_ts below match the same run already validated in
-- vw_bh_output.sql -- change only if you're validating a different run.
-- ============================================================================

WITH tied_fids AS (
SELECT repayment_fid
FROM analytics.momo_loan_book_tracker_repayments_daily r
WHERE r.ova = 'XTRAFLOAT-AGENT'
AND r.date_key <= 20260731
AND date(try_cast(repayment_ts AS timestamp)) <= date_parse(cast(20260731 AS varchar), '%Y%m%d')
AND inserted_ts <= TIMESTAMP '2026-08-20 00:00:00.000'
AND inserted_ts = (
SELECT MAX(inserted_ts) FROM analytics.momo_loan_book_tracker_repayments_daily r2
WHERE r2.repayment_fid = r.repayment_fid
AND r2.ova = 'XTRAFLOAT-AGENT'
AND r2.date_key <= 20260731
AND date(try_cast(r2.repayment_ts AS timestamp)) <= date_parse(cast(20260731 AS varchar), '%Y%m%d')
AND r2.inserted_ts <= TIMESTAMP '2026-08-20 00:00:00.000'
)
GROUP BY repayment_fid
HAVING COUNT(*) > 1
LIMIT 10
)
SELECT r.repayment_fid, r.customer_msisdn, r.repayment_ts, r.repayment_amount_ugx,
r.inserted_ts, r.date_key, r.ova
FROM analytics.momo_loan_book_tracker_repayments_daily r
JOIN tied_fids t ON t.repayment_fid = r.repayment_fid
WHERE r.ova = 'XTRAFLOAT-AGENT'
ORDER BY r.repayment_fid, r.repayment_ts;
-- RESULT:
-- EYEBALL: for each repayment_fid, are all its rows byte-identical (same
-- customer_msisdn/repayment_ts/repayment_amount_ugx -- a true duplicate
-- load, dedup is fine), or do amount/timestamp differ across rows sharing
-- the same fid (repayment_fid is a shared/batch id, not a unique
-- transaction id -- dedup is dropping real repayments)? Also check whether
-- the differing rows are close in time (same batch, split line items) or
-- far apart (recurring reused id).
