-- ============================================================================
-- clean_loan_raw_trace.sql -- ad-hoc diagnostic, not part of the committed
-- validation suite. clean_loan_mismatch_sample.sql surfaced 10 genuinely
-- clean (SETTLED, single-disbursement, never-anomalous) mismatched loans.
-- Two stand out:
--   - 39604855038 / 256772: repayment_event_count=10, attributed_repaid_abs
--     (7,500,000) is EXACTLY 10 x disbursed_amount (750,000) -- suggests
--     either 10 real daily borrow/repay cycles for this phonenumber, or 10
--     duplicate/retry rows of the same logical repayment.
--   - 39634283120 / 256773: repayment_event_count=2 (true, from
--     loan_state_daily), true gross repaid only 765,500 (~750k principal +
--     small fee), but our window heuristic attributes 5,250,000 -- a ~6.9x
--     inflation that can't come from just 2 events at this loan's own size,
--     meaning the window is scooping up repayment rows that don't belong to
--     this loan_uid at all.
-- Both also happen to fall in the exact same April 1-4 window already
-- flagged for phonenumber 256774's 7-way disbursement merge -- pulling raw
-- rows here checks whether that's a coincidence or a wider batch event.
-- ============================================================================

SELECT '256772' AS traced_phonenumber, 'disbursement' AS event_type,
disbursement_fid AS event_id, disbursement_ts AS event_ts,
disbursement_amount_ugx AS amount_ugx, NULL AS raw_msisdn_field
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND date_key BETWEEN 20260320 AND 20260410
AND regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') = '256772'

UNION ALL

SELECT '256772' AS traced_phonenumber, 'repayment' AS event_type,
repayment_fid AS event_id, repayment_ts AS event_ts,
repayment_amount_ugx AS amount_ugx, cast(customer_msisdn AS varchar) AS raw_msisdn_field
FROM analytics.momo_loan_book_tracker_repayments_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND date_key BETWEEN 20260320 AND 20260410
AND regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') = '256772'

UNION ALL

SELECT '256773' AS traced_phonenumber, 'disbursement' AS event_type,
disbursement_fid AS event_id, disbursement_ts AS event_ts,
disbursement_amount_ugx AS amount_ugx, NULL AS raw_msisdn_field
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND date_key BETWEEN 20260320 AND 20260410
AND regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') = '256773'

UNION ALL

SELECT '256773' AS traced_phonenumber, 'repayment' AS event_type,
repayment_fid AS event_id, repayment_ts AS event_ts,
repayment_amount_ugx AS amount_ugx, cast(customer_msisdn AS varchar) AS raw_msisdn_field
FROM analytics.momo_loan_book_tracker_repayments_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND date_key BETWEEN 20260320 AND 20260410
AND regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') = '256773'

ORDER BY traced_phonenumber, event_ts;
-- RESULT:
-- EYEBALL, per phonenumber:
-- 1. Do repayment_fid values repeat, or are amounts/timestamps suspiciously
--    duplicated (same amount within seconds of each other)? That would
--    indicate retry/duplicate rows inflating repayment_event_count and our
--    SUM, not real distinct repayments.
-- 2. raw_msisdn_field: does the RAW (un-normalized) customer_msisdn differ
--    in format between rows that still normalize to the same phonenumber
--    (e.g. one with a leading '0' or '+', one without, or a wrong-length
--    string that regexp_replace can't tell apart from a genuinely different
--    subscriber)? That would point to normalization collisions rather than
--    a real single borrower.
-- 3. For 256773 specifically: sum every repayment_amount_ugx between
--    2026-04-01 14:06:03 (disbursement_ts) and 2026-04-03 22:32:16
--    (next_disbursement_ts) by hand and compare against both the true
--    gross (765,500) and our attributed (5,250,000) to locate exactly which
--    row(s) account for the gap.
-- 4. For 256772: same check between 2026-03-31 10:00:30 and
--    2026-04-04 13:05:20 -- count the raw repayment rows and compare against
--    repayment_event_count=10 and against attributed_repaid_abs=7,500,000.
