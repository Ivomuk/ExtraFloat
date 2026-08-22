-- ============================================================================
-- 256774_full_47day_trace.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. loan_uid_aware_reconciliation_test.sql's
-- modest result (71% vs 68.6% -- a real but small improvement) suggests the
-- "ripple effect" narrative may be incomplete or wrong. Rather than keep
-- reasoning from aggregate percentages, this pulls phonenumber 256774's
-- FULL chronological transaction history -- every disbursement AND every
-- repayment -- across the entire merged loan_uid's life (first disbursement
-- 2026-04-01 to confirmed closure 2026-05-18, plus a few days margin), so
-- the actual activity can be traced by eye instead of inferred from
-- percentages.
-- ============================================================================

SELECT 'disbursement' AS event_type, disbursement_fid AS event_id,
disbursement_ts AS event_ts, disbursement_amount_ugx AS amount_ugx
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND date_key BETWEEN 20260401 AND 20260525
AND regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') = '256774'

UNION ALL

SELECT 'repayment' AS event_type, repayment_fid AS event_id,
repayment_ts AS event_ts, repayment_amount_ugx AS amount_ugx
FROM analytics.momo_loan_book_tracker_repayments_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND date_key BETWEEN 20260401 AND 20260525
AND regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') = '256774'

ORDER BY event_ts;
-- RESULT:
-- EYEBALL: with disbursements at 2026-04-01 12:07-14:07 (the 7-way merge),
-- 2026-04-04 13:05 (the separate loan, 39711478829), and 2026-05-03/05-04+
-- (later loans), and knowing the merged loan_uid didn't reach lifetime_
-- repaid_ugx=5,250,000 until closure on 2026-05-18 -- do the repayment
-- amounts/timestamps in between actually look like they're paying down the
-- merged loan specifically, or could a reasonable person attribute them to
-- the April 4th loan (or later loans) instead? Sum repayments between
-- 2026-04-04 13:05:20 (right after the merge's last disbursement) and
-- 2026-05-03 10:27:05 (the next real disbursement) by hand -- that's
-- exactly the window borrower_history.txt's heuristic currently assigns
-- entirely to loan 39711478829, and compare it against that loan's actual
-- lifetime_repaid_ugx (750,000, confirmed) to see the true size of the
-- leak directly.
