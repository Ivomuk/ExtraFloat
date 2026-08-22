-- ============================================================================
-- loan_state_sharing_trace.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. b2b_reconciliation_diagnostic.sql's Step 2
-- showed phonenumber 256774 with THREE loans clustered within a few days
-- (39711478829, 39645290153, 39631605547), each showing lifetime_repaid_ugx
-- far larger than its own disbursed_amount -- consistent with loan_state_
-- daily tracking a SHARED/CUMULATIVE balance across a borrower's rapid-fire
-- loans rather than isolating each disbursement_fid independently (an
-- extension of the already-confirmed same-day-merge behavior). This pulls
-- the RAW rows from both source tables directly to see it firsthand,
-- instead of reasoning from the already-aggregated view output.
-- ============================================================================

-- Step 1: every disbursement this phonenumber made in the relevant window
-- (a few days either side of the three known loans) -- confirms the full
-- set of disbursement_fid values to check in Step 2, in case there are more
-- than the three already seen.
SELECT disbursement_fid, customer_msisdn, disbursement_ts, disbursement_amount_ugx, inserted_ts
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND date_key BETWEEN 20260330 AND 20260510
AND regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') = '256774'
ORDER BY disbursement_ts;
-- RESULT:

-- Step 2: every loan_state_daily row (every date_key snapshot, not just the
-- latest) for those exact disbursement_fid values -- shows the full daily
-- progression of lifetime_disbursed_ugx/lifetime_repaid_ugx per loan. If
-- these values are IDENTICAL or overlapping across the three different
-- disbursement_fid values (rather than each loan showing its own,
-- independent, smaller totals), that directly confirms shared/cumulative
-- tracking rather than per-loan isolation.
SELECT disbursement_fid, date_key, loan_status, aging_bucket, days_aging,
lifetime_disbursed_ugx, lifetime_repaid_ugx, lifetime_gross_repaid_ugx,
is_anomaly_open, inserted_ts
FROM analytics.momo_loan_book_tracker_loan_state_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND date_key BETWEEN 20260330 AND 20260510
AND disbursement_fid IN (39711478829, 39645290153, 39631605547)
ORDER BY disbursement_fid, date_key;
-- RESULT:

-- Step 3: same trace for phonenumber 256772 (the most dramatic single
-- example from Step 2 -- disbursement_fid 39604855038, attributed
-- 7,500,000 vs lifetime_repaid_ugx 750,000, a 10x gap on a 750,000
-- principal loan) -- confirms whether the pattern holds for a completely
-- different borrower too, not just 256774.
SELECT disbursement_fid, customer_msisdn, disbursement_ts, disbursement_amount_ugx, inserted_ts
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND date_key BETWEEN 20260325 AND 20260410
AND regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') = '256772'
ORDER BY disbursement_ts;
-- RESULT:

-- Step 4: loan_state_daily rows for 256772's disbursement_fid(s) from Step 3
-- -- substitute the actual fid list Step 3 returns if it differs from just
-- 39604855038.
SELECT disbursement_fid, date_key, loan_status, aging_bucket, days_aging,
lifetime_disbursed_ugx, lifetime_repaid_ugx, lifetime_gross_repaid_ugx,
is_anomaly_open, inserted_ts
FROM analytics.momo_loan_book_tracker_loan_state_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND date_key BETWEEN 20260325 AND 20260410
AND disbursement_fid = 39604855038
ORDER BY date_key;
-- RESULT:
-- INTERPRETATION (all steps): if Step 2 shows the three disbursement_fids
-- carrying overlapping or identical lifetime_disbursed_ugx/lifetime_
-- repaid_ugx progressions (e.g. each one's total keeps climbing even after
-- ITS OWN disbursed_amount is already exceeded, tracking what looks like a
-- combined running balance across all three), that's direct, row-level
-- confirmation of shared/cumulative tracking -- not something fixable in
-- borrower_history.txt or the validation suite, since it's how the source
-- table itself represents these loans.
