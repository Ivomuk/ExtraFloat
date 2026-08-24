-- ============================================================================
-- loan_uid_grain_prevalence.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. loan_state_daily carries a loan_uid column
-- (the source system's own "loan" identifier) distinct from
-- disbursement_fid -- confirmed directly: one sampled loan_uid
-- (LN_20e4f63013c313490bd91e52821dc80b) spans 7 different disbursement_fid
-- values (disbursement_event_count=7), all sharing one lifetime_
-- disbursed_ugx/lifetime_repaid_ugx total. borrower_history.txt and this
-- validation suite treat disbursement_fid as the loan grain throughout --
-- wrong whenever a loan_uid spans more than one disbursement_fid. This
-- quantifies how much of the book that actually affects.
-- ============================================================================

SELECT
COUNT(*) AS total_rows_at_latest_snapshot,
COUNT(DISTINCT disbursement_fid) AS distinct_disbursement_fids,
COUNT(DISTINCT loan_uid) AS distinct_loan_uids,
SUM(CASE WHEN disbursement_event_count > 1 THEN 1 ELSE 0 END) AS rows_with_multi_disbursement_loan_uid,
ROUND(100.0 * SUM(CASE WHEN disbursement_event_count > 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_rows_multi_disbursement_loan_uid,
MAX(disbursement_event_count) AS max_disbursement_events_in_one_loan_uid,
approx_percentile(disbursement_event_count, 0.5) AS median_disbursement_event_count,
approx_percentile(disbursement_event_count, 0.95) AS p95_disbursement_event_count
FROM (
SELECT *
FROM (
SELECT lsd.*,
ROW_NUMBER() OVER (
PARTITION BY disbursement_fid
ORDER BY date_key DESC, inserted_ts DESC
) rn
FROM analytics.momo_loan_book_tracker_loan_state_daily lsd
WHERE ova = 'XTRAFLOAT-AGENT'
AND disbursement_fid IS NOT NULL
AND date_key <= 20260731
AND inserted_ts <= TIMESTAMP '2026-08-20 00:00:00.000'
)
WHERE rn = 1
);
-- RESULT:
-- INTERPRETATION: pct_rows_multi_disbursement_loan_uid is the real scale of
-- the grain mismatch -- how many of our disbursement_fid-keyed "loans" are
-- actually fragments of a larger, multi-disbursement loan_uid. If
-- distinct_loan_uids is meaningfully smaller than distinct_disbursement_
-- fids, that gap IS the number of loan_uids this affects (each one spans
-- multiple fids). This number, not the ~3.9% same-day-repeat figure from
-- Section G, is likely the true prevalence, since loan_uid spans can
-- stretch across many days/weeks (confirmed: the sampled loan_uid closed
-- 47 days after its first disbursement), not just same-calendar-day
-- repeats.
