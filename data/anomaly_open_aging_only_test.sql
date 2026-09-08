-- ============================================================================
-- anomaly_open_aging_only_test.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. The raw timeline for loan 38643775849
-- showed it flip to ANOMALY_OPEN on 2026-02-19 (days_aging=1), while the
-- customer's next loan (38677752027) wasn't disbursed until 2026-02-19
-- 19:13 -- late enough in that same day that, if the daily loan_state
-- snapshot reflects state as of the START of each day, the flag could
-- not have been caused by that specific disbursement. This tests the
-- question directly rather than arguing from one example's clock times:
-- can a loan become ANOMALY_OPEN purely from being overdue, with NO
-- second loan ever disbursed for that customer at all (as of the current
-- load)? If yes, the flag is (at least in part) an aging/overdue
-- threshold, not literally conditional on a concurrent new loan existing
-- -- correcting the documented "agent took a new loan before settling
-- the current one" description, or at least showing it is not the only
-- trigger.
--
-- Population: loans that are the LAST (or only) disbursement ever
-- recorded for their phonenumber in the current load, AND were disbursed
-- at least ~7 days before the load cutoff (2026-06-09) so they have had
-- real time to either resolve or age further -- excludes loans that are
-- simply too recent to have had a chance to do either yet.
--
-- Self-contained (raw tables), bounded to date_key <= 20260609.
-- ============================================================================

WITH disb_dedup AS (
SELECT disbursement_fid, phonenumber, disbursement_ts
FROM (
SELECT d.*,
ROW_NUMBER() OVER (PARTITION BY disbursement_fid ORDER BY inserted_ts DESC) rn
FROM (
SELECT
disbursement_fid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber,
try_cast(disbursement_ts AS timestamp) AS disbursement_ts,
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
disbursement_fid, phonenumber, disbursement_ts,
LEAD(disbursement_fid) OVER (PARTITION BY phonenumber ORDER BY disbursement_ts, disbursement_fid) AS next_disbursement_fid
FROM disb_dedup
),
loan_state_history AS (
SELECT disbursement_fid, date_key, loan_status, is_anomaly_open, days_aging
FROM (
SELECT lsd.*,
ROW_NUMBER() OVER (PARTITION BY disbursement_fid, date_key ORDER BY inserted_ts DESC) rn
FROM analytics.momo_loan_book_tracker_loan_state_daily lsd
WHERE ova = 'XTRAFLOAT-AGENT'
AND disbursement_fid IS NOT NULL
AND date_key <= 20260609
)
WHERE rn = 1
),
loan_summary AS (
SELECT
disbursement_fid,
MAX(days_aging) AS max_days_aging,
MAX(CASE WHEN loan_status = 'ANOMALY_OPEN' OR is_anomaly_open = true THEN 1 ELSE 0 END) AS ever_anomaly_open,
MAX(CASE WHEN loan_status IN ('CLOSED', 'SETTLED') THEN 1 ELSE 0 END) AS ever_resolved
FROM loan_state_history
GROUP BY disbursement_fid
),
no_next_loan_aged AS (
SELECT w.disbursement_fid, ls.max_days_aging, ls.ever_anomaly_open, ls.ever_resolved
FROM disb_windows w
JOIN loan_summary ls ON ls.disbursement_fid = w.disbursement_fid
WHERE w.next_disbursement_fid IS NULL
AND w.disbursement_ts <= TIMESTAMP '2026-06-02 00:00:00.000'
)
SELECT
COUNT(*) AS n_loans_no_next_disbursement_aged_7d_plus,
COUNT_IF(ever_anomaly_open = 1) AS n_ever_anomaly_open,
ROUND(100.0 * COUNT_IF(ever_anomaly_open = 1) / NULLIF(COUNT(*), 0), 2) AS pct_ever_anomaly_open,
COUNT_IF(ever_anomaly_open = 1 AND ever_resolved = 1) AS n_anomaly_then_resolved,
COUNT_IF(ever_anomaly_open = 1 AND ever_resolved = 0) AS n_anomaly_never_resolved,
COUNT_IF(ever_anomaly_open = 0 AND ever_resolved = 0) AS n_never_anomaly_never_resolved,
APPROX_PERCENTILE(max_days_aging, 0.5) AS median_max_days_aging,
MAX(max_days_aging) AS max_max_days_aging
FROM no_next_loan_aged;
-- RESULT:
-- INTERPRETATION:
--   pct_ever_anomaly_open = 0%   -- no loan ever becomes ANOMALY_OPEN
--                                   without a second loan existing,
--                                   however overdue it gets (check
--                                   max_max_days_aging is genuinely high
--                                   here, not just barely over 7 days,
--                                   to make sure this isn't just "not
--                                   aged enough yet") -- CONFIRMS the
--                                   documented "new loan required"
--                                   mechanism; the Feb 19 timing in the
--                                   38643775849 example must mean the
--                                   snapshot reflects END-of-day state
--                                   (i.e. it DID already know about the
--                                   19:13 disbursement that same day).
--   pct_ever_anomaly_open > 0%  -- some loans DO get flagged purely from
--                                   aging, with no second loan ever
--                                   taken -- CONTRADICTS "new loan
--                                   required" as a strict precondition;
--                                   the flag is (at least partly) an
--                                   aging/overdue threshold, and the
--                                   documented description needs
--                                   correcting, not just re-confirming.
--   n_never_anomaly_never_resolved with a high max_max_days_aging -- some
--                                   loans just sit OPEN indefinitely,
--                                   never flagged and never resolved,
--                                   however overdue -- worth separate
--                                   follow-up regardless of which way the
--                                   main question resolves, since that is
--                                   itself a form of untracked exposure.
