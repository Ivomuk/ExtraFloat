-- ============================================================================
-- anomaly_rollover_diagnose_zero_rows.sql -- ad-hoc diagnostic, not part of
-- the committed validation suite. anomaly_open_rollover_examples.sql
-- returned zero rows; this breaks its rollover_candidates filter down into
-- one count per condition so we can see exactly which requirement is
-- eliminating everything, instead of guessing. Self-contained (raw
-- tables), bounded to date_key <= 20260609.
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
loan_state_dedup AS (
SELECT disbursement_fid, loan_uid, loan_status, is_anomaly_open
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
joined AS (
SELECT ls.*, d.disbursement_ts
FROM loan_state_dedup ls
JOIN disb_dedup d ON d.disbursement_fid = ls.disbursement_fid
)
SELECT
COUNT(*) AS total_loan_state_rows,
COUNT_IF(loan_uid IS NOT NULL) AS n_loan_uid_populated,
ROUND(100.0 * COUNT_IF(loan_uid IS NOT NULL) / NULLIF(COUNT(*), 0), 2) AS pct_loan_uid_populated,
COUNT_IF(loan_status = 'ANOMALY_OPEN' OR is_anomaly_open = true) AS n_anomaly_open_total,
(SELECT COUNT(*) FROM (
SELECT loan_uid FROM joined
WHERE loan_uid IS NOT NULL
GROUP BY loan_uid
HAVING COUNT(DISTINCT disbursement_fid) >= 2
)) AS n_loan_uids_with_multiple_fids,
(SELECT COUNT(*) FROM (
SELECT loan_uid FROM joined
WHERE loan_uid IS NOT NULL
GROUP BY loan_uid
HAVING COUNT(DISTINCT disbursement_fid) >= 2
AND MAX(CASE WHEN loan_status = 'ANOMALY_OPEN' OR is_anomaly_open = true THEN 1 ELSE 0 END) = 1
)) AS n_multi_fid_loan_uids_with_anomaly,
(SELECT COUNT(*) FROM (
SELECT loan_uid FROM joined
WHERE loan_uid IS NOT NULL
GROUP BY loan_uid
HAVING COUNT(DISTINCT disbursement_fid) >= 2
AND MAX(CASE WHEN loan_status = 'ANOMALY_OPEN' OR is_anomaly_open = true THEN 1 ELSE 0 END) = 1
AND MAX(disbursement_ts) >= DATE '2026-04-01'
)) AS n_matching_final_filter
FROM joined;
-- RESULT:
-- INTERPRETATION: read top to bottom -- whichever count first drops to (or
-- near) zero is the filter eliminating everything.
--   pct_loan_uid_populated near 0%      -> loan_uid isn't populated in this
--                                          load; the rollover mechanism
--                                          can't be traced via loan_uid at
--                                          all in this dataset, regardless
--                                          of anything else.
--   n_anomaly_open_total is healthy but
--   n_loan_uids_with_multiple_fids is 0 -> loan_uid IS populated, but no
--                                          disbursement_fid ever shares one
--                                          with another -- i.e. every loan
--                                          gets its own distinct loan_uid
--                                          even under a rollover, so the
--                                          "shared loan_uid" mechanism
--                                          documented for the OLD schema
--                                          may not hold post-rebuild.
--   n_multi_fid_loan_uids_with_anomaly
--     is 0 but multiple-fid groups exist -> loan_uid sharing happens, but
--                                          never co-occurs with the
--                                          ANOMALY_OPEN flag specifically
--                                          -- worth checking what other
--                                          loan_status values appear on
--                                          those shared-loan_uid rows
--                                          instead.
--   n_matching_final_filter is 0 but the
--     row above it isn't                -> just the April 2026+ recency
--                                          filter; drop it and use the full
--                                          date range instead.
