-- ============================================================================
-- spot_check_loan_trace.sql
-- ============================================================================
-- Cheap, human-readable spot check: pick a handful of disbursements and
-- trace each one by hand through repayments_daily and loan_state_daily,
-- instead of running a full validation suite over the whole table set.
-- Every query below is scoped to a tiny IN-list of msisdns/disbursement_fids
-- picked in STEP 1 -- nothing here does a full-table join, so this is cheap
-- enough to run interactively even against the 120M-row source tables.
--
-- STEP 1's date bounds are anchored to 2026-07-31 -- the same snapshot_dt
-- literal currently used in borrower_history.txt and loan_summary_query.txt.
-- Update it here too if those files' snapshot_dt ever changes, so the
-- sample stays anchored to the same cutoff the production queries use.
--
-- What this checks that the aggregate GATE-style validation queries don't:
-- whether the repayment-attribution heuristic and the derived cure-timing
-- (used by both borrower_history.txt's on_time/default flags and
-- loan_summary_query.txt's penalty_events) actually make sense for real,
-- individual loans -- not just in aggregate.
--
-- What to eyeball once you have results:
--   1. STEP 3's running repayment total for a msisdn, summed by hand,
--      against STEP 4's lifetime_repaid_ugx for that same loan.
--   2. Whether principal was reached within 24h/48h of disbursement_ts
--      (STEP 2) -- does that match loan_state's loan_status/aging_bucket
--      (STEP 4)?
--   3. Do the penalty events borrower_history.txt/loan_summary_query.txt
--      would derive (0/1/2, per the 24h/48h rule) match what you'd
--      conclude by eye from steps 2-4?
--   4. If a msisdn has more than one loan in the sample, does the
--      time-window attribution (bounding each repayment by the loan's
--      disbursement_ts and the NEXT disbursement) assign repayments to the
--      loan you'd expect, or does it look ambiguous/wrong?
-- ============================================================================

-- STEP 1: a small, mixed sample -- 3 older (mature/fully-resolved) loans
-- and 3 very recent (immature) loans, so you can eyeball both a completed
-- cure-timing outcome and a still-open one. Both buckets are DATE-BOUNDED
-- (not a bare ORDER BY ... LIMIT over the full 120M-row table) for two
-- reasons: (1) cost/performance -- an unbounded ORDER BY needs the engine
-- to consider the entire table to find the true min/max, which is not
-- cheap; a date bound prunes that scan (and, if these tables are date-
-- partitioned, prunes partitions outright); (2) meaningfulness -- the
-- literal oldest/newest rows ever recorded aren't necessarily "a clearly
-- mature loan" vs "a clearly immature loan," just whatever extremes happen
-- to exist. Bounding to "60-200 days before snapshot_dt" guarantees the
-- old bucket is well past the 48h penalty window; "within 2 days of
-- snapshot_dt" guarantees the recent bucket is still immature.
-- Record the disbursement_fid / customer_msisdn values returned here --
-- Athena/Trino has no cross-statement variables, so substitute them by
-- hand into :sample_fids / :sample_msisdns in steps 2-4 below (same manual
-- substitution pattern as :snapshot_dt elsewhere in this project).
WITH sample_old AS (
SELECT disbursement_fid, customer_msisdn, disbursement_ts, disbursement_amount_ugx
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE try_cast(disbursement_ts AS timestamp) IS NOT NULL
AND disbursement_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date(try_cast(disbursement_ts AS timestamp)) BETWEEN date_add('day', -200, date '2026-07-31') AND date_add('day', -60, date '2026-07-31')
ORDER BY disbursement_ts ASC
LIMIT 3
),
sample_recent AS (
SELECT disbursement_fid, customer_msisdn, disbursement_ts, disbursement_amount_ugx
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE try_cast(disbursement_ts AS timestamp) IS NOT NULL
AND disbursement_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date(try_cast(disbursement_ts AS timestamp)) BETWEEN date_add('day', -2, date '2026-07-31') AND date '2026-07-31'
ORDER BY disbursement_ts DESC
LIMIT 3
)
SELECT 'old' AS bucket, * FROM sample_old
UNION ALL
SELECT 'recent' AS bucket, * FROM sample_recent;
-- RESULT:


-- STEP 2: the disbursement rows themselves, for reference while eyeballing
-- steps 3-4. Substitute the disbursement_fid values from STEP 1.
SELECT disbursement_fid, customer_msisdn, disbursement_ts, disbursement_amount_ugx, inserted_ts
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE disbursement_fid IN (:sample_fids)
ORDER BY disbursement_ts;
-- RESULT:


-- STEP 3: every repayment for those SAME msisdns (not just matching
-- disbursement_fid -- repayments carry no loan key at all, which is exactly
-- why the attribution heuristic exists). Scoped to a tiny msisdn IN-list
-- picked in STEP 1, so this stays cheap despite repayments_daily being huge.
SELECT repayment_fid, customer_msisdn, repayment_ts, repayment_amount_ugx, inserted_ts
FROM analytics.momo_loan_book_tracker_repayments_daily
WHERE customer_msisdn IN (:sample_msisdns)
ORDER BY customer_msisdn, repayment_ts;
-- RESULT:
-- EYEBALL: for a msisdn with only ONE disbursement in your sample, every
-- repayment here unambiguously belongs to it -- sum repayment_amount_ugx in
-- repayment_ts order by hand and note when (if ever) the running total
-- reaches disbursement_amount_ugx from STEP 2, and whether that's within
-- 24h / 48h / never. If a msisdn has multiple loans in the sample, this is
-- exactly the case where the time-window attribution heuristic (bounded by
-- the loan's own disbursement_ts and the NEXT disbursement) has to make a
-- judgment call -- check whether that call looks right.
-- Also note here whether any repayment_amount_ugx values are negative --
-- the open sign-convention question (principal-only vs. gross, and
-- negative-as-reversal vs. negative-as-debit-convention) flagged in
-- borrower_history_validation_queries.sql Section B0 is directly
-- observable in this raw data.


-- STEP 4: loan_state_daily's latest snapshot for those disbursement_fids --
-- the authoritative status/lifetime totals to compare STEP 3's manual
-- running total against.
SELECT disbursement_fid, date_key, loan_status, aging_bucket, days_aging,
lifetime_disbursed_ugx, lifetime_repaid_ugx, lifetime_gross_repaid_ugx,
interest_and_penalty_ugx, is_anomaly_open
FROM analytics.momo_loan_book_tracker_loan_state_daily
WHERE disbursement_fid IN (:sample_fids)
ORDER BY disbursement_fid, date_key DESC;
-- RESULT:
-- EYEBALL: lifetime_repaid_ugx here should track (not necessarily equal --
-- see the gross-vs-principal-only open question) the running total you
-- summed by hand in STEP 3. loan_status/aging_bucket should be consistent
-- with whether/when principal was reached. is_anomaly_open = true means
-- this loan is excluded from both borrower_history.txt and
-- loan_summary_query.txt entirely -- if one of your sampled loans is
-- flagged here, don't expect it to show up in either file's output at all.
