-- ============================================================================
-- spot_check_loan_trace.sql
-- ============================================================================
-- Cheap, human-readable spot check: pick a handful of disbursements and
-- trace each one by hand through repayments_daily and loan_state_daily,
-- instead of running a full validation suite over the whole table set.
--
-- WINDOW SIZE: kept as tight as the goal allows. 48h maturity only needs a
-- handful of days of margin, not months, so the old bucket looks back just
-- 10-20 days (not 60-200) -- comfortably past the 48h penalty window while
-- still a small partition range. Steps 2-4 share one combined range: 20
-- days before the anchor through 7 days after it (27 partition-days total,
-- not ~207) -- the 7-day forward extension exists because a loan disbursed
-- ON the anchor date itself (the recent bucket's edge case) has its 24h/48h
-- penalty checkpoints falling on the day(s) AFTER the anchor; a cutoff
-- capped exactly at the anchor would silently miss those, making a loan
-- look defaulted when the query just never looked far enough forward.
--
-- PARTITION PRUNING: all three momo_loan_book_tracker_* tables are
-- partitioned by date_key. Every WHERE clause below filters on date_key
-- directly (not on a derived expression like date(try_cast(disbursement_ts
-- AS timestamp)), which the engine can't use for partition elimination),
-- and lists that filter FIRST, before the disbursement_fid/customer_msisdn
-- IN-list -- partition elimination should prune down to a handful of
-- partitions before the ID lookup even runs, rather than the ID lookup
-- doing the work of scanning every partition on its own. date_key bounds
-- are computed as constant expressions (date_add/date_format applied to
-- the literal anchor, not to the column itself), so they fold to a plain
-- range comparison against the raw partition column and still prune
-- correctly regardless of clause order -- listing the partition filter
-- first here is about making the intent legible, not changing the plan.
--
-- STEP 1's anchor is 2026-04-30 -- deliberately earlier than
-- borrower_history.txt/loan_summary_query.txt's current snapshot_dt
-- (2026-07-31), not tied to it, so both sample buckets sit comfortably
-- away from the edges of whatever's actually loaded (the old bucket
-- doesn't risk landing before the data starts; the recent bucket doesn't
-- depend on rows disbursed in the last couple of days before the live
-- cutoff, which may be sparser or still trickling in). Change the anchor
-- literal to try a different date; it doesn't need to match the production
-- files' snapshot_dt. If STEP 1 comes back empty, that's a sign the anchor
-- sits outside the real data range -- rerun
-- `SELECT MIN(disbursement_ts), MAX(disbursement_ts) FROM
-- analytics.momo_loan_book_tracker_disbursements_daily` and re-anchor.
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
-- cure-timing outcome and a still-open one.
-- Record the disbursement_fid / customer_msisdn values returned here --
-- Athena/Trino has no cross-statement variables, so substitute them by
-- hand into :sample_fids / :sample_msisdns in steps 2-4 below (same manual
-- substitution pattern as :snapshot_dt elsewhere in this project).
WITH sample_old AS (
SELECT disbursement_fid, customer_msisdn, disbursement_ts, disbursement_amount_ugx
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE date_key BETWEEN cast(date_format(date_add('day', -20, date '2026-04-30'), '%Y%m%d') AS bigint)
AND cast(date_format(date_add('day', -10, date '2026-04-30'), '%Y%m%d') AS bigint)
AND try_cast(disbursement_ts AS timestamp) IS NOT NULL
AND disbursement_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
ORDER BY disbursement_ts ASC
LIMIT 3
),
sample_recent AS (
SELECT disbursement_fid, customer_msisdn, disbursement_ts, disbursement_amount_ugx
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE date_key BETWEEN cast(date_format(date_add('day', -2, date '2026-04-30'), '%Y%m%d') AS bigint)
AND cast(date_format(date '2026-04-30', '%Y%m%d') AS bigint)
AND try_cast(disbursement_ts AS timestamp) IS NOT NULL
AND disbursement_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
ORDER BY disbursement_ts DESC
LIMIT 3
)
SELECT 'old' AS bucket, * FROM sample_old
UNION ALL
SELECT 'recent' AS bucket, * FROM sample_recent;
-- RESULT:


-- STEP 2: the disbursement rows themselves, for reference while eyeballing
-- steps 3-4. Substitute the disbursement_fid values from STEP 1. The
-- date_key bound here is the same combined range STEP 1 searched across
-- (covers both buckets) -- these rows came from that same table/range, so
-- it's a safe, still-pruning bound, not just the IN-list on its own.
SELECT disbursement_fid, customer_msisdn, disbursement_ts, disbursement_amount_ugx, inserted_ts
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE date_key BETWEEN cast(date_format(date_add('day', -20, date '2026-04-30'), '%Y%m%d') AS bigint)
AND cast(date_format(date_add('day', 7, date '2026-04-30'), '%Y%m%d') AS bigint)
AND disbursement_fid IN (:sample_fids)
ORDER BY disbursement_ts;
-- RESULT:


-- STEP 3: every repayment for those SAME msisdns (not just matching
-- disbursement_fid -- repayments carry no loan key at all, which is exactly
-- why the attribution heuristic exists). Same combined date_key range as
-- STEP 2.
SELECT repayment_fid, customer_msisdn, repayment_ts, repayment_amount_ugx, inserted_ts
FROM analytics.momo_loan_book_tracker_repayments_daily
WHERE date_key BETWEEN cast(date_format(date_add('day', -20, date '2026-04-30'), '%Y%m%d') AS bigint)
AND cast(date_format(date_add('day', 7, date '2026-04-30'), '%Y%m%d') AS bigint)
AND customer_msisdn IN (:sample_msisdns)
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
-- running total against. Same combined date_key bound as steps 2-3 --
-- loan_state_daily is itself a daily-snapshot table, so this also limits
-- how many per-day rows come back per loan before ORDER BY picks the
-- latest one.
SELECT disbursement_fid, date_key, loan_status, aging_bucket, days_aging,
lifetime_disbursed_ugx, lifetime_repaid_ugx, lifetime_gross_repaid_ugx,
interest_and_penalty_ugx, is_anomaly_open
FROM analytics.momo_loan_book_tracker_loan_state_daily
WHERE date_key BETWEEN cast(date_format(date_add('day', -20, date '2026-04-30'), '%Y%m%d') AS bigint)
AND cast(date_format(date_add('day', 7, date '2026-04-30'), '%Y%m%d') AS bigint)
AND disbursement_fid IN (:sample_fids)
ORDER BY disbursement_fid, date_key DESC;
-- RESULT:
-- EYEBALL: lifetime_repaid_ugx here should track (not necessarily equal --
-- see the gross-vs-principal-only open question) the running total you
-- summed by hand in STEP 3. loan_status/aging_bucket should be consistent
-- with whether/when principal was reached. is_anomaly_open = true means
-- this loan is excluded from both borrower_history.txt and
-- loan_summary_query.txt entirely -- if one of your sampled loans is
-- flagged here, don't expect it to show up in either file's output at all.
