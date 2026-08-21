-- Isolation step: does the tied-fid CTE alone still find ties right now,
-- independent of the outer join? Same logic as Section F and
-- section_f_tie_sample.sql's tied_fids CTE, just wrapped in COUNT(*).
-- If this returns 0 too, the values substituted here don't match whatever
-- was substituted when Section F itself found 59,424,552 -- double check
-- :snapshot_dt / :as_of_load_ts against that run before concluding anything
-- about the data itself.
SELECT COUNT(*) AS tied_fid_count
FROM (
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
);
