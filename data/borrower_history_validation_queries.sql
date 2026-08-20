-- ============================================================================
-- borrower_history_validation_queries.sql
-- ============================================================================
-- Diagnostics for the borrower_history.txt rewrite (xtrafloat_daily_trans ->
-- momo_loan_book_tracker_* tables). None of these have been run -- this repo
-- has no live warehouse connection, so every query here is prepared, not
-- executed. Treat the rewrite as unverified until GATE 6 at the bottom of
-- this file is filled in with real execution evidence.
--
-- ARCHITECTURE (v2): every earlier version of this file had each section
-- (B1/B2/B3/C/D/...) re-implement its own copy of the disbursement/repayment/
-- loan-state dedup logic, and those copies drifted from borrower_history.txt
-- and from each other -- missing ROW_NUMBER dedup in some places, missing the
-- as_of_load_ts freeze in others. That's a structural problem, not a series
-- of one-off typos, so this version fixes it structurally: GATE 0 below
-- creates a small set of views ONCE, matching borrower_history.txt's own
-- dedup/window/anomaly-exclusion logic exactly, and every later section
-- queries those views instead of re-deriving the logic. There is now exactly
-- one place this logic is defined.
--
-- Substitute before running:
--   :validation_schema  -- a database you have CREATE VIEW rights in (a
--                            scratch/sandbox schema, not production)
--   :snapshot_dt         -- e.g. 20260731 -- MUST match the borrower_history.txt
--                            run being validated
--   :as_of_load_ts       -- e.g. TIMESTAMP '2026-08-20 00:00:00.000' -- MUST
--                            match the borrower_history.txt run being validated
-- ============================================================================
--
-- ============================================================================
-- ACCEPTANCE THRESHOLDS (proposed defaults -- ratify with the team before
-- using any of these as a go/no-go gate; they are starting points, not
-- something I can unilaterally declare correct)
-- ============================================================================
--   Attributed repayment coverage (count)  (B1: attributed_repayments /
--                                            total_repayments)            >= 99%
--   Attributed repayment coverage (value)  (B1: attributed UGX / total
--                                            repayment UGX)               >= 99%
--   Boundary-sensitive repayment rate      (B1: near_next_disbursement_
--                                            boundary / attributed_
--                                            repayments)                  <= 1%
--   Rapid-reborrow rate                    (B1: rapid_reborrow_disbursements
--                                            / total_disbursements)       <= 2%
--   Reconciliation match rate (count)      (B2: n_exact_match_abs /
--                                            n_matched)                   >= 98%
--   Reconciliation match rate (value)      (B2: matched UGX within
--                                            tolerance / total matched
--                                            lifetime_repaid_ugx)         >= 98%
--   p95 abs reconciliation error, relative
--   to median matched-loan principal        (B2)                          <= 2%
--   Negative repayment share               (B0: negative_repayments /
--                                            total_repayments)            ~0%,
--                                            investigate any nonzero count
--   Dedup tie collisions                   (GATE 0 views are built to
--                                            resolve ties the same way
--                                            production does; F reports the
--                                            count separately)             0
--   Final-output grain violations          (GATE 1: rows with phonenumber
--                                            not unique, null, or empty)    0
--   Final-output range violations          (GATE 1: rates outside [0,1],
--                                            negative counts, first_loan_ts
--                                            > latest_loan_ts)              0
-- ============================================================================


-- ============================================================================
-- GATE 0 -- Canonical source layer (create once, reuse everywhere below)
-- ============================================================================
-- Mirrors borrower_history.txt's disb_raw -> disb_dedup -> disb_windows,
-- repay_raw -> repay_dedup, and loan_state_loads_dedup -> loan_state_snapshot
-- -> loan_state_anomalies chain exactly: same casts, same filters (including
-- ova = 'XTRAFLOAT-AGENT' -- all three momo_loan_book_tracker_* tables carry
-- other MoMo services' activity too, confirmed against the warehouse), same
-- dedup ORDER BY, same as_of_load_ts freeze. If you change
-- borrower_history.txt's dedup/window/anomaly logic, update these views to
-- match -- this is the only place that logic should exist in this file.

-- Includes disbursement_external_id even though most sections don't need it,
-- so Section E's identity-bridge check can use this single deduped view
-- instead of joining back to the raw, un-deduped source table (which would
-- reintroduce the exact fan-out risk this canonical layer exists to remove).
CREATE OR REPLACE VIEW :validation_schema.vw_bh_disb_dedup AS
SELECT disbursement_fid, phonenumber, disbursement_ts, disbursed_amount, disbursement_external_id
FROM (
SELECT d.*,
ROW_NUMBER() OVER (
PARTITION BY disbursement_fid
ORDER BY inserted_ts DESC
) rn
FROM (
SELECT
disbursement_fid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber,
try_cast(disbursement_ts AS timestamp) AS disbursement_ts,
cast(disbursement_amount_ugx AS double) AS disbursed_amount,
disbursement_external_id,
inserted_ts
FROM analytics.momo_loan_book_tracker_disbursements_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND try_cast(disbursement_ts AS timestamp) IS NOT NULL
AND disbursement_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date_key <= :snapshot_dt
AND date(try_cast(disbursement_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
AND inserted_ts <= :as_of_load_ts
) d
)
WHERE rn = 1;

CREATE OR REPLACE VIEW :validation_schema.vw_bh_disb_windows AS
SELECT
disbursement_fid,
phonenumber,
disbursement_ts,
disbursed_amount,
LEAD(disbursement_ts) OVER (
PARTITION BY phonenumber ORDER BY disbursement_ts, disbursement_fid
) AS next_disbursement_ts
FROM :validation_schema.vw_bh_disb_dedup;

CREATE OR REPLACE VIEW :validation_schema.vw_bh_repay_dedup AS
SELECT repayment_fid, phonenumber, repayment_ts, repayment_amount
FROM (
SELECT r.*,
ROW_NUMBER() OVER (
PARTITION BY repayment_fid
ORDER BY inserted_ts DESC
) rn
FROM (
SELECT
repayment_fid,
regexp_replace(trim(cast(customer_msisdn AS varchar)), '[^0-9]', '') AS phonenumber,
try_cast(repayment_ts AS timestamp) AS repayment_ts,
cast(repayment_amount_ugx AS double) AS repayment_amount,
inserted_ts
FROM analytics.momo_loan_book_tracker_repayments_daily
WHERE ova = 'XTRAFLOAT-AGENT'
AND try_cast(repayment_ts AS timestamp) IS NOT NULL
AND repayment_fid IS NOT NULL
AND customer_msisdn IS NOT NULL
AND date_key <= :snapshot_dt
AND date(try_cast(repayment_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
AND inserted_ts <= :as_of_load_ts
) r
)
WHERE rn = 1;

CREATE OR REPLACE VIEW :validation_schema.vw_bh_loan_state_snapshot AS
SELECT disbursement_fid, loan_status, aging_bucket, days_aging, is_active_loan,
is_anomaly_open, has_pre_window_history, lifetime_disbursed_ugx,
lifetime_repaid_ugx, lifetime_gross_repaid_ugx, interest_and_penalty_ugx,
expected_total_charge_ugx, charge_variance_ugx, charge_variance_pct,
is_charge_anomaly
FROM (
SELECT lsld.*,
ROW_NUMBER() OVER (
PARTITION BY disbursement_fid
ORDER BY date_key DESC
) rn2
FROM (
SELECT *
FROM (
SELECT lsd.*,
ROW_NUMBER() OVER (
PARTITION BY disbursement_fid, date_key
ORDER BY inserted_ts DESC
) rn
FROM analytics.momo_loan_book_tracker_loan_state_daily lsd
WHERE ova = 'XTRAFLOAT-AGENT'
AND disbursement_fid IS NOT NULL
AND date_key <= :snapshot_dt
AND inserted_ts <= :as_of_load_ts
)
WHERE rn = 1
) lsld
)
WHERE rn2 = 1;

CREATE OR REPLACE VIEW :validation_schema.vw_bh_loan_state_anomalies AS
SELECT disbursement_fid
FROM :validation_schema.vw_bh_loan_state_snapshot
WHERE loan_status = 'ANOMALY_OPEN' OR is_anomaly_open = true;

-- Surviving loan-level population: exactly what feeds classified/loan_core
-- in borrower_history.txt (disbursements, ANOMALY_OPEN excluded). Every
-- section below that needs "the population borrower_history.txt actually
-- produces loans from" should use this view, not disb_dedup directly.
CREATE OR REPLACE VIEW :validation_schema.vw_bh_surviving_loans AS
SELECT d.disbursement_fid, d.phonenumber, d.disbursement_ts, d.disbursed_amount
FROM :validation_schema.vw_bh_disb_dedup d
WHERE d.disbursement_fid NOT IN (SELECT disbursement_fid FROM :validation_schema.vw_bh_loan_state_anomalies);

-- The production query itself, as a view -- gives us (a) a real compile
-- check (CREATE VIEW fails if the query doesn't parse/analyze) and (b) a
-- single queryable target for the GATE 1 final-output checks below, without
-- pasting the 700+ line query into this file a second time and letting the
-- two copies drift.
-- ACTION REQUIRED: do NOT hand-paste data/borrower_history.txt here --
-- manual assembly risks validating a different revision than what's
-- actually checked in. Instead run:
--   python scripts/build_vw_bh_output.py <validation_schema> > vw_bh_output.sql
-- and execute the generated vw_bh_output.sql. It contains SEVEN statements,
-- in order: DROP TABLE IF EXISTS + CREATE TABLE ... AS SELECT for a
-- <validation_schema>.tbl_bh_classified checkpoint (transaction grain), then
-- the same for a <validation_schema>.tbl_bh_loan_final checkpoint (built
-- from tbl_bh_classified, per-loan grain), then the same for a
-- <validation_schema>.tbl_bh_loan_level checkpoint (built from
-- tbl_bh_loan_final, adds the borrower-level window-function cascade), then
-- CREATE OR REPLACE VIEW for vw_bh_output itself (built from
-- tbl_bh_loan_level). borrower_history.txt is split at its
-- ##BORROWER_HISTORY_CHECKPOINT_0##/_1##/_2## markers because the query
-- exceeds the warehouse's stage-count ceiling as one statement, and even as
-- two -- see those markers' comments for why. The generated file stamps the
-- git commit SHA of data/borrower_history.txt as a comment -- record that
-- SHA in GATE 6. The statements below are a structural placeholder only,
-- showing what the generated file's shape looks like -- they are NOT meant
-- to be run as written.
-- DROP TABLE IF EXISTS :validation_schema.tbl_bh_classified;
-- CREATE TABLE :validation_schema.tbl_bh_classified AS
-- <generated from data/borrower_history.txt Part A0 by scripts/build_vw_bh_output.py>
-- ;
-- DROP TABLE IF EXISTS :validation_schema.tbl_bh_loan_final;
-- CREATE TABLE :validation_schema.tbl_bh_loan_final AS
-- <generated from data/borrower_history.txt Part A1 by scripts/build_vw_bh_output.py>
-- ;
-- DROP TABLE IF EXISTS :validation_schema.tbl_bh_loan_level;
-- CREATE TABLE :validation_schema.tbl_bh_loan_level AS
-- <generated from data/borrower_history.txt Part A2 by scripts/build_vw_bh_output.py>
-- ;
-- CREATE OR REPLACE VIEW :validation_schema.vw_bh_output AS
-- <generated from data/borrower_history.txt Part A3 by scripts/build_vw_bh_output.py>
-- ;


-- ============================================================================
-- GATE 1 -- Final-output checks (against vw_bh_output)
-- ============================================================================
-- Nothing in earlier versions of this file validated the actual production
-- query's output shape at all -- every section only tested the new source
-- adapter (disbursements/repayments/loan-state) in isolation. This is the
-- first check against the real thing.

-- 1a. Grain: exactly one row per phonenumber, no null/empty identifiers.
SELECT
COUNT(*) AS total_rows,
COUNT(DISTINCT phonenumber) AS distinct_phonenumbers,
COUNT(*) - COUNT(DISTINCT phonenumber) AS grain_violation_count,
SUM(CASE WHEN phonenumber IS NULL OR phonenumber = '' THEN 1 ELSE 0 END) AS null_or_empty_phonenumber
FROM :validation_schema.vw_bh_output;
-- RESULT:
-- INTERPRETATION: grain_violation_count and null_or_empty_phonenumber must
-- both be 0 -- prepare_borrower_limit_features() dedupes on msisdn and
-- assumes one row per borrower; a violation here means something upstream
-- (dedup, join fan-out) is broken.

-- 1b. Range and logical-identity checks. Joins vw_bh_output back to
-- vw_bh_loan_state_snapshot on latest_requestid so the missing-snapshot
-- check (below) can tell "no match" apart from "matched, status null" --
-- everything else here reads only from vw_bh_output (aliased o; note
-- has_pre_window_history exists on BOTH views with different meanings --
-- borrower-level on o, per-loan on ls -- so every column reference below
-- is deliberately prefixed to avoid ambiguity once the join is present).
SELECT
SUM(CASE WHEN o.lifetime_on_time_24h_rate NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS bad_on_time_24h_rate,
SUM(CASE WHEN o.lifetime_default_24h_rate NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS bad_default_24h_rate,
SUM(CASE WHEN o.lifetime_on_time_26h_rate NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS bad_on_time_26h_rate,
SUM(CASE WHEN o.lifetime_default_26h_rate NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS bad_default_26h_rate,
SUM(CASE WHEN o.lifetime_severe_default_48h_rate NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS bad_severe_default_48h_rate,
SUM(CASE WHEN o.lifetime_zero_recovery_rate NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS bad_zero_recovery_rate,
SUM(CASE WHEN o.total_loans < 1 THEN 1 ELSE 0 END) AS bad_total_loans,
SUM(CASE WHEN o.num_prior_loans < 0 THEN 1 ELSE 0 END) AS bad_num_prior_loans,
SUM(CASE WHEN o.first_loan_ts > o.latest_loan_ts THEN 1 ELSE 0 END) AS bad_first_vs_latest_ts,
-- on_time_24h_flag and default_24h_flag are exact complements per loan
-- (see loan_final in borrower_history.txt), so their lifetime rates should
-- sum to exactly 1 for every borrower with qualifying loans -- same for the
-- 26h pair. A violation means the two derivations of the same threshold
-- (timestamp comparison vs. checkpoint FILTER) have drifted apart.
SUM(CASE WHEN ABS(o.lifetime_on_time_24h_rate + o.lifetime_default_24h_rate - 1.0) > 0.001 THEN 1 ELSE 0 END) AS bad_24h_complement_identity,
SUM(CASE WHEN ABS(o.lifetime_on_time_26h_rate + o.lifetime_default_26h_rate - 1.0) > 0.001 THEN 1 ELSE 0 END) AS bad_26h_complement_identity,
-- for the latest loan specifically, num_prior_loans must equal total_loans - 1
-- (it's the last loan in the same ordering total_loans counts over)
SUM(CASE WHEN o.num_prior_loans != o.total_loans - 1 THEN 1 ELSE 0 END) AS bad_num_prior_loans_identity,
-- NULL RATES: quantify, don't silently let NOT BETWEEN/ABS(...)>x pass NULL
-- rows uncounted (both evaluate to unknown/false for NULL inputs, so the
-- SUM(CASE...) checks above never flag them). A NULL lifetime_*_rate here
-- means exactly "this borrower has zero MATURE loans for that horizon" --
-- AVG() over zero non-null rows is NULL, there's no other way to get one.
SUM(CASE WHEN o.lifetime_on_time_24h_rate IS NULL THEN 1 ELSE 0 END) AS n_null_on_time_24h_rate,
SUM(CASE WHEN o.lifetime_default_24h_rate IS NULL THEN 1 ELSE 0 END) AS n_null_default_24h_rate,
SUM(CASE WHEN o.lifetime_on_time_26h_rate IS NULL THEN 1 ELSE 0 END) AS n_null_on_time_26h_rate,
SUM(CASE WHEN o.lifetime_default_26h_rate IS NULL THEN 1 ELSE 0 END) AS n_null_default_26h_rate,
SUM(CASE WHEN o.lifetime_severe_default_48h_rate IS NULL THEN 1 ELSE 0 END) AS n_null_severe_default_48h_rate,
SUM(CASE WHEN o.lifetime_zero_recovery_rate IS NULL THEN 1 ELSE 0 END) AS n_null_zero_recovery_rate,
-- ASYMMETRIC NULL: a complementary pair should be NULL together or not at
-- all (same maturity gate feeds both). If this is ever nonzero, the two
-- flags' maturity gates have drifted apart from each other.
SUM(CASE WHEN (o.lifetime_on_time_24h_rate IS NULL) != (o.lifetime_default_24h_rate IS NULL) THEN 1 ELSE 0 END) AS bad_24h_asymmetric_null,
SUM(CASE WHEN (o.lifetime_on_time_26h_rate IS NULL) != (o.lifetime_default_26h_rate IS NULL) THEN 1 ELSE 0 END) AS bad_26h_asymmetric_null,
-- CORRECTION to an earlier version of this check: `latest_loan_status IS
-- NULL` cannot actually distinguish "no snapshot row matched" from "a
-- snapshot row matched but its own loan_status happens to be null" -- both
-- produce the same NULL in vw_bh_output, since latest_loan_status is just
-- the raw (nullable) ls.loan_status carried through the LEFT JOIN in
-- loan_level. ls.disbursement_fid IS NULL from the explicit join above is
-- the only way to tell these apart -- it's NULL if and only if the join
-- found nothing, regardless of what a matched row's own fields contain.
SUM(CASE WHEN ls.disbursement_fid IS NULL THEN 1 ELSE 0 END) AS n_latest_loan_missing_state_snapshot,
SUM(CASE WHEN ls.disbursement_fid IS NOT NULL AND o.latest_loan_status IS NULL THEN 1 ELSE 0 END) AS n_latest_loan_matched_but_status_null
FROM :validation_schema.vw_bh_output o
LEFT JOIN :validation_schema.vw_bh_loan_state_snapshot ls ON ls.disbursement_fid = o.latest_requestid;
-- RESULT:
-- INTERPRETATION: bad_* columns should all be 0 -- concrete defects, not
-- thresholds open to interpretation. n_null_* and n_latest_loan_missing_
-- state_snapshot are expected to be nonzero in normal operation (recent
-- unseasoned loans, loans not yet reflected in a loan_state_daily load) --
-- report the counts so reviewers know the scale, rather than treating a
-- passing range/identity check as "nothing to see here."

-- 1c. Column contract: exact set equality against the 51 documented columns
-- (41 legacy + 10 enrichment) -- both directions. An earlier version of
-- this check only used NOT IN, which can only ever detect unexpected
-- extras: a genuinely missing column plus a genuinely unexpected extra
-- would leave the count at 51 and pass silently. This version does a real
-- two-way anti-join against an explicit expected-columns list.
-- Order is not independently verifiable from information_schema, which
-- does not guarantee ordinal stability across all engines -- verify order
-- by reading the view/query definition directly if consumers rely on it.
WITH expected(column_name) AS (
VALUES
('phonenumber'),('total_loans'),('first_loan_ts'),('latest_loan_ts'),('total_disbursed_amount'),
('avg_loan_size_lifetime'),('max_loan_size_lifetime'),('lifetime_on_time_24h_rate'),
('lifetime_on_time_26h_rate'),('lifetime_default_24h_rate'),('lifetime_default_26h_rate'),
('lifetime_severe_default_48h_rate'),('lifetime_zero_recovery_rate'),
('lifetime_avg_hours_to_principal_cure'),('lifetime_worst_hours_to_principal_cure'),
('lifetime_cure_time_volatility'),('latest_requestid'),('latest_disbursement_ts'),
('latest_disbursed_amount'),('num_prior_loans'),('prior_on_time_24h_rate'),
('avg_prior_hours_to_cure'),('worst_prior_hours_to_cure'),('cure_time_volatility'),
('recent_3_on_time_rate'),('recent_3_avg_cure_time'),('lifetime_prior_default_24h_rate'),
('recent_5_default_24h_rate'),('cure_time_trend'),('borrower_trend'),('borrower_profile_type'),
('loans_last_50_loans'),('defaults_last_50_loans'),('default_rate_last_50_loans'),
('prior_on_time_streak'),('prior_default_streak'),('avg_prior_loan_size'),('max_prior_loan_size'),
('loan_size_vs_avg_ratio'),('loan_size_vs_max_ratio'),('loan_above_prior_max_flag'),
-- additive enrichment columns
('has_pre_window_history'),('latest_loan_status'),('latest_aging_bucket'),('latest_days_aging'),
('latest_is_active_loan'),('latest_interest_and_penalty_ugx'),('latest_expected_total_charge_ugx'),
('latest_charge_variance_ugx'),('latest_charge_variance_pct'),('latest_is_charge_anomaly')
),
actual AS (
SELECT column_name
FROM information_schema.columns
WHERE table_schema = ':validation_schema_as_quoted_string'  -- substitute the schema name here AS A QUOTED STRING, e.g. 'sandbox' -- unlike every :validation_schema.object_name reference elsewhere in this file, this one is a string comparison, not an identifier prefix, and needs quotes
AND table_name = 'vw_bh_output'
)
SELECT 'MISSING (documented but not present in vw_bh_output)' AS issue, e.column_name
FROM expected e
LEFT JOIN actual a ON a.column_name = e.column_name
WHERE a.column_name IS NULL
UNION ALL
SELECT 'UNEXPECTED (present but not documented)', a.column_name
FROM actual a
LEFT JOIN expected e ON e.column_name = a.column_name
WHERE e.column_name IS NULL;
-- RESULT:
-- INTERPRETATION: this query should return ZERO rows. Any MISSING row is a
-- required column absent from the output -- a genuine contract break. Any
-- UNEXPECTED row is an undocumented column that drifted into the SELECT
-- list. Both directions are checked independently, so a missing column and
-- an extra column can't cancel out into a false pass the way a bare count
-- comparison could.

-- 1d. Type-family contract: 1c only checks column NAMES -- a column could
-- keep its name and silently change from double to varchar, or boolean to
-- string, and still pass. Checked against FAMILIES (matched by data_type
-- prefix), not exact Trino type names, deliberately: I can't confirm exact
-- warehouse types without live access, and asserting one specific type
-- (e.g. bigint vs integer for a count) would risk flagging a difference
-- that doesn't actually matter to the consumer as a false defect.
-- Families are derived from prepare_borrower_limit_features()'s own
-- coercion code (extrafloat/engine/extrafloat_limit_engine_features.py) --
-- what it actually does to each column -- not guessed from column names.
-- NOTE: latest_requestid is deliberately EXCLUDED from this pass/fail
-- assertion (see the separate informational query right after this one).
-- Traced in borrower_history.txt to `w.disbursement_fid AS requestid` with
-- no cast, so its real SQL type is whatever disbursement_fid natively is
-- (source samples suggest bigint) -- while the Python consumer does
-- `.astype(str)` on it, which works on either a numeric or string column
-- and so doesn't resolve the ambiguity either way. Asserting a family here
-- that isn't actually a ratified contract would make this gate cry wolf on
-- every run regardless of whether anything is actually wrong; instead its
-- real type is reported separately for a human to judge.
WITH expected_family(column_name, family) AS (
VALUES
('phonenumber','VARCHAR'),
('total_loans','NUMERIC'),('first_loan_ts','TIMESTAMP'),('latest_loan_ts','TIMESTAMP'),
('total_disbursed_amount','NUMERIC'),('avg_loan_size_lifetime','NUMERIC'),('max_loan_size_lifetime','NUMERIC'),
('lifetime_on_time_24h_rate','NUMERIC'),('lifetime_on_time_26h_rate','NUMERIC'),
('lifetime_default_24h_rate','NUMERIC'),('lifetime_default_26h_rate','NUMERIC'),
('lifetime_severe_default_48h_rate','NUMERIC'),('lifetime_zero_recovery_rate','NUMERIC'),
('lifetime_avg_hours_to_principal_cure','NUMERIC'),('lifetime_worst_hours_to_principal_cure','NUMERIC'),
('lifetime_cure_time_volatility','NUMERIC'),
-- latest_requestid intentionally omitted -- see NOTE above and the
-- informational query following this one.
('latest_disbursement_ts','TIMESTAMP'),('latest_disbursed_amount','NUMERIC'),
('num_prior_loans','NUMERIC'),('prior_on_time_24h_rate','NUMERIC'),('avg_prior_hours_to_cure','NUMERIC'),
('worst_prior_hours_to_cure','NUMERIC'),('cure_time_volatility','NUMERIC'),('recent_3_on_time_rate','NUMERIC'),
('recent_3_avg_cure_time','NUMERIC'),('lifetime_prior_default_24h_rate','NUMERIC'),
('recent_5_default_24h_rate','NUMERIC'),('cure_time_trend','NUMERIC'),
('borrower_trend','VARCHAR'),('borrower_profile_type','VARCHAR'),
('loans_last_50_loans','NUMERIC'),('defaults_last_50_loans','NUMERIC'),('default_rate_last_50_loans','NUMERIC'),
('prior_on_time_streak','NUMERIC'),('prior_default_streak','NUMERIC'),('avg_prior_loan_size','NUMERIC'),
('max_prior_loan_size','NUMERIC'),('loan_size_vs_avg_ratio','NUMERIC'),('loan_size_vs_max_ratio','NUMERIC'),
('loan_above_prior_max_flag','NUMERIC'),
('has_pre_window_history','BOOLEAN'),
('latest_loan_status','VARCHAR'),('latest_aging_bucket','VARCHAR'),('latest_days_aging','NUMERIC'),
('latest_is_active_loan','BOOLEAN'),
('latest_interest_and_penalty_ugx','NUMERIC'),('latest_expected_total_charge_ugx','NUMERIC'),
('latest_charge_variance_ugx','NUMERIC'),('latest_charge_variance_pct','NUMERIC'),
('latest_is_charge_anomaly','BOOLEAN')
),
actual_types AS (
SELECT
column_name,
data_type,
CASE
WHEN data_type LIKE 'timestamp%' THEN 'TIMESTAMP'
WHEN data_type LIKE 'date%' THEN 'TIMESTAMP'
WHEN data_type IN ('double','real','bigint','integer','smallint','tinyint') OR data_type LIKE 'decimal%' THEN 'NUMERIC'
WHEN data_type = 'boolean' THEN 'BOOLEAN'
WHEN data_type LIKE 'varchar%' OR data_type LIKE 'char%' THEN 'VARCHAR'
ELSE 'OTHER: ' || data_type
END AS actual_family
FROM information_schema.columns
WHERE table_schema = ':validation_schema_as_quoted_string'  -- see the note on 1c's identical placeholder
AND table_name = 'vw_bh_output'
)
SELECT
ef.column_name,
ef.family AS expected_family,
at.data_type AS actual_data_type,
at.actual_family
FROM expected_family ef
JOIN actual_types at ON at.column_name = ef.column_name
WHERE at.actual_family != ef.family;
-- RESULT:
-- INTERPRETATION: this query should return ZERO rows, full stop -- every
-- column it checks has a ratified expected family. Any row is a column
-- whose type family doesn't match what the Python consumer expects -- e.g.
-- a rate column that became VARCHAR would silently coerce to NaN/garbage
-- in pandas rather than erroring loudly.

-- 1d-info. latest_requestid's actual type, reported (not asserted) --
-- excluded from 1d above because its correct family is an open question,
-- not a known contract (see the NOTE on 1d). This is diagnostic, not a
-- pass/fail gate: read actual_data_type and make the numeric-vs-string
-- call directly, e.g. by asking whoever owns disbursement_fid's schema.
SELECT
column_name,
data_type AS actual_data_type,
CASE
WHEN data_type LIKE 'timestamp%' OR data_type LIKE 'date%' THEN 'TIMESTAMP'
WHEN data_type IN ('double','real','bigint','integer','smallint','tinyint') OR data_type LIKE 'decimal%' THEN 'NUMERIC'
WHEN data_type = 'boolean' THEN 'BOOLEAN'
WHEN data_type LIKE 'varchar%' OR data_type LIKE 'char%' THEN 'VARCHAR'
ELSE 'OTHER: ' || data_type
END AS actual_family
FROM information_schema.columns
WHERE table_schema = ':validation_schema_as_quoted_string'  -- see the note on 1c's identical placeholder
AND table_name = 'vw_bh_output'
AND column_name = 'latest_requestid';
-- RESULT:
-- INTERPRETATION: informational only -- record actual_family here in GATE 6
-- once known, so this stops being an open question on every future review.


-- ============================================================================
-- SOURCE DATA QUALITY GATE -- null/non-finite/implausible amounts
-- ============================================================================
-- Run this before A/B0-B4 -- a null or non-finite amount silently
-- propagating through SUM/ABS/ratios/cure-timing logic would corrupt every
-- downstream check without necessarily producing an obviously wrong result.
--
-- CAVEAT ON REACHABILITY: vw_bh_disb_dedup/vw_bh_repay_dedup (like
-- borrower_history.txt itself) cast amounts with plain `cast(... AS
-- double)`, not `try_cast`. If the underlying disbursement_amount_ugx/
-- repayment_amount_ugx columns are ever non-numeric (e.g. stored as varchar
-- with a malformed value), that cast throws and the ENTIRE view -- and
-- every check in this file, including this one -- fails to even run,
-- rather than surfacing as a null/bad row here. This section can only
-- catch quality problems that survive a successful cast (null, zero,
-- negative, non-finite double values); it cannot catch a cast failure
-- itself. If GATE 0's views fail to build with a cast/conversion error,
-- that error IS the finding -- it means switching to try_cast (and then
-- deciding what a failed-cast row should mean for the query) needs to be a
-- deliberate decision, not something to silently code around here.
SELECT
'disbursements' AS source,
COUNT(*) AS total_rows,
SUM(CASE WHEN disbursed_amount IS NULL THEN 1 ELSE 0 END) AS n_null_amount,
SUM(CASE WHEN disbursed_amount <= 0 THEN 1 ELSE 0 END) AS n_zero_or_negative_amount,
SUM(CASE WHEN is_nan(disbursed_amount) THEN 1 ELSE 0 END) AS n_nan_amount,
SUM(CASE WHEN is_infinite(disbursed_amount) THEN 1 ELSE 0 END) AS n_infinite_amount,
-- heuristic threshold, not a confirmed business limit -- flag for review,
-- not an automatic fail; adjust once the actual expected loan-size range
-- is confirmed with the business owner
SUM(CASE WHEN disbursed_amount > 1000000000 THEN 1 ELSE 0 END) AS n_implausibly_large_amount
FROM :validation_schema.vw_bh_disb_dedup
UNION ALL
SELECT
'repayments',
COUNT(*),
SUM(CASE WHEN repayment_amount IS NULL THEN 1 ELSE 0 END),
SUM(CASE WHEN repayment_amount = 0 THEN 1 ELSE 0 END),  -- zero is plausible for a repayment (a $0 correction row); negative is covered by Section B0's sign diagnostic, not repeated here
SUM(CASE WHEN is_nan(repayment_amount) THEN 1 ELSE 0 END),
SUM(CASE WHEN is_infinite(repayment_amount) THEN 1 ELSE 0 END),
SUM(CASE WHEN ABS(repayment_amount) > 1000000000 THEN 1 ELSE 0 END)
FROM :validation_schema.vw_bh_repay_dedup;
-- RESULT:
-- INTERPRETATION: n_null_amount, n_nan_amount, and n_infinite_amount should
-- all be 0 for both sources -- any of these would propagate as NULL/NaN/Inf
-- through every downstream SUM/ratio in loan_final and silently corrupt
-- results for that loan and (via the borrower-level AVG/streak windows)
-- potentially other loans for the same borrower too. n_zero_or_negative_
-- amount for disbursements should be 0 (a loan can't disburse <= 0).
-- n_implausibly_large_amount is a heuristic flag for manual review, not a
-- hard failure -- confirm what "implausible" actually means for this
-- product with the business owner before treating any nonzero count here
-- as a defect.


-- ============================================================================
-- SECTION A -- Repayment amount semantics: gross or principal-only?
-- ============================================================================
-- Blocks cure-timing feature validity if wrong. Uses ONLY
-- vw_bh_loan_state_snapshot's own authoritative totals -- independent of
-- repayment attribution, so this is a clean check even if attribution turns
-- out to be imperfect.
--
-- CAVEAT: this section tests what lifetime_repaid_ugx/lifetime_gross_repaid_ugx
-- mean, NOT directly what the atomic repayment_amount_ugx field in
-- repayments_daily means -- those are different columns in different tables.
-- If lifetime_repaid_ugx is itself a derived allocation (tracker-side split of
-- gross cash flow into a principal-recovered component) rather than a raw sum
-- of repayment_amount_ugx, it would read as principal-only by construction
-- regardless of what the atomic field contains. Section B3 tests the atomic
-- field directly, on a subset where attribution is unambiguous, and is the
-- more decisive check -- treat A as corroborating context, not proof.
WITH closed_charged_loans AS (
SELECT
disbursement_fid,
lifetime_disbursed_ugx,
lifetime_repaid_ugx,
lifetime_gross_repaid_ugx,
interest_and_penalty_ugx,
lifetime_repaid_ugx / NULLIF(lifetime_disbursed_ugx, 0) AS repaid_over_principal,
lifetime_repaid_ugx / NULLIF(lifetime_disbursed_ugx + interest_and_penalty_ugx, 0) AS repaid_over_gross,
lifetime_gross_repaid_ugx / NULLIF(lifetime_disbursed_ugx, 0) AS gross_repaid_over_principal,
lifetime_gross_repaid_ugx / NULLIF(lifetime_disbursed_ugx + interest_and_penalty_ugx, 0) AS gross_repaid_over_gross,
(lifetime_gross_repaid_ugx - lifetime_repaid_ugx) / NULLIF(interest_and_penalty_ugx, 0) AS gross_minus_repaid_over_charge
FROM :validation_schema.vw_bh_loan_state_snapshot
WHERE loan_status = 'CLOSED'
AND interest_and_penalty_ugx > 0
)
SELECT
COUNT(*) AS n_closed_charged_loans,
approx_percentile(repaid_over_principal, 0.5) AS median_repaid_over_principal,
approx_percentile(repaid_over_gross, 0.5) AS median_repaid_over_gross,
approx_percentile(gross_repaid_over_principal, 0.5) AS median_gross_repaid_over_principal,
approx_percentile(gross_repaid_over_gross, 0.5) AS median_gross_repaid_over_gross,
approx_percentile(gross_minus_repaid_over_charge, 0.5) AS median_gross_minus_repaid_over_charge,
SUM(CASE WHEN ABS(repaid_over_principal - 1.0) <= 0.02 THEN 1 ELSE 0 END) AS n_repaid_matches_principal,
SUM(CASE WHEN ABS(gross_repaid_over_gross - 1.0) <= 0.02 THEN 1 ELSE 0 END) AS n_gross_repaid_matches_gross,
SUM(CASE WHEN ABS(gross_minus_repaid_over_charge - 1.0) <= 0.02 THEN 1 ELSE 0 END) AS n_gap_matches_charge
FROM closed_charged_loans;
-- RESULT:
-- INTERPRETATION:
-- median_gross_minus_repaid_over_charge near 1.0 (and n_gap_matches_charge
-- high) confirms lifetime_repaid_ugx = principal-recovered and
-- lifetime_gross_repaid_ugx = principal + charge, exactly as the field names
-- imply. median_repaid_over_principal near 1.0 with median_gross_repaid_over_
-- principal clearly above 1.0 is the expected pattern regardless of what the
-- ATOMIC repayment_amount_ugx field contains, per the caveat above -- do not
-- treat this alone as resolving whether borrower_history.txt's cure-timing
-- logic is correct. Go to Section B3 for that.


-- ============================================================================
-- SECTION B0 -- Repayment sign diagnostic
-- ============================================================================
-- classified in borrower_history.txt applies ABS() unconditionally to both
-- disbursement and repayment amounts. That's only safe for repayments if
-- every atomic row is genuine positive cash flow and any negative rows are
-- purely a sign convention. If repayments_daily instead contains reversals,
-- refunds, chargebacks, or correction entries as negative rows, ABS()
-- silently converts those into additional positive recovery.
--
-- CAVEAT: repayments_daily (per the schema sample this file was built from)
-- has no explicit transaction-type/status/reversal-code column, so this
-- diagnostic can only detect negative SIGN as a proxy. Uses vw_bh_repay_dedup
-- (post-dedup, post-as_of_load_ts-freeze) so this reflects the same
-- repayment population borrower_history.txt actually classifies, not the
-- raw un-deduped table.
SELECT
COUNT(*) AS total_repayments,
SUM(CASE WHEN repayment_amount < 0 THEN 1 ELSE 0 END) AS negative_repayments,
SUM(CASE WHEN repayment_amount < 0 THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) AS negative_repayment_share,
SUM(CASE WHEN repayment_amount < 0 THEN repayment_amount ELSE 0 END) AS negative_repayment_total_ugx,
SUM(repayment_amount) AS raw_sum_ugx,
SUM(ABS(repayment_amount)) AS abs_sum_ugx,
SUM(ABS(repayment_amount)) - SUM(repayment_amount) AS overstatement_from_abs_ugx
FROM :validation_schema.vw_bh_repay_dedup;
-- RESULT:
-- INTERPRETATION: negative_repayment_share should be ~0%. overstatement_
-- from_abs_ugx is exactly how much extra "recovery" borrower_history.txt's
-- ABS()-based classified CTE adds versus the raw signed total -- if this is
-- nonzero and not explained as a benign sign convention (confirm with the
-- data owner), the unconditional ABS() needs to change to source-semantic
-- handling rather than blindly flipping sign.


-- ============================================================================
-- SECTION B -- Repayment attribution quality (repay_attributed heuristic)
-- ============================================================================
-- All three of B1/B2/B3 now read from the GATE 0 views -- same dedup, same
-- as_of_load_ts freeze, same population as borrower_history.txt actually
-- uses. B1/B2/B3 report BOTH the raw signed sum and the ABS()-basis sum
-- (matching what classified actually computes) -- see Section B0.

-- B1. Coverage: how many repayments got dropped for lack of a matching
-- disbursement (count AND value), and how many disbursements look like
-- rapid reborrows (a proxy for "overlapping loans", since there is no exact
-- concurrency signal)?
WITH attributed AS (
SELECT
w.disbursement_fid,
r.repayment_fid,
r.repayment_ts,
r.repayment_amount,
w.disbursement_ts,
w.next_disbursement_ts,
CASE WHEN w.next_disbursement_ts IS NOT NULL
AND date_diff('second', r.repayment_ts, w.next_disbursement_ts) <= 3600
THEN 1 ELSE 0 END AS near_boundary_flag
FROM :validation_schema.vw_bh_repay_dedup r
JOIN :validation_schema.vw_bh_disb_windows w
ON r.phonenumber = w.phonenumber
AND r.repayment_ts >= w.disbursement_ts
AND (w.next_disbursement_ts IS NULL OR r.repayment_ts < w.next_disbursement_ts)
)
SELECT
(SELECT COUNT(*) FROM :validation_schema.vw_bh_repay_dedup) AS total_repayments,
(SELECT COUNT(*) FROM attributed) AS attributed_repayments,
(SELECT COUNT(*) FROM :validation_schema.vw_bh_repay_dedup) - (SELECT COUNT(*) FROM attributed) AS dropped_repayments,
(SELECT SUM(ABS(repayment_amount)) FROM :validation_schema.vw_bh_repay_dedup) AS total_repayment_ugx_abs,
(SELECT SUM(ABS(repayment_amount)) FROM attributed) AS attributed_repayment_ugx_abs,
(SELECT COUNT(*) FROM attributed WHERE near_boundary_flag = 1) AS near_next_disbursement_boundary,
(SELECT COUNT(*) FROM :validation_schema.vw_bh_disb_windows) AS total_disbursements,
(SELECT COUNT(*) FROM :validation_schema.vw_bh_disb_windows
 WHERE next_disbursement_ts IS NOT NULL
 AND date_diff('hour', disbursement_ts, next_disbursement_ts) <= 26) AS rapid_reborrow_disbursements;
-- RESULT:
-- INTERPRETATION: dropped_repayments (count) and the gap between total_
-- repayment_ugx_abs and attributed_repayment_ugx_abs (value) should both be
-- small -- report both per the acceptance thresholds, since a small count-
-- based drop rate can still hide a large monetary one if what's dropped
-- skews toward high-value loans. near_next_disbursement_boundary flags
-- repayments that could plausibly belong to either loan. rapid_reborrow_
-- disbursements / total_disbursements is the closest available proxy for
-- "rate of borrowers with overlapping or near-simultaneous loans" -- a
-- proxy, not a direct measurement, since neither table exposes an explicit
-- "loan closed" event to test true concurrency against.

-- B2 is a BLOCKING gate for production go/no-go, not an informational
-- diagnostic -- treat a failing threshold here as reason to hold the
-- rewrite, the same as B3 below.
--
-- Reported TWICE, clearly labeled, because they answer different
-- questions: B2a reconciles every deduplicated disbursement regardless of
-- ANOMALY_OPEN status (a general source-quality signal); B2b restricts to
-- vw_bh_surviving_loans, the exact population that actually feeds
-- borrower_history.txt's output. A few anomalous loans in B2a could worsen
-- (or mask) the reconciliation number for loans production doesn't even
-- use -- B2b is the one that gates production; B2a is context.

-- B2a. Reconciliation over ALL deduplicated disbursements (source-quality
-- signal, includes ANOMALY_OPEN loans -- NOT the production population).
WITH attributed AS (
SELECT w.disbursement_fid, r.repayment_amount
FROM :validation_schema.vw_bh_repay_dedup r
JOIN :validation_schema.vw_bh_disb_windows w
ON r.phonenumber = w.phonenumber
AND r.repayment_ts >= w.disbursement_ts
AND (w.next_disbursement_ts IS NULL OR r.repayment_ts < w.next_disbursement_ts)
),
per_loan_attributed AS (
-- LEFT JOIN from disb_windows so every known disbursement gets a row,
-- including loans with zero repayments so far.
SELECT
w.disbursement_fid,
w.disbursed_amount,
COALESCE(SUM(a.repayment_amount), 0) AS attributed_repaid_raw,
COALESCE(SUM(ABS(a.repayment_amount)), 0) AS attributed_repaid_abs
FROM :validation_schema.vw_bh_disb_windows w
LEFT JOIN attributed a ON a.disbursement_fid = w.disbursement_fid
GROUP BY w.disbursement_fid, w.disbursed_amount
),
joined AS (
SELECT
COALESCE(pla.disbursement_fid, lsl.disbursement_fid) AS disbursement_fid,
pla.disbursed_amount,
pla.attributed_repaid_raw,
pla.attributed_repaid_abs,
lsl.lifetime_repaid_ugx,
CASE WHEN pla.disbursement_fid IS NULL THEN 1 ELSE 0 END AS state_only,
CASE WHEN lsl.disbursement_fid IS NULL THEN 1 ELSE 0 END AS disbursement_only
FROM per_loan_attributed pla
FULL OUTER JOIN :validation_schema.vw_bh_loan_state_snapshot lsl
ON lsl.disbursement_fid = pla.disbursement_fid
)
SELECT
'ALL_DEDUPED_LOANS (context, not the production population)' AS population,
COUNT(*) AS n_total,
SUM(state_only) AS n_state_only,
SUM(disbursement_only) AS n_disbursement_only,
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0 THEN 1 ELSE 0 END) AS n_matched,
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_raw - lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_exact_match_raw,
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_exact_match_abs,
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0 THEN lifetime_repaid_ugx ELSE 0 END) AS total_matched_lifetime_repaid_ugx,
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1 THEN lifetime_repaid_ugx ELSE 0 END) AS matched_within_tolerance_ugx,
approx_percentile(
CASE WHEN state_only = 0 AND disbursement_only = 0
THEN ABS(attributed_repaid_raw - lifetime_repaid_ugx) END, 0.95) AS p95_abs_diff_raw_ugx,
approx_percentile(
CASE WHEN state_only = 0 AND disbursement_only = 0
THEN ABS(attributed_repaid_abs - lifetime_repaid_ugx) END, 0.95) AS p95_abs_diff_abs_ugx,
approx_percentile(
CASE WHEN state_only = 0 AND disbursement_only = 0
THEN disbursed_amount END, 0.5) AS median_matched_principal_ugx
FROM joined;

-- B2b. Same reconciliation, restricted to vw_bh_surviving_loans -- THIS IS
-- THE RESULT THAT GATES PRODUCTION (matches the population borrower_
-- history.txt actually builds loan_final from).
WITH attributed AS (
SELECT w.disbursement_fid, r.repayment_amount
FROM :validation_schema.vw_bh_repay_dedup r
JOIN :validation_schema.vw_bh_disb_windows w
ON r.phonenumber = w.phonenumber
AND r.repayment_ts >= w.disbursement_ts
AND (w.next_disbursement_ts IS NULL OR r.repayment_ts < w.next_disbursement_ts)
WHERE w.disbursement_fid IN (SELECT disbursement_fid FROM :validation_schema.vw_bh_surviving_loans)
),
per_loan_attributed AS (
SELECT
w.disbursement_fid,
w.disbursed_amount,
COALESCE(SUM(a.repayment_amount), 0) AS attributed_repaid_raw,
COALESCE(SUM(ABS(a.repayment_amount)), 0) AS attributed_repaid_abs
FROM :validation_schema.vw_bh_surviving_loans w
LEFT JOIN attributed a ON a.disbursement_fid = w.disbursement_fid
GROUP BY w.disbursement_fid, w.disbursed_amount
),
joined AS (
SELECT
COALESCE(pla.disbursement_fid, lsl.disbursement_fid) AS disbursement_fid,
pla.disbursed_amount,
pla.attributed_repaid_raw,
pla.attributed_repaid_abs,
lsl.lifetime_repaid_ugx,
CASE WHEN pla.disbursement_fid IS NULL THEN 1 ELSE 0 END AS state_only,
CASE WHEN lsl.disbursement_fid IS NULL THEN 1 ELSE 0 END AS disbursement_only
FROM per_loan_attributed pla
FULL OUTER JOIN (
SELECT lsl.* FROM :validation_schema.vw_bh_loan_state_snapshot lsl
JOIN :validation_schema.vw_bh_surviving_loans sl ON sl.disbursement_fid = lsl.disbursement_fid
) lsl
ON lsl.disbursement_fid = pla.disbursement_fid
)
SELECT
'PRODUCTION_SURVIVING_LOANS (gates production go/no-go)' AS population,
COUNT(*) AS n_total,
SUM(state_only) AS n_state_only,
SUM(disbursement_only) AS n_disbursement_only,
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0 THEN 1 ELSE 0 END) AS n_matched,
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_raw - lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_exact_match_raw,
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_exact_match_abs,
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0 THEN lifetime_repaid_ugx ELSE 0 END) AS total_matched_lifetime_repaid_ugx,
SUM(CASE WHEN state_only = 0 AND disbursement_only = 0
AND ABS(attributed_repaid_abs - lifetime_repaid_ugx) <= 1 THEN lifetime_repaid_ugx ELSE 0 END) AS matched_within_tolerance_ugx,
approx_percentile(
CASE WHEN state_only = 0 AND disbursement_only = 0
THEN ABS(attributed_repaid_raw - lifetime_repaid_ugx) END, 0.95) AS p95_abs_diff_raw_ugx,
approx_percentile(
CASE WHEN state_only = 0 AND disbursement_only = 0
THEN ABS(attributed_repaid_abs - lifetime_repaid_ugx) END, 0.95) AS p95_abs_diff_abs_ugx,
approx_percentile(
CASE WHEN state_only = 0 AND disbursement_only = 0
THEN disbursed_amount END, 0.5) AS median_matched_principal_ugx
FROM joined;
-- RESULT:
-- INTERPRETATION: n_state_only and n_disbursement_only should both be small
-- and explainable -- if either is large, the two source tables disagree on
-- coverage, a data-pipeline issue to raise separately from the
-- reconciliation rate. n_exact_match_abs / n_matched and matched_within_
-- tolerance_ugx / total_matched_lifetime_repaid_ugx should both be high per
-- the acceptance thresholds -- report both, since a high count-based match
-- rate can still conceal a large monetary discrepancy concentrated in a few
-- high-value loans. Compute p95_abs_diff_abs_ugx / median_matched_principal_ugx
-- yourself from this row's two columns to evaluate the "p95 error relative
-- to principal" threshold -- that ratio is deliberately not precomputed here
-- since dividing two aggregates inside the same SELECT can hide which raw
-- numbers produced it. CAVEAT: a mismatch on either basis is still ambiguous
-- between (1) the time-window heuristic misattributing payments, or (2)
-- repayment_amount being gross-of-charges (see Section A's caveat). B3
-- below isolates cause (2) by removing cause (1) entirely -- read B2b and B3
-- together. Comparing n_exact_match_raw against n_exact_match_abs also
-- answers Section B0's question: if raw matches much better than abs,
-- negative rows are real adjustments ABS() is wrongly inflating.
-- Use B2b (not B2a) against the acceptance thresholds -- it's the
-- population that actually determines what ships.

-- B3 is a BLOCKING gate for production go/no-go, same as B2 -- a failing
-- result here means the cure-timing logic's core assumption (atomic
-- repayment amounts are principal-only) is wrong, not just "worth noting."
--
-- B3. Isolate amount semantics from attribution error: single-loan borrowers
-- only, restricted to matched, closed, charged loans with no left-censoring
-- and no boundary ambiguity -- a stricter eligibility set than earlier
-- versions of this check, per the "not as unambiguous as claimed" critique:
--   - exactly one DEDUPED disbursement (via vw_bh_disb_dedup, not raw rows)
--   - a matching loan_state row exists (not disbursement-only)
--   - loan_status = 'CLOSED' and interest_and_penalty_ugx > 0 (comparable
--     to Section A's population)
--   - has_pre_window_history is not true (excludes borrowers whose "only"
--     loan in this feed may not be their only loan ever)
--   - zero-repayment loans are reported separately, not silently dropped by
--     the inner join
WITH single_loan_borrowers AS (
SELECT phonenumber
FROM :validation_schema.vw_bh_disb_dedup
GROUP BY phonenumber
HAVING COUNT(*) = 1
),
eligible AS (
SELECT
d.disbursement_fid,
d.phonenumber,
d.disbursement_ts,
ls.lifetime_disbursed_ugx,
ls.lifetime_repaid_ugx,
ls.lifetime_gross_repaid_ugx,
ls.interest_and_penalty_ugx
FROM :validation_schema.vw_bh_disb_dedup d
JOIN single_loan_borrowers slb ON slb.phonenumber = d.phonenumber
JOIN :validation_schema.vw_bh_loan_state_snapshot ls ON ls.disbursement_fid = d.disbursement_fid
WHERE ls.loan_status = 'CLOSED'
AND ls.interest_and_penalty_ugx > 0
AND COALESCE(ls.has_pre_window_history, false) = false
),
atomic_repaid AS (
SELECT
e.disbursement_fid,
COALESCE(SUM(r.repayment_amount), 0) AS atomic_repaid_raw,
COALESCE(SUM(ABS(r.repayment_amount)), 0) AS atomic_repaid_abs,
COUNT(r.repayment_fid) AS n_repayments
FROM eligible e
LEFT JOIN :validation_schema.vw_bh_repay_dedup r
ON r.phonenumber = e.phonenumber
AND r.repayment_ts >= e.disbursement_ts
GROUP BY e.disbursement_fid
)
SELECT
COUNT(*) AS n_eligible_single_loan_closed_charged,
SUM(CASE WHEN n_repayments = 0 THEN 1 ELSE 0 END) AS n_zero_repayment_excluded_from_ratios,
approx_percentile(
CASE WHEN n_repayments > 0 THEN ar.atomic_repaid_abs / NULLIF(e.lifetime_disbursed_ugx, 0) END, 0.5
) AS median_atomic_over_principal,
approx_percentile(
CASE WHEN n_repayments > 0 THEN ar.atomic_repaid_abs / NULLIF(e.lifetime_disbursed_ugx + e.interest_and_penalty_ugx, 0) END, 0.5
) AS median_atomic_over_gross,
SUM(CASE WHEN n_repayments > 0 AND ABS(ar.atomic_repaid_abs - e.lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_abs_matches_lifetime_repaid,
SUM(CASE WHEN n_repayments > 0 AND ABS(ar.atomic_repaid_abs - e.lifetime_gross_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_abs_matches_lifetime_gross_repaid,
SUM(CASE WHEN n_repayments > 0 AND ABS(ar.atomic_repaid_raw - e.lifetime_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_raw_matches_lifetime_repaid,
SUM(CASE WHEN n_repayments > 0 AND ABS(ar.atomic_repaid_raw - e.lifetime_gross_repaid_ugx) <= 1 THEN 1 ELSE 0 END) AS n_raw_matches_lifetime_gross_repaid
FROM eligible e
JOIN atomic_repaid ar ON ar.disbursement_fid = e.disbursement_fid;
-- RESULT:
-- INTERPRETATION: this is the decisive check, not Section A -- but read
-- "decisive" as "strongly indicative," not "proof": these borrowers have
-- only one deduped disbursement WITHIN THIS FEED, not necessarily one loan
-- in their complete history (has_pre_window_history=false reduces but does
-- not eliminate that risk), and a reused/reassigned phone number could still
-- misattribute payments to the wrong person. n_zero_repayment_excluded_from_
-- ratios should be reported alongside the ratio columns, not treated as
-- implicitly zero.
-- Amount semantics: if n_abs_matches_lifetime_repaid (or n_raw_matches_
-- lifetime_repaid) is high, repayment_amount_ugx is principal-only. If the
-- *_gross_repaid variants are high instead, repayments are gross-of-charges.
-- Sign handling: raw vs abs match-rate comparison answers Section B0's
-- question on this low-ambiguity subset.
-- CORRECTION to an earlier version of this note: if repayments turn out to
-- be gross-of-charges, "floor the cure comparison at disbursed_amount" is
-- NOT by itself a fix -- it only changes the threshold, not the allocation
-- order. If a payment applies to interest/penalty before principal,
-- cumulative gross cash crossing disbursed_amount does not prove principal
-- itself has been repaid. A correct fix needs the tracker's actual
-- allocation waterfall, not just a different threshold.

-- B4. Same-timestamp disbursement ordering: borrower_history.txt orders a
-- borrower's loans by `disbursement_ts, requestid` to determine prior-loan
-- windows and which loan is "latest" (reversed for latest-loan selection).
-- That's deterministic as long as disbursement_fid/requestid sorts
-- consistently, but it assumes requestid ordering is a meaningful
-- tie-breaker when two disbursements share an identical disbursement_ts --
-- not necessarily true (e.g. if both were backfilled in the same batch
-- with fid assigned by an unrelated process). Count how often this
-- actually happens before trusting that assumption.
SELECT
COUNT(*) AS borrowers_with_same_timestamp_tie,
SUM(n_tied) AS total_tied_disbursements
FROM (
SELECT phonenumber, disbursement_ts, COUNT(*) AS n_tied
FROM :validation_schema.vw_bh_disb_dedup
GROUP BY phonenumber, disbursement_ts
HAVING COUNT(*) > 1
);
-- RESULT:
-- INTERPRETATION: borrowers_with_same_timestamp_tie should be small/zero.
-- If it's material, num_prior_loans, prior_on_time_24h_rate, and which loan
-- gets called "latest" for those borrowers depend on requestid ordering
-- alone -- confirm with the data owner whether disbursement_fid/requestid
-- is contractually guaranteed to reflect true creation order, or find a
-- genuine sequence/timestamp-with-higher-precision field to order by
-- instead.


-- ============================================================================
-- SECTION C -- ANOMALY_OPEN exclusion impact
-- ============================================================================
SELECT
(SELECT COUNT(*) FROM :validation_schema.vw_bh_disb_dedup) AS total_loans,
(SELECT COUNT(*) FROM :validation_schema.vw_bh_loan_state_anomalies) AS anomaly_open_loans,
(SELECT COUNT(DISTINCT phonenumber) FROM :validation_schema.vw_bh_disb_dedup) AS total_borrowers,
(SELECT COUNT(DISTINCT d.phonenumber)
 FROM :validation_schema.vw_bh_disb_dedup d
 JOIN :validation_schema.vw_bh_loan_state_anomalies a ON a.disbursement_fid = d.disbursement_fid
) AS borrowers_with_an_anomaly_loan;
-- RESULT:
-- INTERPRETATION: report anomaly_open_loans / total_loans and borrowers_
-- with_an_anomaly_loan / total_borrowers as the population-shift cost of the
-- exclusion decision -- both counts now come from the same deduped
-- disbursement population borrower_history.txt actually uses, not raw
-- loan_state_daily.


-- ============================================================================
-- SECTION D -- has_pre_window_history impact
-- ============================================================================
-- Uses vw_bh_surviving_loans (deduped, normalized, ANOMALY_OPEN-excluded) --
-- the same population borrower_history.txt's loan_level actually produces.
WITH per_loan_flag AS (
SELECT sl.phonenumber, ls.has_pre_window_history
FROM :validation_schema.vw_bh_surviving_loans sl
LEFT JOIN :validation_schema.vw_bh_loan_state_snapshot ls ON ls.disbursement_fid = sl.disbursement_fid
)
SELECT
COUNT(*) AS total_borrowers,
SUM(CASE WHEN flagged THEN 1 ELSE 0 END) AS borrowers_with_pre_window_history,
SUM(CASE WHEN flagged THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) AS pct_left_censored
FROM (
SELECT phonenumber, bool_or(has_pre_window_history) AS flagged
FROM per_loan_flag
GROUP BY phonenumber
);
-- RESULT:
-- INTERPRETATION: pct_left_censored is the share of borrowers whose total_
-- loans/first_loan_ts/lifetime rates from borrower_history.txt should be
-- treated as incomplete, not zero-history.


-- ============================================================================
-- SECTION E -- Candidate old->new loan identity bridge
-- ============================================================================
-- Tightened from an existence-only check: deduped new-side population,
-- frozen to the same as_of_load_ts, plus cardinality and agreement checks
-- -- a high raw match rate alone doesn't prove disbursement_external_id is
-- safe to use as a loan key.
WITH old_dispatch_raw AS (
SELECT
transactionid,
regexp_replace(trim(cast(phonenumber AS varchar)), '[^0-9]', '') AS phonenumber,
cast(requestid AS varchar) AS requestid,
try_cast(original_timestamp_enrich AS timestamp) AS event_ts,
cast(amount AS double) AS raw_amount
FROM devdata.xtrafloat_daily_trans
WHERE lower(trim(tranname)) = 'xtrafloat dispatch'
AND requestid IS NOT NULL
AND try_cast(original_timestamp_enrich AS timestamp) IS NOT NULL
),
-- dedupe to one row per transactionid (same pattern as borrower_history_
-- original.txt's own dedup), then collapse to one row per requestid (the
-- earliest dispatch event) -- without this, un-deduped old-table ingestion
-- artifacts would masquerade as genuine cardinality collisions below.
old_dispatch_txn_dedup AS (
SELECT * FROM (
SELECT o.*,
ROW_NUMBER() OVER (PARTITION BY transactionid ORDER BY event_ts DESC) rn
FROM old_dispatch_raw o
)
WHERE rn = 1
),
old_dispatch AS (
SELECT * FROM (
SELECT t.*,
ROW_NUMBER() OVER (PARTITION BY requestid ORDER BY event_ts ASC) rn2
FROM old_dispatch_txn_dedup t
)
WHERE rn2 = 1
),
bridge AS (
SELECT
d.disbursement_fid,
d.phonenumber,
d.disbursement_ts,
d.disbursed_amount,
t.requestid AS old_requestid,
t.phonenumber AS old_phonenumber,
t.event_ts AS old_event_ts,
t.raw_amount AS old_raw_amount
FROM :validation_schema.vw_bh_disb_dedup d
LEFT JOIN old_dispatch t
ON cast(t.requestid AS varchar) = cast(d.disbursement_external_id AS varchar)
WHERE d.disbursement_external_id IS NOT NULL
)
SELECT
COUNT(*) AS new_disbursements_with_external_id,
SUM(CASE WHEN old_requestid IS NOT NULL THEN 1 ELSE 0 END) AS matched_to_old_requestid,
-- cardinality: does one new disbursement ever map to >1 old requestid, or
-- vice versa?
(SELECT COUNT(*) FROM (
SELECT disbursement_fid FROM bridge WHERE old_requestid IS NOT NULL
GROUP BY disbursement_fid HAVING COUNT(DISTINCT old_requestid) > 1
)) AS new_ids_matching_multiple_old,
(SELECT COUNT(*) FROM (
SELECT old_requestid FROM bridge WHERE old_requestid IS NOT NULL
GROUP BY old_requestid HAVING COUNT(DISTINCT disbursement_fid) > 1
)) AS old_ids_matching_multiple_new,
-- agreement on the matched subset: do phone/amount/timestamp actually line
-- up, or does the id merely coincide?
SUM(CASE WHEN old_requestid IS NOT NULL AND phonenumber = old_phonenumber THEN 1 ELSE 0 END) AS matched_phone_agrees,
SUM(CASE WHEN old_requestid IS NOT NULL AND ABS(disbursed_amount - old_raw_amount) <= 1 THEN 1 ELSE 0 END) AS matched_amount_agrees,
approx_percentile(
CASE WHEN old_requestid IS NOT NULL THEN ABS(date_diff('second', disbursement_ts, old_event_ts)) END, 0.5
) AS median_abs_timestamp_diff_seconds
FROM bridge;
-- RESULT:
-- INTERPRETATION: matched_to_old_requestid / new_disbursements_with_
-- external_id being high is necessary but not sufficient -- new_ids_
-- matching_multiple_old and old_ids_matching_multiple_new must both be 0
-- for this to be a safe 1:1 key; if either is nonzero, disbursement_
-- external_id is not a reliable bridge on its own. matched_phone_agrees and
-- matched_amount_agrees should both be ~100% of the matched subset --
-- disagreement means the id match is coincidental, not a real link. If all
-- of these pass, use disbursement_external_id (not msisdn+timestamp
-- proximity) for old-vs-new loan-level comparison.


-- ============================================================================
-- SECTION F -- Dedup tie-breaker collisions
-- ============================================================================
-- Deliberately does NOT use the GATE 0 views -- the views already resolve
-- ties (that's what ROW_NUMBER does), so querying them can never reveal a
-- tie. This section has to look at the pre-dedup population directly. Fixed
-- from an earlier version that checked the CURRENT global max inserted_ts
-- regardless of as_of_load_ts -- that could miss a tie that existed at the
-- frozen cutoff (superseded by a later unique load) or report a tie that
-- arose only after the cutoff (irrelevant to the run being validated).
-- Every check below is bounded to the same date/inserted_ts cutoffs
-- borrower_history.txt itself uses.
SELECT 'disbursements_daily' AS source_table, COUNT(*) AS keys_with_tie
FROM (
SELECT disbursement_fid
FROM analytics.momo_loan_book_tracker_disbursements_daily d
WHERE d.ova = 'XTRAFLOAT-AGENT'
AND d.date_key <= :snapshot_dt
AND date(try_cast(disbursement_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
AND inserted_ts <= :as_of_load_ts
AND inserted_ts = (
SELECT MAX(inserted_ts) FROM analytics.momo_loan_book_tracker_disbursements_daily d2
WHERE d2.disbursement_fid = d.disbursement_fid
AND d2.ova = 'XTRAFLOAT-AGENT'
AND d2.date_key <= :snapshot_dt
AND date(try_cast(d2.disbursement_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
AND d2.inserted_ts <= :as_of_load_ts
)
GROUP BY disbursement_fid
HAVING COUNT(*) > 1
)
UNION ALL
SELECT 'repayments_daily', COUNT(*)
FROM (
SELECT repayment_fid
FROM analytics.momo_loan_book_tracker_repayments_daily r
WHERE r.ova = 'XTRAFLOAT-AGENT'
AND r.date_key <= :snapshot_dt
AND date(try_cast(repayment_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
AND inserted_ts <= :as_of_load_ts
AND inserted_ts = (
SELECT MAX(inserted_ts) FROM analytics.momo_loan_book_tracker_repayments_daily r2
WHERE r2.repayment_fid = r.repayment_fid
AND r2.ova = 'XTRAFLOAT-AGENT'
AND r2.date_key <= :snapshot_dt
AND date(try_cast(r2.repayment_ts AS timestamp)) <= date_parse(cast(:snapshot_dt AS varchar), '%Y%m%d')
AND r2.inserted_ts <= :as_of_load_ts
)
GROUP BY repayment_fid
HAVING COUNT(*) > 1
)
UNION ALL
SELECT 'loan_state_daily (per disbursement_fid, date_key)', COUNT(*)
FROM (
SELECT disbursement_fid, date_key
FROM analytics.momo_loan_book_tracker_loan_state_daily l
WHERE l.ova = 'XTRAFLOAT-AGENT'
AND date_key <= :snapshot_dt
AND inserted_ts <= :as_of_load_ts
AND inserted_ts = (
SELECT MAX(inserted_ts) FROM analytics.momo_loan_book_tracker_loan_state_daily l2
WHERE l2.disbursement_fid = l.disbursement_fid AND l2.date_key = l.date_key
AND l2.ova = 'XTRAFLOAT-AGENT'
AND l2.date_key <= :snapshot_dt
AND l2.inserted_ts <= :as_of_load_ts
)
GROUP BY disbursement_fid, date_key
HAVING COUNT(*) > 1
);
-- RESULT:
-- INTERPRETATION: keys_with_tie should be 0 for all three rows. Any nonzero
-- count means dedup is choosing arbitrarily for that many business keys, AS
-- OF THIS SPECIFIC (snapshot_dt, as_of_load_ts) RUN -- ask the table owner
-- whether a real tie-breaker field exists before shipping.


-- ============================================================================
-- SECTION G -- Same-day disbursement loan_state_daily coverage
-- ============================================================================
-- Quantifies a confirmed limitation (not a guess -- see
-- data/spot_check_loan_trace.sql's msisdn 25672546 trace and the comment at
-- borrower_history.txt's ls_* join): when a borrower takes a second
-- disbursement on the SAME calendar day as an already-open loan, that
-- second disbursement's own disbursement_fid appears to get no independent
-- loan_state_daily row at all -- its principal seems folded into the first
-- same-day loan's ongoing state instead. borrower_history.txt's LEFT JOIN
-- handles this safely (NULL ls_* columns, no error), but this section
-- measures how often it actually happens across the whole book, not just
-- the one borrower it was first observed on.
WITH same_day_ranked AS (
SELECT
d.disbursement_fid,
d.phonenumber,
d.disbursement_ts,
ROW_NUMBER() OVER (
PARTITION BY d.phonenumber, date(d.disbursement_ts)
ORDER BY d.disbursement_ts
) AS same_day_rn
FROM :validation_schema.vw_bh_disb_dedup d
)
SELECT
COUNT(*) AS total_disbursements,
SUM(CASE WHEN sdr.same_day_rn = 1 THEN 1 ELSE 0 END) AS first_of_day_disbursements,
SUM(CASE WHEN sdr.same_day_rn > 1 THEN 1 ELSE 0 END) AS same_day_repeat_disbursements,
SUM(CASE WHEN sdr.same_day_rn > 1 AND ls.disbursement_fid IS NULL THEN 1 ELSE 0 END) AS same_day_repeats_missing_loan_state,
SUM(CASE WHEN sdr.same_day_rn > 1 AND ls.disbursement_fid IS NOT NULL THEN 1 ELSE 0 END) AS same_day_repeats_with_loan_state,
ROUND(100.0 * SUM(CASE WHEN sdr.same_day_rn > 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS pct_disbursements_same_day_repeat,
ROUND(100.0 * SUM(CASE WHEN sdr.same_day_rn > 1 AND ls.disbursement_fid IS NULL THEN 1 ELSE 0 END)
       / NULLIF(SUM(CASE WHEN sdr.same_day_rn > 1 THEN 1 ELSE 0 END), 0), 2) AS pct_same_day_repeats_missing_loan_state
FROM same_day_ranked sdr
LEFT JOIN :validation_schema.vw_bh_loan_state_snapshot ls
ON ls.disbursement_fid = sdr.disbursement_fid;
-- RESULT:
-- INTERPRETATION: pct_disbursements_same_day_repeat is how common
-- multi-loan-per-day behavior is across the whole book -- context, not a
-- pass/fail. pct_same_day_repeats_missing_loan_state is the number that
-- matters: it's the share of those repeats where the confirmed gap
-- actually bites, i.e. what fraction of borrower_history.txt's ls_*
-- enrichment columns are NULL specifically because of this pattern (as
-- opposed to a genuinely still-open/unseen loan). If that percentage isn't
-- close to 100%, the merge behavior is NOT universal for same-day
-- repeats -- worth a closer look at what distinguishes the ones that DO
-- get their own loan_state_daily row from the ones that don't, rather than
-- assuming every same-day repeat is affected.


-- ============================================================================
-- GATE 6 -- Record approval evidence
-- ============================================================================
-- Fill this in every time this file is actually run. An unfilled template
-- is not evidence, per the earlier review's point that this file states a
-- validation PLAN, not validation EVIDENCE, until results are recorded.
--
--   Execution date:            ____________________
--   borrower_history.txt git SHA: ________________ (from scripts/build_vw_bh_output.py's
--                                  output comment -- proves which revision was actually validated)
--   snapshot_dt used:          ____________________
--   as_of_load_ts used:        ____________________
--   snapshot_ts used:          ____________________
--   Query engine/version:      ____________________ (e.g. Athena engine v3)
--   GATE 1 grain violations:   ____________________
--   GATE 1 range violations:   ____________________
--   GATE 1 null-rate counts:   ____________________ (n_null_* columns, and whether they're explained by recent unseasoned loans)
--   GATE 1 missing-snapshot vs matched-but-null: ___ (n_latest_loan_missing_state_snapshot / n_latest_loan_matched_but_status_null)
--   GATE 1 column contract (1c names): ___________ (missing / unexpected columns, if any)
--   GATE 1 column contract (1d types): ___________ (any type-family mismatches -- should be ZERO rows, latest_requestid is excluded, not exempted)
--   GATE 1d-info latest_requestid actual_family: ___________ (record here once known; not a pass/fail check)
--   Source data quality gate:  ____________________ (null/NaN/Inf/implausible amount counts, both sources)
--   Section A conclusion:      ____________________ (principal-only / gross / unclear)
--   B0 negative repayment share: __________________
--   B1 coverage (count/value): ____________________
--   B2a reconciliation (all loans): _______________
--   B2b reconciliation (production-surviving loans -- the gating result): ___
--   B3 conclusion:             ____________________
--   B4 same-timestamp ties:    ____________________
--   Section C impact:          ____________________
--   Section D pct_left_censored: __________________
--   Section E bridge verdict:  ____________________ (safe key / not safe / partial)
--   Section F tie count:       ____________________
--   Section G same-day-repeat %/missing-loan-state %: ___________ (context, not pass/fail)
--   Threshold decision:        ____________________ (pass / fail / conditional, and why)
--   Investigation links:       ____________________
--   Approved by:               ____________________


-- ============================================================================
-- TEARDOWN -- drop the GATE 0 views once validation is complete, if
-- :validation_schema is a shared scratch database other work might collide
-- with names in.
-- ============================================================================
-- DROP VIEW IF EXISTS :validation_schema.vw_bh_output;
-- DROP TABLE IF EXISTS :validation_schema.tbl_bh_loan_level;
-- DROP TABLE IF EXISTS :validation_schema.tbl_bh_loan_final;
-- DROP TABLE IF EXISTS :validation_schema.tbl_bh_classified;
-- DROP VIEW IF EXISTS :validation_schema.vw_bh_surviving_loans;
-- DROP VIEW IF EXISTS :validation_schema.vw_bh_loan_state_anomalies;
-- DROP VIEW IF EXISTS :validation_schema.vw_bh_loan_state_snapshot;
-- DROP VIEW IF EXISTS :validation_schema.vw_bh_repay_dedup;
-- DROP VIEW IF EXISTS :validation_schema.vw_bh_disb_windows;
-- DROP VIEW IF EXISTS :validation_schema.vw_bh_disb_dedup;
