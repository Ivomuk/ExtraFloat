-- ============================================================================
-- 24h_decimal_division_test.sql -- ad-hoc diagnostic, not part of the
-- committed validation suite. 24h_filter_vs_case_test.sql ruled out FILTER
-- itself as the problem -- an independently-written, logically-equivalent
-- CASE expression agreed exactly with FILTER's (wrong) answer. That means
-- the hours_since_disbursement VALUE being compared against <= 24 must
-- itself be wrong at the point of comparison, not the aggregate construct.
--
-- Leading hypothesis: date_diff('second', disbursement_ts, event_ts) /
-- 3600.0 -- a BIGINT divided by the literal 3600.0. In Trino/Presto,
-- unsuffixed decimal literals like 3600.0 are typed as DECIMAL(5,1), not
-- DOUBLE. BIGINT / DECIMAL division follows SQL decimal arithmetic scale-
-- inference rules, which can allocate less precision to the result than
-- true floating-point division would carry, silently truncating/rounding
-- the fractional part. This directly compares the CURRENT formula against
-- an explicit DOUBLE-forced version on the same raw elapsed_seconds values
-- already confirmed > 86400 for these events.
-- ============================================================================

WITH raw AS (
SELECT * FROM (VALUES
(86442), (86493), (86502), (86548), (86553)
) AS t(elapsed_seconds)
)
SELECT
elapsed_seconds,
elapsed_seconds / 3600.0 AS hours_via_decimal_literal,
CAST(elapsed_seconds AS DOUBLE) / 3600.0 AS hours_via_explicit_double,
elapsed_seconds / CAST(3600 AS DOUBLE) AS hours_via_double_cast_divisor,
(elapsed_seconds / 3600.0) <= 24 AS current_formula_says_within_24h,
(CAST(elapsed_seconds AS DOUBLE) / 3600.0) <= 24 AS double_cast_says_within_24h
FROM raw;
-- RESULT:
-- INTERPRETATION: elapsed_seconds is already confirmed ground truth (all
-- > 86,400 = 24h exactly), so both hours_via_explicit_double and
-- current_formula_says_within_24h/double_cast_says_within_24h SHOULD show
-- values slightly over 24 and FALSE respectively, for every row. If
-- hours_via_decimal_literal comes back rounded/truncated to exactly 24.0
-- (or current_formula_says_within_24h comes back TRUE) while hours_via_
-- explicit_double correctly shows ~24.01-24.04 (and double_cast_says_
-- within_24h comes back FALSE), that confirms the decimal-literal-division
-- truncation hypothesis precisely -- the fix is to force DOUBLE arithmetic
-- (CAST(... AS DOUBLE) / 3600.0, or divide by 3600e0) everywhere
-- borrower_history.txt computes hours_since_disbursement or hours_to_
-- principal_cure.
