@echo off
echo === Reconciling commission (raw) vs cash_out_comm_6m+cash_in_comm_6m+voucher_comm_6m+payment_comm_6m ===
echo Full population check against data\mfs_daily_agent_mart_20260731.csv.
echo If output\engine_test_output.csv exists (from a run with --keep-intermediate),
echo the mismatch is also broken down by agent_category.

python scripts\check_commission_raw_vs_recomputed.py --transaction-file data\mfs_daily_agent_mart_20260731.csv --output-file output\engine_test_output.csv --out commission_raw_vs_recomputed_full.csv

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: check_commission_raw_vs_recomputed.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Full per-agent comparison written to commission_raw_vs_recomputed_full.csv
echo For deeper follow-up on any mismatched bucket, see:
echo   scripts\sample_commission_mismatch_rows.py     (pull sample rows for manual inspection)
echo   scripts\profile_commission_mismatch_bucket.py  (what distinguishes the mismatched agents)
echo   scripts\count_zero_commission_agents.py        (how many agents have commission_raw == 0)
pause
