@echo off

echo === Scoring the ablation model (pd_model/artifacts_ablation) ===
echo Reuses the same retail-filtered input files as run_retail_filtered.bat --
echo does NOT re-run the audit/classify/filter steps. Run run_retail_filtered.bat
echo first if those filtered files don't exist yet.
echo.
echo Writes to a SEPARATE output file (output/engine_test_output_ablation.csv)
echo so your existing output/engine_test_output.csv is untouched.

python run_credit_risk_pipeline.py ^
    --transaction-file retail_agents_filtered.csv ^
    --loan-file data\loan_summary_retail_filtered.csv ^
    --borrower-file borrower_history_retail_filtered.csv ^
    --loan-history-file data\loan_history_snapshot_20260817_retail_filtered.csv ^
    --snapshot-date 20260817 ^
    --artifacts-dir pd_model/artifacts_ablation/ ^
    --scorecard-path scorecards/capacity_scorecard_v1.json ^
    --output output/engine_test_output_ablation.csv > engine_log_ablation.txt 2>&1

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: run_credit_risk_pipeline.py failed with exit code %ERRORLEVEL%
    echo.
    echo Last 20 lines of engine_log_ablation.txt:
    echo ----------------------------------------
    powershell -Command "Get-Content engine_log_ablation.txt -Tail 20"
    echo ----------------------------------------
    echo Full log: engine_log_ablation.txt
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Ablation scoring complete. Output written to output/engine_test_output_ablation.csv
echo Full log: engine_log_ablation.txt
echo.
echo Next: run check_blacklist-style comparison and the cal_pd diagnostic against
echo this file, e.g.:
echo   python scripts\check_whitelist_blacklist_eval.py --whitelist-file data\whitelist_aug_20260804.csv --blacklist-file data\blacklist_aug_20260804.csv --output-file output\engine_test_output_ablation.csv --borrower-file borrower_history_retail_filtered.csv --score-col cal_pd --out-prefix wl_bl_eval_ablation
echo   python scripts\check_scoring_output_diagnostics.py --scoring-output output\engine_test_output_ablation.csv --excluded-file retail_agents_excluded.csv
pause
