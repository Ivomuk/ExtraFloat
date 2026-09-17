@echo off
echo === Checking ABLATION model output against the August whitelist/blacklist ===
echo (model trained without the 10 prior-loan magnitude/repayment-ratio/duration
echo  columns, re-run to measure the real impact of dropping the two confirmed
echo  look-ahead-leaked features (most_recent_prior_days_past_due_within_30d,
echo  prior_avg_repayment_ratio) -- compare this run's Part C "Defaulter..." AUC
echo  and the business-summary numbers against check_blacklist.bat's with-features
echo  results from the current, tenure_days-fixed model)

python scripts\check_whitelist_blacklist_eval.py --whitelist-file data\whitelist_aug_20260804.csv --blacklist-file data\blacklist_aug_20260804.csv --output-file output\engine_test_output_ablation.csv --borrower-file borrower_history_retail_filtered.csv --loan-summary-file data\loan_summary.csv --score-col cal_pd --out-prefix wl_bl_eval_ablation

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: check_whitelist_blacklist_eval.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Ablation whitelist/blacklist check complete. See wl_bl_eval_ablation_matched_agents.csv
echo and the JSON baseline summary above.
echo.
echo Next: python scripts\check_defaulter_visibility_at_snapshot.py --matched-file wl_bl_eval_ablation_matched_agents.csv --ever-anomaly-open-file data\ever_anomaly_open.csv
pause
