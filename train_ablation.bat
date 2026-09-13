@echo off

echo === Ablation training run: same retail-filtered inputs as train_retail_filtered.bat,
echo but excluding the 10 recently-added prior-loan magnitude/repayment-ratio/duration
echo columns, to check whether they're actually improving val/bootstrap AUC.
echo.
echo NOTE: assumes train_retail_filtered.bat has already been run at least once, so
echo the retail-filtered input files below already exist. This script does NOT
echo re-run the audit/classify/filter steps -- if those filtered files are missing
echo or stale, run train_retail_filtered.bat first.
echo.
echo Writing to a SEPARATE --output-dir (pd_model/artifacts_ablation) so your
echo existing pd_model/artifacts/ from the with-features run is untouched.

python -m pd_model.run_pipeline ^
  --train-file            data/mfs_daily_agent_mart_20260531_retail_filtered.csv ^
  --val-file              data/mfs_daily_agent_mart_20260731_retail_filtered.csv ^
  --loan-training-file    data/state_data_20260910_retail_filtered.csv ^
  --train-snapshot-date   20260531 ^
  --val-snapshot-date     20260731 ^
  --output-dir            pd_model/artifacts_ablation ^
  --champion              xgb ^
  --exclude-features      prior_max_loan_seq,prior_max_principal_outstanding_ugx,prior_max_total_outstanding_ugx,prior_avg_repayment_ratio,prior_principal_unsettled_count,prior_total_late_fee_owed_ugx,most_recent_prior_days_past_due_within_30d,prior_loan_has_no_history_flag,prior_loan_confirmed_no_lateness_flag,prior_loan_censored_flag ^
  > training_log_ablation.txt 2>&1

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: pd_model training failed with exit code %ERRORLEVEL%
    echo.
    echo Last 20 lines of training_log_ablation.txt:
    echo ----------------------------------------
    powershell -Command "Get-Content training_log_ablation.txt -Tail 20"
    echo ----------------------------------------
    echo Full log: training_log_ablation.txt
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Ablation training complete. Artifacts written to pd_model/artifacts_ablation/
echo Full log: training_log_ablation.txt
echo.
echo Next: compare this log's XGBoost/LightGBM val AUC and bootstrap comparison
echo against training_log.txt (the with-features run), then run
echo run_ablation_scoring.bat to score this model for the whitelist/blacklist check.
pause
