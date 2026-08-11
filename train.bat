@echo off
python -m pd_model.run_pipeline ^
    --train-file data/agent_snapshot_train.csv ^
    --val-file data/agent_snapshot_val.csv ^
    --repayment-file data/snapshots_202601112144.csv ^
    --train-snapshot-date 20250831 ^
    --val-snapshot-date 20251130 ^
    --train-cutoff 2025-08-31 ^
    --output-dir pd_model/artifacts/ ^
    --champion xgb

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: pd_model training failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Training complete. Artifacts written to pd_model/artifacts/
pause
