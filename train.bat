@echo off
python -m pd_model.run_pipeline ^
  --train-file            data/mfs_daily_agent_mart_20260531.csv ^
  --val-file              data/mfs_daily_agent_mart_20260731.csv ^
  --loan-training-file    data/state_data_202608131214.csv ^
  --train-snapshot-date   20260531 ^
  --val-snapshot-date     20260731 ^
  --output-dir            pd_model/artifacts ^
  --champion              xgb > training_log.txt 2>&1

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: pd_model training failed with exit code %ERRORLEVEL%
    echo.
    echo Last 20 lines of training_log.txt:
    echo ----------------------------------------
    powershell -Command "Get-Content training_log.txt -Tail 20"
    echo ----------------------------------------
    echo Full log: training_log.txt
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Training complete. Artifacts written to pd_model/artifacts/
echo Full log: training_log.txt
pause
