@echo off
python calibrate_scorecard.py ^
    --agents data/mfs_daily_agent_mart_20260731.csv ^
    --out scorecards/capacity_scorecard_v1.json ^
    --final ^
    --force

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: calibrate_scorecard.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Calibration complete. Scorecard written to scorecards/capacity_scorecard_v1.json
pause
