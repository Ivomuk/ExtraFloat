@echo off
python calibrate_scorecard.py ^
    --agents data/agent_profile_snapshot.csv ^
    --out scorecards/capacity_scorecard_v1.json ^
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
