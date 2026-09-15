@echo off
echo === Calibrating capacity scorecard (provisional_v0) ===
echo Writes scorecards\capacity_scorecard_v0.json -- does NOT wire it into
echo production scoring. Point run_extrafloat_segmentation's
echo scoring.scorecard_path at it yourself once reviewed.

python calibrate_scorecard.py --agents data\mfs_daily_agent_mart_20260731.csv --out scorecards\capacity_scorecard_v0.json --target-proportions "{\"Below Threshold\": 0.25, \"New Bronze\": 0.20, \"Bronze\": 0.15, \"Silver\": 0.12, \"Gold\": 0.10, \"Platinum\": 0.08, \"Titanium\": 0.06, \"Diamond\": 0.04}" --cutoff-version provisional_v0 --population-description "mfs_daily_agent_mart_20260731.csv, ~467,830 agents, 1m/3m/6m KPI windows"

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: calibrate_scorecard.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Scorecard written to scorecards\capacity_scorecard_v0.json (is_provisional=true).
echo Review the tier proportions printed above, then re-run with --final and
echo a reviewed cutoff-version once signed off.
pause
