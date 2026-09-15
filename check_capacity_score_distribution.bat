@echo off
echo === Capacity-score distribution + tier-proportion candidate preview (writes nothing) ===
echo Edit the three --target-proportions JSON blocks below to try your own numbers.

python scripts\check_capacity_score_distribution.py --agents data\mfs_daily_agent_mart_20260731.csv --target-proportions "{\"_name\": \"Even\", \"Below Threshold\": 0.125, \"New Bronze\": 0.125, \"Bronze\": 0.125, \"Silver\": 0.125, \"Gold\": 0.125, \"Platinum\": 0.125, \"Titanium\": 0.125, \"Diamond\": 0.125}" --target-proportions "{\"_name\": \"Mild pyramid\", \"Below Threshold\": 0.15, \"New Bronze\": 0.20, \"Bronze\": 0.20, \"Silver\": 0.15, \"Gold\": 0.12, \"Platinum\": 0.10, \"Titanium\": 0.05, \"Diamond\": 0.03}" --target-proportions "{\"_name\": \"Steep pyramid\", \"Below Threshold\": 0.30, \"New Bronze\": 0.25, \"Bronze\": 0.15, \"Silver\": 0.10, \"Gold\": 0.08, \"Platinum\": 0.06, \"Titanium\": 0.04, \"Diamond\": 0.02}"

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: check_capacity_score_distribution.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Preview complete -- no scorecard file was written. Once you've picked a
echo proportion set, run calibrate_scorecard.py with it for real.
pause
