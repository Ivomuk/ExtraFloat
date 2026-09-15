@echo off
echo === Calibrating capacity scorecard (provisional_v1) ===
echo Writes scorecards\capacity_scorecard_v1.json (v0 kept as-is for
echo comparison) -- does NOT wire it into production scoring. Point
echo run_extrafloat_segmentation's scoring.scorecard_path at it yourself
echo once reviewed.
echo.
echo Two changes vs. v0: (1) calibrated against the retail-filtered
echo population, matching the PD model's population; (2)
echo expected_tier_proportions now reflects the FULL production
echo post-processing (tenure safety cap + dormancy override), not just
echo raw score-quantile bucketing -- see calibrate_capacity_scorecard's
echo docstring. Target proportions are unchanged from v0 so the printed
echo raw vs. FINAL breakdown isolates the effect of these two changes.
echo Uses --force -- v1 is still provisional/iterative, so each re-run
echo overwrites the previous v1 rather than accumulating stale files.

python calibrate_scorecard.py --agents data\mfs_daily_agent_mart_20260731_retail_filtered.csv --out scorecards\capacity_scorecard_v1.json --target-proportions "{\"Below Threshold\": 0.25, \"New Bronze\": 0.20, \"Bronze\": 0.15, \"Silver\": 0.12, \"Gold\": 0.10, \"Platinum\": 0.08, \"Titanium\": 0.06, \"Diamond\": 0.04}" --cutoff-version provisional_v1 --population-description "mfs_daily_agent_mart_20260731_retail_filtered.csv, ~444,938 agents, 1m/3m/6m KPI windows" --force

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: calibrate_scorecard.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Scorecard written to scorecards\capacity_scorecard_v1.json (is_provisional=true).
echo Review the raw vs. FINAL tier proportions printed above, then re-run
echo with --final and a reviewed cutoff-version once signed off.
pause
