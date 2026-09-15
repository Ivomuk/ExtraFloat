@echo off
echo === Running agent segmentation standalone (provisional scorecard test) ===
echo Scorecard: scorecards\capacity_scorecard_v0.json (is_provisional=true)
echo segmentation_allow_provisional.json overrides allow_provisional_scorecard
echo for THIS TEST RUN ONLY -- remove that override once the scorecard is
echo reviewed and recalibrated with --final, and never use it for a real
echo production run.

python -m segmentation.run_extrafloat_segmentation --agents data\mfs_daily_agent_mart_20260731.csv --scorecard scorecards\capacity_scorecard_v0.json --config segmentation_allow_provisional.json --output segmentation_outputs

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: segmentation failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Segmentation complete. See segmentation_outputs\agent_segments.csv and
echo segmentation_outputs\run_manifest.json (tier distribution, drift/quality
echo gate reports, and alerts are also printed above).
pause
