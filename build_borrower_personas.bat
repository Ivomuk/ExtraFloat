@echo off
echo === Building borrower personas (MoMo transaction capacity + loan/credit behavior) ===
echo Joins mfs_daily_agent_mart_20260731.csv with borrower_history_features.csv,
echo then clusters the joined population via segmentation's own feature
echo engineering + HDBSCAN/LOF pipeline (reused via direct import, nothing
echo under segmentation/ is modified).
echo.
echo Requires (see segmentation\borrower_persona_clustering.py's module
echo docstring for the full prerequisites):
echo   - data\mfs_daily_agent_mart_20260731.csv (MOMO_PATH)
echo   - data\borrower_history_features.csv (LOANS_PATH) -- produced by
echo     pipeline\run_pipeline.py; run that first if this doesn't exist yet.
echo     NOTE: this is NOT the same file as data\borrower_history.csv used
echo     elsewhere in this repo -- a separately-produced loan/credit-behavior
echo     features file.
echo   - output\engine_test_output.csv (ENGINE_OUTPUT_PATH) -- OPTIONAL, only
echo     used to attach assigned_limit/risk_tier for profiling. Script
echo     degrades gracefully with a warning if missing.
echo.
echo NOTE: EXPECTED_JOIN_COUNT in the script is a hardcoded sanity check.
echo If the source files changed since it was last set, this run will fail
echo with the actual count in the error message -- update EXPECTED_JOIN_COUNT
echo to that number and re-run; a changed count is not itself a bug.

python segmentation\borrower_persona_clustering.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: borrower_persona_clustering.py failed with exit code %ERRORLEVEL%
    echo If this is an EXPECTED_JOIN_COUNT assertion failure, see the NOTE above.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Complete. Output written to segmentation_outputs\borrower_persona_output\:
echo   borrower_persona_clusters.csv         -- one row per borrower, persona_cluster + anomaly flags
echo   borrower_persona_cluster_profile.csv  -- per-cluster median features, size, assigned_limit/risk_tier
pause
