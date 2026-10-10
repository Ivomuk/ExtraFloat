@echo off
REM run_stage4a_full_audit.bat
REM ===========================================================================
REM Runs the full Stage 4A.1 -> Stage 4A chain in one go:
REM   1. Full Phase 2.1 PIT reconstruction (writes the reconstruction-evidence
REM      JSON that step 4 needs).
REM   2. Champion-feature-coverage check (schema-only, fast).
REM   3. Upstream cluster/peer schema-overlap check (schema-only, fast).
REM   4. Stage 4A audit, wired to all three evidence files plus the real
REM      champion/shadow artifacts.
REM
REM Edit the paths below if your real data files live somewhere else.
REM ===========================================================================

setlocal

set LOAN_TRAINING_FILE=data\state_data_20260910_retail_filtered.csv
set TRANSACTION_MART_FILE=data\mfs_daily_agent_mart_202607_retail_filtered.csv
set PD_MODEL_ARTIFACTS_DIR=pd_model\artifacts
set SHADOW_ARTIFACTS_DIR=pd_model\artifacts
set STAGE4A1_OUT_PREFIX=stage4a1_phase21_pit
set STAGE4A_OUT_PREFIX=stage4a_pd_rescoring_audit

echo.
echo ===========================================================================
echo STEP 1/4: Full Phase 2.1 PIT reconstruction
echo ===========================================================================
python scripts\build_historical_pit_transaction_features.py ^
    --loan-training-file %LOAN_TRAINING_FILE% ^
    --transaction-mart-file %TRANSACTION_MART_FILE% ^
    --out-prefix %STAGE4A1_OUT_PREFIX%
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: step 1 (full reconstruction^) failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ===========================================================================
echo STEP 2/4: Champion-feature-coverage check
echo ===========================================================================
python scripts\build_historical_pit_transaction_features.py --champion-feature-coverage-only ^
    --pd-model-artifacts-dir %PD_MODEL_ARTIFACTS_DIR% ^
    --out-prefix %STAGE4A1_OUT_PREFIX%
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: step 2 (champion-feature-coverage^) failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ===========================================================================
echo STEP 3/4: Upstream cluster/peer schema-overlap check
echo ===========================================================================
python scripts\build_historical_pit_transaction_features.py --schema-check-only ^
    --pd-model-artifacts-dir %PD_MODEL_ARTIFACTS_DIR% ^
    --out-prefix %STAGE4A1_OUT_PREFIX%
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: step 3 (schema-check^) failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ===========================================================================
echo STEP 4/4: Stage 4A PIT PD rescoring feasibility audit
echo ===========================================================================
python scripts\audit_historical_pd_rescoring_feasibility.py ^
    --pd-model-artifacts-dir %PD_MODEL_ARTIFACTS_DIR% ^
    --shadow-artifacts-dir %SHADOW_ARTIFACTS_DIR% ^
    --phase21-reconstruction-evidence-json %STAGE4A1_OUT_PREFIX%_phase21_pit_reconstruction_evidence.json ^
    --phase21-champion-feature-coverage-json %STAGE4A1_OUT_PREFIX%_champion_feature_coverage.json ^
    --phase21-upstream-schema-check-json %STAGE4A1_OUT_PREFIX%_phase21_upstream_schema_check.json ^
    --out-prefix %STAGE4A_OUT_PREFIX%
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: step 4 (Stage 4A audit^) failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ===========================================================================
echo DONE. See %STAGE4A_OUT_PREFIX%_disposition.json for the final disposition.
echo ===========================================================================
pause
