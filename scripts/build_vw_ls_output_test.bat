@echo off
REM ============================================================================
REM build_vw_ls_output_test.bat
REM ============================================================================
REM Windows wrapper for scripts/build_vw_ls_output.py, pinned to the April
REM test cutoff (data is currently loaded only through 2026-04-14, short of
REM loan_summary_query.txt's checked-in production snapshot_dt of 20260731).
REM data/loan_summary_query.txt itself is never modified -- see that
REM script's own header for why the override exists.
REM
REM Run this from the repo root (the folder containing scripts\ and data\):
REM     scripts\build_vw_ls_output_test.bat your_schema
REM
REM If you omit the schema argument it defaults to xtrafloat_validation.
REM Output is written to vw_ls_output_test.sql in the current folder --
REM open it and run its single CREATE OR REPLACE VIEW statement in your
REM Athena/Trino SQL client.
REM ============================================================================

setlocal

set SCHEMA=%1
if "%SCHEMA%"=="" set SCHEMA=xtrafloat_validation

set SNAPSHOT_DT=20260414
set AS_OF_LOAD_TS=2026-04-15 00:00:00.000
set OUT_FILE=vw_ls_output_test.sql

set PYTHON_CMD=
where python >nul 2>nul
if not errorlevel 1 set PYTHON_CMD=python
if "%PYTHON_CMD%"=="" (
    where py >nul 2>nul
    if not errorlevel 1 set PYTHON_CMD=py
)
if "%PYTHON_CMD%"=="" (
    echo ERROR: neither "python" nor "py" was found on PATH. Install Python 3 and try again.
    exit /b 1
)

echo Building %OUT_FILE% for schema "%SCHEMA%" ^(snapshot_dt=%SNAPSHOT_DT%, as_of_load_ts=%AS_OF_LOAD_TS%^) ...

%PYTHON_CMD% scripts\build_vw_ls_output.py %SCHEMA% ^
    --snapshot-dt %SNAPSHOT_DT% ^
    --as-of-load-ts "%AS_OF_LOAD_TS%" ^
    > %OUT_FILE%

if errorlevel 1 (
    echo ERROR: build_vw_ls_output.py failed -- see output above.
    exit /b 1
)

echo Done. Wrote %OUT_FILE%.
echo Next: run %OUT_FILE% in your Athena/Trino SQL client, then GATE 0 in
echo data\loan_summary_query_validation_queries.sql using the SAME
echo snapshot_dt / as_of_load_ts values printed inside %OUT_FILE%'s header.

endlocal
