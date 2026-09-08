@echo off
REM ============================================================================
REM build_vw_bh_output_test.bat
REM ============================================================================
REM Windows wrapper for scripts/build_vw_bh_output.py, pinned to the June 9
REM test cutoff (data is currently loaded through 2026-06-09, short of
REM borrower_history.txt's checked-in production snapshot_dt of 20260731).
REM data/borrower_history.txt itself is never modified -- see that script's
REM own header for why the override exists.
REM
REM as_of_load_ts/snapshot_ts are WAREHOUSE LOAD timestamps (inserted_ts),
REM not event-date cutoffs -- confirmed via MIN/MAX(inserted_ts) against
REM both source tables that this reload landed 2026-09-05 to 2026-09-08.
REM If a later reload changes that, update AS_OF_LOAD_TS below to something
REM past the new MAX(inserted_ts), or this will silently return 0 rows.
REM
REM Run this from the repo root (the folder containing scripts\ and data\):
REM     scripts\build_vw_bh_output_test.bat your_schema
REM
REM If you omit the schema argument it defaults to xtrafloat_validation.
REM Output is written to vw_bh_output_test.sql in the current folder -- it
REM contains SEVEN statements (three checkpoint CREATE TABLEs, then the
REM final CREATE OR REPLACE VIEW vw_bh_output) -- run them IN ORDER against
REM the same session/connection in your Athena/Trino SQL client.
REM ============================================================================

setlocal

set SCHEMA=%1
if "%SCHEMA%"=="" set SCHEMA=xtrafloat_validation

set SNAPSHOT_DT=20260609
set AS_OF_LOAD_TS=2026-09-09 00:00:00.000
set SNAPSHOT_TS=2026-09-09 00:00:00.000
set OUT_FILE=vw_bh_output_test.sql

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

%PYTHON_CMD% scripts\build_vw_bh_output.py %SCHEMA% ^
    --snapshot-dt %SNAPSHOT_DT% ^
    --as-of-load-ts "%AS_OF_LOAD_TS%" ^
    --snapshot-ts "%SNAPSHOT_TS%" ^
    > %OUT_FILE%

if errorlevel 1 (
    echo ERROR: build_vw_bh_output.py failed -- see output above.
    exit /b 1
)

echo Done. Wrote %OUT_FILE%.
echo Next: run %OUT_FILE%'s SEVEN statements IN ORDER in your Athena/Trino SQL
echo client, then GATE 0 in data\borrower_history_validation_queries.sql using
echo the SAME snapshot_dt / as_of_load_ts / snapshot_ts values printed inside
echo %OUT_FILE%'s header.

endlocal
