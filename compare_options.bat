@echo off
python compare_options.py ^
    --input  output/engine_test_output.csv ^
    --output output/options_comparison.csv

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: compare_options.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Comparison written to output/options_comparison.csv
pause
