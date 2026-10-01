@echo off
python scripts\benchmark_persona_pca_clustering.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: benchmark_persona_pca_clustering.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Complete. See segmentation_outputs\persona_pca_benchmark\ for the plot and CSV.
pause
