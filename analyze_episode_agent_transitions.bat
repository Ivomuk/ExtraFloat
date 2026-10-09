@echo off
python scripts\analyze_episode_agent_transitions.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: analyze_episode_agent_transitions.py failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
