@echo off
title Zvy Copilot Desktop
echo ========================================================
echo   Launching Zvy Copilot (Groq LLM Engine)
echo ========================================================
cd /d "%~dp0"
python zvy_desktop.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Opening in browser as fallback...
    start http://127.0.0.1:8080
)
