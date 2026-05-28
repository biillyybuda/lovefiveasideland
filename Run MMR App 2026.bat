@echo off
cd /d "%~dp0"

REM PowerShell loads .env.local safely, including passwords with percent signs.
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\run_local.ps1"
pause
