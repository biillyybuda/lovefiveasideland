@echo off
cd /d "C:\Users\Bills PC\Documents\5 a side stats\MMRApp\LoveFiveASideApp_2026"

REM PowerShell loads .env.local safely, including passwords with percent signs.
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\run_local.ps1"
pause
