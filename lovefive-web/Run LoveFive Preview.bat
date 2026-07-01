@echo off
cd /d "%~dp0"
node.exe node_modules\next\dist\bin\next dev --hostname 127.0.0.1 --port 3000
pause
