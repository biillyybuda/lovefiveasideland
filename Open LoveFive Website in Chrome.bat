@echo off
setlocal

set "ROOT=%~dp0"
set "WEB_DIR=%ROOT%lovefive-web"
set "URL=http://127.0.0.1:3000/app/matchday"

if not exist "%WEB_DIR%\package.json" (
  echo Could not find the website folder:
  echo "%WEB_DIR%"
  pause
  exit /b 1
)

where npm.cmd >nul 2>nul
if errorlevel 1 (
  echo Node.js / npm was not found.
  echo Install Node.js, then try this launcher again.
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if errorlevel 1 (
  echo Starting LoveFive website locally...
  start "LoveFive Local Website" cmd /k "cd /d ""%WEB_DIR%"" && npm.cmd run dev -- --hostname 127.0.0.1 --port 3000"
) else (
  echo Local website is already running on port 3000.
)

echo Waiting for the website to be ready...
for /l %%i in (1,1,30) do (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $response = Invoke-WebRequest -Uri '%URL%' -UseBasicParsing -TimeoutSec 2; if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { exit 0 } } catch { exit 1 }"
  if not errorlevel 1 goto open_site
  timeout /t 1 /nobreak >nul
)

echo The website did not answer within 30 seconds.
echo The server window may show what went wrong.
pause
exit /b 1

:open_site
set "CHROME_EXE="
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" set "CHROME_EXE=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" set "CHROME_EXE=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
if exist "%LocalAppData%\Google\Chrome\Application\chrome.exe" set "CHROME_EXE=%LocalAppData%\Google\Chrome\Application\chrome.exe"

echo Opening LoveFive in Chrome...
if defined CHROME_EXE (
  start "" "%CHROME_EXE%" "%URL%"
) else (
  start "" "%URL%"
)

endlocal
