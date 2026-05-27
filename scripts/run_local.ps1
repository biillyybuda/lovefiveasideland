param(
    [switch]$Check
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Import-LocalEnv {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    foreach ($line in Get-Content -LiteralPath $Path) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        if ($line.TrimStart().StartsWith("#")) {
            continue
        }

        $parts = $line -split "=", 2
        if ($parts.Count -ne 2) {
            continue
        }

        [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1], "Process")
    }
}

$localEnvPath = Join-Path $root ".env.local"

# Avoid stale Windows/terminal settings overriding the local launcher config.
# Render can still use DATABASE_URL; this only affects local runs via this script.
[Environment]::SetEnvironmentVariable("DATABASE_URL", $null, "Process")

Import-LocalEnv $localEnvPath

if (-not $env:PGPASSWORD -and -not $env:DATABASE_URL) {
    Write-Host "Missing database credentials."
    Write-Host "Create .env.local from .env.example, or set DATABASE_URL / PG* variables first."
    exit 1
}

$venvPython = Join-Path $root "venv312\Scripts\python.exe"
$python = $venvPython
$oldErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $venvPython --version *> $null
$venvOk = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $oldErrorActionPreference

if (-not $venvOk) {
    $python = "C:\Users\Bills PC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    $env:PYTHONPATH = Join-Path $root "venv312\Lib\site-packages"
}

Write-Host "Using Python:"
& $python --version
Write-Host "Database route: $($env:PGHOST):$($env:PGPORT)"

if ($Check) {
    & $python -c "import os, streamlit, psycopg2; print('Launcher check OK'); print('PGPORT=' + str(os.getenv('PGPORT'))); print('DATABASE_URL set=' + str(bool(os.getenv('DATABASE_URL'))))"
    exit $LASTEXITCODE
}

Write-Host "Launching Streamlit..."
Start-Process "http://localhost:8501"
& $python -m streamlit run app.py --server.address localhost --server.port 8501
