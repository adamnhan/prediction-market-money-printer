# PowerShell script to run tests with coverage and open the HTML report
param(
    [switch]$InstallDeps
)

# Directory of this script (scripts/)
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
# Repo root is parent of scripts/
$repoRoot = Resolve-Path (Join-Path $scriptRoot '..')

if ($InstallDeps) {
    Write-Host "Installing dependencies..."
    $req = Join-Path $repoRoot 'requirements.txt'
    if (Test-Path $req) {
        pip install -r $req
    } else {
        Write-Host "Requirements file not found at $req"
    }
}

Write-Host "Running tests with coverage..."
$rc = Join-Path $repoRoot '.coveragerc'
if (Test-Path $rc) {
    coverage run --rcfile=$rc -m pytest
} else {
    Write-Host "Couldn't find $rc, running coverage without rcfile"
    coverage run -m pytest
}

Write-Host "Generating coverage report..."
if (Test-Path $rc) {
    coverage html --rcfile=$rc
} else {
    coverage html
}

$report = Join-Path $repoRoot 'htmlcov\index.html'
if (Test-Path $report) {
    Write-Host "Opening coverage report: $report"
    Start-Process $report
} else {
    Write-Host "Coverage report not found at $report"
}
