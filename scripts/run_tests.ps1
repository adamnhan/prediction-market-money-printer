# PowerShell script to run tests with coverage and open the HTML report
param(
    [switch]$InstallDeps
)

if ($InstallDeps) {
    Write-Host "Installing dependencies..."
    pip install -r ..\requirements.txt
}

Write-Host "Running tests with coverage..."
coverage run -m pytest

Write-Host "Generating coverage report..."
coverage html

$report = Join-Path -Path (Get-Location) -ChildPath "..\htmlcov\index.html"
if (Test-Path $report) {
    Write-Host "Opening coverage report: $report"
    Start-Process $report
} else {
    Write-Host "Coverage report not found at $report"
}
