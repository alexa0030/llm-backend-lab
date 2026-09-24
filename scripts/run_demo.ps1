$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    python -m unittest discover -s tests -v
    python benchmark/run_benchmark.py
    python analysis/compare_results.py
} finally {
    Pop-Location
}

