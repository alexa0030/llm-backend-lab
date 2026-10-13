$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ($ProjectRoot -match '[^\x00-\x7F]') {
    throw "OpenVINO Windows native components may fail under non-ASCII paths. Clone or move the repository to an English-only path before running real experiments."
}
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Optimum = Join-Path $ProjectRoot ".venv\Scripts\optimum-cli.exe"
$OpenVinoModel = Join-Path $ProjectRoot "models\qwen2.5-0.5b-instruct-ov-2025b"

if (-not (Test-Path $Python)) {
    throw "Missing .venv. Run: python -m venv .venv; .\.venv\Scripts\python -m pip install -r requirements-real.txt"
}

Push-Location $ProjectRoot
try {
    if (-not (Test-Path $OpenVinoModel)) {
        & $Optimum export openvino --model Qwen/Qwen2.5-0.5B-Instruct `
            --task text-generation-with-past --weight-format int8 $OpenVinoModel
    }
    & $Python benchmark/run_benchmark.py --config config.yaml
    & $Python analysis/compare_results.py --config config.yaml
} finally {
    Pop-Location
}
