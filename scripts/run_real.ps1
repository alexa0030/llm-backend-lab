param(
    [string]$Python,
    [string]$HfModel,
    [string]$OpenVinoModel,
    [string]$Device = "CPU"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
}
$Optimum = Join-Path $ProjectRoot ".venv\Scripts\optimum-cli.exe"
if (-not $HfModel) {
    $HfModel = Join-Path $ProjectRoot "models\qwen2.5-0.5b-instruct-hf"
}
if (-not $OpenVinoModel) {
    $OpenVinoModel = Join-Path $ProjectRoot "models\qwen2.5-0.5b-instruct-ov-2025b"
}

if (-not (Test-Path $Python)) {
    throw "Missing .venv. Run: python -m venv .venv; .\.venv\Scripts\python -m pip install -r requirements-real.txt"
}

& $Python -c "import sys; raise SystemExit(1 if 'Anaconda' in sys.version or 'Continuum' in sys.version else 0)"
if ($LASTEXITCODE -ne 0) {
    throw "Real OpenVINO GenAI validation requires a venv created from official CPython. An Anaconda-derived interpreter caused native DLL conflicts and 0xC0000005 on this host."
}

Push-Location $ProjectRoot
try {
    if (-not (Test-Path $OpenVinoModel)) {
        & $Optimum export openvino --model Qwen/Qwen2.5-0.5B-Instruct `
            --task text-generation-with-past --weight-format int8 $OpenVinoModel
    }
    & $Python smoke/run_real_smoke.py --hf-model $HfModel `
        --openvino-model-dir $OpenVinoModel --device $Device
    if ($LASTEXITCODE -ne 0) {
        throw "Real smoke test failed. Full comparison was not run; inspect reports/real_smoke_report.md."
    }
    & $Python benchmark/run_benchmark.py --config config.yaml
    & $Python analysis/compare_results.py --config config.yaml
} finally {
    Pop-Location
}
