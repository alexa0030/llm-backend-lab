param(
    [string]$Model = "Qwen/Qwen2.5-0.5B-Instruct",
    [string]$Output = "models/qwen2.5-0.5b-instruct-ov",
    [ValidateSet("fp16", "int8", "int4")][string]$WeightFormat = "int8"
)
$ErrorActionPreference = "Stop"
optimum-cli export openvino --model $Model --task text-generation-with-past --weight-format $WeightFormat $Output

