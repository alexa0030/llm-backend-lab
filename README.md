# LLM Backend Migration & Validation

[![CI](https://github.com/alexa0030/llm-backend-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/alexa0030/llm-backend-lab/actions/workflows/ci.yml)

一个小而完整的 LLM 推理后端迁移验证 demo：对同一批 prompt 分别运行 baseline 与 candidate backend，同时验证输出正确性、性能与回归门禁。

当前默认配置不下载模型，使用两个确定性的 mock backend 演示完整工程闭环。真实适配器已经实现，可切换到 Hugging Face Transformers 和 OpenVINO GenAI。

## 30 秒运行

环境要求：Python 3.10+。默认 demo 无第三方依赖。

```powershell
./scripts/run_demo.ps1
```

也可分步运行：

```powershell
python -m unittest discover -s tests -v
python benchmark/run_benchmark.py
python analysis/compare_results.py
```

无需修改配置文件也可以临时覆盖 workload：

```powershell
python benchmark/run_benchmark.py --backends hf-mock openvino-mock --limit 3 --runs 1
python analysis/compare_results.py
```

结果写入：

- `reports/raw_results.csv`：逐请求原始指标
- `reports/summary.json`：机器可读汇总
- `reports/run_metadata.json`：Python、操作系统和运行参数等复现实验所需信息
- `reports/benchmark_report.md`：可直接阅读的迁移验证报告

## 项目闭环

```text
JSONL prompts
    ├─> baseline backend ─┐
    └─> candidate backend ├─> raw CSV ─> similarity + performance ─> regression report
                         ┘
```

统一结果结构包含成功状态、输出文本、输入/输出 token、TTFT、端到端延迟、tokens/s 和内存变化。`config.yaml` 定义 workload、生成参数与回归阈值；文件内容采用 JSON 语法（同时也是合法 YAML），因此核心 demo 无需 PyYAML。

## 切换真实模型

真实运行会下载模型并消耗数 GB 磁盘与内存：

```powershell
python -m pip install -r requirements-real.txt
./scripts/export_openvino.ps1
```

随后把 `config.yaml` 中的 backend 改为：

```json
"backends": ["huggingface", "openvino"],
"baseline": "huggingface",
"candidate": "openvino"
```

建议第一次把 `limit` 和 `measured_runs` 都设为 `1`，确认模型和设备可用后再扩大 workload。OpenVINO 模型目录由 `models.openvino_model_dir` 指定。

## 指标口径

- **TTFT**：OpenVINO 使用 GenAI `PerfMetrics`；HF 的简单同步实现无法可靠拆分首 token，因此记为空值，不伪造数字。
- **E2E latency**：单次 `generate` 调用的墙钟时间。
- **Tokens/s**：输出 token 数除以 E2E latency；OpenVINO 使用官方 throughput 指标。
- **Similarity**：当前 demo 使用标准库 `SequenceMatcher`，适合作为轻量 smoke gate；生产级验证可替换为 embedding、ROUGE 或 Who What Benchmark。
- **Memory delta**：安装 `psutil` 后记录请求前后 RSS 差值；它不是峰值内存。

## 目录

```text
benchmark/       benchmark runner
analysis/        comparison and report generation
data/            small public-workload-style sample
src/             backend adapters, result models, metrics
tests/           inference, boundary and regression tests
scripts/         one-command demo and OpenVINO export
reports/         generated evidence
.github/         GitHub Actions CI
```

每次 push 和 pull request 都会在 Python 3.10、3.12 上运行测试、benchmark demo 与回归门禁。

## 下一步

1. 用真实 Qwen2.5-0.5B-Instruct 跑 10 条 prompt，记录机器配置和软件版本。
2. 加 INT8/INT4 两组 OpenVINO 模型，比较正确性与性能权衡。
3. 将 HF 的同步生成改成 streamer，以同一口径采集 TTFT/TPOT。
4. 增加 vLLM serving backend、并发与 P95/P99 指标。

> 不要把 mock 报告中的性能数字写进简历；它只证明 harness 与 regression gate 可运行。
