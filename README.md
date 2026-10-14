# LLM Backend Migration & Validation

[![CI](https://github.com/alexa0030/llm-backend-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/alexa0030/llm-backend-lab/actions/workflows/ci.yml)

一个可复现的 LLM 推理后端迁移验证框架：对同一批 prompt 分别运行 baseline 与 candidate backend，同时验证输出正确性、性能、资源占用与回归门禁。

默认配置运行真实的 Qwen2.5-0.5B-Instruct，对比 Hugging Face Transformers CPU baseline 与 OpenVINO GenAI。Optimum/OpenVINO 是单独的 fallback 配置，不能替代 GenAI 验收。Mock backend 仅用于 CI 和框架单元测试，不属于模型性能实验。

## 框架自测（30 秒，无模型下载）

环境要求：Python 3.10+。默认 demo 无第三方依赖。

```powershell
./scripts/run_demo.ps1
```

也可分步运行：

```powershell
python -m unittest discover -s tests -v
python benchmark/run_benchmark.py --config config.mock.yaml
python analysis/compare_results.py --config config.mock.yaml
```

无需修改配置文件也可以临时覆盖 workload：

```powershell
python benchmark/run_benchmark.py --config config.mock.yaml --limit 3 --runs 1
python analysis/compare_results.py --config config.mock.yaml
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

Mock 报告会被自动标记为 `MOCK FRAMEWORK SELF-TEST`，其中人工延迟产生的性能变化不得作为 OpenVINO 性能结论引用。

## 真实模型实验

真实运行会下载模型并消耗数 GB 磁盘与内存：

```powershell
python -m pip install -r requirements-real.txt
./scripts/run_real.ps1
```

Windows 上请用 [python.org](https://www.python.org/downloads/windows/) 的官方 CPython 创建独立英文路径 venv。不要从 Anaconda/Miniconda 的 `python.exe` 创建真实 GenAI 实验环境；即使 `pip check` 正常，继承的原生 DLL 搜索环境仍可能让 `LLMPipeline` 以 `0xC0000005` 退出。

默认 `config.yaml` 已使用真实 backend：

```json
"backends": ["huggingface", "openvino-genai"],
"baseline": "huggingface",
"candidate": "openvino-genai"
```

先运行隔离式真实 Smoke Test；它逐项验证模型加载、正常生成、空输入、最大生成长度、结果记录和异常处理。任何原生崩溃都会被父进程记录为 `FAIL`，依赖项记录为 `NOT_RUN`：

```powershell
python smoke/run_real_smoke.py --hf-model models/qwen2.5-0.5b-instruct-hf --openvino-model-dir models/qwen2.5-0.5b-instruct-ov-2025b --device CPU
```

只有两个目标 backend 全部通过后，才应运行 `config.yaml` 的性能对比。若只需定位 OpenVINO Runtime/IR，可显式使用 `config.optimum.yaml`；该结果必须标为 Optimum fallback，不能写成 GenAI 性能。

第一次可把 `limit` 和 `measured_runs` 都设为 `1`，确认链路后再扩大 workload。真实报告只有在模型、软件版本、生成参数和硬件环境全部记录后才可作为实验结果引用。

## 指标口径

- **TTFT**：HF 扩展 `TextIteratorStreamer`，在收到首个生成 token ID 时打点；OpenVINO 使用 GenAI `PerfMetrics`。
- **TPOT**：首 token 之后的平均单 token 时间；吞吐量同样按 decode 阶段计算。
- **E2E latency**：单次 `generate` 调用的墙钟时间。
- **Tokens/s**：decode 阶段生成速率；Transformers 以首 token 后的墙钟时间计算，OpenVINO GenAI 使用官方 `PerfMetrics` throughput。报告必须同时保留 token 数和 E2E latency，避免混用口径。
- **Similarity**：当前 demo 使用标准库 `SequenceMatcher`，适合作为轻量 smoke gate；生产级验证可替换为 embedding、ROUGE 或 Who What Benchmark。
- **Peak RSS**：后台线程以 10ms 间隔采样当前进程的主机内存峰值。它不是 GPU 显存，也不代表整个系统的内存峰值。
- **Model load**：backend 构造及模型加载耗时，和逐请求 latency 分开记录。
- **Input tokens**：Transformers 读取套用 chat template 后的 `input_ids`；GenAI 读取 `PerfMetrics.get_num_input_tokens()`，包含 GenAI 自动应用的模型 chat template。

## 回归验证设计

- warm-up 数据不进入统计分布；正式请求重复执行并计算 P50/P95。
- 相同 prompt、生成参数和随机性设置用于两个 backend。
- 逐 case 保存输出与相似度，失败请求保存异常类型和消息。
- 正确性、延迟退化和成功率分别设置 gate，任一失败都会返回非零退出码。
- `run_metadata.json` 固化 Git revision、Python/依赖版本、模型、设备和生成参数。
- 只有真实 backend、至少 1 次 warm-up、5 条 prompt、每条 3 次测量同时满足时，运行才标记为 performance-publishable。

### Windows 路径注意事项

OpenVINO 的部分 Windows 原生组件对非 ASCII 安装路径兼容性有限。真实实验建议把仓库和虚拟环境放在纯英文路径；框架自测不受影响。若 GenAI `LLMPipeline` 原生崩溃，可先用 `openvino-optimum` 验证 Runtime/IR，并保留环境版本与复现日志，不应把崩溃误报为模型转换失败。

## 项目定位

本项目定位为**推理后端自动化验证与性能回归框架**，不是聊天机器人，也不预设 OpenVINO 一定更快。可在面试中重点说明：迁移 backend 后，如何使用同一 workload 同时验证输出质量、延迟分布、decode 吞吐、资源占用与异常 case，并保证实验可复现。

详细设计与排障记录：

- [实验方法](docs/experiment_methodology.md)
- [已知问题与兼容性排查](docs/known_issues.md)

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
