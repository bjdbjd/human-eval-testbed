# HumanEval 模型评估工具

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于 OpenAI HumanEval 基准测试，评估大语言模型代码生成能力的工具。

## 项目结构

```
.
├── evaluate_models.py     # 评估脚本（支持全量/快速/自定义测试）
├── results/               # 评估结果输出目录
├── requirements.txt       # 依赖清单
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 快速验证（5 道题）

先跑 5 道题验证生成+评测流程是否正常：

```bash
python evaluate_models.py \
    --api-type anthropic \
    --api-key YOUR_API_KEY \
    --base-url YOUR_BASE_URL \
    --models model-name \
    --quick
```

### 3. 全量评估（164 道题）

验证通过后跑完整的 164 题：

```bash
python evaluate_models.py \
    --api-type anthropic \
    --api-key YOUR_API_KEY \
    --base-url YOUR_BASE_URL \
    --models model1 model2 model3 \
    --max-tokens 32768
```

## 命令行参数

### 基础参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--models` | 模型名称列表，可同时指定多个 | 必填 |
| `--api-type` | API 类型：`anthropic` 或 `openai` | `anthropic` |
| `--api-key` | API 密钥 | 必填 |
| `--base-url` | API 服务地址（`/v1` 后缀可选，自动处理） | 必填 |
| `--output-dir` | 结果输出目录 | `results` |
| `--max-tokens` | 每次生成允许的最大 token 数 | `16384` |

### 题目选择参数（三选一，优先级从高到低）

| 参数 | 说明 | 示例 |
|------|------|------|
| `--tasks ID...` | 指定具体题目，支持题号或全名 | `--tasks 0 2 4` 或 `--tasks HumanEval/0 HumanEval/2` |
| `--num-problems N` | 测试前 N 道题目 | `--num-problems 10` |
| `--quick` | 快速测试模式（固定 5 道题） | `--quick` |

若不指定以上参数，默认运行全部 164 道题目。

### base-url 兼容性

`/v1` 后缀可选，程序会根据 API 类型自动处理：

- **OpenAI 类型**：自动添加 `/v1` 后缀（如 `https://api.example.com` → `https://api.example.com/v1`）
- **Anthropic 类型**：自动移除 `/v1` 后缀（如 `https://api.example.com/v1` → `https://api.example.com`）

示例：无论输入 `https://api.example.com` 还是 `https://api.example.com/v1`，均可正常使用。

## 使用示例

### 快速测试（5 题）

```bash
python evaluate_models.py --api-type openai --api-key sk-xxx --base-url https://api.example.com --models gpt-4o --quick
```

### 测试前 20 题

```bash
python evaluate_models.py --api-type openai --api-key sk-xxx --base-url https://api.example.com --models gpt-4o --num-problems 20
```

### 指定具体题目

```bash
# 使用题号
python evaluate_models.py --api-type openai --api-key sk-xxx --base-url https://api.example.com --models gpt-4o --tasks 0 2 4 10 21

# 使用全名
python evaluate_models.py --api-type openai --api-key sk-xxx --base-url https://api.example.com --models gpt-4o --tasks HumanEval/0 HumanEval/2 HumanEval/4
```

### 全量测试（164 题）

```bash
python evaluate_models.py --api-type openai --api-key sk-xxx --base-url https://api.example.com --models gpt-4o claude-4
```

## API 类型支持

### Anthropic 兼容接口

```bash
python evaluate_models.py --api-type anthropic --api-key sk-xxx --base-url https://api.example.com --models claude-4
```

### OpenAI 兼容接口

```bash
python evaluate_models.py --api-type openai --api-key sk-xxx --base-url https://api.example.com --models gpt-4o
```

支持所有兼容 OpenAI `/v1/chat/completions` 和 Anthropic `/v1/messages` 格式的服务。

## 输出说明

评估结果保存在 `results/` 目录下：

**全量测试（164 题）：**
- `<模型名>_samples_{timestamp}.jsonl` — 生成的代码补全
- `<模型名>_samples_{timestamp}_results.jsonl` — 逐题评测结果
- `<模型名>_report_{timestamp}.md` — Markdown 格式评估报告
- `summary_{timestamp}.json` — 多模型对比汇总

**部分测试（如 5 题）：**
- `<模型名>_samples_5of164_{timestamp}.jsonl`
- `<模型名>_samples_5of164_{timestamp}_results.jsonl`
- `<模型名>_report_5of164_{timestamp}.md`

> `{timestamp}` 格式为 `yyyymmddhhmmss`，如 `20260427210000`

## 环境要求

- Python >= 3.10
- pip（用于安装依赖）

## 安全提示

建议使用环境变量传递 API Key，避免在命令行历史中暴露：

```bash
# Linux/macOS
export ANTHROPIC_API_KEY="your-api-key"
python evaluate_models.py --api-key $ANTHROPIC_API_KEY ...

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="your-api-key"
python evaluate_models.py --api-key $env:ANTHROPIC_API_KEY ...
```

## 常见问题

### Q: 评测失败提示 "Some problems are not attempted"

A: 这是旧版本问题，新版本已自动处理部分测试场景。请确保使用最新代码。

### Q: 模型返回内容但解析失败（0 chars）

A: 脚本会自动重试 3 次。如果仍然失败，可能是模型输出格式不符合预期，查看日志中的 `[DEBUG] raw output` 了解原始输出。

### Q: 如何测试本地模型？

A: 使用 OpenAI 兼容的本地推理服务（如 LM Studio、Ollama），设置 `--api-type openai` 和对应的 `--base-url`。

## 致谢

- [HumanEval](https://github.com/openai/human-eval) - OpenAI 的代码生成基准测试数据集
- Chen, M., et al. "Evaluating Large Language Models Trained on Code", arXiv:2107.03374, 2021

## License

[MIT License](LICENSE)