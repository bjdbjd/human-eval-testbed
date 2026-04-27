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
    --base-url YOUR_BASE_URL \
    --api-key YOUR_API_KEY \
    --models model-name \
    --quick
```

### 3. 全量评估（164 道题）

验证通过后跑完整的 164 题：

```bash
python evaluate_models.py \
    --api-type anthropic \
    --base-url YOUR_BASE_URL \
    --api-key YOUR_API_KEY \
    --models model1 model2 model3 \
    --max-tokens 32768
```

## 命令行参数

### 基础参数

| 参数             | 说明                            | 默认值         |
|----------------|-------------------------------|-------------|
| `--models`     | 模型名称列表，可同时指定多个                | 必填          |
| `--channel`    | 模型渠道标识，用于区分不同供应商              | 空（不区分）      |
| `--api-type`   | API 类型：`anthropic` 或 `openai` | `anthropic` |
| `--base-url`   | API 服务地址（`/v1` 后缀可选，自动处理）     | 必填          |
| `--api-key`    | API 密钥                        | 必填          |
| `--output-dir` | 结果输出目录                        | `results`   |
| `--max-tokens` | 每次生成允许的最大 token 数             | `16384`     |

渠道参数用于对比同一模型在不同供应商的表现：

- 支持中英文渠道名称
- 文件名添加渠道标识，如 `model_硅基流动_samples_{timestamp}.jsonl`
- 报告基本信息中显示渠道

### 题目选择参数（三选一，优先级从高到低）

| 参数                 | 说明              | 示例                                                  |
|--------------------|-----------------|-----------------------------------------------------|
| `--tasks ID...`    | 指定具体题目，支持题号或全名  | `--tasks 0 2 4` 或 `--tasks HumanEval/0 HumanEval/2` |
| `--num-problems N` | 测试前 N 道题目       | `--num-problems 10`                                 |
| `--quick`          | 快速测试模式（固定 5 道题） | `--quick`                                           |

若不指定以上参数，默认运行全部 164 道题目。

### 多轮重试参数

| 参数                 | 说明                                  | 默认值      |
|--------------------|-------------------------------------|----------|
| `--retry-rounds N` | 错误重试轮数，如 3 表示最多执行 3 轮，记录 pass@1/2/3 | `1`（不重试） |

多轮重试模式下：

- 每轮执行后，收集错误题目，下一轮只跑这些错误题
- 记录每轮累计正确率（pass@1、pass@2、pass@3...）
- 报告中展示每轮结果和代码

### 重复执行参数

| 参数           | 说明                         | 默认值      |
|--------------|----------------------------|----------|
| `--repeat N` | 重复执行测试的次数，如 2 表示执行 2 次完整测试 | `1`（不重复） |

重复执行模式下：

- 每次执行生成独立的数据文件和报告
- 文件名添加 `run{N}` 标识，如 `model_samples_run1_{timestamp}.jsonl`
- 汇总报告中显示所有执行的 pass@1 及平均值

### base-url 兼容性

`/v1` 后缀可选，程序会根据 API 类型自动处理：

- **OpenAI 类型**：自动添加 `/v1` 后缀（如 `https://api.example.com` → `https://api.example.com/v1`）
- **Anthropic 类型**：自动移除 `/v1` 后缀（如 `https://api.example.com/v1` → `https://api.example.com`）

示例：无论输入 `https://api.example.com` 还是 `https://api.example.com/v1`，均可正常使用。

## 使用示例

### 快速测试（5 题）

```bash
python evaluate_models.py --api-type openai --base-url https://api.example.com --api-key sk-xxx --models gpt-4o --quick
```

### 测试前 20 题

```bash
python evaluate_models.py --api-type openai --base-url https://api.example.com --api-key sk-xxx --models gpt-4o --num-problems 20
```

### 指定具体题目

```bash
# 使用题号
python evaluate_models.py --api-type openai --base-url https://api.example.com --api-key sk-xxx --models gpt-4o --tasks 0 2 4 10 21

# 使用全名
python evaluate_models.py --api-type openai --base-url https://api.example.com --api-key sk-xxx --models gpt-4o --tasks HumanEval/0 HumanEval/2 HumanEval/4
```

### 全量测试（164 题）

```bash
python evaluate_models.py --api-type openai --base-url https://api.example.com --api-key sk-xxx --models gpt-4o claude-4
```

### 渠道对比测试

```bash
# 测试火山引擎渠道
python evaluate_models.py --api-type openai --base-url URL1 --api-key KEY1 --models minimax-m2.5 --channel volcengine

# 测试硅基流动渠道（中文渠道名）
python evaluate_models.py --api-type openai --base-url URL2 --api-key KEY2 --models minimax-m2.5 --channel 硅基流动

# 完整示例：多参数组合
python evaluate_models.py --api-type anthropic --base-url https://api.example.com --api-key sk-xxx --models minimax-m2.5 --channel 超算互联网 --max-tokens 16384 --tasks 0 2 4 --retry-rounds 3 --repeat 3
```


### 多轮重试测试

```bash
# 最多执行 3 轮，记录 pass@1/2/3
python evaluate_models.py --api-type openai --base-url https://api.example.com --api-key sk-xxx --models gpt-4o --retry-rounds 3
```

### 重复执行测试

```bash
# 执行 2 次完整测试，生成 2 份报告
python evaluate_models.py --api-type openai --base-url https://api.example.com --api-key sk-xxx --models gpt-4o --repeat 2
```

## API 类型支持

### Anthropic 兼容接口

```bash
python evaluate_models.py --api-type anthropic --base-url https://api.example.com --api-key sk-xxx --models claude-4
```

### OpenAI 兼容接口

```bash
python evaluate_models.py --api-type openai --base-url https://api.example.com --api-key sk-xxx --models gpt-4o
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

**多轮重试测试（如 3 轮）：**

- `<模型名>_samples_retry3_{timestamp}.jsonl`
- `<模型名>_report_retry3_{timestamp}.md` — 包含每轮 pass@N 结果

**重复执行测试（如 2 次）：**

- `<模型名>_samples_run1_{timestamp}.jsonl` — 第 1 次执行
- `<模型名>_samples_run2_{timestamp}.jsonl` — 第 2 次执行
- `<模型名>_report_run1_{timestamp}.md` — 第 1 次报告
- `<模型名>_report_run2_{timestamp}.md` — 第 2 次报告

> `{timestamp}` 格式为 `yyyymmddhhmmss`，如 `20260427210000`

## 环境要求

- Python >= 3.10
- pip（用于安装依赖）

## 安全提示

建议使用环境变量传递 API Key，避免在命令行历史中暴露：

```bash
# Linux/macOS
export ANTHROPIC_API_KEY="your-api-key"
python evaluate_models.py --api-type anthropic --base-url YOUR_BASE_URL --models model-name

export OPENAI_API_KEY="your-api-key"
python evaluate_models.py --api-type openai --base-url YOUR_BASE_URL --models model-name

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="your-api-key"
python evaluate_models.py --api-type anthropic --base-url YOUR_BASE_URL --models model-name
```

程序会自动读取对应的环境变量，无需通过 `--api-key` 参数传递。

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