"""
HumanEval 模型评估工具：评估大语言模型的代码生成能力。

用途：
    - 全面评估模型的 pass@1 分数（164 题）
    - 快速验证 API 配置（--quick 模式，5 题）
    - 自定义测试范围（--tasks 或 --num-problems）
    - 多轮重试错误题目（--retry-rounds）
    - 重复执行测试（--repeat）
    - 渠道对比测试（--channel）
    - 生成详细的评测报告（逐题结果、生成代码）

使用示例：
    # 全量测试（164 题）
    python evaluate_models.py --api-type openai --base-url URL --api-key KEY --models gpt-4o

    # 快速测试（5 题）
    python evaluate_models.py --api-type openai --base-url URL --api-key KEY --models gpt-4o --quick

    # 测试前 10 题
    python evaluate_models.py --api-type openai --base-url URL --api-key KEY --models gpt-4o --num-problems 10

    # 指定具体题目（支持题号或全名）
    python evaluate_models.py --api-type openai --base-url URL --api-key KEY --models gpt-4o --tasks 0 2 4
    python evaluate_models.py --api-type openai --base-url URL --api-key KEY --models gpt-4o --tasks HumanEval/0 HumanEval/2

    # 多轮重试测试（最多执行 3 轮，记录 pass@1/2/3）
    python evaluate_models.py --api-type openai --base-url URL --api-key KEY --models gpt-4o --retry-rounds 3

    # 重复执行测试（执行 2 次，生成 2 份报告）
    python evaluate_models.py --api-type openai --base-url URL --api-key KEY --models gpt-4o --repeat 2

    # 渠道对比测试
    python evaluate_models.py --api-type openai --base-url URL --api-key KEY --models gpt-4o --channel 硅基流动

    # 多参数组合
    python evaluate_models.py --api-type anthropic --base-url URL --api-key KEY --models gpt-4o --channel 测试渠道 --max-tokens 16384 --tasks 0 2 4 --retry-rounds 3 --repeat 2

注意：
    - base_url 的 /v1 后缀可选，程序会自动处理
    - OpenAI 类型自动添加 /v1，Anthropic 类型自动移除 /v1
"""

import argparse
import ast
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from anthropic.types import MessageParam
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from tqdm import tqdm

# 快速测试默认题目集
QUICK_TEST_TASKS = ["HumanEval/0", "HumanEval/2", "HumanEval/4", "HumanEval/10", "HumanEval/21"]
# HumanEval 总题数
HUMANEVAL_TOTAL_PROBLEMS = 164


class Spinner:
    """命令行等待动画。"""

    def __init__(self, message="Waiting"):
        self.message = message
        self.spinning = False
        self.thread = None
        self.chars = "|/-\\"

    def _spin(self):
        """动画循环。"""
        idx = 0
        while self.spinning:
            char = self.chars[idx % len(self.chars)]
            sys.stdout.write(f"\r  [{char}] {self.message}...")
            sys.stdout.flush()
            time.sleep(0.1)
            idx += 1
        sys.stdout.write("\r" + " " * (len(self.message) + 15) + "\r")
        sys.stdout.flush()

    def start(self):
        """启动动画。"""
        self.spinning = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()

    def stop(self):
        """停止动画。"""
        self.spinning = False
        if self.thread:
            self.thread.join()


def fix_indentation(code: str) -> str:
    """修复缩进，确保函数体从 4 空格开始，同时保持相对缩进。"""
    if not code:
        return code

    lines = code.splitlines()
    if not lines:
        return code

    # 收集所有非空行的缩进
    indents = []
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            indents.append(indent)

    if not indents:
        return code

    min_indent = min(indents)

    # 检测顶层语句的缩进模式（函数体顶层语句应有相同缩进）
    top_level_indents = []
    prev_indent = None
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        # 顶层语句特征：首行、缩进回退、或前一行为空
        if prev_indent is None or indent <= prev_indent:
            top_level_indents.append(indent)
        prev_indent = indent

    if top_level_indents:
        # 使用最常见的顶层缩进作为基准
        most_common_indent = Counter(top_level_indents).most_common(1)[0][0]
        target_indent = 4
        offset = target_indent - most_common_indent
    else:
        offset = 4 - min_indent

    # 应用偏移量
    fixed_lines = []
    for line in lines:
        if line.strip() == "":
            fixed_lines.append("")
        else:
            current_indent = len(line) - len(line.lstrip())
            new_indent = max(0, current_indent + offset)
            fixed_lines.append(" " * new_indent + line.lstrip())

    return "\n".join(fixed_lines)


def fix_indentation_smart(code: str) -> str:
    """基于 Python 块结构智能修复缩进。

    策略：
    1. 先尝试简单修复
    2. 若语法错误，基于块结构（冒号结尾的行）重新缩进
    3. 处理 dedent 关键字（else、elif、except、finally）
    """
    if not code:
        return code

    # 先尝试简单修复
    fixed = fix_indentation(code)

    # 检查语法是否有效
    try:
        ast.parse("def _test_():\n" + fixed)
        return fixed
    except SyntaxError:
        pass

    # 语法无效，基于块结构重新缩进
    lines = code.splitlines()
    if not lines:
        return code

    fixed_lines = []
    indent_stack = [4]  # 缩进栈，起始为 4（函数体）

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            fixed_lines.append("")
            continue

        current_indent = indent_stack[-1]

        # 处理 dedent 关键字
        if stripped.startswith(("else:", "elif ", "except:", "finally:")):
            if len(indent_stack) > 1:
                indent_stack.pop()
                current_indent = indent_stack[-1]

        # 注释行与下一行同缩进
        if stripped.startswith("#"):
            next_indent = current_indent
            for j in range(i + 1, len(lines)):
                next_stripped = lines[j].strip()
                if next_stripped and not next_stripped.startswith("#"):
                    next_indent = indent_stack[-1]
                    break
            fixed_lines.append(" " * next_indent + stripped)
            continue

        fixed_lines.append(" " * current_indent + stripped)

        # 更新缩进栈
        if stripped.endswith(":"):
            indent_stack.append(current_indent + 4)
        elif stripped.startswith(("return ", "break", "continue", "raise ", "pass")):
            if len(indent_stack) > 1:
                indent_stack.pop()

    result = "\n".join(fixed_lines)

    # 验证修复结果
    try:
        ast.parse("def _test_():\n" + result)
        return result
    except SyntaxError:
        return fix_indentation_aggressive(code)


def fix_indentation_aggressive(code: str) -> str:
    """激进缩进修复：将缩进归一化为 4 的倍数，起始为 4。"""
    if not code:
        return code

    lines = code.splitlines()
    if not lines:
        return code

    # 收集所有缩进级别
    indent_levels = set()
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            indent_levels.add(indent)

    if not indent_levels:
        return code

    # 映射到 4 的倍数
    sorted_levels = sorted(indent_levels)
    indent_map = {}
    for i, level in enumerate(sorted_levels):
        indent_map[level] = 4 + i * 4

    # 应用映射
    fixed_lines = []
    for line in lines:
        if line.strip() == "":
            fixed_lines.append("")
        else:
            current_indent = len(line) - len(line.lstrip())
            new_indent = indent_map.get(current_indent, 4)
            fixed_lines.append(" " * new_indent + line.lstrip())

    return "\n".join(fixed_lines)


def extract_code_from_markdown(text: str) -> str:
    """从 markdown 文本中提取 ```python 代码块内容。"""
    if not text:
        return text

    python_block_pattern = r'```python\s*\n(.*?)```'
    matches = list(re.finditer(python_block_pattern, text, re.DOTALL | re.IGNORECASE))

    if matches:
        code_parts = [match.group(1).strip() for match in matches]
        return "\n\n".join(code_parts)

    return text


def strip_redundant_prefix(completion: str, prompt: str) -> str:
    """移除补全中冗余重复的部分（已在 prompt 中定义的导入、函数签名、文档字符串）。"""
    lines = completion.splitlines()
    stripped = []

    in_docstring = False
    docstring_delim = None
    past_prefix = False

    for i, line in enumerate(lines):
        content = line.strip()

        # 跳过前导空行
        if not past_prefix and content == "":
            continue

        # 跳过已在 prompt 中存在的导入语句
        if not past_prefix and content.startswith(("import ", "from ")):
            if content in prompt:
                continue

        # 跳过已在 prompt 中存在的函数签名
        if not past_prefix and content.startswith("def "):
            match = re.search(r'def\s+(\w+)\s*\(', prompt)
            if match and match.group(1) in content:
                continue

        # 跳过文档字符串
        if not past_prefix and not in_docstring:
            if content.startswith('"""') or content.startswith("'''"):
                delim = content[:3]
                if content.count(delim) >= 2 and content.endswith(delim):
                    continue
                in_docstring = True
                docstring_delim = delim
                continue

        if in_docstring:
            if docstring_delim in content:
                in_docstring = False
                docstring_delim = None
                continue
            continue

        past_prefix = True
        stripped.append(line)

    if not stripped:
        return ""

    return "\n".join(stripped)


def extract_body_from_full_function(raw_code: str, prompt: str) -> str:
    """兜底方案：当模型重新声明整个函数时，提取函数体。"""
    func_match = re.search(r'def\s+(\w+)\s*\(', prompt)
    if not func_match:
        return ""
    func_name = func_match.group(1)

    # 查找函数定义位置
    func_start = -1
    for i, line in enumerate(raw_code.splitlines()):
        if re.search(r'def\s+' + func_name + r'\s*\(', line.strip()):
            func_start = i
            break
    if func_start == -1:
        return ""

    lines = raw_code.splitlines()
    i = func_start + 1

    # 跳过文档字符串
    if i < len(lines) and lines[i].strip().startswith(('"""', "'''")):
        delim = lines[i].strip()[:3]
        if lines[i].strip().count(delim) >= 2 and lines[i].strip().endswith(delim):
            i += 1
        else:
            i += 1
            while i < len(lines):
                if delim in lines[i].strip():
                    i += 1
                    break
                i += 1

    # 提取函数体（缩进大于函数定义的行）
    def_indent = len(lines[func_start]) - len(lines[func_start].lstrip())
    body_lines = []
    while i < len(lines):
        line = lines[i]
        if line.strip() == "":
            body_lines.append(line)
            i += 1
            continue
        line_indent = len(line) - len(line.lstrip())
        if line_indent <= def_indent and line.strip() != "":
            break
        body_lines.append(line)
        i += 1

    if not body_lines:
        return ""

    return "\n".join(body_lines)


def fix_truncated_code(code: str) -> str:
    """检测并修复截断的代码（如 if/for/while/try 块缺少函数体）。"""
    if not code:
        return code

    # 语法已有效则无需修复
    try:
        ast.parse("def _test_():\n" + code)
        return code
    except SyntaxError:
        pass

    lines = code.splitlines()
    last_line = lines[-1].rstrip()

    # 不完整块：以冒号结尾
    if last_line.endswith(":"):
        stripped = last_line.strip()
        if any(stripped.startswith(kw) for kw in
               ("if ", "elif ", "for ", "while ", "try:", "with ", "else:", "finally:", "def ", "class ")):
            # 若前面已有 return，移除空块
            has_return = any(l.strip().startswith("return ") for l in lines[:-1])
            if has_return:
                return "\n".join(lines[:-1])
            lines.append("        pass")
            return "\n".join(lines)

    # 不完整表达式：行末非语句结束符
    stripped = last_line.strip()
    if stripped and not stripped.endswith((";", ")", "]", "}", "'", '"', "#truncated")):
        truncated_prefixes = ("if ", "elif ", "for ", "while ", "with ", "return ", "import ", "from ", "assert ",
                              "raise ", "yield ", "else ", "not ", "and ", "or ")
        if any(stripped.startswith(p) for p in truncated_prefixes) or stripped in ("if", "elif", "else", "for", "while",
                                                                                   "with", "return", "raise", "yield",
                                                                                   "assert", "not", "and", "or", "try",
                                                                                   "finally", "except"):
            has_return = any(l.strip().startswith("return ") for l in lines)
            if has_return:
                lines[-1] = last_line + "  # truncated"
                return "\n".join(lines)
            lines[-1] = last_line + "  # truncated expression"
            lines.append("        pass")
            return "\n".join(lines)

    # 未闭合的括号或反斜杠
    if last_line.endswith(("(", "[", "{", "\\")):
        lines[-1] = last_line.rstrip("\\") + "  # truncated"
    return "\n".join(lines)


def remove_nested_function(code: str, prompt: str) -> str:
    """移除嵌套的同名函数定义。"""
    func_matches = list(re.finditer(r'def\s+(\w+)\s*\(', prompt))
    if not func_matches:
        return code
    func_name = func_matches[-1].group(1)

    lines = code.splitlines()
    if not lines:
        return code

    result = []
    in_nested_func = False
    nested_indent = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        current_indent = len(line) - len(line.lstrip()) if stripped else 0

        # 检测嵌套的同名函数（缩进 > 0）
        if stripped.startswith(f'def {func_name}(') and current_indent > 0:
            in_nested_func = True
            nested_indent = current_indent
            continue

        # 跳过嵌套函数内的行
        if in_nested_func:
            if stripped and current_indent <= nested_indent:
                in_nested_func = False
                result.append(line)
            continue

        result.append(line)

    return '\n'.join(result)


def process_raw_output(raw_code: str, prompt: str) -> str:
    """对模型原始输出应用所有后处理步骤。"""
    code = raw_code

    # 提取 markdown 代码块
    code = extract_code_from_markdown(code)

    # 移除代码围栏
    if code.startswith("```"):
        lines = code.splitlines()
        if lines[-1].strip().startswith("```"):
            code = "\n".join(lines[1:-1])
        else:
            code = "\n".join(lines[1:])
        code = code.strip()

    # 移除冗余的函数重声明/导入/文档字符串
    code = strip_redundant_prefix(code, prompt)

    # 兜底：提取函数体
    if not code:
        extracted = extract_body_from_full_function(raw_code, prompt)
        if extracted:
            code = extracted

    if not code:
        return ""

    # 修复截断代码
    code = fix_truncated_code(code)

    # 智能修复缩进
    code = fix_indentation_smart(code)

    # 移除嵌套函数定义
    code = remove_nested_function(code, prompt)

    return code


def normalize_base_url(base_url: str, api_type: str) -> str:
    """规范化 base_url，确保 API 端点格式正确。

    OpenAI 类型需要 /v1 后缀，Anthropic 类型不需要。
    """
    base_url = base_url.rstrip("/")

    if api_type == "openai":
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
    elif api_type == "anthropic":
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]

    return base_url


def normalize_task_ids(task_ids: list[str]) -> list[str]:
    """将题目 ID 转换为标准格式 'HumanEval/N'。

    示例：
        ["0", "2", "4"] -> ["HumanEval/0", "HumanEval/2", "HumanEval/4"]
        ["HumanEval/0", "HumanEval/2"] -> ["HumanEval/0", "HumanEval/2"]
    """
    normalized = []
    for task_id in task_ids:
        if task_id.isdigit():
            normalized.append(f"HumanEval/{task_id}")
        elif task_id.startswith("HumanEval/"):
            normalized.append(task_id)
        else:
            raise ValueError(f"无效的题目 ID: {task_id}。请使用 'N' 或 'HumanEval/N' 格式。")
    return normalized


def generate_completion(prompt: str, model: str, api_type: str, api_key: str, base_url: str, max_tokens: int = 16384,
                        max_retries: int = 8, parse_retry_count: int = 3) -> tuple[str, float, int, int]:
    """调用模型生成代码补全，返回 (代码, 响应时间, 输入token, 输出token)。

    Args:
        prompt: 代码补全提示词
        model: 模型名称
        api_type: API 类型（anthropic 或 openai）
        api_key: API 密钥
        base_url: API 地址（自动规范化）
        max_tokens: 最大生成 token 数
        max_retries: API 错误重试次数（429、500 等）
        parse_retry_count: 解析失败重试次数
    """
    func_match = re.search(r'def\s+(\w+)\s*\(', prompt)
    func_name = func_match.group(1) if func_match else "the function"

    base_url = normalize_base_url(base_url, api_type)

    # 创建客户端
    if api_type == "anthropic":
        client = Anthropic(api_key=api_key, base_url=base_url)
    elif api_type == "openai":
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        raise ValueError(f"不支持的 api_type: {api_type}")

    # 两种提示策略：详细指令优先，简单回退在后
    prompts = [
        (
            f"{prompt}"
            f"    # Complete the body of {func_name}. Do NOT re-write the function signature, "
            f"docstring, or imports — they are already provided above. "
            f"Only write the implementation code starting here.\n",
            "detailed"
        ),
        (f"{prompt}", "fallback"),
    ]

    for prompt_idx, (completion_prompt, strategy) in enumerate(prompts):
        # 解析失败重试循环
        for parse_attempt in range(parse_retry_count):
            # API 调用重试循环
            for attempt in range(max_retries):
                spinner = None
                start_time = time.time()
                try:
                    spinner = Spinner(f"Calling {model}")
                    spinner.start()

                    if api_type == "anthropic":
                        user_msg: MessageParam = {
                            "role": "user",
                            "content": completion_prompt,
                        }
                        response = client.messages.create(
                            model=model,
                            max_tokens=max_tokens,
                            temperature=0.0,
                            messages=[user_msg]
                        )
                        spinner.stop()
                        response_time = time.time() - start_time
                        # 获取 token 使用量
                        input_tokens = response.usage.input_tokens
                        output_tokens = response.usage.output_tokens
                        # 处理 ThinkingBlock
                        raw_code = ""
                        thinking_code = ""
                        for block in response.content:
                            if hasattr(block, "text"):
                                if type(block).__name__ == "ThinkingBlock":
                                    thinking_code += block.text
                                else:
                                    raw_code += block.text
                        if not raw_code.strip() and thinking_code.strip():
                            raw_code = thinking_code
                        raw_code = raw_code.strip()
                    elif api_type == "openai":
                        user_msg: ChatCompletionUserMessageParam = {
                            "role": "user",
                            "content": completion_prompt,
                        }
                        response = client.chat.completions.create(
                            model=model,
                            max_tokens=max_tokens,
                            temperature=0.0,
                            messages=[user_msg]
                        )
                        spinner.stop()
                        response_time = time.time() - start_time
                        # 获取 token 使用量
                        input_tokens = response.usage.prompt_tokens
                        output_tokens = response.usage.completion_tokens
                        raw_code = response.choices[0].message.content or ""
                        raw_code = raw_code.strip()

                    if not raw_code:
                        print(f"  [DEBUG] 模型返回空内容 (strategy={strategy})")
                        continue

                    print(f"  [DEBUG] 原始输出 ({len(raw_code)} chars, strategy={strategy}): {raw_code[:80]}...")

                    code = process_raw_output(raw_code, prompt)
                    if code:
                        return code, response_time, input_tokens, output_tokens

                    print(f"  [DEBUG] 解析失败 (第 {parse_attempt + 1}/{parse_retry_count} 次, strategy={strategy})")
                    break

                except Exception as e:
                    response_time = time.time() - start_time
                    if spinner:
                        spinner.stop()
                    error_str = str(e)
                    if "429" in error_str:
                        wait = min(2 ** (attempt + 1), 60)
                        print(f"  TPM 限流，等待 {wait}s 后重试 (第{attempt + 1}次)")
                        time.sleep(wait)
                    elif any(code in error_str for code in ["500", "502", "503", "504"]):
                        wait = min(2 ** (attempt + 1), 30)
                        print(f"  服务异常({error_str})，等待 {wait}s 后重试 (第{attempt + 1}次)")
                        time.sleep(wait)
                    else:
                        print(f"调用模型 {model} 出错: {e}")
                        return "", response_time, 0, 0

        # 所有解析重试失败，切换下一个策略
        if prompt_idx < len(prompts) - 1:
            print(f"  [DEBUG] 所有解析重试失败，切换到回退策略")

    print(f"错误: 模型 {model} 所有策略均失败")
    return "", 0.0, 0, 0


def generate_report(model: str, samples_file: str, results_file: str, overall_result: dict, problems: dict,
                    response_times: dict = None, is_partial: bool = False,
                    api_type: str = None, api_key: str = None, base_url: str = None, max_tokens: int = None,
                    round_stats: list = None, task_round_results: dict = None, channel: str = "",
                    token_usage: dict = None):
    """生成 Markdown 格式的测试报告。

    Args:
        model: 模型名称
        samples_file: 样本文件路径
        results_file: 结果文件路径
        overall_result: 总体评估结果（含 pass@1 等）
        problems: 题目定义字典
        response_times: 题目响应时间字典
        is_partial: 是否为部分测试
        api_type: API 类型
        api_key: API 密钥（会被掩码）
        base_url: API 地址
        max_tokens: 最大 token 数
        round_stats: 多轮重试统计
        task_round_results: 题目每轮结果
        channel: 渠道标识
        token_usage: 题目 token 使用量字典 {"task_id": {"input": N, "output": N}}
    """
    task_results = {}
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                r = json.loads(line)
                task_results[r["task_id"]] = r

    task_samples = {}
    if os.path.exists(samples_file):
        with open(samples_file) as f:
            for line in f:
                s = json.loads(line)
                task_samples[s["task_id"]] = s

    total = len(task_results)
    passed = sum(1 for r in task_results.values() if r.get("passed", False))
    pass_rate = overall_result.get("pass@1", 0)

    # 多轮重试时使用最后的累计正确率，并显示对应的 pass@N
    if round_stats and len(round_stats) > 1:
        final_round = round_stats[-1]["round"]
        final_pass_rate = round_stats[-1]["cumulative_rate"]
        pass_label = f"pass@{final_round}"
    else:
        final_pass_rate = pass_rate
        pass_label = "pass@1"

    # 计算响应时间统计
    avg_response_time = 0.0
    min_response_time = 0.0
    max_response_time = 0.0
    total_response_time = 0.0
    std_response_time = 0.0
    if response_times:
        times = list(response_times.values())
        avg_response_time = sum(times) / len(times) if times else 0.0
        min_response_time = min(times) if times else 0.0
        max_response_time = max(times) if times else 0.0
        total_response_time = sum(times) if times else 0.0
        if len(times) > 1:
            variance = sum((t - avg_response_time) ** 2 for t in times) / len(times)
            std_response_time = variance ** 0.5

    # 计算 token 使用量统计
    total_input_tokens = 0
    total_output_tokens = 0
    avg_input_tokens = 0.0
    avg_output_tokens = 0.0
    min_input_tokens = 0
    max_input_tokens = 0
    min_output_tokens = 0
    max_output_tokens = 0
    if token_usage:
        input_tokens_list = [v["input"] for v in token_usage.values()]
        output_tokens_list = [v["output"] for v in token_usage.values()]
        total_input_tokens = sum(input_tokens_list)
        total_output_tokens = sum(output_tokens_list)
        avg_input_tokens = total_input_tokens / len(input_tokens_list) if input_tokens_list else 0.0
        avg_output_tokens = total_output_tokens / len(output_tokens_list) if output_tokens_list else 0.0
        min_input_tokens = min(input_tokens_list) if input_tokens_list else 0
        max_input_tokens = max(input_tokens_list) if input_tokens_list else 0
        min_output_tokens = min(output_tokens_list) if output_tokens_list else 0
        max_output_tokens = max(output_tokens_list) if output_tokens_list else 0

    # 掩码 API Key（显示前 4 位和后 4 位）
    masked_api_key = "—"
    if api_key:
        if len(api_key) <= 8:
            masked_api_key = api_key[:2] + "****" + api_key[-2:] if len(api_key) >= 4 else "****"
        else:
            masked_api_key = api_key[:4] + "****" + api_key[-4:]

    # 报告标题：部分测试时标注题目数量
    if is_partial:
        title = f"# HumanEval 测试报告 — {model}（{total}/{HUMANEVAL_TOTAL_PROBLEMS} 题）"
    else:
        title = f"# HumanEval 测试报告 — {model}"

    lines = [
        title,
        "",
        "## 基本信息",
        "",
        "| 项目 | 值 |",
        "|------|-----|",
        f"| 测试时间 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |",
        f"| 模型 | {model} |",
    ]

    # 渠道信息（如有）
    if channel:
        lines.append(f"| 渠道 | {channel} |")

    lines.extend([
        f"| 测试题目数 | {total} |",
        f"| 通过题目数 | {passed} |",
        f"| 未通过题目数 | {total - passed} |",
        f"| {pass_label} | {final_pass_rate:.1%} |",
    ])

    # 多轮重试结果表格
    if round_stats and len(round_stats) > 1:
        lines.extend([
            "",
            "## 多轮重试结果",
            "",
            "| 轮次 | 本轮题目数 | 本轮正确 | 累计正确率 |",
            "|------|-----------|---------|-----------|",
        ])
        for rs in round_stats:
            rate_str = f"{rs['cumulative_rate']:.1%} (pass@{rs['round']})"
            lines.append(f"| {rs['round']} | {rs['attempted']} | {rs['passed']} | {rate_str} |")
        final_passed = sum(1 for r in task_results.values() if r.get("passed", False))
        lines.append(f"")
        lines.append(f"**最终结果：** {final_passed}/{total} 通过 ({round_stats[-1]['cumulative_rate']:.1%})")

    lines.extend([
        "",
        "## 模型/API 配置",
        "",
        "| 项目 | 值 |",
        "|------|-----|",
        f"| 模型 | {model} |",
        f"| API 类型 | {api_type or '—'} |",
        f"| API 地址 | {base_url or '—'} |",
        f"| API Key | {masked_api_key} |",
        f"| 最大 Token | {max_tokens or '—'} |",
        "",
        "## 响应时间统计",
        "",
        "| 项目 | 值 |",
        "|------|-----|",
        f"| 平均响应时间 | {avg_response_time:.2f}s |",
        f"| 最快响应 | {min_response_time:.2f}s |",
        f"| 最慢响应 | {max_response_time:.2f}s |",
        f"| 响应时间标准差 | {std_response_time:.2f}s |",
        f"| 总耗时 | {total_response_time:.2f}s |",
        "",
        "## Token 使用量统计",
        "",
        "| 项目 | 值 |",
        "|------|-----|",
        f"| 总输入 Token | {total_input_tokens:,} |",
        f"| 总输出 Token | {total_output_tokens:,} |",
        f"| 总 Token | {total_input_tokens + total_output_tokens:,} |",
        f"| 平均输入 Token | {avg_input_tokens:.1f} |",
        f"| 平均输出 Token | {avg_output_tokens:.1f} |",
        f"| 最少输入 Token | {min_input_tokens:,} |",
        f"| 最多输入 Token | {max_input_tokens:,} |",
        f"| 最少输出 Token | {min_output_tokens:,} |",
        f"| 最多输出 Token | {max_output_tokens:,} |",
        "",
        "## 逐题结果",
        "",
        "| 题目 | 函数名 | 是否通过 | 响应时间 | 输入Token | 输出Token | 生成代码（前3行） |",
        "|------|--------|----------|----------|-----------|-----------|-------------------|",
    ])

    # 逐题结果表格
    for task_id, r in task_results.items():
        func_match = re.search(r'def\s+(\w+)\s*\(', problems[task_id]["prompt"])
        func_name = func_match.group(1) if func_match else "—"
        is_passed = "通过" if r.get("passed", False) else "未通过"

        # 多轮重试时显示通过的轮次
        if task_round_results and task_id in task_round_results:
            round_results = task_round_results[task_id]
            passed_round = next((rr["round"] for rr in round_results if rr["passed"]), None)
            if passed_round:
                is_passed = f"通过 (第{passed_round}轮)"

        resp_time = response_times.get(task_id, 0.0) if response_times else 0.0
        # 获取 token 使用量（最后一次）
        task_tokens = token_usage.get(task_id, {"input": 0, "output": 0}) if token_usage else {"input": 0, "output": 0}
        input_tok = task_tokens["input"]
        output_tok = task_tokens["output"]
        completion = task_samples.get(task_id, {}).get("completion", "")
        comp_lines = [l.strip() for l in completion.splitlines() if l.strip()][:3]
        comp_preview = "<br>".join(comp_lines) if comp_lines else "（空）"
        lines.append(f"| {task_id} | {func_name} | {is_passed} | {resp_time:.2f}s | {input_tok} | {output_tok} | {comp_preview} |")

    lines.append("")
    lines.append("## 未通过题目详情")
    lines.append("")

    failed_tasks = [tid for tid, r in task_results.items() if not r.get("passed", False)]
    if not failed_tasks:
        lines.append("所有题目均通过，无失败详情。")
    else:
        for task_id in failed_tasks:
            prompt = problems[task_id]["prompt"]
            func_match = re.search(r'def\s+(\w+)\s*\(', prompt)
            func_name = func_match.group(1) if func_match else "—"

            # 多轮重试时展示每轮结果
            if task_round_results and task_id in task_round_results:
                round_results = task_round_results[task_id]
                lines.append(f"### {task_id} — `{func_name}`")
                lines.append("")

                for rr in round_results:
                    round_num = rr["round"]
                    passed = rr["passed"]
                    code = rr.get("code", "")
                    resp_time = rr.get("response_time", 0.0)
                    status = "通过" if passed else "未通过"

                    lines.append(f"**第 {round_num} 轮** ({status}, {resp_time:.2f}s)")
                    lines.append("```python")
                    if passed:
                        # 正确：只展示前三行
                        code_lines = [l for l in code.splitlines()][:3]
                        for l in code_lines:
                            lines.append(l)
                        if len(code.splitlines()) > 3:
                            lines.append("# ... (仅展示前三行)")
                    else:
                        # 错误：展示完整代码
                        for l in code.splitlines():
                            lines.append(l)
                    lines.append("```")
                    lines.append("")
            else:
                # 单轮测试，保持原有格式
                completion = task_samples.get(task_id, {}).get("completion", "")
                lines.append(f"### {task_id} — `{func_name}`")
                lines.append("")
                lines.append("**生成的补全代码：**")
                lines.append("```python")
                for l in completion.splitlines():
                    lines.append(l)
                lines.append("```")
                lines.append("")
                lines.append("**完整拼接代码（prompt + completion）：**")
                lines.append("```python")
                full_code = prompt + completion
                for l in full_code.splitlines():
                    lines.append(l)
                lines.append("```")
                lines.append("")

    report_text = "\n".join(lines)

    # 文件名：将 _samples_ 替换为 _report_，保留时间戳
    # 格式：
    #   {model}_samples_{timestamp}.jsonl -> {model}_report_{timestamp}.md
    #   {model}_samples_{N}of164_{timestamp}.jsonl -> {model}_report_{N}of164_{timestamp}.md
    #   {model}_samples_retry{N}_{timestamp}.jsonl -> {model}_report_retry{N}_{timestamp}.md
    #   {model}_samples_{N}of164_retry{N}_{timestamp}.jsonl -> {model}_report_{N}of164_retry{N}_{timestamp}.md
    #   {model}_samples_run{N}_{timestamp}.jsonl -> {model}_report_run{N}_{timestamp}.md
    #   {model}_samples_{N}of164_retry{N}_run{N}_{timestamp}.jsonl -> {model}_report_{N}of164_retry{N}_run{N}_{timestamp}.md
    report_file = re.sub(r'_samples_((\d+of\d+_)?(retry\d+_)?(run\d+_)?)(\d{14})\.jsonl$', r'_report_\1\5.md',
                         samples_file)
    if report_file == samples_file:
        # 兜底处理
        report_file = samples_file.replace("_samples_", "_report_").replace(".jsonl", ".md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  报告已生成: {report_file}")
    return report_file


def run_human_eval(model: str, output_file: str, api_type: str, api_key: str, base_url: str,
                   max_tokens: int = 16384, k=None, task_ids: list[str] = None, retry_rounds: int = 1,
                   channel: str = ""):
    """运行 HumanEval 评估，支持多轮重试。

    Args:
        model: 模型名称
        output_file: 输出文件路径
        api_type: API 类型（anthropic 或 openai）
        api_key: API 密钥
        base_url: API 地址
        max_tokens: 最大生成 token 数
        k: pass@k 计算的 k 值列表
        task_ids: 指定测试的题目 ID（None 表示全部）
        retry_rounds: 重试轮数（默认 1，不重试）
        channel: 渠道标识
    """
    if k is None:
        k = [1]
    all_problems = read_problems()

    # 筛选要测试的题目
    if task_ids:
        problems = {tid: all_problems[tid] for tid in task_ids if tid in all_problems}
    else:
        problems = all_problems

    total_problems = len(problems)

    # 多轮重试数据结构
    task_round_results = {}  # {task_id: [{"round": 1, "passed": False, "code": "...", "response_time": 2.5}, ...]}
    round_stats = []  # [{"round": 1, "attempted": 164, "passed": 143, "cumulative_rate": 0.872}, ...]
    token_usage = {}  # {task_id: {"input": N, "output": N}}

    # 初始题目集：全部题目
    failed_task_ids = list(problems.keys())

    print(f"Running HumanEval for model: {model} ({total_problems} problems, {retry_rounds} round(s))")

    for round_num in range(1, retry_rounds + 1):
        if not failed_task_ids:
            print(f"  All problems passed, skipping remaining rounds")
            break

        print(f"\n=== Round {round_num}/{retry_rounds} ({len(failed_task_ids)} problems) ===")

        round_results = {}  # {task_id: {"passed": bool, "code": str, "response_time": float}}

        # 第一步：生成所有代码（不显示进度条）
        for task_id in failed_task_ids:
            # 打印当前时间
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}] Generating {task_id}...")

            problem = problems[task_id]
            prompt = problem["prompt"]
            completion, resp_time, input_tokens, output_tokens = generate_completion(prompt, model, api_type, api_key, base_url,
                                                        max_tokens=max_tokens)

            round_results[task_id] = {
                "round": round_num,
                "passed": False,  # 待测试
                "code": completion,
                "response_time": resp_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }

            time.sleep(1)

        # 第二步：批量执行代码验证（显示进度条）
        print("Running test suites...")
        pbar = tqdm(failed_task_ids, desc="Testing", unit="problem")

        for task_id in pbar:
            problem = problems[task_id]
            prompt = problem["prompt"]
            completion = round_results[task_id]["code"]

            # 运行单题测试判断是否通过
            passed = test_single_problem(prompt, completion, problem)
            round_results[task_id]["passed"] = passed

            # 更新 task_round_results
            if task_id not in task_round_results:
                task_round_results[task_id] = []
            task_round_results[task_id].append(round_results[task_id])

            # 更新进度条后缀信息
            status = "OK" if passed else "FAIL"
            pbar.set_postfix_str(f"{task_id} -> {status}")

        pbar.close()

        # 统计本轮结果
        round_passed = sum(1 for r in round_results.values() if r["passed"])
        cumulative_passed = sum(1 for tid, rounds in task_round_results.items()
                                if any(r["passed"] for r in rounds))
        cumulative_rate = cumulative_passed / total_problems

        round_stats.append({
            "round": round_num,
            "attempted": len(failed_task_ids),
            "passed": round_passed,
            "cumulative_rate": cumulative_rate
        })

        print(f"  Round {round_num}: {round_passed}/{len(failed_task_ids)} passed")
        print(f"  Cumulative: {cumulative_passed}/{total_problems} ({cumulative_rate:.1%})")

        # 收集失败题目，准备下一轮
        failed_task_ids = [tid for tid, r in round_results.items() if not r["passed"]]

    # 构建最终 samples（使用最后通过的代码，或最后生成的代码）
    samples = []
    response_times = {}
    for task_id in problems.keys():
        if task_id in task_round_results:
            rounds = task_round_results[task_id]
            # 找到通过的代码，或使用最后一轮的代码
            passed_round = next((r for r in rounds if r["passed"]), None)
            if passed_round:
                code = passed_round["code"]
                resp_time = passed_round["response_time"]
                in_tok = passed_round.get("input_tokens", 0)
                out_tok = passed_round.get("output_tokens", 0)
            else:
                code = rounds[-1]["code"]
                resp_time = rounds[-1]["response_time"]
                in_tok = rounds[-1].get("input_tokens", 0)
                out_tok = rounds[-1].get("output_tokens", 0)
        else:
            code = ""
            resp_time = 0.0
            in_tok = 0
            out_tok = 0
        samples.append({"task_id": task_id, "completion": code})
        response_times[task_id] = resp_time
        token_usage[task_id] = {"input": in_tok, "output": out_tok}

    # 保存生成结果
    write_jsonl(output_file, samples)
    print(f"Results saved to {output_file}")

    # 运行评测
    ignore_incomplete = task_ids is not None
    result = evaluate_functional_correctness(
        sample_file=output_file,
        k=k,
        n_workers=4,
        timeout=3.0,
        ignore_incomplete=ignore_incomplete
    )
    print(f"Model: {model}, Results: {result}")

    # 生成测试报告
    results_file = output_file + "_results.jsonl"
    is_partial = task_ids is not None
    generate_report(model, output_file, results_file, result, problems, response_times,
                    is_partial=is_partial, api_type=api_type, api_key=api_key,
                    base_url=base_url, max_tokens=max_tokens,
                    round_stats=round_stats if retry_rounds > 1 else None,
                    task_round_results=task_round_results if retry_rounds > 1 else None,
                    channel=channel, token_usage=token_usage)

    return result


def test_single_problem(prompt: str, completion: str, problem: dict = None) -> bool:
    """测试单个题目，判断补全代码是否通过。

    Args:
        prompt: 题目提示词（函数签名 + 文档字符串）
        completion: 生成的代码补全
        problem: 题目定义（可选，不传则根据 prompt 查找）

    Returns:
        True 表示通过，False 表示失败
    """
    if not completion:
        return False

    # 语法检查
    try:
        ast.parse(prompt + completion)
    except SyntaxError:
        return False

    # 使用 human_eval 评测
    try:
        from human_eval.execution import check_correctness

        if problem is None:
            all_problems = read_problems()
            for tid, p in all_problems.items():
                if p["prompt"] == prompt:
                    problem = p
                    break

        if problem is None:
            return True

        result = check_correctness(problem, completion, timeout=3.0)
        return result.get("passed", False)
    except (ImportError, KeyError, TypeError, RuntimeError):
        return False


def main():
    parser = argparse.ArgumentParser(description="HumanEval 模型评估工具")
    parser.add_argument("--models", nargs="+", required=True, help="要评估的模型列表")
    parser.add_argument("--api-type", default="anthropic", help="API 类型: anthropic 或 openai")
    parser.add_argument("--api-key", help="API 密钥（也可通过环境变量 ANTHROPIC_API_KEY 或 OPENAI_API_KEY 设置）")
    parser.add_argument("--base-url", required=True, help="API 服务地址")
    parser.add_argument("--max-tokens", type=int, default=16384, help="每次生成的最大 token 数")
    parser.add_argument("--output-dir", default="results", help="结果输出目录")

    # 题目选择参数（优先级从高到低）
    parser.add_argument("--tasks", nargs="+",
                        help="指定测试题目，支持题号或全名，如 --tasks 0 2 4 或 --tasks HumanEval/0 HumanEval/2")
    parser.add_argument("--num-problems", type=int, help="测试前 N 道题目")
    parser.add_argument("--quick", action="store_true", help="快速测试模式（固定 5 道题）")

    # 多轮重试参数
    parser.add_argument("--retry-rounds", type=int, default=1,
                        help="错误重试轮数，如 3 表示最多执行 3 轮，记录 pass@1/2/3")

    # 重复执行参数
    parser.add_argument("--repeat", type=int, default=1,
                        help="重复执行测试的次数，如 2 表示执行 2 次完整测试，生成 2 份报告")

    # 渠道参数
    parser.add_argument("--channel", default="", help="模型渠道标识，如 'volcengine'、'硅基流动'，用于区分不同供应商")

    args = parser.parse_args()

    # API Key 支持环境变量
    api_key = args.api_key
    if not api_key:
        if args.api_type == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("API 密钥未提供。请通过 --api-key 参数或环境变量设置。")

    # 确定要测试的题目
    task_ids = None
    if args.tasks:
        task_ids = normalize_task_ids(args.tasks)
    elif args.num_problems:
        all_problems = read_problems()
        task_ids = list(all_problems.keys())[:args.num_problems]
    elif args.quick:
        task_ids = QUICK_TEST_TASKS

    Path(args.output_dir).mkdir(exist_ok=True)

    # 渠道标识处理：用于文件命名（中文需保留）
    channel = args.channel.strip() if args.channel else ""
    channel_safe = channel.replace("/", "_").replace("\\", "_") if channel else ""

    # 重复执行测试
    all_results = {}  # {model_channel: [result1, result2, ...]}
    for repeat_num in range(1, args.repeat + 1):
        if args.repeat > 1:
            print(f"\n{'=' * 50}")
            print(f"Repeat {repeat_num}/{args.repeat}")
            print(f"{'=' * 50}")

        # 每次执行使用新的时间戳
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        for model in args.models:
            # 文件名：模型名 + 渠道（如有） + 题目数量 + retry + run + 时间戳
            model_safe = model.replace('/', '_')
            parts = [model_safe]
            if channel_safe:
                parts.append(channel_safe)
            parts.append("samples")
            if task_ids:
                parts.append(f"{len(task_ids)}of164")
            if args.retry_rounds > 1:
                parts.append(f"retry{args.retry_rounds}")
            if args.repeat > 1:
                parts.append(f"run{repeat_num}")
            parts.append(timestamp)
            output_file = os.path.join(args.output_dir, "_".join(parts) + ".jsonl")

            result = run_human_eval(model, output_file, args.api_type, api_key, args.base_url,
                                    max_tokens=args.max_tokens, k=[1], task_ids=task_ids,
                                    retry_rounds=args.retry_rounds, channel=channel)

            # 结果 key：模型名 + 渠道（如有）
            result_key = f"{model}_{channel}" if channel else model
            if result_key not in all_results:
                all_results[result_key] = []
            all_results[result_key].append({
                "run": repeat_num,
                "timestamp": timestamp,
                "result": result
            })

    # 保存汇总结果
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    summary_file = os.path.join(args.output_dir, f"summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved. Summary: {summary_file}")

    # 打印对比
    print("\n=== Final Comparison ===")
    for model, runs in all_results.items():
        if args.repeat > 1:
            # 多次执行时显示每次的结果
            rates = [run["result"].get('pass@1', 0) for run in runs]
            avg_rate = sum(rates) / len(rates) if rates else 0
            print(f"{model}: pass@1 = {rates} (avg: {avg_rate:.3f})")
        else:
            rate = runs[0]["result"].get('pass@1', 0)
            print(f"{model}: pass@1 = {rate:.3f}")


if __name__ == "__main__":
    main()
