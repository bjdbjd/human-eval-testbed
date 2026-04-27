"""
HumanEval 模型评估工具：评估大语言模型的代码生成能力。

用途：
    - 全面评估模型的 pass@1 分数（164 题）
    - 快速验证 API 配置（--quick 模式，5 题）
    - 自定义测试范围（--tasks 或 --num-problems）
    - 生成详细的评测报告（逐题结果、生成代码）

使用示例：
    # 全量测试（164 题）
    python evaluate_models.py --api-type openai --api-key KEY --base-url URL --models gpt-4o

    # 快速测试（5 题）
    python evaluate_models.py --api-type openai --api-key KEY --base-url URL --models gpt-4o --quick

    # 测试前 10 题
    python evaluate_models.py --api-type openai --api-key KEY --base-url URL --models gpt-4o --num-problems 10

    # 指定具体题目（支持题号或全名）
    python evaluate_models.py --api-type openai --api-key KEY --base-url URL --models gpt-4o --tasks 0 2 4
    python evaluate_models.py --api-type openai --api-key KEY --base-url URL --models gpt-4o --tasks HumanEval/0 HumanEval/2

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
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from anthropic.types import MessageParam
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam

# 快速测试默认题目集
QUICK_TEST_TASKS = ["HumanEval/0", "HumanEval/2", "HumanEval/4", "HumanEval/10", "HumanEval/21"]


class Spinner:
    """命令行等待动画"""

    def __init__(self, message="Waiting"):
        self.message = message
        self.spinning = False
        self.thread = None
        self.chars = "|/-\\"

    def _spin(self):
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
        self.spinning = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()

    def stop(self):
        self.spinning = False
        if self.thread:
            self.thread.join()


def fix_indentation(code: str) -> str:
    """Fix indentation to ensure code starts with 4 spaces while preserving relative indentation.

    This function handles cases where model output has inconsistent indentation at the top level.
    It detects the intended indentation pattern and normalizes it.
    """
    if not code:
        return code

    lines = code.splitlines()
    if not lines:
        return code

    # Find all non-empty lines and their indentation
    indents = []
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            indents.append(indent)

    if not indents:
        return code

    min_indent = min(indents)

    # Detect if there are multiple distinct indentation levels at "top level"
    # Top level statements should all have the same indentation (4 spaces in function body)
    # We look for the most common indentation among lines that look like top-level statements
    top_level_indents = []
    prev_indent = None
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        # A line is likely top-level if:
        # 1. It's the first non-empty line, OR
        # 2. Its indent is <= previous line's indent (dedent), OR
        # 3. Previous line was empty
        if prev_indent is None or indent <= prev_indent:
            top_level_indents.append(indent)
        prev_indent = indent

    if top_level_indents:
        # Use the most common top-level indent as reference
        from collections import Counter
        indent_counts = Counter(top_level_indents)
        most_common_indent = indent_counts.most_common(1)[0][0]
        target_indent = 4  # Function body should start at 4 spaces
        offset = target_indent - most_common_indent
    else:
        offset = 4 - min_indent

    # Apply offset to all lines
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
    """Smart indentation fix based on Python block structure.

    Strategy:
    1. First try simple fix_indentation
    2. If syntax error, re-indent based on Python block structure (lines ending with :)
    3. Track dedent keywords (else, elif, except, finally)
    """
    if not code:
        return code

    # First try the simple fix
    fixed = fix_indentation(code)

    # Check if it's syntactically valid
    try:
        ast.parse("def _test_():\n" + fixed)
        return fixed
    except SyntaxError:
        pass

    # If not valid, try smart re-indent based on block structure
    lines = code.splitlines()
    if not lines:
        return code

    # Strategy: Re-indent based on Python block structure
    fixed_lines = []
    indent_stack = [4]  # Stack of indentation levels, start at 4 (function body)

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            fixed_lines.append("")
            continue

        current_indent = indent_stack[-1]

        # Handle dedent keywords
        if stripped.startswith(("else:", "elif ", "except:", "finally:")):
            # These should dedent to match the corresponding if/try
            if len(indent_stack) > 1:
                indent_stack.pop()
                current_indent = indent_stack[-1]

        # Handle comments - they should have same indent as next non-comment line
        if stripped.startswith("#"):
            # Look ahead to find next non-comment, non-empty line
            next_indent = current_indent
            for j in range(i + 1, len(lines)):
                next_stripped = lines[j].strip()
                if next_stripped and not next_stripped.startswith("#"):
                    # Use the indent that the next line will have
                    next_indent = indent_stack[-1]
                    break
            fixed_lines.append(" " * next_indent + stripped)
            continue

        # Add the line with current indent
        fixed_lines.append(" " * current_indent + stripped)

        # Update indent stack for next line
        if stripped.endswith(":"):
            # This line starts a new block
            indent_stack.append(current_indent + 4)
        elif stripped.startswith(("return ", "break", "continue", "raise ", "pass")):
            # These typically end a block, pop the indent for next statement
            # But only if we're not at the base level
            if len(indent_stack) > 1:
                indent_stack.pop()

    result = "\n".join(fixed_lines)

    # Verify the fix
    try:
        ast.parse("def _test_():\n" + result)
        return result
    except SyntaxError:
        # If still failing, try a more aggressive approach
        # Just normalize all lines to start at 4 spaces, preserving relative indents
        return fix_indentation_aggressive(code)


def fix_indentation_aggressive(code: str) -> str:
    """Aggressive indentation fix: normalize to multiples of 4, starting at 4."""
    if not code:
        return code

    lines = code.splitlines()
    if not lines:
        return code

    # Find all unique indent levels
    indent_levels = set()
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            indent_levels.add(indent)

    if not indent_levels:
        return code

    # Sort indent levels and map to multiples of 4
    sorted_levels = sorted(indent_levels)
    indent_map = {}
    for i, level in enumerate(sorted_levels):
        indent_map[level] = 4 + i * 4  # Start at 4, increment by 4

    # Apply mapping
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
    """Extract code from markdown text that may contain ```python blocks.

    Handles cases where model outputs explanation text with code blocks inside.
    """
    if not text:
        return text

    # Check if text contains ```python blocks
    python_block_pattern = r'```python\s*\n(.*?)```'
    matches = list(re.finditer(python_block_pattern, text, re.DOTALL | re.IGNORECASE))

    if matches:
        # Extract code from all python blocks and join them
        code_parts = [match.group(1).strip() for match in matches]
        return "\n\n".join(code_parts)

    # No python blocks found, return original text
    return text


def strip_redundant_prefix(completion: str, prompt: str) -> str:
    """Remove parts of the completion that redundantly re-declare what's already in the prompt."""
    lines = completion.splitlines()
    stripped = []

    in_docstring = False
    docstring_delim = None
    past_prefix = False

    for i, line in enumerate(lines):
        content = line.strip()

        if not past_prefix and content == "":
            continue

        if not past_prefix and content.startswith(("import ", "from ")):
            if content in prompt:
                continue

        if not past_prefix and content.startswith("def "):
            match = re.search(r'def\s+(\w+)\s*\(', prompt)
            if match and match.group(1) in content:
                continue

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
    """Fallback: when the model re-declares the whole function, extract just the body."""
    func_match = re.search(r'def\s+(\w+)\s*\(', prompt)
    if not func_match:
        return ""
    func_name = func_match.group(1)

    func_start = -1
    for i, line in enumerate(raw_code.splitlines()):
        if re.search(r'def\s+' + func_name + r'\s*\(', line.strip()):
            func_start = i
            break
    if func_start == -1:
        return ""

    lines = raw_code.splitlines()
    i = func_start + 1
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
    """Detect and fix truncated code (e.g., if/for/while/try blocks with no body)."""
    if not code:
        return code

    # If code is already syntactically valid, no fix needed
    try:
        ast.parse("def _test_():\n" + code)
        return code
    except SyntaxError:
        pass

    lines = code.splitlines()
    last_line = lines[-1].rstrip()

    # Incomplete block ending with ':'
    if last_line.endswith(":"):
        stripped = last_line.strip()
        if any(stripped.startswith(kw) for kw in
               ("if ", "elif ", "for ", "while ", "try:", "with ", "else:", "finally:", "def ", "class ")):
            # Don't add pass if there's already a return/break/continue in the block
            has_return = any(l.strip().startswith("return ") for l in lines[:-1])
            if has_return:
                return "\n".join(lines[:-1])
            lines.append("        pass")
            return "\n".join(lines)

    # Incomplete expressions — line ends mid-statement (e.g. "if not", "return", "for x in")
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

    # Open bracket / backslash
    if last_line.endswith(("(", "[", "{", "\\")):
        lines[-1] = last_line.rstrip("\\") + "  # truncated"
    return "\n".join(lines)


def remove_nested_function(code: str, prompt: str) -> str:
    """Remove nested function definition with same name as the target function."""
    # Find all function definitions in prompt, use the last one (the one being completed)
    func_matches = list(re.finditer(r'def\s+(\w+)\s*\(', prompt))
    if not func_matches:
        return code
    func_name = func_matches[-1].group(1)  # Use the last function (the one being completed)

    lines = code.splitlines()
    if not lines:
        return code

    result = []
    in_nested_func = False
    nested_indent = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        current_indent = len(line) - len(line.lstrip()) if stripped else 0

        # 检测嵌套的同名函数定义（缩进 > 0 说明是嵌套的）
        if stripped.startswith(f'def {func_name}(') and current_indent > 0:
            in_nested_func = True
            nested_indent = current_indent
            continue

        # 如果在嵌套函数内，跳过直到缩进回到嵌套函数之前
        if in_nested_func:
            if stripped and current_indent <= nested_indent:
                in_nested_func = False
                result.append(line)
            continue

        result.append(line)

    return '\n'.join(result)


def process_raw_output(raw_code: str, prompt: str) -> str:
    """Apply all post-processing steps to raw model output."""
    code = raw_code

    # Extract code from markdown text (handles ```python blocks inside explanation)
    code = extract_code_from_markdown(code)

    # Strip markdown code fences (if the whole output is wrapped in ```)
    if code.startswith("```"):
        lines = code.splitlines()
        if lines[-1].strip().startswith("```"):
            code = "\n".join(lines[1:-1])
        else:
            code = "\n".join(lines[1:])
        code = code.strip()

    # Remove redundant function re-declarations / imports / docstrings
    code = strip_redundant_prefix(code, prompt)

    # Fallback: extract body if model re-declares the whole function
    if not code:
        extracted = extract_body_from_full_function(raw_code, prompt)
        if extracted:
            code = extracted

    if not code:
        return ""

    # Fix truncated code
    code = fix_truncated_code(code)

    # Fix indentation (smart fix that handles inconsistent indentation jumps)
    code = fix_indentation_smart(code)

    # Remove nested function definitions (model sometimes defines same function inside)
    code = remove_nested_function(code, prompt)

    return code


def normalize_base_url(base_url: str, api_type: str) -> str:
    """Normalize base_url to ensure correct API endpoint format.

    Handles cases where user provides URL with or without /v1 suffix.
    - OpenAI: needs /v1 suffix (e.g., https://api.example.com/v1)
    - Anthropic: typically no /v1 suffix (e.g., https://api.anthropic.com)
    """
    base_url = base_url.rstrip("/")

    if api_type == "openai":
        # OpenAI-compatible APIs need /v1
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
    elif api_type == "anthropic":
        # Anthropic APIs typically don't use /v1
        # But some proxies might add it, so we handle both
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]  # Remove /v1

    return base_url


def normalize_task_ids(task_ids: list[str]) -> list[str]:
    """Convert task IDs to standard format 'HumanEval/N'.

    Examples:
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
            raise ValueError(f"Invalid task ID: {task_id}. Use 'N' or 'HumanEval/N' format.")
    return normalized


def generate_completion(prompt: str, model: str, api_type: str, api_key: str, base_url: str, max_tokens: int = 16384,
                        max_retries: int = 8, parse_retry_count: int = 3) -> tuple[str, float]:
    """Generate completion and return (code, response_time_seconds).

    Args:
        prompt: The code completion prompt
        model: Model name to use
        api_type: API type, e.g. "anthropic" or "openai"
        api_key: API key
        base_url: API base URL (will be normalized automatically)
        max_tokens: Maximum tokens to generate
        max_retries: Maximum retry attempts for API errors (429, 500, etc.)
        parse_retry_count: Number of retries when parsing fails (default 3)
    """
    func_match = re.search(r'def\s+(\w+)\s*\(', prompt)
    func_name = func_match.group(1) if func_match else "the function"

    # Normalize base_url for the API type
    base_url = normalize_base_url(base_url, api_type)

    # Create client based on api_type
    if api_type == "anthropic":
        client = Anthropic(api_key=api_key, base_url=base_url)
    elif api_type == "openai":
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        raise ValueError(f"Unsupported api_type: {api_type}")

    # Two prompt strategies: detailed instruction first, simpler fallback second
    prompts = [
        (
            f"{prompt}"
            f"    # Complete the body of {func_name}. Do NOT re-write the function signature, "
            f"docstring, or imports — they are already provided above. "
            f"Only write the implementation code starting here.\n",
            "detailed"
        ),
        (f"{prompt}", "fallback"),  # Fallback: just the raw prompt, no instruction
    ]

    for prompt_idx, (completion_prompt, strategy) in enumerate(prompts):
        # 解析失败重试循环
        for parse_attempt in range(parse_retry_count):
            # API 调用重试循环（处理 429/500 等错误）
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
                        # Handle ThinkingBlock — some models return content blocks of different types
                        raw_code = ""
                        thinking_code = ""
                        for block in response.content:
                            if hasattr(block, "text"):
                                if type(block).__name__ == "ThinkingBlock":
                                    thinking_code += block.text
                                else:
                                    raw_code += block.text
                        # Fallback: if all TextBlocks are empty but ThinkingBlock has content
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
                        raw_code = response.choices[0].message.content or ""
                        raw_code = raw_code.strip()

                    if not raw_code:
                        print(f"  [DEBUG] model returned empty content (strategy={strategy})")
                        continue  # API 返回空，继续 API 重试

                    print(f"  [DEBUG] raw output ({len(raw_code)} chars, strategy={strategy}): {raw_code[:80]}...")

                    code = process_raw_output(raw_code, prompt)
                    if code:
                        return code, response_time  # 成功，直接返回

                    # 解析失败
                    print(
                        f"  [DEBUG] parse failed (attempt {parse_attempt + 1}/{parse_retry_count}, strategy={strategy})")
                    break  # 跳出 API 重试循环，进入下一次解析重试

                except Exception as e:
                    response_time = time.time() - start_time
                    if spinner:
                        spinner.stop()
                    error_str = str(e)
                    if "429" in error_str:
                        wait = min(2 ** (attempt + 1), 60)
                        print(f"  TPM限流，等待 {wait}s 后重试 (第{attempt + 1}次)")
                        time.sleep(wait)
                    elif any(code in error_str for code in ["500", "502", "503", "504"]):
                        wait = min(2 ** (attempt + 1), 30)
                        print(f"  服务异常({error_str})，等待 {wait}s 后重试 (第{attempt + 1}次)")
                        time.sleep(wait)
                    else:
                        print(f"Error calling model {model}: {e}")
                        return "", response_time

        # 所有解析重试都失败，切换下一个 strategy
        if prompt_idx < len(prompts) - 1:
            print(f"  [DEBUG] all parse retries failed, switching to fallback strategy")

    print(f"Error: model {model} all strategies failed")
    return "", 0.0


def generate_report(model: str, samples_file: str, results_file: str, overall_result: dict, problems: dict,
                    response_times: dict = None, is_partial: bool = False,
                    api_type: str = None, api_key: str = None, base_url: str = None, max_tokens: int = None):
    """Generate a human-readable markdown test report.

    Args:
        model: Model name
        samples_file: Path to samples JSONL file
        results_file: Path to results JSONL file
        overall_result: Overall evaluation result dict with pass@1 etc.
        problems: Dict of problem definitions
        response_times: Dict mapping task_id to response time in seconds
        is_partial: True if this is a partial test (not all 164 problems)
        api_type: API type used for this test
        api_key: API key (will be masked for security)
        base_url: API base URL
        max_tokens: Maximum tokens setting
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

    # Calculate average response time
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
        # Calculate standard deviation
        if len(times) > 1:
            variance = sum((t - avg_response_time) ** 2 for t in times) / len(times)
            std_response_time = variance ** 0.5

    # Mask API key (show first 4 and last 4 chars, mask the rest)
    masked_api_key = "—"
    if api_key:
        if len(api_key) <= 8:
            masked_api_key = api_key[:2] + "****" + api_key[-2:] if len(api_key) >= 4 else "****"
        else:
            masked_api_key = api_key[:4] + "****" + api_key[-4:]

    # 报告标题：部分测试时标注题目数量
    total_problems = 164  # HumanEval 总题数
    if is_partial:
        title = f"# HumanEval 测试报告 — {model}（{total}/{total_problems} 题）"
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
        f"| 测试题目数 | {total} |",
        f"| 通过题目数 | {passed} |",
        f"| 未通过题目数 | {total - passed} |",
        f"| pass@1 | {pass_rate:.1%} |",
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
        "## 逐题结果",
        "",
        "| 题目 | 函数名 | 是否通过 | 响应时间 | 生成代码（前3行） |",
        "|------|--------|----------|----------|-------------------|",
    ]

    for task_id, r in task_results.items():
        func_match = re.search(r'def\s+(\w+)\s*\(', problems[task_id]["prompt"])
        func_name = func_match.group(1) if func_match else "—"
        is_passed = "通过" if r.get("passed", False) else "未通过"
        resp_time = response_times.get(task_id, 0.0) if response_times else 0.0
        completion = task_samples.get(task_id, {}).get("completion", "")
        comp_lines = [l.strip() for l in completion.splitlines() if l.strip()][:3]
        comp_preview = "<br>".join(comp_lines) if comp_lines else "（空）"
        lines.append(f"| {task_id} | {func_name} | {is_passed} | {resp_time:.2f}s | {comp_preview} |")

    lines.append(f"")
    lines.append(f"## 未通过题目详情")
    lines.append(f"")

    failed_tasks = [tid for tid, r in task_results.items() if not r.get("passed", False)]
    if not failed_tasks:
        lines.append(f"所有题目均通过，无失败详情。")
    else:
        for task_id in failed_tasks:
            prompt = problems[task_id]["prompt"]
            func_match = re.search(r'def\s+(\w+)\s*\(', prompt)
            func_name = func_match.group(1) if func_match else "—"
            completion = task_samples.get(task_id, {}).get("completion", "")

            lines.append(f"### {task_id} — `{func_name}`")
            lines.append(f"")
            lines.append(f"**生成的补全代码：**")
            lines.append(f"```python")
            for l in completion.splitlines():
                lines.append(l)
            lines.append(f"```")
            lines.append(f"")
            lines.append(f"**完整拼接代码（prompt + completion）：**")
            lines.append(f"```python")
            full_code = prompt + completion
            for l in full_code.splitlines():
                lines.append(l)
            lines.append(f"```")
            lines.append(f"")

    report_text = "\n".join(lines)

    # 文件名：将 _samples_ 替换为 _report_，保留时间戳
    # 格式：{model}_samples_{N}of164_{timestamp}.jsonl -> {model}_report_{N}of164_{timestamp}.md
    # 或：{model}_samples_{timestamp}.jsonl -> {model}_report_{timestamp}.md
    report_file = re.sub(r'_samples_(\d+of\d+_)?(\d{14})\.jsonl$', r'_report_\1\2.md', samples_file)
    if report_file == samples_file:
        # 兜底处理
        report_file = samples_file.replace("_samples_", "_report_").replace(".jsonl", ".md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  报告已生成: {report_file}")
    return report_file


def run_human_eval(model: str, output_file: str, api_type: str, api_key: str, base_url: str,
                   max_tokens: int = 16384, k=None, task_ids: list[str] = None):
    if k is None:
        k = [1]
    all_problems = read_problems()

    # 筛选要测试的题目
    if task_ids:
        problems = {tid: all_problems[tid] for tid in task_ids if tid in all_problems}
    else:
        problems = all_problems

    samples = []
    response_times = {}  # task_id -> response_time

    print(f"Running HumanEval for model: {model} ({len(problems)} problems)")
    for i, (task_id, problem) in enumerate(problems.items()):
        prompt = problem["prompt"]
        completion, resp_time = generate_completion(prompt, model, api_type, api_key, base_url,
                                                    max_tokens=max_tokens)
        response_times[task_id] = resp_time
        samples.append({
            "task_id": task_id,
            "completion": completion
        })
        status = "OK" if completion else "FAIL"
        print(f"  [{i + 1}/{len(problems)}] {task_id} -> {status} ({len(completion)} chars, {resp_time:.2f}s)")
        time.sleep(1)

    # 保存生成结果
    write_jsonl(output_file, samples)
    print(f"Results saved to {output_file}")

    # 运行评测
    # 如果只测试部分题目，需要设置 ignore_incomplete=True
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
                    base_url=base_url, max_tokens=max_tokens)

    return result


def main():
    parser = argparse.ArgumentParser(description="HumanEval 模型评估工具")
    parser.add_argument("--models", nargs="+", required=True, help="要评估的模型列表")
    parser.add_argument("--api-type", default="anthropic", help="API 类型: anthropic 或 openai")
    parser.add_argument("--api-key", required=True, help="API 密钥")
    parser.add_argument("--base-url", required=True, help="API 服务地址")
    parser.add_argument("--max-tokens", type=int, default=16384, help="每次生成的最大 token 数")
    parser.add_argument("--output-dir", default="results", help="结果输出目录")

    # 题目选择参数（优先级从高到低）
    parser.add_argument("--tasks", nargs="+",
                        help="指定测试题目，支持题号或全名，如 --tasks 0 2 4 或 --tasks HumanEval/0 HumanEval/2")
    parser.add_argument("--num-problems", type=int, help="测试前 N 道题目")
    parser.add_argument("--quick", action="store_true", help="快速测试模式（固定 5 道题）")

    args = parser.parse_args()

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

    # 时间戳后缀
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    results = {}
    for model in args.models:
        # 文件名：部分测试时添加题目数量标识，末尾加时间戳
        model_safe = model.replace('/', '_')
        if task_ids:
            output_file = os.path.join(args.output_dir, f"{model_safe}_samples_{len(task_ids)}of164_{timestamp}.jsonl")
        else:
            output_file = os.path.join(args.output_dir, f"{model_safe}_samples_{timestamp}.jsonl")
        result = run_human_eval(model, output_file, args.api_type, args.api_key, args.base_url,
                                max_tokens=args.max_tokens, k=[1], task_ids=task_ids)
        results[model] = result

    # 保存汇总结果
    summary_file = os.path.join(args.output_dir, f"summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved. Summary: {summary_file}")

    # 打印对比
    print("\n=== Final Comparison ===")
    for model, res in results.items():
        print(f"{model}: pass@1 = {res.get('pass@1', 0):.3f}")


if __name__ == "__main__":
    main()
