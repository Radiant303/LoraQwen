# =========================
# 合并模型验收测试：FIM 补全 + tool 调用 + think 思考 + persona 聊天
# 位置：test/test_merged.py
#
# 与旧版的差异：
# - 默认加载合并后的 persona 模型（旧版指向纯 FIM 底座，persona/tool/think
#   三项验收根本没测到 persona 权重）；可用 --model 指定任意模型
# - system prompt 统一为线上原版（旧版用了一套更长的"增强版"，训练、测试、
#   线上三方不一致，测试结果没有代表性）
# - 全部用例改为断言式，输出 PASS/FAIL 汇总并以退出码标记结果，
#   报告存档到 test/results/ 供版本间回归对比
# - 解码统一 greedy（do_sample=False），结果可复现
#   （线上 tools 请求用 temperature=0.2，接近确定性；抽样评测每次不可比）
#
# 用法：
#   uv run python test/test_merged.py
#   uv run python test/test_merged.py --model models/fim/SpringNote-Qwen3-1.7B-FIM-V7
#
# strict=False 的用例是新数据才训练的行为（如 /no_think、免工具直答），
# 对旧模型只警告不计失败。
# =========================

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

DEFAULT_MODEL = ROOT / "models" / "persona" / "SpringNote-Qwen3-1.7B-FIM-Persona-V7"

MAX_ITER = 5

# 伪造的笔记根路径（仅用于测试数据的 path 字段）
FAKE_NOTES_BASE = "D:/SpringNote/notes"

# 与线上 MEMORY_TOOL_SYSTEM_PROMPT 原文一致（同 train/build_tools_data.py）
SYSTEM = (
    "你是 SpringNote 的回忆书问答助手。你必须基于用户的历史日报、周报、月报回答问题。\n"
    "你可以自主调用工具检索或读取记录；需要信息时先调用工具，不要让应用预先替你检索。\n"
    "连续追问时结合完整消息历史理解省略指代，例如“什么时候”“这个配置”“刚才说的”等。\n"
    "回答必须只依据工具返回和对话上下文；材料不足时明确说明缺少依据，不要编造事实。\n"
    "最终回答使用自然中文和清晰 Markdown，不要输出工具调用 JSON。"
)


def _fn(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
        "strict": True,
    }


_KW = {
    "keywords": {
        "type": "array",
        "items": {"type": "string", "minLength": 2},
        "minItems": 1,
        "maxItems": 8,
    }
}

_DATE = {"type": "string", "pattern": r"^20\d{2}-(0[1-9]|1[0-2])-([0-2][0-9]|3[0-1])$"}
_WEEK = {"type": "string", "pattern": r"^20\d{2}-W(0[1-9]|[1-4]\d|5[0-3])$"}
_MONTH = {"type": "string", "pattern": r"^20\d{2}-(0[1-9]|1[0-2])$"}

_TOOLS_KW_SUFFIX = (
    " Submit all distinctive keywords in this single call."
    " Every keyword must contain at least two Unicode characters."
)

TOOLS = [
    _fn(
        "get_current_date",
        "Get the current local date, ISO week label, and week number."
        " Use this before resolving relative dates such as today,"
        " yesterday, this week, this month.",
        {},
        [],
    ),
    _fn(
        "keyword_search",
        "Run one global indexed search across SpringNote daily, weekly,"
        " and monthly Markdown records. Use this only when the record type"
        " is unknown or the answer may span multiple types; prefer a scoped"
        " search when the user names daily, weekly, or monthly records."
        " When calling keyword_search, submit all distinctive keywords in"
        " this single call. Every keyword must contain at least two"
        " Unicode characters.",
        {"keywords": _KW["keywords"]},
        ["keywords"],
    ),
    _fn(
        "search_daily_notes",
        "Search only SpringNote daily Markdown notes. Use this instead of"
        " keyword_search when the request is limited to daily notes or"
        " day-level records." + _TOOLS_KW_SUFFIX,
        {"keywords": _KW["keywords"]},
        ["keywords"],
    ),
    _fn(
        "search_weekly_notes",
        "Search only SpringNote weekly Markdown reports. Use this instead"
        " of keyword_search when the request is limited to weekly"
        " reports." + _TOOLS_KW_SUFFIX,
        {"keywords": _KW["keywords"]},
        ["keywords"],
    ),
    _fn(
        "search_monthly_notes",
        "Search only SpringNote monthly Markdown reports. Use this instead"
        " of keyword_search when the request is limited to monthly"
        " reports." + _TOOLS_KW_SUFFIX,
        {"keywords": _KW["keywords"]},
        ["keywords"],
    ),
    _fn(
        "read_daily_note",
        "Read the full daily Markdown note for a specific date.",
        {"date": _DATE},
        ["date"],
    ),
    _fn(
        "read_week_daily_notes",
        "Read all available daily notes in a date range, typically one week.",
        {"startDate": _DATE, "endDate": _DATE},
        ["startDate", "endDate"],
    ),
    _fn(
        "read_weekly_note",
        "Read only the full SpringNote weekly report Markdown for a"
        " specific ISO week. Do not return daily notes.",
        {"week": _WEEK},
        ["week"],
    ),
    _fn(
        "read_month_weekly_notes",
        "Read all available SpringNote weekly report Markdown files whose"
        " ISO weeks overlap a specific calendar month. Return weekly"
        " reports only, not daily notes or the monthly report.",
        {"month": _MONTH},
        ["month"],
    ),
    _fn(
        "read_month_report",
        "Read only the monthly report Markdown for a specific month."
        " Do not return daily notes.",
        {"month": _MONTH},
        ["month"],
    ),
]

# ---------- 结果记录 ----------

RESULTS = []


def check(name, ok, detail="", strict=True):
    level = "PASS" if ok else ("FAIL" if strict else "WARN")
    RESULTS.append({
        "name": name,
        "level": level,
        "strict": strict,
        "detail": detail,
    })
    mark = {"PASS": "✓", "FAIL": "✗", "WARN": "!"}[level]
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail and not ok else ""))


# ---------- 模型 ----------

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=str(DEFAULT_MODEL), help="待测模型目录")
parser.add_argument("--max-new-tokens", type=int, default=256)
parser.add_argument("--tag", default="", help="报告文件名附加标签")
args = parser.parse_args()

MODEL = Path(args.model)
MAX_NEW_TOKENS = args.max_new_tokens

print(f"加载模型: {MODEL}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model.eval()


# =========================
# 通用生成函数（greedy，可复现）
# =========================

def gen_raw(prompt, n=MAX_NEW_TOKENS):
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=n,
            do_sample=False,
        )

    return tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=False,
    )


def generate(messages, tools=None, enable_thinking=False):
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return gen_raw(prompt)


# =========================
# 1. FIM 补全
# =========================

def test_fim():
    prefix = "Graph 微服务的交互流程也跑通了。整体逻辑:"
    suffix = (
        "返回需要调用的工具以及是否需要调用的决策。后续节点判断是否中断，"
        "如果信息不足，就通过 SSE 把提示信息发给调用方，调用方可以继续补充信息；"
        "信息齐全后进入工具执行节点，执行结果以 PCM 语音格式通过 SSE 推送给用户，"
        "最后由 finishNode 发送完成信息。整个交互过程中的消息推送都走 SSE 的 PCM 格式。"
    )
    fim_prompt = (
        "<|fim_prefix|>"
        + prefix
        + "<|fim_suffix|>"
        + suffix
        + "<|fim_middle|>"
    )

    print("\n" + "=" * 70)
    print("【FIM 补全测试】")
    raw = gen_raw(fim_prompt)
    middle = raw.split("<|im_end|>")[0].strip()

    print("=======middle=======")
    print(middle)

    check("FIM 补全非空", len(middle) > 0)
    check(
        "FIM 正常停止（输出 <|im_end|>）",
        "<|im_end|>" in raw,
    )
    leaked = [w for w in ("陈果果", "Radiant303", "463423961", "QQ群") if w in middle]
    check("FIM 不泄漏 persona 信息", not leaked, f"泄漏词: {leaked}")


# =========================
# 2. tool 调用
# =========================

FAKE_DAILY = {
    "2026-07-18": "排查订单支付成功后积分未增加的问题，发现支付回调处理时只更新了订单状态，没有触发积分发放流程，在支付回调方法中增加了积分发放的异步调用。",
    "2026-07-17": "完成 Graph 微服务交互流程联调，各节点按定义顺序执行，工具执行结果以 PCM 语音格式通过 SSE 推送给用户。",
    "2026-07-16": "排查消息通知服务的消息积压，消费者同步执行数据库写入和外部接口调用耗时较长，改为批量处理模式，消费速度提升到原来的三倍。",
    "2026-07-15": "排查商品详情页加载慢的问题，热点商品查询缓存失效导致大量请求同时访问数据库，增加请求控制避免多个请求执行相同查询。",
    "2026-07-14": "调整周报生成逻辑，按 ISO 周聚合日报内容，修复跨月周统计遗漏的问题。",
    "2026-07-11": "完成回忆书工具调用链路验证，keyword_search 一次提交全部关键词，重复调用会被缓存拦截。",
}

FAKE_WEEKLY = {
    "2026-W29": "# 周报 2026-W29\n\n- 完成 Graph 微服务交互流程联调\n- 修复消息通知服务消息积压\n- 调整周报生成逻辑",
    "2026-W28": "# 周报 2026-W28\n\n- 排查支付回调积分未增加问题\n- 优化商品详情页缓存失效问题\n- 完成回忆书工具调用链路验证",
}

FAKE_MONTHLY = {
    "2026-07": "# 月报 2026-07\n\n本月完成：\n- Graph 微服务交互流程跑通\n- 消息通知服务积压问题修复\n- 商品详情页缓存问题修复\n- 回忆书工具调用链路验证完成",
}

CURRENT_DATE = "2026-07-19"
CURRENT_WEEK = "2026-W29"
CURRENT_WEEK_NUMBER = 29


def _matches(text, keywords):
    return any(k in text for k in keywords)


def execute_tool(name, arguments):
    if name == "get_current_date":
        return {
            "date": CURRENT_DATE,
            "isoWeek": CURRENT_WEEK,
            "weekNumber": CURRENT_WEEK_NUMBER,
        }

    if name == "read_daily_note":
        d = arguments.get("date")
        if d in FAKE_DAILY:
            return {"results": [{
                "title": f"日报 {d}",
                "path": f"{FAKE_NOTES_BASE}/daily/{d}.md",
                "snippet": FAKE_DAILY[d],
                "score": 100,
            }]}
        return {"results": []}

    if name == "read_week_daily_notes":
        start = arguments.get("startDate")
        end = arguments.get("endDate")
        results = []
        for d, t in sorted(FAKE_DAILY.items()):
            if start <= d <= end:
                results.append({
                    "title": f"日报 {d}",
                    "path": f"{FAKE_NOTES_BASE}/daily/{d}.md",
                    "snippet": t,
                    "score": 100,
                })
        return {"results": results}

    if name == "read_weekly_note":
        w = arguments.get("week")
        if w in FAKE_WEEKLY:
            return {"results": [{
                "title": f"周报 {w}",
                "path": f"{FAKE_NOTES_BASE}/weekly/{w}.md",
                "snippet": FAKE_WEEKLY[w],
                "score": 120,
            }]}
        return {"results": []}

    if name == "read_month_weekly_notes":
        month = arguments.get("month")
        results = []
        for w, t in FAKE_WEEKLY.items():
            if w.startswith(month):
                results.append({
                    "title": f"周报 {w}",
                    "path": f"{FAKE_NOTES_BASE}/weekly/{w}.md",
                    "snippet": t,
                    "score": 120,
                })
        return {"results": results}

    if name == "read_month_report":
        month = arguments.get("month")
        if month in FAKE_MONTHLY:
            return {"results": [{
                "title": f"月报 {month}",
                "path": f"{FAKE_NOTES_BASE}/monthly/{month}.md",
                "snippet": FAKE_MONTHLY[month],
                "score": 140,
            }]}
        return {"results": []}

    if name in ("keyword_search", "search_daily_notes",
                "search_weekly_notes", "search_monthly_notes"):
        kws = arguments.get("keywords", [])
        results = []
        scope_daily = name in ("keyword_search", "search_daily_notes")
        scope_weekly = name in ("keyword_search", "search_weekly_notes")
        scope_monthly = name in ("keyword_search", "search_monthly_notes")

        if scope_daily:
            for d, t in FAKE_DAILY.items():
                if _matches(t, kws):
                    results.append({
                        "title": f"日报 {d}",
                        "path": f"{FAKE_NOTES_BASE}/daily/{d}.md",
                        "snippet": t,
                        "score": sum(t.count(k) for k in kws),
                    })
        if scope_weekly:
            for w, t in FAKE_WEEKLY.items():
                if _matches(t, kws):
                    results.append({
                        "title": f"周报 {w}",
                        "path": f"{FAKE_NOTES_BASE}/weekly/{w}.md",
                        "snippet": t,
                        "score": sum(t.count(k) for k in kws),
                    })
        if scope_monthly:
            for m, t in FAKE_MONTHLY.items():
                if _matches(t, kws):
                    results.append({
                        "title": f"月报 {m}",
                        "path": f"{FAKE_NOTES_BASE}/monthly/{m}.md",
                        "snippet": t,
                        "score": sum(t.count(k) for k in kws),
                    })
        return {"results": results}

    return {"results": [], "error": f"unknown tool: {name}"}


TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.S,
)


def parse_tool_calls(text):
    calls = []
    for raw in TOOL_CALL_RE.findall(text):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if "name" in obj and "arguments" in obj:
            calls.append(obj)
    return calls


def run_tool_case(name, user_content, history=None,
                  expected_tools=None, expect_in_answer=None,
                  strict=True):
    print("\n" + "-" * 70)
    print(f"[用例] {name}")
    print(f"用户: {user_content}")

    messages = [{"role": "system", "content": SYSTEM}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    all_tool_calls = []
    final_answer = None

    for step in range(MAX_ITER):
        text = generate(messages, tools=TOOLS)
        calls = parse_tool_calls(text)

        if not calls:
            final_answer = text.split("<|im_end|>")[0].strip()
            break

        all_tool_calls.extend(calls)

        messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call_{step + 1}",
                    "type": "function",
                    "function": c,
                }
                for c in calls
            ],
        })

        for c in calls:
            result = execute_tool(c["name"], c["arguments"])
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{step + 1}",
                "content": json.dumps(result, ensure_ascii=False),
            })
    else:
        final_answer = "[达到最大迭代次数，未得到最终回答]"

    actual_tools = [c["name"] for c in all_tool_calls]
    print(f"调用链路: {actual_tools or '(无工具调用，直接回答)'}")
    print(f"最终回答: {final_answer}")

    if expected_tools is not None:
        check(
            f"{name} - 工具调用序列",
            actual_tools == expected_tools,
            f"预期 {expected_tools}，实际 {actual_tools}",
            strict=strict,
        )

    if expect_in_answer is not None and final_answer is not None:
        check(
            f"{name} - 回答内容",
            expect_in_answer in final_answer,
            f"回答中未找到 {expect_in_answer!r}",
            strict=strict,
        )

    return all_tool_calls, final_answer


def test_tools():
    print("\n" + "=" * 70)
    print("【tool 调用测试】")

    run_tool_case(
        "相对日期-昨天",
        "我昨天的日报写了什么？",
        expected_tools=["get_current_date", "read_daily_note"],
        expect_in_answer="积分",
    )

    run_tool_case(
        "相对日期-本周日报",
        "这周我都做了什么？",
        expected_tools=["get_current_date", "read_week_daily_notes"],
    )

    run_tool_case(
        "类型明确-日报搜索",
        "在日报里搜一下缓存",
        expected_tools=["search_daily_notes"],
        expect_in_answer="缓存",
    )

    run_tool_case(
        "类型不明-全局搜索",
        "我有没有记录过 Kafka 相关的事？",
        expected_tools=["keyword_search"],
        expect_in_answer="没有",
    )

    run_tool_case(
        "无结果-诚实说明",
        "我记过健身相关的内容吗？",
        expected_tools=["keyword_search"],
        expect_in_answer="没有",
    )

    run_tool_case(
        "读取缺失-诚实说明",
        "2026-01-01 的日报写了什么？",
        expected_tools=["read_daily_note"],
        expect_in_answer="没有",
    )

    # 追问
    history = []
    _, ans1 = run_tool_case(
        "追问-第一轮",
        "我昨天的日报写了什么？",
        history=history,
    )
    history.append({"role": "user", "content": "我昨天的日报写了什么？"})
    history.append({"role": "assistant", "content": ans1})
    run_tool_case(
        "追问-第二轮（前天）",
        "那前天呢？",
        history=history,
        expected_tools=["read_daily_note"],
        expect_in_answer="Graph",
    )

    # 新行为：免工具直答（新数据才训练，旧模型只警告）
    run_tool_case(
        "免工具-寒暄直答",
        "你好",
        expected_tools=[],
        strict=False,
    )


# =========================
# 3. think 思考
# =========================

def test_thinking():
    print("\n" + "=" * 70)
    print("【think 思考测试】")

    messages = [
        {"role": "system", "content": "你是一个乐于助人的AI助手。"},
        {"role": "user", "content": "9.11 和 9.9 哪个大？"}
    ]
    text = generate(messages, enable_thinking=True)
    answer = text.split("<|im_end|>")[0].strip()

    print("模型输出:")
    print(answer[:500])

    check("思考块出现", "<think>" in answer)
    check("结论正确（9.9 更大）", "9.9" in answer)

    # 新行为：/no_think 软开关（新数据才训练，旧模型只警告）
    messages = [
        {"role": "system", "content": "你是一个乐于助人的AI助手。"},
        {"role": "user", "content": "/no_think 9.11 和 9.9 哪个大？"}
    ]
    text = generate(messages, enable_thinking=True)
    answer = text.split("<|im_end|>")[0].strip()
    think_body = ""
    m = re.search(r"<think>\s*(.*?)\s*</think>", answer, re.S)
    if m:
        think_body = m.group(1)

    print("/no_think 输出:")
    print(answer[:300])
    check(
        "/no_think 不产生思考内容",
        "<think>" not in answer or len(think_body) < 10,
        strict=False,
    )
    check(
        "/no_think 结论正确",
        "9.9 更大" in answer or "9.9更大" in answer,
        strict=False,
    )


# =========================
# 4. persona 聊天
# =========================

def test_persona():
    persona_sys = (
        "你是SpringNote官方AI助手。\n\n"
        "你的职责是帮助用户了解SpringNote、整理知识、处理笔记相关任务。\n\n"
        "回答要求：\n"
        "- 准确\n"
        "- 简洁\n"
        "- 不编造信息\n"
        "- 不知道的信息明确说明"
    )

    # (问题, 期望出现在回答中的字符串, strict)
    cases = [
        ("SpringNote是谁开发的？", "陈果果", True),
        ("作者的GitHub账号是什么？", "Radiant303", True),
        ("SpringNote官方QQ群是多少？", "463423961", True),
        ("SpringNote官网在哪里？", "radiant303.github.io", True),
        ("陈果果毕业于哪里？", "没有公开", True),
        ("陈果果的微信号是多少？", "没有公开", False),  # 新拒答数据
    ]

    print("\n" + "=" * 70)
    print("【persona 聊天测试】")

    for q, expect, strict in cases:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": persona_sys},
                {"role": "user", "content": q},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ans = gen_raw(prompt).split("<|im_end|>")[0].strip()

        print(f"\nQ: {q}")
        print(f"A: {ans}")
        check(
            f"persona - {q}",
            expect in ans,
            f"回答中未找到 {expect!r}",
            strict=strict,
        )


# =========================
# 汇总与存档
# =========================

def summarize():
    n_pass = sum(1 for r in RESULTS if r["level"] == "PASS")
    n_fail = sum(1 for r in RESULTS if r["level"] == "FAIL")
    n_warn = sum(1 for r in RESULTS if r["level"] == "WARN")

    print("\n" + "=" * 70)
    print(f"验收汇总: PASS {n_pass}  FAIL {n_fail}  WARN(新行为待重训) {n_warn}")
    if n_fail:
        print("失败用例:")
        for r in RESULTS:
            if r["level"] == "FAIL":
                print(f"  - {r['name']}: {r['detail']}")

    out_dir = ROOT / "test" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    report = out_dir / f"{ts}_{MODEL.name}{tag}.json"
    report.write_text(
        json.dumps({
            "model": str(MODEL),
            "time": ts,
            "summary": {"pass": n_pass, "fail": n_fail, "warn": n_warn},
            "results": RESULTS,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"报告已存档: {report}")

    return n_fail == 0


if __name__ == "__main__":
    test_fim()
    test_tools()
    test_thinking()
    test_persona()
    ok = summarize()
    sys.exit(0 if ok else 1)
