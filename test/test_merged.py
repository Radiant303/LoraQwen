# =========================
# 合并模型测试：FIM 补全 + tool 调用 + think 思考 + persona 聊天
# 位置：test/test_merged.py
#
# 加载合并后的模型，对四类能力进行端到端验证。
# =========================

import json
import re
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

MODEL = ROOT / "models" / "fim" / "SpringNote-Qwen3-1.7B-FIM-V7"

# 伪造的笔记根路径（仅用于测试数据的 path 字段）
FAKE_NOTES_BASE = "D:/SpringNote/notes"

MAX_ITER = 5
MAX_NEW_TOKENS = 256

# ---------- 内联工具定义（不依赖外部数据生成脚本） ----------

SYSTEM = (
    "你是 SpringNote 的回忆书问答助手。\n"
    "你的唯一数据来源是用户保存的日报、周报、月报等历史记录。\n"
    "禁止根据模型记忆、常识、示例内容推测用户历史。\n\n"

    "====================\n"
    "【最高优先级规则】\n"
    "====================\n"

    "1. 用户询问任何个人历史记录时，必须调用工具获取真实数据。\n"
    "2. 工具调用必须形成完整链路，不能停留在中间步骤。\n"
    "3. 如果调用了辅助工具（例如 get_current_date），必须继续调用最终数据读取工具。\n"
    "4. 获取日期不是回答目的，只是为了确定查询参数。\n\n"

    "错误示例：\n"
    "用户：我昨天日报写了什么？\n"
    "调用：get_current_date\n"
    "回答：昨天是xxx\n"
    "这是错误行为。\n\n"

    "正确流程：\n"
    "用户：我昨天日报写了什么？\n"
    "调用：get_current_date\n"
    "计算昨天日期\n"
    "调用：read_daily_note(date=昨天日期)\n"
    "根据日报内容回答。\n\n"


    "====================\n"
    "【时间解析规则】\n"
    "====================\n"

    "1. 出现以下时间表达：\n"
    "今天、昨天、前天、最近、本周、这周、上周、本月、这个月\n"
    "必须先获取当前日期。\n\n"

    "2. 相对日期必须转换为绝对日期后继续查询。\n\n"

    "例如：\n"
    "昨天 -> get_current_date -> read_daily_note\n"
    "前天 -> 根据上下文已有日期 -> read_daily_note\n"
    "本周 -> get_current_date -> read_week_daily_notes\n\n"


    "====================\n"
    "【工具选择规则】\n"
    "====================\n"

    "用户问题 -> 必须使用工具：\n\n"

    "1. '我的日报写了什么'\n"
    "   '某天日报是什么'\n"
    "   '昨天日报'\n"
    "   '前天日报'\n"
    "   -> read_daily_note\n\n"

    "2. '这周做了什么'\n"
    "   '本周总结'\n"
    "   '最近一周工作'\n"
    "   -> read_week_daily_notes\n\n"

    "3. '有没有记录过xxx'\n"
    "   '我做过xxx吗'\n"
    "   '关于xxx有没有内容'\n"
    "   不确定记录类型\n"
    "   -> keyword_search\n\n"

    "4. 用户明确说：\n"
    "'在日报里搜xxx'\n"
    "'日报搜索xxx'\n"
    "   -> search_daily_notes\n\n"


    "====================\n"
    "【禁止行为】\n"
    "====================\n"

    "禁止：\n"
    "1. 没调用工具直接回答历史事实。\n"
    "2. 编造不存在的日报、周报内容。\n"
    "3. 调用 get_current_date 后直接结束。\n"
    "4. 调用工具后说'如果需要我可以查询'。\n"
    "5. 用户已经明确需求后要求用户再次确认。\n\n"


    "====================\n"
    "【多轮对话规则】\n"
    "====================\n"

    "必须结合完整历史消息理解省略表达。\n\n"

    "例如：\n"
    "用户：我昨天的日报写了什么？\n"
    "助手：正在查询昨天日报。\n"
    "用户：那前天呢？\n\n"

    "此时：\n"
    "前天 = 昨天日期 - 1天\n"
    "直接调用 read_daily_note。\n"
    "不要重新解释日期。\n"
    "不要只返回日期。\n\n"


    "====================\n"
    "【回答规则】\n"
    "====================\n"

    "1. 必须等待工具返回结果后回答。\n"
    "2. 只使用工具返回的信息。\n"
    "3. 没有记录：明确说明没有找到。\n"
    "4. 不允许推测、补充不存在的信息。\n"
    "5. 使用自然中文 Markdown 输出。\n"
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

# ---------- 加载模型 ----------

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
# 通用生成函数
# =========================

def gen_raw(prompt, n=128, enable_thinking=False):
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    # Qwen3 推荐参数：思考 temp=0.6/top_p=0.95；非思考 temp=0.7/top_p=0.8
    if enable_thinking:
        gen_kwargs = {"temperature": 0.6, "top_p": 0.95}
    else:
        gen_kwargs = {"temperature": 0.7, "top_p": 0.8}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=n,
            **gen_kwargs,
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
    return gen_raw(prompt, n=MAX_NEW_TOKENS, enable_thinking=enable_thinking)


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

    print("=" * 70)
    print("【FIM 补全测试】")
    middle = gen_raw(fim_prompt).split("<|im_end|>")[0].strip()
    print("=======prefix======")
    print(prefix)
    print("=======middle======")
    print(middle)
    print("=======suffix======")
    print(suffix)


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


def run_tool_case(name, user_content, history=None, expected_tools=None):
    print("\n" + "=" * 70)
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

        assistant_tool_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call_{step + 1}",
                    "type": "function",
                    "function": c,
                }
                for c in calls
            ],
        }
        messages.append(assistant_tool_msg)

        for idx, c in enumerate(calls):
            result = execute_tool(c["name"], c["arguments"])
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{step + 1}",
                "content": json.dumps(result, ensure_ascii=False),
            })
    else:
        final_answer = "[达到最大迭代次数，未得到最终回答]"

    print("\n调用链路:")
    if not all_tool_calls:
        print("  (无工具调用，直接回答)")
    for c in all_tool_calls:
        print(f"  -> {c['name']}({c['arguments']})")

    print("\n最终回答:")
    print(final_answer)

    if expected_tools is not None:
        actual = [c["name"] for c in all_tool_calls]
        if actual == expected_tools:
            print("\n[✓] 工具调用顺序符合预期")
        else:
            print(f"\n[✗] 预期: {expected_tools}")
            print(f"    实际: {actual}")

    return all_tool_calls, final_answer


def test_tools():
    print("\n" + "=" * 70)
    print("【tool 调用测试】")

    run_tool_case(
        "相对日期 - 昨天",
        "我昨天的日报写了什么？",
        expected_tools=["get_current_date", "read_daily_note"],
    )

    run_tool_case(
        "相对日期 - 本周日报",
        "这周我都做了什么？",
        expected_tools=["get_current_date", "read_week_daily_notes"],
    )

    run_tool_case(
        "类型明确 - 日报搜索",
        "在日报里搜一下缓存",
        expected_tools=["search_daily_notes"],
    )

    run_tool_case(
        "类型不明 - 全局搜索",
        "我有没有记录过 Kafka 相关的事？",
        expected_tools=["keyword_search"],
    )

    run_tool_case(
        "无结果 - 诚实说明",
        "我记过健身相关的内容吗？",
        expected_tools=["search_daily_notes"],
    )

    run_tool_case(
        "读取缺失 - 诚实说明",
        "2026-01-01 的日报写了什么？",
        expected_tools=["read_daily_note"],
    )

    # 追问
    history = []
    _, ans1 = run_tool_case(
        "追问 - 第一轮",
        "我昨天的日报写了什么？",
        history=history,
    )
    history.append({"role": "user", "content": "我昨天的日报写了什么？"})
    history.append({"role": "assistant", "content": ans1})
    run_tool_case(
        "追问 - 第二轮（前天）",
        "那前天呢？",
        history=history,
        expected_tools=["read_daily_note"],
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
    if "<think>" in answer:
        print("\n[✓] 包含思考块")
    else:
        print("\n[✗] 未出现思考块")


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

    questions = [
        "SpringNote是谁开发的？",
        "陈果果毕业于哪里？",
        "SpringNote官方QQ群是多少？",
        "怎么联系SpringNote作者？",
        "SpringNote官网在哪里？",
    ]

    print("\n" + "=" * 70)
    print("【persona 聊天测试】")

    for q in questions:
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

        print()
        print("Q:", q)
        print("A:", ans)


# =========================
# 主入口
# =========================

if __name__ == "__main__":
    test_fim()
    test_tools()
    test_thinking()
    test_persona()
