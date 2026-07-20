import json

import re

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel

# 工具 schema / 系统提示词复用生成器的定义

from data_tools import TOOLS, SYSTEM


# =========================
# 配置
# =========================

MODEL_PATH = "./SpringNote-Qwen3-0.6B-FIM-Persona-V5"

# 如果 MODEL_PATH 是 adapter 目录, 改成 BASE_MODEL = "./SpringNote-Qwen3-0.6B-FIM-V2"

# 本脚本会自动检测：含 adapter_config.json 则按 adapter 加载，否则按合并模型加载

MAX_ITER = 5

MAX_NEW_TOKENS = 256


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)


if (
    MODEL_PATH
    and
    (MODEL_PATH + "/adapter_config.json").__class__ is str
):
    pass


import os

if os.path.exists(os.path.join(MODEL_PATH, "adapter_config.json")):

    base = AutoModelForCausalLM.from_pretrained(
        "./SpringNote-Qwen3-0.6B-FIM-V2",
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base, MODEL_PATH)

else:

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

model.eval()


# =========================
# 假数据（与训练数据的格式一致即可，不需要真实内容）
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


# =========================
# 假工具执行器
# =========================

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
            "weekNumber": CURRENT_WEEK_NUMBER
        }

    if name == "read_daily_note":

        d = arguments.get("date")

        if d in FAKE_DAILY:
            return {"results": [{
                "title": f"日报 {d}",
                "path": f"D:/SpringNote/notes/daily/{d}.md",
                "snippet": FAKE_DAILY[d],
                "score": 100
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
                    "path": f"D:/SpringNote/notes/daily/{d}.md",
                    "snippet": t,
                    "score": 100
                })

        return {"results": results}

    if name == "read_weekly_note":

        w = arguments.get("week")

        if w in FAKE_WEEKLY:
            return {"results": [{
                "title": f"周报 {w}",
                "path": f"D:/SpringNote/notes/weekly/{w}.md",
                "snippet": FAKE_WEEKLY[w],
                "score": 120
            }]}

        return {"results": []}

    if name == "read_month_weekly_notes":

        month = arguments.get("month")

        results = []

        for w, t in FAKE_WEEKLY.items():

            if w.startswith(month):
                results.append({
                    "title": f"周报 {w}",
                    "path": f"D:/SpringNote/notes/weekly/{w}.md",
                    "snippet": t,
                    "score": 120
                })

        return {"results": results}

    if name == "read_month_report":

        month = arguments.get("month")

        if month in FAKE_MONTHLY:
            return {"results": [{
                "title": f"月报 {month}",
                "path": f"D:/SpringNote/notes/monthly/{month}.md",
                "snippet": FAKE_MONTHLY[month],
                "score": 140
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
                        "path": f"D:/SpringNote/notes/daily/{d}.md",
                        "snippet": t,
                        "score": sum(t.count(k) for k in kws)
                    })

        if scope_weekly:

            for w, t in FAKE_WEEKLY.items():

                if _matches(t, kws):
                    results.append({
                        "title": f"周报 {w}",
                        "path": f"D:/SpringNote/notes/weekly/{w}.md",
                        "snippet": t,
                        "score": sum(t.count(k) for k in kws)
                    })

        if scope_monthly:

            for m, t in FAKE_MONTHLY.items():

                if _matches(t, kws):
                    results.append({
                        "title": f"月报 {m}",
                        "path": f"D:/SpringNote/notes/monthly/{m}.md",
                        "snippet": t,
                        "score": sum(t.count(k) for k in kws)
                    })

        return {"results": results}

    return {"results": [], "error": f"unknown tool: {name}"}


# =========================
# 解析模型输出
# =========================

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.S
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


def generate(messages, tools=None, enable_thinking=False):

    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    text = tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=False
    )

    return text


# =========================
# 单条测试
# =========================

def run_case(name, user_content, history=None, expected_tools=None,
             enable_thinking=False):

    print("\n" + "=" * 70)

    print(f"[用例] {name}")

    print(f"用户: {user_content}")

    messages = [
        {"role": "system", "content": SYSTEM},
    ]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_content})

    all_tool_calls = []

    final_answer = None

    for step in range(MAX_ITER):

        text = generate(messages, tools=TOOLS,
                        enable_thinking=enable_thinking)

        calls = parse_tool_calls(text)

        if not calls:

            final_answer = text.split("<|im_end|>")[0].strip()

            break

        all_tool_calls.extend(calls)

        # assistant tool 消息

        assistant_tool_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call_{step + 1}",
                    "type": "function",
                    "function": c
                }
                for c in calls
            ]
        }

        messages.append(assistant_tool_msg)

        # tool 结果消息

        for idx, c in enumerate(calls):

            result = execute_tool(c["name"], c["arguments"])

            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{step + 1}",
                "content": json.dumps(result, ensure_ascii=False)
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


# =========================
# 测试用例
# =========================

def test_relative_yesterday():

    run_case(
        "相对日期 - 昨天",
        "我昨天的日报写了什么？",
        expected_tools=["get_current_date", "read_daily_note"]
    )


def test_this_week_dailies():

    run_case(
        "相对日期 - 本周日报",
        "这周我都做了什么？",
        expected_tools=["get_current_date", "read_week_daily_notes"]
    )


def test_search_daily():

    run_case(
        "类型明确 - 日报搜索",
        "在日报里搜一下缓存",
        expected_tools=["search_daily_notes"]
    )


def test_keyword_unknown():

    run_case(
        "类型不明 - 全局搜索",
        "我有没有记录过 Kafka 相关的事？",
        expected_tools=["keyword_search"]
    )


def test_no_result():

    run_case(
        "无结果 - 诚实说明",
        "我记过健身相关的内容吗？",
        expected_tools=["search_daily_notes"]
    )


def test_read_missing():

    run_case(
        "读取缺失 - 诚实说明",
        "2026-01-01 的日报写了什么？",
        expected_tools=["read_daily_note"]
    )


def test_followup():

    history = []

    calls1, ans1 = run_case(
        "追问 - 第一轮",
        "我昨天的日报写了什么？",
        history=history
    )

    history.append({"role": "user", "content": "我昨天的日报写了什么？"})

    history.append({"role": "assistant", "content": ans1})

    calls2, _ = run_case(
        "追问 - 第二轮（前天）",
        "那前天呢？",
        history=history,
        expected_tools=["read_daily_note"]
    )


def test_thinking():

    print("\n" + "=" * 70)

    print("[用例] 思考能力 - 9.11 vs 9.9")

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


if __name__ == "__main__":

    test_relative_yesterday()

    test_this_week_dailies()

    test_search_daily()

    test_keyword_unknown()

    test_no_result()

    test_read_missing()

    test_followup()

    test_thinking()
