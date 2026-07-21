# =========================
# 回忆书工具调用训练数据生成器
# 位置：train/build_tools_data.py
#
# 按 场景工具.md 的正样本路径合成多轮对话，
# 用 chat template 渲染成训练文本，保证与线上一致。
# 直接输出 data/train/tools_train_v2.jsonl。
# =========================

import json
import random
from datetime import date, timedelta
from pathlib import Path

from transformers import AutoTokenizer

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "models" / "base" / "Qwen3-1.7B"
DST = ROOT / "data" / "train" / "tools_train_v2.jsonl"

# 伪造的笔记根路径（仅用于合成训练数据中的 path 字段）
FAKE_NOTES_BASE = "D:/SpringNote/notes"

N_TARGET = 2400

tokenizer = AutoTokenizer.from_pretrained(
    BASE,
    trust_remote_code=True
)

# 系统提示词（与线上 MEMORY_TOOL_SYSTEM_PROMPT 原文一致）

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
                "additionalProperties": False
            }
        },
        "strict": True
    }


_KW = {
    "keywords": {
        "type": "array",
        "items": {"type": "string", "minLength": 2},
        "minItems": 1,
        "maxItems": 8
    }
}

_DATE = {
    "type": "string",
    "pattern": r"^20\d{2}-(0[1-9]|1[0-2])-([0-2][0-9]|3[0-1])$"
}

_WEEK = {
    "type": "string",
    "pattern": r"^20\d{2}-W(0[1-9]|[1-4]\d|5[0-3])$"
}

_MONTH = {
    "type": "string",
    "pattern": r"^20\d{2}-(0[1-9]|1[0-2])$"
}

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
        []
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
        ["keywords"]
    ),

    _fn(
        "search_daily_notes",
        "Search only SpringNote daily Markdown notes. Use this instead of"
        " keyword_search when the request is limited to daily notes or"
        " day-level records." + _TOOLS_KW_SUFFIX,
        {"keywords": _KW["keywords"]},
        ["keywords"]
    ),

    _fn(
        "search_weekly_notes",
        "Search only SpringNote weekly Markdown reports. Use this instead"
        " of keyword_search when the request is limited to weekly"
        " reports." + _TOOLS_KW_SUFFIX,
        {"keywords": _KW["keywords"]},
        ["keywords"]
    ),

    _fn(
        "search_monthly_notes",
        "Search only SpringNote monthly Markdown reports. Use this instead"
        " of keyword_search when the request is limited to monthly"
        " reports." + _TOOLS_KW_SUFFIX,
        {"keywords": _KW["keywords"]},
        ["keywords"]
    ),

    _fn(
        "read_daily_note",
        "Read the full daily Markdown note for a specific date.",
        {"date": _DATE},
        ["date"]
    ),

    _fn(
        "read_week_daily_notes",
        "Read all available daily notes in a date range, typically one week.",
        {"startDate": _DATE, "endDate": _DATE},
        ["startDate", "endDate"]
    ),

    _fn(
        "read_weekly_note",
        "Read only the full SpringNote weekly report Markdown for a"
        " specific ISO week. Do not return daily notes.",
        {"week": _WEEK},
        ["week"]
    ),

    _fn(
        "read_month_weekly_notes",
        "Read all available SpringNote weekly report Markdown files whose"
        " ISO weeks overlap a specific calendar month. Return weekly"
        " reports only, not daily notes or the monthly report.",
        {"month": _MONTH},
        ["month"]
    ),

    _fn(
        "read_month_report",
        "Read only the monthly report Markdown for a specific month."
        " Do not return daily notes.",
        {"month": _MONTH},
        ["month"]
    ),
]

# =========================
# 伪造的笔记内容池
# =========================

DAILY_POOL = [
    ("排查订单支付成功后积分未增加的问题，发现支付回调处理时只更新了订单状态，没有触发积分发放流程，在支付回调方法中增加了积分发放的异步调用。",
     ["积分", "支付回调"]),

    ("完成 Graph 微服务交互流程联调，各节点按定义顺序执行，工具执行结果以 PCM 语音格式通过 SSE 推送给用户。",
     ["Graph", "微服务"]),

    ("排查消息通知服务的消息积压，消费者同步执行数据库写入和外部接口调用耗时较长，改为批量处理模式，消费速度提升到原来的三倍。",
     ["消息积压", "批量处理"]),

    ("排查商品详情页加载慢的问题，热点商品查询缓存失效导致大量请求同时访问数据库，增加请求控制避免多个请求执行相同查询。",
     ["缓存", "热点商品"]),

    ("调整周报生成逻辑，按 ISO 周聚合日报内容，修复跨月周统计遗漏的问题。",
     ["周报", "ISO"]),

    ("优化笔记列表的慢查询，给修改时间字段补充索引，接口耗时从 800ms 降到 60ms。",
     ["慢查询", "索引"]),

    ("完成回忆书工具调用链路验证，keyword_search 一次提交全部关键词，重复调用会被缓存拦截。",
     ["回忆书", "keyword_search"]),

    ("修复日记编辑器偶发内容丢失，自动保存防抖时间从 2 秒调整为 800 毫秒。",
     ["编辑器", "自动保存"]),

    ("联调语音转写链路，PCM 音频流分段上传，转写结果实时回填到日报正文。",
     ["PCM", "语音转写"]),

    ("重构设置页配置存储，把散落的配置项收敛到统一的配置服务，补齐默认值兜底。",
     ["配置", "设置页"]),
]

ABSENT_TOPICS = [
    ["健身", "跑步"],
    ["旅游", "机票"],
    ["装修", "建材"],
    ["炒股", "基金"],
    ["相亲", "婚礼"],
]


def iso_week(d):
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}", w


def rand_today():
    return date(2026, random.choice([5, 6, 7]), random.randint(1, 28))


def daily_item(d, text):
    return {
        "title": f"日报 {d.isoformat()}",
        "path": f"{FAKE_NOTES_BASE}/daily/{d.isoformat()}.md",
        "snippet": text,
        "score": 100
    }


def weekly_item(week, texts):
    body = "\n".join(f"- {t}" for t in texts)
    return {
        "title": f"周报 {week}",
        "path": f"{FAKE_NOTES_BASE}/weekly/{week}.md",
        "snippet": f"# 周报 {week}\n\n本周完成：\n{body}",
        "score": 120
    }


def monthly_item(month, texts):
    body = "\n".join(f"- {t}" for t in texts)
    return {
        "title": f"月报 {month}",
        "path": f"{FAKE_NOTES_BASE}/monthly/{month}.md",
        "snippet": f"# 月报 {month}\n\n本月完成：\n{body}",
        "score": 140
    }


def acall(call_id, name, arguments):
    return {
        "role": "assistant",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments}
        }]
    }


def tresp(call_id, payload):
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(payload, ensure_ascii=False)
    }


def base_msgs(question):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question}
    ]


def current_date_call(today):
    _, w = iso_week(today)
    return {
        "date": today.isoformat(),
        "isoWeek": iso_week(today)[0],
        "weekNumber": w
    }


def pick_topic():
    return random.choice(DAILY_POOL)


# =========================
# 场景生成函数，每个返回 messages
# =========================

def sc_read_relative_day():
    today = rand_today()
    offset, label, templates = random.choice([
        (1, "昨天", ["我昨天的日报写了什么？", "帮我看看昨天的日报。", "昨天我记录了些什么？"]),
        (0, "今天", ["我今天的日报记了什么？", "今天我写日报了吗？", "看看我今天的日报。"]),
        (2, "前天", ["前天的日报帮我翻一下。", "我前天记了什么？"]),
    ])

    target = today - timedelta(days=offset)
    text, _ = pick_topic()
    q = random.choice(templates)

    return base_msgs(q) + [
        acall("call_1", "get_current_date", {}),
        tresp("call_1", current_date_call(today)),
        acall("call_2", "read_daily_note", {"date": target.isoformat()}),
        tresp("call_2", {"results": [daily_item(target, text)]}),
        {"role": "assistant", "content":
            f"你{label}（{target.isoformat()}）的日报主要记录了：\n\n"
            f"- {text}\n\n"
            "以上是你当天日报的内容。"}
    ]


def sc_read_week_dailies():
    today = rand_today()
    monday = today - timedelta(days=today.weekday())

    q = random.choice([
        "这周我都做了什么？",
        "本周我的日报里记了哪些事？",
        "帮我回顾一下这一周的工作。",
    ])

    days = []
    d = monday
    while d <= today:
        days.append((d, pick_topic()[0]))
        d += timedelta(days=1)

    results = [daily_item(d, t) for d, t in days]
    answer = "\n".join(
        f"- **{d.isoformat()}**：{t}" for d, t in days
    )

    return base_msgs(q) + [
        acall("call_1", "get_current_date", {}),
        tresp("call_1", current_date_call(today)),
        acall("call_2", "read_week_daily_notes", {
            "startDate": monday.isoformat(),
            "endDate": today.isoformat()
        }),
        tresp("call_2", {"results": results}),
        {"role": "assistant", "content":
            f"本周（{monday.isoformat()} 至 {today.isoformat()}）"
            f"你的日报记录如下：\n\n{answer}\n\n"
            "需要查看某一天的完整内容可以继续问我。"}
    ]


def sc_read_weekly_report_relative():
    today = rand_today()
    week, _ = iso_week(today)
    texts = [pick_topic()[0] for _ in range(3)]

    q = random.choice([
        "本周周报讲了什么？",
        "这周的周报帮我总结一下。",
        "看看我这周的周报。",
    ])

    body = "\n".join(f"- {t}" for t in texts)

    return base_msgs(q) + [
        acall("call_1", "get_current_date", {}),
        tresp("call_1", current_date_call(today)),
        acall("call_2", "read_weekly_note", {"week": week}),
        tresp("call_2", {"results": [weekly_item(week, texts)]}),
        {"role": "assistant", "content":
            f"本周（{week}）周报的主要内容：\n\n{body}\n\n"
            "以上是你本周周报记录的工作。"}
    ]


def sc_read_month_report_relative():
    today = rand_today()
    first = today.replace(day=1)
    prev = first - timedelta(days=1)
    month = prev.strftime("%Y-%m")
    texts = [pick_topic()[0] for _ in range(4)]

    q = random.choice([
        "上个月的月报讲了什么？",
        "帮我看看上月的月报。",
        "我上个月都做了些什么，看月报。",
    ])

    body = "\n".join(f"- {t}" for t in texts)

    return base_msgs(q) + [
        acall("call_1", "get_current_date", {}),
        tresp("call_1", current_date_call(today)),
        acall("call_2", "read_month_report", {"month": month}),
        tresp("call_2", {"results": [monthly_item(month, texts)]}),
        {"role": "assistant", "content":
            f"你上个月（{month}）的月报主要内容：\n\n{body}\n\n"
            "以上是月报记录的内容。"}
    ]


def sc_read_month_weeklies():
    month = f"2026-{random.choice(['04', '05', '06', '07'])}"
    y, m = int(month[:4]), int(month[5:])
    first = date(y, m, 1)
    last = date(y, m, 28)
    w1 = iso_week(first)[1]
    w2 = iso_week(last)[1]
    weeks = [f"{y}-W{w:02d}" for w in range(w1, w2 + 1)]

    results = [
        weekly_item(w, [pick_topic()[0] for _ in range(2)])
        for w in weeks
    ]

    q = random.choice([
        f"{m}月有哪几篇周报？",
        f"帮我列出{month}的周报。",
        f"{m}月的周报都有哪几篇？",
    ])

    listing = "\n".join(f"- {w}" for w in weeks)

    return base_msgs(q) + [
        acall("call_1", "read_month_weekly_notes", {"month": month}),
        tresp("call_1", {"results": results}),
        {"role": "assistant", "content":
            f"{month} 共有 {len(weeks)} 篇周报：\n\n{listing}\n\n"
            "需要读取某一篇的完整内容可以告诉我。"}
    ]


def _search_scenario(scope, tool_name, item_builder, label):
    text, kws = pick_topic()
    n = random.randint(1, min(3, len(kws)))
    keywords = kws[:n]
    d = rand_today()
    hit = item_builder(d, text)

    q = random.choice([
        f"在{label}里搜一下{keywords[0]}。",
        f"帮我查查{label}里有没有关于{keywords[0]}的记录。",
        f"搜搜{label}：{keywords[0]}。",
    ])

    msgs = base_msgs(q) + [
        acall("call_1", tool_name, {"keywords": keywords}),
        tresp("call_1", {"results": [hit]}),
    ]

    if random.random() < 0.5:
        msgs += [
            {"role": "assistant", "content":
                f"在{label}中查到 1 条相关记录：\n\n"
                f"- **{hit['title']}**：{text}\n\n"
                "需要查看完整内容可以告诉我。"}
        ]
    else:
        if label == "日报":
            # 日报可二次读取全文确认
            msgs += [
                acall("call_2", "read_daily_note", {"date": d.isoformat()}),
                tresp("call_2", {"results": [daily_item(d, text)]}),
                {"role": "assistant", "content":
                    f"查到了，{d.isoformat()} 的日报中有相关记录：\n\n"
                    f"- {text}\n\n"
                    "以上就是日报中的相关内容。"}
            ]
        else:
            # 周报/月报直接用搜索结果回答
            msgs += [
                {"role": "assistant", "content":
                    f"在{label}中查到 1 条相关记录：\n\n"
                    f"- **{hit['title']}**：{text}\n\n"
                    "需要查看完整内容可以告诉我。"}
            ]

    return msgs


def sc_search_daily():
    return _search_scenario(
        "日报", "search_daily_notes",
        lambda d, t: daily_item(d, t), "日报"
    )


def sc_search_weekly():
    def build(d, t):
        week, _ = iso_week(d)
        item = weekly_item(week, [t])
        item["score"] = 3
        return item

    return _search_scenario(
        "周报", "search_weekly_notes", build, "周报"
    )


def sc_search_monthly():
    def build(d, t):
        month = d.strftime("%Y-%m")
        item = monthly_item(month, [t])
        item["score"] = 2
        return item

    return _search_scenario(
        "月报", "search_monthly_notes", build, "月报"
    )


def sc_keyword_unknown():
    text, kws = pick_topic()
    d = rand_today()

    q = random.choice([
        f"我有没有记录过{kws[0]}相关的事？",
        f"找找关于{kws[0]}的记录。",
        f"我好像记过{kws[0]}，帮我找找。",
    ])

    return base_msgs(q) + [
        acall("call_1", "keyword_search", {"keywords": kws}),
        tresp("call_1", {"results": [daily_item(d, text)]}),
        acall("call_2", "read_daily_note", {"date": d.isoformat()}),
        tresp("call_2", {"results": [daily_item(d, text)]}),
        {"role": "assistant", "content":
            f"有记录。{d.isoformat()} 的日报中提到了{kws[0]}：\n\n"
            f"- {text}\n\n"
            "以上就是相关记录的内容。"}
    ]


def sc_no_result_search():
    kws = random.choice(ABSENT_TOPICS)
    scope, tool_name = random.choice([
        ("日报", "search_daily_notes"),
        ("周报", "search_weekly_notes"),
        ("月报", "search_monthly_notes"),
        ("全部记录", "keyword_search"),
    ])

    q = random.choice([
        f"帮我查查有没有关于{kws[0]}的记录。",
        f"我记过{kws[0]}吗？",
        f"搜一下{kws[0]}。",
    ])

    return base_msgs(q) + [
        acall("call_1", tool_name, {"keywords": kws}),
        tresp("call_1", {"results": []}),
        {"role": "assistant", "content":
            f"没有找到相关记录。我用关键词「{'、'.join(kws)}」"
            f"搜索了{scope}，没有匹配的内容。\n\n"
            "如果你记得大概的日期，可以告诉我，"
            "我可以直接读取那一天的记录帮你确认。"}
    ]


def sc_no_result_read():
    today = rand_today()
    target = today - timedelta(days=random.randint(3, 30))

    q = random.choice([
        f"帮我看看{target.isoformat()}那天的日报。",
        f"{target.isoformat()}的日报写了什么？",
    ])

    return base_msgs(q) + [
        acall("call_1", "read_daily_note", {"date": target.isoformat()}),
        tresp("call_1", {"results": []}),
        {"role": "assistant", "content":
            f"{target.isoformat()} 这一天没有找到日报记录。\n\n"
            "有可能那天没有写日报。如果你需要，"
            "我可以帮你搜索前后几天的记录看看。"}
    ]


def sc_followup():
    today = rand_today()
    d1 = today - timedelta(days=1)
    d2 = today - timedelta(days=2)
    t1, _ = pick_topic()
    t2, _ = pick_topic()

    return base_msgs("我昨天的日报写了什么？") + [
        acall("call_1", "get_current_date", {}),
        tresp("call_1", current_date_call(today)),
        acall("call_2", "read_daily_note", {"date": d1.isoformat()}),
        tresp("call_2", {"results": [daily_item(d1, t1)]}),
        {"role": "assistant", "content":
            f"你昨天（{d1.isoformat()}）的日报主要记录了：\n\n"
            f"- {t1}"},
        {"role": "user", "content": "那前天呢？"},
        acall("call_3", "read_daily_note", {"date": d2.isoformat()}),
        tresp("call_3", {"results": [daily_item(d2, t2)]}),
        {"role": "assistant", "content":
            f"你前天（{d2.isoformat()}）的日报记录了：\n\n"
            f"- {t2}"}
    ]


def sc_month_topic_summary():
    today = rand_today()
    text, kws = pick_topic()
    d = today.replace(day=random.randint(1, 25))

    q = random.choice([
        f"我这个月关于{kws[0]}做了哪些工作？",
        f"这个月{kws[0]}相关的事情帮我汇总一下。",
    ])

    return base_msgs(q) + [
        acall("call_1", "get_current_date", {}),
        tresp("call_1", current_date_call(today)),
        acall("call_2", "keyword_search", {"keywords": kws}),
        tresp("call_2", {"results": [daily_item(d, text)]}),
        {"role": "assistant", "content":
            f"这个月与{kws[0]}相关的工作记录如下：\n\n"
            f"- **{d.isoformat()}**：{text}\n\n"
            "目前检索到以上 1 条相关记录。"}
    ]


SCENARIOS = [
    sc_read_relative_day,
    sc_read_week_dailies,
    sc_read_weekly_report_relative,
    sc_read_month_report_relative,
    sc_read_month_weeklies,
    sc_search_daily,
    sc_search_weekly,
    sc_search_monthly,
    sc_keyword_unknown,
    sc_no_result_search,
    sc_no_result_read,
    sc_followup,
    sc_month_topic_summary,
]


def main():
    out = []
    seen = set()

    while len(out) < N_TARGET:
        msgs = random.choice(SCENARIOS)()
        text = tokenizer.apply_chat_template(
            msgs,
            tools=TOOLS,
            tokenize=False
        )

        if text in seen:
            continue

        seen.add(text)
        out.append({"text": text})

    DST.parent.mkdir(parents=True, exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"{len(out)} 条 -> {DST}")


if __name__ == "__main__":
    main()
