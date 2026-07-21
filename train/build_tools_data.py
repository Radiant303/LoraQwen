# =========================
# 回忆书工具调用训练数据生成器
# 位置：train/build_tools_data.py
#
# 按 场景工具.md 的正样本路径合成多轮对话，
# 用 chat template 渲染成训练文本，保证与线上一致。
# 直接输出 data/train/tools_train_v2.jsonl。
#
# 与旧版的差异（针对多样性不足、场景覆盖有洞）：
# - 笔记内容池从 10 条手写扩到 ~150 条：手写 + 程序化模板合成，
#   降低模型背诵固定 snippet、复述固定答案模板的风险
# - 新增场景：cached 缓存拦截、local_tool_execution_failed 错误处理、
#   多结果打分排序、3 次调用链（date→search→read）、搜索无结果后放宽关键词、
#   无需工具的负样本（防过度触发）、跨月/跨年相对日期
# - 日期范围从 2026-05~07 放宽到 2025~2026 全年，20% 落在月末/年初边界
# - 问法模板每场景扩到 5+ 条并加入少量英文问法；首问频次封顶 3%
# - 结尾套话（"需要查看完整内容可以告诉我"）随机化，50% 不带套话
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

N_TARGET = 3000

# 同一首问在全量数据中的占比上限
MAX_QUESTION_SHARE = 0.03

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

# ---- 程序化内容池：模板 × 槽位合成，关键词随槽位确定 ----

_MOD = [
    "订单", "支付", "积分", "消息推送", "搜索", "笔记编辑器", "同步",
    "登录", "导出", "任务调度", "缓存", "网关", "日报", "周报", "回收站",
]
_TECH = [
    "Redis", "Kafka", "SQLite", "Vue", "Flutter", "Rust", "gRPC",
    "WebSocket", "Docker", "Nginx", "PostgreSQL", "JWT", "OAuth2",
]


def _g_slow_query():
    mod = random.choice(_MOD)
    ms1 = random.choice([600, 800, 1200, 1500, 2000])
    ms2 = random.choice([40, 60, 80, 120, 200])
    return (
        f"优化{mod}模块的慢查询，给高频过滤字段补充索引，"
        f"接口耗时从 {ms1}ms 降到 {ms2}ms。",
        [mod, "索引"],
    )


def _g_conn_pool():
    mod = random.choice(_MOD)
    n1, n2 = random.choice([(10, 30), (8, 32), (16, 64)])
    return (
        f"排查{mod}模块偶发超时，定位为数据库连接池过小，"
        f"最大连接数从 {n1} 调整到 {n2} 后超时消失。",
        [mod, "连接池"],
    )


def _g_cache():
    tech = random.choice(["Redis", "本地缓存"])
    mod = random.choice(_MOD)
    return (
        f"排查{mod}数据不一致，{tech}缓存与数据库双写时序有问题，"
        f"改为先写库再删缓存，并给缓存补了过期时间。",
        [mod, tech],
    )


def _g_retry():
    mod = random.choice(_MOD)
    return (
        f"修复{mod}接口重试导致的重复提交，给请求加了幂等键，"
        f"重复请求直接返回首次结果。",
        [mod, "幂等"],
    )


def _g_tech_migrate():
    tech = random.choice(_TECH)
    mod = random.choice(_MOD)
    return (
        f"完成{mod}模块向 {tech} 的迁移验证，旧实现保留灰度开关，"
        f"观察一周后下线。",
        [mod, tech],
    )


def _g_mq():
    tech = random.choice(["Kafka", "Redis Stream"])
    return (
        f"排查{tech}消费积压，消费者单条处理耗时过长，"
        f"改为批量拉取加并行处理，积压在一小时内清完。",
        [tech, "积压"],
    )


def _g_frontend():
    tech = random.choice(["Vue", "Flutter"])
    n = random.choice([3, 5, 8])
    return (
        f"优化{tech}页面首屏加载，拆包并按需加载组件，"
        f"首屏体积减少了 {n}0%。",
        [tech, "首屏"],
    )


def _g_log():
    mod = random.choice(_MOD)
    return (
        f"给{mod}模块补充结构化日志，关键路径打上 traceId，"
        f"排查问题不用再翻散落日志。",
        [mod, "日志"],
    )


def _g_test():
    mod = random.choice(_MOD)
    n = random.choice([12, 20, 35, 48])
    return (
        f"给{mod}模块补了 {n} 个单元测试，覆盖主要分支，"
        f"集成到 CI 后拦住了一个边界条件 bug。",
        [mod, "单元测试"],
    )


def _g_memory():
    tech = random.choice(["Rust", "Flutter"])
    return (
        f"排查{tech}侧内存占用持续上涨，定位到资源未释放，"
        f"修复后长时间运行内存稳定。",
        [tech, "内存"],
    )


def _g_deploy():
    tech = random.choice(["Docker", "Nginx"])
    return (
        f"调整{tech}部署配置，把健康检查间隔从 30 秒改为 10 秒，"
        f"异常实例下线更快。",
        [tech, "部署"],
    )


def _g_deadlock():
    mod = random.choice(_MOD)
    return (
        f"排查{mod}模块偶发死锁，两个事务加锁顺序不一致，"
        f"统一加锁顺序并缩短事务范围后解决。",
        [mod, "死锁"],
    )


_GENERATORS = [
    _g_slow_query, _g_conn_pool, _g_cache, _g_retry, _g_tech_migrate,
    _g_mq, _g_frontend, _g_log, _g_test, _g_memory, _g_deploy, _g_deadlock,
]

# 用户笔记里不存在的话题（用于"无结果"场景，不能与内容池撞车）
ABSENT_TOPICS = [
    ["健身", "跑步"],
    ["旅游", "机票"],
    ["装修", "建材"],
    ["炒股", "基金"],
    ["相亲", "婚礼"],
    ["烹饪", "菜谱"],
    ["宠物", "猫粮"],
    ["学车", "驾照"],
]


def build_pool(n=140):
    """手写 10 条 + 程序化合成 n 条（按文本去重）。"""
    pool = list(DAILY_POOL)
    seen = {t for t, _ in pool}
    while len(pool) < len(DAILY_POOL) + n:
        text, kws = random.choice(_GENERATORS)()
        if text in seen:
            continue
        seen.add(text)
        pool.append((text, kws))
    return pool


def iso_week(d):
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}", w


def rand_today():
    """随机"今天"。20% 落在月末/年初边界，锻炼跨月跨年相对日期。"""
    if random.random() < 0.2:
        return random.choice([
            date(2026, 1, 1), date(2026, 1, 5), date(2026, 3, 1),
            date(2026, 3, 2), date(2026, 7, 1), date(2025, 12, 31),
            date(2026, 2, 28), date(2026, 6, 1), date(2026, 4, 1),
            date(2025, 12, 30),
        ])
    y = random.choice([2025, 2025, 2026, 2026, 2026])
    return date(y, random.randint(1, 12), random.randint(1, 28))


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


# 结尾引导语：50% 不带，避免模型复读固定套话
CLOSERS = [
    "",
    "",
    "需要查看完整内容可以告诉我。",
    "需要了解更多细节可以继续问我。",
    "如果你想看某篇原文，我可以帮你读取。",
]


def closer():
    c = random.choice(CLOSERS)
    return f"\n\n{c}" if c else ""


POOL = build_pool()


def pick_topic():
    return random.choice(POOL)


# =========================
# 场景生成函数，每个返回 messages
# =========================

def sc_read_relative_day():
    today = rand_today()
    offset, label, templates = random.choice([
        (1, "昨天", [
            "我昨天的日报写了什么？", "帮我看看昨天的日报。",
            "昨天我记录了些什么？", "昨天的日报内容帮我翻一下。",
            "昨天记了啥？",
        ]),
        (0, "今天", [
            "我今天的日报记了什么？", "今天我写日报了吗？",
            "看看我今天的日报。", "今天日报里都有什么？",
        ]),
        (2, "前天", [
            "前天的日报帮我翻一下。", "我前天记了什么？",
            "前天写了什么内容？",
        ]),
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
            f"- {text}"
            + closer()}
    ]


def sc_read_week_dailies():
    today = rand_today()
    monday = today - timedelta(days=today.weekday())

    q = random.choice([
        "这周我都做了什么？",
        "本周我的日报里记了哪些事？",
        "帮我回顾一下这一周的工作。",
        "这周到目前为止干了哪些活？",
        "What have I done this week?",
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
            f"你的日报记录如下：\n\n{answer}"
            + closer()}
    ]


def sc_read_weekly_report_relative():
    today = rand_today()
    week, _ = iso_week(today)
    texts = [pick_topic()[0] for _ in range(3)]

    q = random.choice([
        "本周周报讲了什么？",
        "这周的周报帮我总结一下。",
        "看看我这周的周报。",
        "本周周报里汇总了什么？",
    ])

    body = "\n".join(f"- {t}" for t in texts)

    return base_msgs(q) + [
        acall("call_1", "get_current_date", {}),
        tresp("call_1", current_date_call(today)),
        acall("call_2", "read_weekly_note", {"week": week}),
        tresp("call_2", {"results": [weekly_item(week, texts)]}),
        {"role": "assistant", "content":
            f"本周（{week}）周报的主要内容：\n\n{body}"
            + closer()}
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
        "上月月报帮我概括一下。",
    ])

    body = "\n".join(f"- {t}" for t in texts)

    return base_msgs(q) + [
        acall("call_1", "get_current_date", {}),
        tresp("call_1", current_date_call(today)),
        acall("call_2", "read_month_report", {"month": month}),
        tresp("call_2", {"results": [monthly_item(month, texts)]}),
        {"role": "assistant", "content":
            f"你上个月（{month}）的月报主要内容：\n\n{body}"
            + closer()}
    ]


def sc_read_month_weeklies():
    y = random.choice([2025, 2026])
    month = f"{y}-{random.randint(1, 12):02d}"
    yy, m = int(month[:4]), int(month[5:])
    first = date(yy, m, 1)
    last = date(yy, m, 28)
    w1 = iso_week(first)[1]
    w2 = iso_week(last)[1]
    weeks = [f"{yy}-W{w:02d}" for w in range(w1, w2 + 1)]

    results = [
        weekly_item(w, [pick_topic()[0] for _ in range(2)])
        for w in weeks
    ]

    q = random.choice([
        f"{m}月有哪几篇周报？",
        f"帮我列出{month}的周报。",
        f"{m}月的周报都有哪几篇？",
        f"{month}一共有几篇周报？",
    ])

    listing = "\n".join(f"- {w}" for w in weeks)

    return base_msgs(q) + [
        acall("call_1", "read_month_weekly_notes", {"month": month}),
        tresp("call_1", {"results": results}),
        {"role": "assistant", "content":
            f"{month} 共有 {len(weeks)} 篇周报：\n\n{listing}"
            + closer()}
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
        f"{label}里有没有{keywords[0]}相关的内容？",
    ])

    msgs = base_msgs(q) + [
        acall("call_1", tool_name, {"keywords": keywords}),
        tresp("call_1", {"results": [hit]}),
    ]

    if random.random() < 0.5:
        msgs += [
            {"role": "assistant", "content":
                f"在{label}中查到 1 条相关记录：\n\n"
                f"- **{hit['title']}**：{text}"
                + closer()}
        ]
    else:
        if label == "日报":
            # 日报可二次读取全文确认
            msgs += [
                acall("call_2", "read_daily_note", {"date": d.isoformat()}),
                tresp("call_2", {"results": [daily_item(d, text)]}),
                {"role": "assistant", "content":
                    f"查到了，{d.isoformat()} 的日报中有相关记录：\n\n"
                    f"- {text}"
                    + closer()}
            ]
        else:
            # 周报/月报直接用搜索结果回答
            msgs += [
                {"role": "assistant", "content":
                    f"在{label}中查到 1 条相关记录：\n\n"
                    f"- **{hit['title']}**：{text}"
                    + closer()}
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
        f"笔记里有没有提过{kws[0]}？",
    ])

    return base_msgs(q) + [
        acall("call_1", "keyword_search", {"keywords": kws}),
        tresp("call_1", {"results": [daily_item(d, text)]}),
        acall("call_2", "read_daily_note", {"date": d.isoformat()}),
        tresp("call_2", {"results": [daily_item(d, text)]}),
        {"role": "assistant", "content":
            f"有记录。{d.isoformat()} 的日报中提到了{kws[0]}：\n\n"
            f"- {text}"
            + closer()}
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
        f"笔记里有没有{kws[0]}相关的内容？",
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
        f"读一下{target.isoformat()}的日报。",
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
    d3 = today - timedelta(days=3)
    t1, _ = pick_topic()
    t2, _ = pick_topic()

    msgs = base_msgs("我昨天的日报写了什么？") + [
        acall("call_1", "get_current_date", {}),
        tresp("call_1", current_date_call(today)),
        acall("call_2", "read_daily_note", {"date": d1.isoformat()}),
        tresp("call_2", {"results": [daily_item(d1, t1)]}),
        {"role": "assistant", "content":
            f"你昨天（{d1.isoformat()}）的日报主要记录了：\n\n"
            f"- {t1}"},
        {"role": "user", "content": random.choice(["那前天呢？", "前天呢？", "再往前一天呢？"])},
        acall("call_3", "read_daily_note", {"date": d2.isoformat()}),
        tresp("call_3", {"results": [daily_item(d2, t2)]}),
        {"role": "assistant", "content":
            f"你前天（{d2.isoformat()}）的日报记录了：\n\n"
            f"- {t2}"}
    ]

    # 30% 追问第三轮，锻炼长上下文指代
    if random.random() < 0.3:
        t3, _ = pick_topic()
        msgs += [
            {"role": "user", "content": "那大前天呢？"},
            acall("call_4", "read_daily_note", {"date": d3.isoformat()}),
            tresp("call_4", {"results": [daily_item(d3, t3)]}),
            {"role": "assistant", "content":
                f"你大前天（{d3.isoformat()}）的日报记录了：\n\n"
                f"- {t3}"}
        ]

    return msgs


def sc_month_topic_summary():
    today = rand_today()
    text, kws = pick_topic()
    d = today.replace(day=random.randint(1, 25))

    q = random.choice([
        f"我这个月关于{kws[0]}做了哪些工作？",
        f"这个月{kws[0]}相关的事情帮我汇总一下。",
        f"本月有没有{kws[0]}方面的进展？",
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


# =========================
# 新增场景
# =========================

def sc_cached_repeat():
    """重复调用同一工具+参数被缓存拦截，模型应基于缓存结果回答。"""
    today = rand_today()
    d1 = today - timedelta(days=1)
    text, _ = pick_topic()
    payload = {"results": [daily_item(d1, text)]}

    cached = {
        "cached": True,
        "note": "This exact tool call already returned earlier in this"
                " conversation. Use the cached result and continue to"
                " answer instead of calling the same tool again.",
        "result": payload,
    }

    q = random.choice([
        "我昨天的日报写了什么？",
        "帮我看看昨天的日报。",
        "昨天我记录了些什么？",
    ])

    return base_msgs(q) + [
        acall("call_1", "get_current_date", {}),
        tresp("call_1", current_date_call(today)),
        acall("call_2", "read_daily_note", {"date": d1.isoformat()}),
        tresp("call_2", payload),
        acall("call_3", "read_daily_note", {"date": d1.isoformat()}),
        tresp("call_3", cached),
        {"role": "assistant", "content":
            f"你昨天（{d1.isoformat()}）的日报主要记录了：\n\n"
            f"- {text}"
            + closer()}
    ]


def sc_tool_error():
    """工具本地执行失败：明确说明失败原因，不编造内容。"""
    today = rand_today()
    d1 = today - timedelta(days=random.randint(1, 5))

    q = random.choice([
        f"帮我看看{d1.isoformat()}那天的日报。",
        f"{d1.isoformat()}的日报写了什么？",
    ])

    return base_msgs(q) + [
        acall("call_1", "read_daily_note", {"date": d1.isoformat()}),
        tresp("call_1", {"results": [], "error": "local_tool_execution_failed"}),
        {"role": "assistant", "content":
            f"读取 {d1.isoformat()} 的日报时本地检索执行失败，"
            "暂时无法获取记录内容。\n\n"
            "可能是笔记文件读取异常，建议稍后重试；"
            "如果持续失败，可以检查一下当天的笔记文件是否存在异常。"}
    ]


def sc_multi_result_search():
    """搜索命中多条结果：按相关度/日期逐条列出。"""
    topic = pick_topic()
    text, kws = topic
    n_results = random.randint(2, 4)
    today = rand_today()
    days = sorted(
        random.sample(range(3, 40), n_results),
        reverse=False,
    )
    items = []
    for off in days:
        d = today - timedelta(days=off)
        t, _ = pick_topic()
        item = daily_item(d, t)
        item["score"] = sum(t.count(k) for k in kws) or random.randint(1, 4)
        items.append(item)
    items.sort(key=lambda x: -x["score"])

    scope, tool_name = random.choice([
        ("日报", "search_daily_notes"),
        ("全部记录", "keyword_search"),
    ])

    kw = kws[0]
    q = random.choice([
        f"在{scope}里搜一下{kw}。",
        f"帮我查查有没有关于{kw}的记录。",
        f"搜搜{scope}：{kw}。",
    ])

    listing = "\n".join(
        f"- **{it['title']}**：{it['snippet']}" for it in items
    )

    return base_msgs(q) + [
        acall("call_1", tool_name, {"keywords": [kw]}),
        tresp("call_1", {"results": items}),
        {"role": "assistant", "content":
            f"共查到 {len(items)} 条相关记录：\n\n{listing}"
            + closer()}
    ]


def sc_chain3():
    """三次调用链：相对时间 + 类型不明 → date → keyword_search → read。"""
    today = rand_today()
    text, kws = pick_topic()
    d = today - timedelta(days=random.randint(2, 20))

    q = random.choice([
        f"我最近有没有记录过{kws[0]}相关的事？",
        f"最近我好像搞过{kws[0]}，帮我找找是哪天。",
        f"最近有没有关于{kws[0]}的记录？",
    ])

    return base_msgs(q) + [
        acall("call_1", "get_current_date", {}),
        tresp("call_1", current_date_call(today)),
        acall("call_2", "keyword_search", {"keywords": kws}),
        tresp("call_2", {"results": [daily_item(d, text)]}),
        acall("call_3", "read_daily_note", {"date": d.isoformat()}),
        tresp("call_3", {"results": [daily_item(d, text)]}),
        {"role": "assistant", "content":
            f"有记录。{d.isoformat()} 的日报中提到了{kws[0]}：\n\n"
            f"- {text}"
            + closer()}
    ]


def sc_broaden_search():
    """窄关键词无结果 → 放宽关键词再搜 → 命中。教查询改写。"""
    text, kws = pick_topic()
    d = rand_today()
    narrow = [f"{kws[0]}详细方案"]  # 过窄的词搜不到
    broad = kws[:1]

    q = random.choice([
        f"帮我找找{kws[0]}相关的记录。",
        f"搜一下{kws[0]}。",
    ])

    return base_msgs(q) + [
        acall("call_1", "keyword_search", {"keywords": narrow}),
        tresp("call_1", {"results": []}),
        acall("call_2", "keyword_search", {"keywords": broad}),
        tresp("call_2", {"results": [daily_item(d, text)]}),
        {"role": "assistant", "content":
            f"找到了。先用较窄的关键词没有命中，"
            f"放宽后查到 {d.isoformat()} 的日报有相关记录：\n\n"
            f"- {text}"
            + closer()}
    ]


def sc_no_tool_needed():
    """无需工具的输入：直接回答，不调用任何工具。"""
    q, answers = random.choice([
        ("你好", [
            "你好！我是 SpringNote 的回忆书问答助手，"
            "可以基于你的日报、周报、月报回答问题和检索记录。"
            "比如可以问我：我昨天的日报写了什么？",
            "你好！我可以帮你查询和回顾日报、周报、月报。"
            "想查哪天的记录，直接告诉我就行。",
        ]),
        ("你是谁？", [
            "我是 SpringNote 的回忆书问答助手，"
            "专门基于你的历史日报、周报、月报回答问题。",
        ]),
        ("你能做什么？", [
            "我可以帮你检索和回顾历史记录："
            "查找某天的日报、汇总某周的工作、按关键词搜索笔记内容等。"
            "所有回答都基于你的日报、周报、月报。",
            "我的职责是基于你的日报、周报、月报回答问题："
            "可以按日期读取记录、按关键词搜索、汇总一段时间的工作。",
        ]),
        ("谢谢", [
            "不客气！需要查询或回顾记录时随时问我。",
            "不客气，有其他关于笔记记录的问题随时找我。",
        ]),
        ("早上好", [
            "早上好！需要回顾昨天的日报，还是查一下本周的记录？",
        ]),
        ("Markdown 怎么写二级标题？", [
            "在 Markdown 中，二级标题用两个井号加空格表示，"
            "例如 `## 标题`。\n\n（这属于通用知识，并非来自你的笔记记录。）",
        ]),
        ("Markdown 表格怎么写？", [
            "用竖线分隔列、第二行用短横线分隔表头即可，例如：\n"
            "| 列1 | 列2 |\n| --- | --- |\n| 内容 | 内容 |\n\n"
            "（这属于通用知识，并非来自你的笔记记录。）",
        ]),
        ("JSON 和 YAML 有什么区别？", [
            "JSON 结构严格、适合数据交换；YAML 更易读、常用于配置。"
            "两者都可以表达层级数据。\n\n"
            "（这属于通用知识，并非来自你的笔记记录。）",
        ]),
        ["帮我写一首诗。", [
            "我主要负责基于你的日报、周报、月报回答问题，"
            "写诗不是我的强项。\n\n"
            "如果你想回顾某天的记录，我可以帮你查。",
        ]],
        ["今天天气怎么样？", [
            "我无法获取实时天气信息，我的数据范围是你的历史笔记记录。"
            "如果你记录过和天气相关的日报内容，我可以帮你检索。",
        ]],
        ("What can you do?", [
            "I can answer questions based on your SpringNote daily,"
            " weekly and monthly records, and help you search or review"
            " them. What would you like to look up?",
        ]),
        ("Hello!", [
            "Hello! I'm the SpringNote memory assistant. I can look up"
            " your daily, weekly and monthly records for you.",
        ]),
    ])
    if isinstance(answers, list):
        a = random.choice(answers)
    else:
        a = answers
    return base_msgs(q) + [{"role": "assistant", "content": a}]


def sc_last_week_cross_month():
    """跨月/跨年的"上周"：今天靠近月初，上周落在上一月。"""
    today = rand_today()
    this_monday = today - timedelta(days=today.weekday())
    last_monday = this_monday - timedelta(days=7)
    last_sunday = this_monday - timedelta(days=1)

    days = []
    d = last_monday
    while d <= last_sunday:
        if random.random() < 0.8:  # 允许部分日期没写日报
            days.append((d, pick_topic()[0]))
        d += timedelta(days=1)

    if not days:
        days = [(last_monday, pick_topic()[0])]

    results = [daily_item(d, t) for d, t in days]
    answer = "\n".join(f"- **{d.isoformat()}**：{t}" for d, t in days)

    q = random.choice([
        "上周我都做了什么？",
        "帮我汇总一下上周的日报。",
        "上周记录了哪些事？",
    ])

    return base_msgs(q) + [
        acall("call_1", "get_current_date", {}),
        tresp("call_1", current_date_call(today)),
        acall("call_2", "read_week_daily_notes", {
            "startDate": last_monday.isoformat(),
            "endDate": last_sunday.isoformat(),
        }),
        tresp("call_2", {"results": results}),
        {"role": "assistant", "content":
            f"上周（{last_monday.isoformat()} 至 {last_sunday.isoformat()}）"
            f"你的日报记录如下：\n\n{answer}"
            + closer()}
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
    # 新增
    sc_cached_repeat,
    sc_tool_error,
    sc_multi_result_search,
    sc_chain3,
    sc_broaden_search,
    sc_no_tool_needed,
    sc_last_week_cross_month,
]


def main():
    out = []
    seen = set()
    question_count = {}
    max_per_question = max(20, int(N_TARGET * MAX_QUESTION_SHARE))

    while len(out) < N_TARGET:
        msgs = random.choice(SCENARIOS)()
        text = tokenizer.apply_chat_template(
            msgs,
            tools=TOOLS,
            tokenize=False
        )

        if text in seen:
            continue

        # 首问频次封顶，防止高频问法压倒长尾
        first_q = msgs[1]["content"]
        if question_count.get(first_q, 0) >= max_per_question:
            continue

        seen.add(text)
        question_count[first_q] = question_count.get(first_q, 0) + 1
        out.append({"text": text})

    DST.parent.mkdir(parents=True, exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    top = sorted(question_count.items(), key=lambda x: -x[1])[:5]
    print(f"{len(out)} 条 -> {DST}")
    print(f"唯一首问: {len(question_count)}，Top5: {top}")


if __name__ == "__main__":
    main()
