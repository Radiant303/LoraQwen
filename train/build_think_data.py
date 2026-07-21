# =========================
# 思考链(think)训练数据生成器
# 位置：train/build_think_data.py
#
# 读取 data/raw/think_train.json，用 chat template 渲染成训练文本。
# 与旧版的差异：
# - 修复重复 bug：旧版把软开关前缀插进 system 块，当样本无 system prompt 时
#   正则匹配不到、原样返回，导致追加完全重复的样本（实测 72/1081）。
#   现改为把 /think、/no_think 软开关放进 user 消息开头（Qwen3 官方用法），
#   与是否有 system prompt 无关
# - 新增 /no_think 样本（旧版只有 /think，开关不对称）：
#   /no_think 时 assistant 只给答案，模板自动补空 <think> 块
# - 新增多轮切换样本：第一轮普通思考问答，第二轮用 /think 或 /no_think
#   切换模式，教模型在对话中途响应软开关
# - 写出前按渲染文本全局去重
# 直接输出 data/train/think_train_v2.jsonl。
# =========================

import json
import random
from pathlib import Path

from transformers import AutoTokenizer

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "models" / "base" / "Qwen3-1.7B"
SRC = ROOT / "data" / "raw" / "think_train.json"
DST = ROOT / "data" / "train" / "think_train_v2.jsonl"

SYS_VARIANTS = [
    None,
    "你是一个乐于助人的AI助手，思考后给出准确回答。",
    "You are a helpful assistant. Think step by step before answering.",
    "请仔细思考，再给出最终结论。",
]

# 软开关样本占基础样本的比例
SWITCH_RATIO = 0.35
# 多轮切换样本占基础样本的比例
MULTI_TURN_RATIO = 0.25


def think_response(item):
    return f"<think>\n{item['think']}\n</think>\n\n{item['answer']}"


def build_messages(turns, system):
    """turns: [(user_content, assistant_content), ...]"""
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    for q, a in turns:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    return messages


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE,
        trust_remote_code=True,
    )

    data = json.load(open(SRC, encoding="utf-8"))

    out = []
    for item in data:
        q = item["instruction"]
        full = think_response(item)
        plain = item["answer"]

        for sys in random.sample(SYS_VARIANTS, 3):
            # 基础样本：带真实思考块
            text = tokenizer.apply_chat_template(
                build_messages([(q, full)], sys),
                tokenize=False,
                add_generation_prompt=False,
            )
            out.append(text)

            # 软开关单轮变体
            if random.random() < SWITCH_RATIO:
                if random.random() < 0.5:
                    # /think 显式开启
                    switched = tokenizer.apply_chat_template(
                        build_messages([(f"/think {q}", full)], sys),
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                else:
                    # /no_think 关闭：只给答案，模板自动补空 think 块
                    switched = tokenizer.apply_chat_template(
                        build_messages([(f"/no_think {q}", plain)], sys),
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                out.append(switched)

        # 多轮切换样本：第一轮普通思考问答（历史中 think 会被模板剥离，
        # 与线上一致），第二轮用软开关切换
        if random.random() < MULTI_TURN_RATIO:
            other = random.choice(data)
            if random.random() < 0.5:
                # 对话中途开启思考
                turns = [
                    (q, full),
                    (f"/think {other['instruction']}", think_response(other)),
                ]
            else:
                # 对话中途关闭思考
                turns = [
                    (q, full),
                    (f"/no_think {other['instruction']}", other["answer"]),
                ]
            text = tokenizer.apply_chat_template(
                build_messages(turns, random.choice(SYS_VARIANTS)),
                tokenize=False,
                add_generation_prompt=False,
            )
            out.append(text)

    # 全局去重
    seen = set()
    unique = []
    for text in out:
        if text in seen:
            continue
        seen.add(text)
        unique.append({"text": text})

    DST.parent.mkdir(parents=True, exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        for item in unique:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    n_switch = sum(
        1 for t in seen if "/think " in t or "/no_think " in t
    )
    print(f"{len(unique)} 条（去重前 {len(out)}）-> {DST}")
    print(f"含软开关样本: {n_switch}")


if __name__ == "__main__":
    main()
