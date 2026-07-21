# =========================
# 人设(persona)训练数据生成器
# 位置：train/build_persona_data.py
#
# 读取 data/raw/persona_train.json，用 Qwen3 chat template 渲染对话，
# 对同一批问答配多种 system prompt 变体，
# 并通过 enable_thinking=False 让 assistant 答案前带空 <think> 块，
# 直接输出 data/train/persona_train_aug_v2.jsonl。
# =========================

import json
import random
from pathlib import Path

from transformers import AutoTokenizer

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "models" / "base" / "Qwen3-1.7B"
SRC = ROOT / "data" / "raw" / "persona_train.json"
DST = ROOT / "data" / "train" / "persona_train_aug_v2.jsonl"

SYSTEM = """
你是SpringNote官方AI助手。

你由陈果果基于Qwen3模型微调开发。

你的职责是帮助用户了解SpringNote、
整理知识、处理笔记相关任务。

回答要求：
- 准确
- 简洁
- 不编造信息
- 不知道的信息明确说明
""".strip()

SYS_VARIANTS = [
    "你是SpringNote官方AI助手，由陈果果基于Qwen3模型微调开发。",
    "你是SpringNote笔记软件的AI助手。",
    "你是一个乐于助人的AI助手，回答准确，不编造信息。",
    "You are a helpful assistant.",
    None,  # 不带 system prompt
]


def build_messages(instruction, response, system):
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.extend([
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ])
    return messages


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE,
        trust_remote_code=True,
    )

    with open(SRC, encoding="utf-8") as f:
        data = json.load(f)

    out = []
    for item in data:
        if "instruction" not in item or "response" not in item:
            continue

        instruction = item["instruction"]
        response = item["response"]

        # 原始 system prompt 保留一份
        messages = build_messages(instruction, response, SYSTEM)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        out.append({"text": text})

        # 再随机配 3 个不同的 system prompt
        for sys in random.sample(SYS_VARIANTS, 3):
            messages = build_messages(instruction, response, sys)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            out.append({"text": text})

    DST.parent.mkdir(parents=True, exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"{len(out)} 条 -> {DST}")


if __name__ == "__main__":
    main()
