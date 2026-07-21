# =========================
# 思考链(think)训练数据生成器
# 位置：train/build_think_data.py
#
# 读取 data/raw/think_train.json，用 chat template 渲染成训练文本，
# 每条样本配 3 种 system prompt 变体，
# 并额外加入 /think 软开关变体，让模型学会可控思考。
# 直接输出 data/train/think_train_v2.jsonl。
# =========================

import json
import random
import re
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

# /think 软开关前缀变体
THINK_PREFIXES = [
    "",
    "/think ",
    "请仔细思考后再回答。",
]


def add_think_prefix_to_system(text, prefix):
    """在已有渲染文本的 system content 末尾插入软开关前缀。"""
    return re.sub(
        r"(<\|im_start\|>system\n.*?)(<\|im_end\|>)",
        lambda m: f"{m.group(1)}\n\n{prefix}{m.group(2)}",
        text,
        count=1,
        flags=re.S,
    )


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE,
        trust_remote_code=True,
    )

    out = []
    for item in json.load(open(SRC, encoding="utf-8")):
        response = (
            f"<think>\n{item['think']}\n</think>\n\n"
            f"{item['answer']}"
        )

        for sys in random.sample(SYS_VARIANTS, 3):
            messages = (
                [{"role": "system", "content": sys}] if sys else []
            ) + [
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": response},
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            out.append({"text": text})

            # 额外生成 /think 软开关变体
            if random.random() < 0.3:
                prefix = random.choice(THINK_PREFIXES)
                if prefix:
                    switched = add_think_prefix_to_system(text, prefix)
                    out.append({"text": switched})

    DST.parent.mkdir(parents=True, exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"{len(out)} 条 -> {DST}")


if __name__ == "__main__":
    main()
