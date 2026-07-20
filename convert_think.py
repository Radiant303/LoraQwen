import json

import random

from transformers import AutoTokenizer

# =========================
# 思考链数据转换
#
# think_train.json -> data/think_train.jsonl
# 每条样本配 3 种 system prompt 变体，防止思考行为与单一 prompt 绑定
# =========================

random.seed(42)

BASE = "./SpringNote-Qwen3-0.6B-FIM-V2"

SRC = "think_train.json"

DST = "./data/think_train.jsonl"

SYS_VARIANTS = [

    None,

    "你是一个乐于助人的AI助手，思考后给出准确回答。",

    "You are a helpful assistant. Think step by step before answering.",

    "请仔细思考，再给出最终结论。"

]

tokenizer = AutoTokenizer.from_pretrained(
    BASE,
    trust_remote_code=True
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
            {"role": "assistant", "content": response}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        out.append({"text": text})


with open(DST, "w", encoding="utf-8") as f:

    for item in out:

        f.write(
            json.dumps(item, ensure_ascii=False) + "\n"
        )

print(f"{len(out)} 条 -> {DST}")
