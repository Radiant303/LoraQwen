import json
import random
import re

# =========================
# system prompt 数据增强
#
# 同一批问答配多种 system prompt（含不带 system 的情况），
# 避免模型把人设知识和单一 system prompt 绑定
# =========================

random.seed(42)

SRC = r".\data\persona_train.jsonl"
DST = r".\data\persona_train_aug.jsonl"

SYS_VARIANTS = [

    "你是SpringNote官方AI助手，由陈果果基于Qwen3模型微调开发。",

    "你是SpringNote笔记软件的AI助手。",

    "你是一个乐于助人的AI助手，回答准确，不编造信息。",

    "You are a helpful assistant.",

    None  # 不带 system prompt

]

pattern = re.compile(

    r"^<\|im_start\|>system\n(.*?)\n<\|im_end\|>\n(.*)$",

    re.S

)

out = []

for line in open(SRC, encoding="utf-8"):

    text = json.loads(line)["text"]

    m = pattern.match(text)

    rest = m.group(2)

    # 原始 system prompt 保留一份

    out.append(text)

    # 再随机配 3 个不同的 system prompt

    for sys in random.sample(SYS_VARIANTS, 3):

        if sys is None:

            out.append(rest)

        else:

            out.append(

                f"<|im_start|>system\n{sys}\n<|im_end|>\n{rest}"

            )

with open(DST, "w", encoding="utf-8") as f:

    for t in out:

        f.write(

            json.dumps({"text": t}, ensure_ascii=False) + "\n"

        )

print(f"{len(out)} 条 -> {DST}")
