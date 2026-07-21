# =========================
# FIM（Fill-in-the-Middle）训练数据生成器
# 位置：train/build_fim_data.py
#
# 从 data/raw/input.txt 构造 FIM 训练样本，输出 data/train/train.jsonl。
#
# 与旧版的差异（针对短段落、上下文过短、比例漂移的修正）：
# - 源段落普遍很短（中位数 ~93 字符），先把相邻段落拼成 800~2600 字符的
#   "文档"再挖空，让训练上下文长度接近真实补全场景
# - prefix/suffix 上下文窗口每样本随机（150~1200 字符），不再固定 300
# - 增加 suffix 为空（光标在文档末尾）的样本，占比 10%
# - 无需补全样本按全局配额精确控制在 6%（旧版因短段落采样丢弃实际漂到 13%+）
# - 按 middle 内容全局去重，并限制每文档采样数与文档长度成正比
# - 输出附带 "doc" 字段（文档 id），供训练脚本按文档分组切分验证集，防止泄漏
# - 固定随机种子，构建可复现
# =========================

import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

random.seed(42)


FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
EOS = "<|im_end|>"


# 全局样本类型配额
NO_COMPLETION_RATIO = 0.06   # 无需补全：middle 为空，模型应直接输出 EOS
END_COMPLETION_RATIO = 0.10  # 文档末尾补全：suffix 为空

# 单样本总字符上限，保证 tokenize 后不超过 MAX_LENGTH=2048
MAX_SAMPLE_CHARS = 2000

# 截断优先落在这些标点后（中英文）
BOUNDARY_PUNCT = r"[，。；：！？、\n\.,;:!?]"


# ==========================
# 清理文本
# ==========================

def clean_text(text):
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ==========================
# 段落 -> 文档
# 源段落太短（中位数 ~93 字符），按顺序拼接相邻段落，
# 拼到目标长度为止，模拟真实长度的笔记文档
# ==========================

def build_documents(paragraphs):
    docs = []
    buf = []
    target = random.randint(800, 2600)
    cur = 0

    for p in paragraphs:
        buf.append(p)
        cur += len(p) + 2
        if cur >= target:
            docs.append(clean_text("\n\n".join(buf)))
            buf = []
            cur = 0
            target = random.randint(800, 2600)

    if buf:
        tail = clean_text("\n\n".join(buf))
        if len(tail) >= 400 and docs:
            # 太短的尾巴并进上一个文档
            docs[-1] = docs[-1] + "\n\n" + tail
        elif len(tail) >= 400:
            docs.append(tail)
        # 不足 400 字符的尾巴直接丢弃

    return [d for d in docs if len(d) >= 400]


# ==========================
# 找 middle 的自然结束点
# ==========================

def find_middle_end(text, start, target_len, allow_expand=True):
    end = min(start + target_len, len(text))

    # 极短补全不要扩展
    if not allow_expand:
        return end

    search_end = min(end + 40, len(text))
    area = text[start:search_end]

    matches = list(re.finditer(BOUNDARY_PUNCT, area))

    if matches:
        # 找最近标点
        return start + matches[-1].end()

    return end


# ==========================
# middle 长度策略
# ==========================

def random_middle_length():
    r = random.random()

    # 极短补全 25%
    if r < 0.25:
        return random.randint(1, 8), False

    # 短补全 35%
    elif r < 0.60:
        return random.randint(8, 30), True

    # 中补全 25%
    elif r < 0.85:
        return random.randint(30, 80), True

    # 长补全 15%
    else:
        return random.randint(80, 200), True


# ==========================
# 每样本随机上下文窗口
# ==========================

def random_context_size():
    return random.randint(150, 1200)


def make_sample(prefix, suffix, middle):
    text = (
        f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}{EOS}"
    )
    if len(text) > MAX_SAMPLE_CHARS:
        return None
    return text


# ==========================
# 普通挖空样本
# ==========================

def create_normal_sample(text, used_middles):
    for _ in range(6):  # 失败重试几次
        pos = random.randint(int(len(text) * 0.05), int(len(text) * 0.95))
        middle_len, expand = random_middle_length()

        end = find_middle_end(text, pos, middle_len, expand)
        if end > len(text):
            continue

        middle = text[pos:end]
        if len(middle) < 1:
            continue

        # 同一 middle 全局只用一次，避免短文档反复挖同一段
        key = middle.strip()
        if key in used_middles:
            continue

        ctx = random_context_size()
        prefix = text[max(0, pos - ctx):pos]
        suffix = text[end:min(len(text), end + ctx)]

        if len(prefix) < 20 or len(suffix) < 10:
            continue

        sample = make_sample(prefix, suffix, middle)
        if sample is None:
            continue

        used_middles.add(key)
        return sample

    return None


# ==========================
# 文档末尾补全样本（suffix 为空）
# middle 是文档结尾的一段，起点切在标点边界后
# ==========================

def create_end_sample(text, used_middles):
    for _ in range(6):
        middle_len, _ = random_middle_length()
        middle_len = max(middle_len, random.randint(10, 60))

        start = max(0, len(text) - middle_len)
        # 起点尽量切在标点之后，让续写从一个干净的位置开始
        m = re.search(BOUNDARY_PUNCT, text[start:len(text)])
        if m and start + m.end() < len(text) - 5:
            start = start + m.end()

        middle = text[start:]
        if len(middle) < 5:
            continue

        key = middle.strip()
        if key in used_middles:
            continue

        ctx = random_context_size()
        prefix = text[max(0, start - ctx):start]
        if len(prefix) < 20:
            continue

        sample = make_sample(prefix, "", middle)
        if sample is None:
            continue

        used_middles.add(key)
        return sample

    return None


# ==========================
# 无需补全样本
# prefix 与 suffix 原本相连，middle 为空，模型应直接输出 EOS
# 其中一部分切在文档末尾（suffix 为空），教模型文末不要乱续
# ==========================

def create_no_completion_sample(text):
    if random.random() < 0.3:
        pos = len(text)  # 文档末尾：suffix 为空
    elif random.random() < 0.6:
        bounds = [m.end() for m in re.finditer(BOUNDARY_PUNCT, text)]
        bounds = [b for b in bounds if int(len(text) * 0.1) <= b <= int(len(text) * 0.9)]
        if bounds:
            pos = random.choice(bounds)
        else:
            pos = random.randint(int(len(text) * 0.1), int(len(text) * 0.9))
    else:
        pos = random.randint(int(len(text) * 0.1), int(len(text) * 0.9))

    ctx = random_context_size()
    prefix = text[max(0, pos - ctx):pos]
    suffix = text[pos:min(len(text), pos + ctx)]

    if len(prefix) < 20:
        return None
    if pos < len(text) and len(suffix) < 10:
        return None

    return make_sample(prefix, suffix, "")


# ==========================
# 主程序
# ==========================

def main():
    input_file = ROOT / "data" / "raw" / "input.txt"
    output_file = ROOT / "data" / "train" / "train.jsonl"

    text = Path(input_file).read_text(encoding="utf-8")
    paragraphs = [
        p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()
    ]

    docs = build_documents(paragraphs)

    # 每文档采样数与长度成正比
    per_doc_n = [max(4, len(d) // 250) for d in docs]
    total = sum(per_doc_n)

    # 全局配额 -> 打散分配到 (doc, slot)
    n_no = round(total * NO_COMPLETION_RATIO)
    n_end = round(total * END_COMPLETION_RATIO)

    slots = []
    for i, n in enumerate(per_doc_n):
        slots.extend([i] * n)
    random.shuffle(slots)

    slot_types = (
        ["no"] * n_no
        + ["end"] * n_end
        + ["normal"] * (total - n_no - n_end)
    )
    random.shuffle(slot_types)

    # 每文档需要生成的各类型数量
    plan = [{} for _ in docs]
    for doc_id, t in zip(slots, slot_types):
        plan[doc_id][t] = plan[doc_id].get(t, 0) + 1

    used_middles = set()
    results = []

    for doc_id, doc in enumerate(docs):
        for t, count in sorted(plan[doc_id].items()):
            for _ in range(count):
                if t == "no":
                    sample = create_no_completion_sample(doc)
                elif t == "end":
                    sample = create_end_sample(doc, used_middles)
                else:
                    sample = create_normal_sample(doc, used_middles)

                # 生成失败时降级为普通样本再试一次
                if sample is None and t != "normal":
                    sample = create_normal_sample(doc, used_middles)
                if sample is None:
                    continue

                results.append({"text": sample, "doc": doc_id})

    random.shuffle(results)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    n_no_actual = sum(
        1 for r in results
        if r["text"].endswith(f"{FIM_MIDDLE}{EOS}")
    )
    n_end_actual = sum(
        1 for r in results
        if f"{FIM_SUFFIX}{FIM_MIDDLE}" in r["text"]
        and not r["text"].endswith(f"{FIM_MIDDLE}{EOS}")
    )
    print("段落数量:", len(paragraphs))
    print("文档数量:", len(docs))
    print("FIM数量:", len(results))
    print(
        f"无需补全: {n_no_actual} ({n_no_actual / max(len(results), 1) * 100:.1f}%)"
    )
    print(
        f"末尾补全: {n_end_actual} ({n_end_actual / max(len(results), 1) * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
