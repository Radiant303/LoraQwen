# =========================
# FIM LoRA 训练脚本
# 位置：train/train_fim.py
#
# 加载 Qwen3-1.7B，用 data/train/train.jsonl 训练 FIM 补全能力。
#
# 与旧版的差异：
# - 精度统一为 fp16（与 persona 训练一致；A10 24G 全量 fp16 + LoRA 无压力）
# - MAX_LENGTH 1024 -> 2048（数据构建器已支持更长上下文，超长样本直接
#   丢弃而不是截断——截断会切掉 middle 和 EOS，产生坏样本）
# - 按"文档"分组切出 3% 验证集（build_fim_data.py 输出的 doc 字段）。
#   注意不能随机按行切：同一文档会生成多个样本，随机切会把同文档
#   泄漏到训练/验证两边，验证 loss 会虚假偏低
# - 每个 epoch 评估，训练结束自动保留验证 loss 最优的 checkpoint
# =========================

import random

import torch
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

from peft import (
    LoraConfig,
    get_peft_model,
)


# =========================
# 配置（A10 24G，fp16）
# =========================

MODEL_PATH = ROOT / "models" / "base" / "Qwen3-1.7B"
OUTPUT_DIR = ROOT / "models" / "adapters" / "output-qwen3-1.7-fim-v8"
DATA_PATH = ROOT / "data" / "train" / "train.jsonl"

MAX_LENGTH = 2048
EVAL_DOC_RATIO = 0.03


# =========================
# Tokenizer
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# =========================
# fp16 加载模型（与 persona 训练精度统一）
# =========================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False

model.gradient_checkpointing_enable()


# =========================
# LoRA
# =========================

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# =========================
# Dataset
# =========================

dataset = load_dataset(
    "json",
    data_files=str(DATA_PATH),
    split="train"
)

print("数据量:", len(dataset))


# =========================
# FIM tokenize
# =========================

def tokenize(example):
    text = example["text"]

    if not text.endswith(tokenizer.eos_token):
        text += tokenizer.eos_token

    # 不截断：超长样本在后面的 filter 中整条丢弃。
    # 截断会切掉 middle 结尾和 EOS，造出"永远学不到停止"的坏样本。
    encoded = tokenizer(
        text,
        truncation=False,
        padding=False,
        add_special_tokens=False,
    )

    input_ids = encoded["input_ids"]

    labels = [-100] * len(input_ids)

    middle_token = "<|fim_middle|>"
    middle_pos = text.find(middle_token)

    if middle_pos != -1:
        prefix_text = text[:middle_pos + len(middle_token)]
        middle_start = len(
            tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        )
        if middle_start < len(input_ids):
            labels[middle_start:] = input_ids[middle_start:]

    return {
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


dataset = dataset.map(
    tokenize,
    remove_columns=["text"],
)

dataset = dataset.filter(
    lambda x: len(x["input_ids"]) <= MAX_LENGTH
    and any(l != -100 for l in x["labels"])
)

print("有效FIM:", len(dataset))


# =========================
# 按文档分组切分验证集（防同文档泄漏）
# =========================

doc_ids = list(set(dataset["doc"]))
random.Random(42).shuffle(doc_ids)

n_eval_docs = max(1, int(len(doc_ids) * EVAL_DOC_RATIO))
eval_docs = set(doc_ids[:n_eval_docs])

eval_dataset = dataset.filter(lambda x: x["doc"] in eval_docs)
train_dataset = dataset.filter(lambda x: x["doc"] not in eval_docs)

train_dataset = train_dataset.remove_columns(["doc"])
eval_dataset = eval_dataset.remove_columns(["doc"])

print(f"训练集: {len(train_dataset)}  验证集: {len(eval_dataset)}")


# =========================
# Collator
# =========================

def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    masks = []
    labels = []

    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids.append(
            item["input_ids"] + [tokenizer.pad_token_id] * pad
        )
        masks.append(item["attention_mask"] + [0] * pad)
        labels.append(item["labels"] + [-100] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# =========================
# Training
# =========================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_ratio=0.03,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    fp16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    optim="adamw_torch_fused",
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    report_to="none",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
)

trainer.train()


# =========================
# 保存（load_best_model_at_end 已把最优 checkpoint 载回）
# =========================

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("最终验证集评估:", trainer.evaluate())
print("完成:", OUTPUT_DIR)
