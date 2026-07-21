# =========================
# Persona / tools / think 多任务 LoRA 训练脚本
# 位置：train/train_persona.py
#
# 在 FIM 模型基础上混合人设、工具调用、思考链与 FIM rehearsal 数据继续训练。
#
# 与旧版的差异：
# - 每个任务单独留出验证集，训练结束自动保留验证 loss 最优的 checkpoint，
#   并在训练后按任务分别报告 eval loss（旧版全程无评估，超参靠肉眼调）
# - 任务配比显式化：persona 靠重复记忆、tools 靠泛化，
#   通过 TASKS 里的 repeat 字段控制各任务采样权重
# - 开启 gradient_checkpointing（旧版注释说开了但实际是 False），
#   batch 提到 4×4，等效 batch 仍为 16
# - 精度保持 fp16，与 FIM 训练统一
# =========================

import os
import torch
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


from datasets import (
    load_dataset,
    concatenate_datasets
)


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)


from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)


# =========================
# 路径与任务配比
# =========================

BASE_MODEL = ROOT / "models" / "fim" / "SpringNote-Qwen3-1.7B-FIM-V8"

# 多任务数据：人设 + 工具调用 + 思考链（文件不存在自动跳过）
# repeat 控制任务在训练集中的采样权重：
#   persona 靠重复记忆事实，权重 2；tools/think 靠泛化，权重 1
# eval_n 为该任务留出的验证集条数
TASKS = [
    {
        "name": "persona",
        "path": ROOT / "data" / "train" / "persona_train_aug_v2.jsonl",
        "repeat": 2,
        "eval_n": 40,
    },
    {
        "name": "tools",
        "path": ROOT / "data" / "train" / "tools_train_v2.jsonl",
        "repeat": 1,
        "eval_n": 50,
    },
    {
        "name": "think",
        "path": ROOT / "data" / "train" / "think_train_v2.jsonl",
        "repeat": 1,
        "eval_n": 25,
    },
]

# FIM rehearsal，防止 chat 训练再次污染补全能力
FIM_REHEARSAL = ROOT / "data" / "train" / "train.jsonl"
FIM_REHEARSAL_N = 500

OUTPUT = ROOT / "models" / "adapters" / "output-qwen3-1.7-persona-v8"


# =========================
# tokenizer
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# =========================
# 4bit（8G 显存可跑；A10 上可改全量 fp16，配方保持不变）
# =========================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model.config.use_cache = False

model = prepare_model_for_kbit_training(model)


# =========================
# LoRA
# =========================

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# =========================
# loss mask 工具
# =========================

assistant_marker = tokenizer.encode(
    "<|im_start|>assistant",
    add_special_tokens=False
)

im_end_id = tokenizer.encode(
    "<|im_end|>",
    add_special_tokens=False
)[0]

FIM_MARKER = "<|fim_middle|>"


def tokenize(example):
    text = example["text"]

    result = tokenizer(
        text,
        max_length=8192,
        truncation=True,
        add_special_tokens=False,
    )

    input_ids = result["input_ids"]

    # 默认全部不训练
    labels = [-100 for _ in input_ids]

    if FIM_MARKER in text:
        # FIM rehearsal：只训练 middle 和结束符
        prefix_text = text[:text.find(FIM_MARKER) + len(FIM_MARKER)]
        start = len(
            tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        )
        for i in range(start, len(input_ids)):
            labels[i] = input_ids[i]
    else:
        # 每个 assistant 段都参与 loss（含 tool_call 段），
        # user/tool 返回段不训练；<|im_end|> 参与，学会停止
        m = len(assistant_marker)
        i = 0
        while i <= len(input_ids) - m:
            if input_ids[i:i + m] == assistant_marker:
                j = i + m
                while j < len(input_ids) and input_ids[j] != im_end_id:
                    labels[j] = input_ids[j]
                    j += 1
                if j < len(input_ids):
                    labels[j] = input_ids[j]
                    j += 1
                i = j
            else:
                i += 1

    result["labels"] = labels
    return result


# =========================
# 数据集：分任务加载、留验证集、按配比混合
# =========================

train_parts = []
eval_parts = []
task_eval_sets = {}

for task in TASKS:
    if not os.path.exists(task["path"]):
        continue

    ds = load_dataset("json", data_files=str(task["path"]))["train"]
    ds = ds.shuffle(seed=42)

    eval_n = min(task["eval_n"], max(1, len(ds) // 10))
    eval_ds = ds.select(range(eval_n))
    train_ds = ds.select(range(eval_n, len(ds)))

    train_ds = train_ds.map(
        tokenize, remove_columns=train_ds.column_names
    )
    eval_ds = eval_ds.map(
        tokenize, remove_columns=eval_ds.column_names
    )

    train_parts.extend([train_ds] * task["repeat"])
    eval_parts.append(eval_ds)
    task_eval_sets[task["name"]] = eval_ds

    print(
        f"任务 {task['name']}: 训练 {len(train_ds)}×{task['repeat']}"
        f"  验证 {len(eval_ds)}"
    )

fim = load_dataset(
    "json", data_files=str(FIM_REHEARSAL)
)["train"].shuffle(seed=42).select(range(FIM_REHEARSAL_N))
fim = fim.map(tokenize, remove_columns=fim.column_names)
train_parts.append(fim)
print(f"任务 fim_rehearsal: 训练 {len(fim)}×1  验证 0（由 FIM 阶段保证）")

train_dataset = concatenate_datasets(train_parts).shuffle(seed=42)
eval_dataset = concatenate_datasets(eval_parts)

# 丢弃没有任何训练 token 的样本
train_dataset = train_dataset.filter(
    lambda x: any(l != -100 for l in x["labels"])
)

print("总训练样本数:", len(train_dataset))
print("总验证样本数:", len(eval_dataset))


# =========================
# 训练
# =========================

args = TrainingArguments(
    output_dir=OUTPUT,
    num_train_epochs=3,

    # 工具样本约 2700 token，logits 很吃显存，
    # batch 4 + gradient checkpointing 保 8G 显存，等效 batch 仍为 16
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,

    learning_rate=1e-4,
    fp16=True,
    logging_steps=10,

    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,

    optim="paged_adamw_8bit",
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    report_to="none",
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer,
        padding=True
    )
)


trainer.train()


# =========================
# 分任务评估报告 + 保存
# =========================

print("\n========== 分任务验证集 loss ==========")
for name, eds in task_eval_sets.items():
    metrics = trainer.evaluate(eval_dataset=eds)
    print(f"  {name}: loss={metrics['eval_loss']:.4f}")

# load_best_model_at_end 已把验证 loss 最优的 checkpoint 载回
model.save_pretrained(OUTPUT)
tokenizer.save_pretrained(OUTPUT)

print("训练完成:", OUTPUT)
