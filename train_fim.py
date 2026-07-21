import torch

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
# 配置 注意此文件建议在5090 32G机器上进行运行
# =========================

MODEL_PATH = "./Qwen3-1.7B"
OUTPUT_DIR = "./output-qwen3-1.7-fim-v6"
DATA_PATH = "./data/train.jsonl"

MAX_LENGTH = 1024


# =========================
# Tokenizer
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print(tokenizer.special_tokens_map)


# =========================
# BF16 加载模型
# =========================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
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


model = get_peft_model(
    model,
    lora_config
)


model.print_trainable_parameters()



# =========================
# Dataset
# =========================

dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train"
)


print(
    "数据量:",
    len(dataset)
)



# =========================
# FIM tokenize
# =========================

def tokenize(example):

    text = example["text"]


    if not text.endswith(tokenizer.eos_token):

        text += tokenizer.eos_token



    encoded = tokenizer(

        text,

        max_length=MAX_LENGTH,

        truncation=True,

        padding=False,

        add_special_tokens=False,

    )


    input_ids = encoded["input_ids"]


    labels = [
        -100
    ] * len(input_ids)



    middle_token = "<|fim_middle|>"


    middle_pos = text.find(
        middle_token
    )


    if middle_pos == -1:

        return {

            "input_ids": input_ids,

            "attention_mask":
                encoded["attention_mask"],

            "labels": labels

        }



    prefix_text = text[

        :middle_pos + len(middle_token)

    ]



    middle_start = len(

        tokenizer(

            prefix_text,

            add_special_tokens=False

        )["input_ids"]

    )



    if middle_start < len(input_ids):

        labels[middle_start:] = (

            input_ids[middle_start:]

        )



    return {

        "input_ids":

            input_ids,


        "attention_mask":

            encoded["attention_mask"],


        "labels":

            labels

    }



dataset = dataset.map(

    tokenize,

    remove_columns=[
        "text"
    ],

)



dataset = dataset.filter(

    lambda x:

        any(
            l != -100
            for l in x["labels"]
        )

)



print(
    "有效FIM:",
    len(dataset)
)



# =========================
# Collator
# =========================

def collate_fn(batch):


    max_len = max(

        len(x["input_ids"])

        for x in batch

    )


    input_ids = []

    masks = []

    labels = []


    for item in batch:


        pad = max_len - len(
            item["input_ids"]
        )


        input_ids.append(

            item["input_ids"]

            +

            [
                tokenizer.pad_token_id
            ] * pad

        )


        masks.append(

            item["attention_mask"]

            +

            [
                0
            ] * pad

        )


        labels.append(

            item["labels"]

            +

            [
                -100
            ] * pad

        )


    return {


        "input_ids":

            torch.tensor(
                input_ids,
                dtype=torch.long
            ),


        "attention_mask":

            torch.tensor(
                masks,
                dtype=torch.long
            ),


        "labels":

            torch.tensor(
                labels,
                dtype=torch.long
            )

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


    bf16=True,


    logging_steps=10,


    save_steps=500,


    save_total_limit=3,


    optim="adamw_torch_fused",


    dataloader_num_workers=8,


    dataloader_pin_memory=True,


    report_to="none",

)



trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=dataset,

    data_collator=collate_fn,

)



trainer.train()



# =========================
# 保存
# =========================

model.save_pretrained(
    OUTPUT_DIR
)


tokenizer.save_pretrained(
    OUTPUT_DIR
)


print(
    "完成:",
    OUTPUT_DIR
)
