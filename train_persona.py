import os

import torch


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
# 路径
# =========================


BASE_MODEL="./SpringNote-Qwen3-0.6B-FIM-V2"

# 多任务数据:人设 + 工具调用 + 思考链(不存在自动跳过)

DATASETS=[

    "./data/persona_train_aug.jsonl",

    "./data/tools_train.jsonl",

    "./data/think_train.jsonl"

]

# FIM rehearsal,防止chat训练再次污染补全能力

FIM_REHEARSAL="./data/train.jsonl"

FIM_REHEARSAL_N=500

OUTPUT="./output-qwen3-0.6-persona-v4"



# =========================
# tokenizer
# =========================


tokenizer=AutoTokenizer.from_pretrained(

    BASE_MODEL,

    trust_remote_code=True

)


if tokenizer.pad_token is None:

    tokenizer.pad_token=tokenizer.eos_token



# =========================
# 4bit
# =========================


bnb_config=BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True

)



model=AutoModelForCausalLM.from_pretrained(

    BASE_MODEL,

    quantization_config=bnb_config,

    device_map="auto",

    trust_remote_code=True

)



model.config.use_cache=False


model=prepare_model_for_kbit_training(
    model
)



# =========================
# LoRA
# =========================


lora_config=LoraConfig(

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



model=get_peft_model(

    model,

    lora_config

)


model.print_trainable_parameters()



# =========================
# 数据集:多任务混合
# =========================


files=[

    f

    for f in DATASETS

    if os.path.exists(f)

]

print(

    "训练文件:",

    files

)


dataset=load_dataset(

    "json",

    data_files=files

)["train"]


fim=load_dataset(

    "json",

    data_files=FIM_REHEARSAL

)["train"].shuffle(

    seed=42

).select(

    range(FIM_REHEARSAL_N)

)


dataset=concatenate_datasets(

    [dataset, fim]

).shuffle(

    seed=42

)


print(

    "总样本数:",

    len(dataset)

)



# =========================
# loss mask
# =========================


assistant_marker=tokenizer.encode(

    "<|im_start|>assistant",

    add_special_tokens=False

)


im_end_id=tokenizer.encode(

    "<|im_end|>",

    add_special_tokens=False

)[0]


FIM_MARKER="<|fim_middle|>"



def tokenize(example):


    text=example["text"]



    result=tokenizer(

        text,

        max_length=8192,

        truncation=True

    )



    input_ids=result["input_ids"]



    # 默认全部不训练

    labels=[

        -100

        for _ in input_ids

    ]



    if FIM_MARKER in text:


        # FIM rehearsal:只训练 middle 和结束符

        prefix_text=text[

            :text.find(FIM_MARKER)+len(FIM_MARKER)

        ]


        start=len(

            tokenizer(

                prefix_text,

                add_special_tokens=False

            )["input_ids"]

        )


        for i in range(

            start,

            len(input_ids)

        ):


            labels[i]=input_ids[i]


    else:


        # 每个 assistant 段都参与 loss(含 tool_call 段),
        # user/tool 返回段不训练;<|im_end|> 参与,学会停止

        m=len(assistant_marker)

        i=0


        while i<=len(input_ids)-m:


            if input_ids[

                i:i+m

            ] == assistant_marker:


                j=i+m


                while (

                    j<len(input_ids)

                    and input_ids[j]!=im_end_id

                ):


                    labels[j]=input_ids[j]

                    j+=1


                if j<len(input_ids):

                    labels[j]=input_ids[j]

                    j+=1


                i=j

            else:

                i+=1



    result["labels"]=labels



    return result





dataset=dataset.map(

    tokenize,

    remove_columns=dataset.column_names

)


dataset=dataset.filter(

    lambda x: any(

        l != -100 for l in x["labels"]

    )

)



# =========================
# 训练
# =========================


args=TrainingArguments(


    output_dir=OUTPUT,


    num_train_epochs=3,


    # 工具样本约2700 token,logits 很吃显存,
    # batch 2 + checkpointing 保 8G 显存,等效 batch 仍为 16

    per_device_train_batch_size=2,


    gradient_accumulation_steps=8,


    learning_rate=1e-4,


    fp16=True,


    logging_steps=10,


    save_strategy="epoch",


    optim="paged_adamw_8bit",


    warmup_ratio=0.05,


    lr_scheduler_type="cosine",


    report_to="none",


    gradient_checkpointing=False

)



trainer=Trainer(

    model=model,

    args=args,

    train_dataset=dataset,

    data_collator=DataCollatorForSeq2Seq(

        tokenizer,

        padding=True

    )

)



trainer.train()



# =========================
# 保存
# =========================


model.save_pretrained(

    OUTPUT

)


tokenizer.save_pretrained(

    OUTPUT

)



print(

    "训练完成:",

    OUTPUT

)
