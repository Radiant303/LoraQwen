import torch


from datasets import load_dataset


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


BASE_MODEL=r".\SpringNote-Qwen3-0.6B-FIM"

DATASET=r".\data\persona_train_aug.jsonl"

OUTPUT=r".\output-qwen3-0.6-persona-v2"



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
# assistant-only loss
# =========================


dataset=load_dataset(

    "json",

    data_files=DATASET

)["train"]



assistant_token_ids=tokenizer.encode(

    "<|im_start|>assistant",

    add_special_tokens=False

)



def tokenize(example):


    text=example["text"]



    result=tokenizer(

        text,

        max_length=512,

        truncation=True

    )



    input_ids=result["input_ids"]



    labels=input_ids.copy()



    # 默认全部不训练

    labels=[

        -100

        for _ in labels

    ]



    # 找 assistant 开始位置

    start=-1



    for i in range(

        len(input_ids)-len(assistant_token_ids)

    ):


        if input_ids[

            i:i+len(assistant_token_ids)

        ] == assistant_token_ids:


            start=i+len(assistant_token_ids)

            break



    if start!=-1:


        for i in range(

            start,

            len(input_ids)

        ):


            # padding不要训练

            if input_ids[i] != tokenizer.pad_token_id:

                labels[i]=input_ids[i]



    result["labels"]=labels



    return result





dataset=dataset.map(

    tokenize,

    remove_columns=dataset.column_names

)



# =========================
# 训练
# =========================


args=TrainingArguments(


    output_dir=OUTPUT,


    num_train_epochs=6,


    per_device_train_batch_size=8,


    gradient_accumulation_steps=2,


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
