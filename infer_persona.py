import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel



BASE="./SpringNote-Qwen3-0.6B-FIM"

LORA="./output-qwen3-0.6-persona-v2"



tokenizer = AutoTokenizer.from_pretrained(
    BASE,
    trust_remote_code=True
)


model = AutoModelForCausalLM.from_pretrained(

    BASE,

    torch_dtype=torch.float16,

    device_map="auto",

    trust_remote_code=True

)


model = PeftModel.from_pretrained(
    model,
    LORA
)


model.eval()



tests=[

"SpringNote是谁开发的？",

"陈果果是谁？毕业于哪里？",

"Radiant303是谁？",

"SpringNote的GitHub地址是什么？",

"SpringNote官网在哪里？",

"SpringNote官方QQ群是多少？",

"怎么联系SpringNote作者？",

]



for q in tests:


    print("\n"+"="*60)

    print("问题:")
    print(q)



    messages=[

        {
            "role":"system",
            "content":
            "你是SpringNote官方AI助手。\n\n你由陈果果基于Qwen3模型微调开发。\n\n你的职责是帮助用户了解SpringNote、\n整理知识、处理笔记相关任务。\n\n回答要求：\n- 准确\n- 简洁\n- 不编造信息\n- 不知道的信息明确说明"
        },

        {
            "role":"user",
            "content":q
        }

    ]


    prompt=tokenizer.apply_chat_template(

        messages,

        tokenize=False,

        add_generation_prompt=True,

        enable_thinking=False

    )


    inputs=tokenizer(

        prompt,

        return_tensors="pt"

    ).to(model.device)



    with torch.no_grad():

        output=model.generate(

            **inputs,

            max_new_tokens=128,

            do_sample=False

        )



    answer=tokenizer.decode(

        output[0][inputs.input_ids.shape[1]:],

        skip_special_tokens=True

    )


    print("回答:")
    print(answer)
