import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel



MODELS = [
   (
        "Qwen3-0.6B 原版",
        r"./Qwen3-0.6B",
        None
    ),
#    (
#        "Qwen3-0.6B + FIM-V5",
#        r"./Qwen3-0.6B",
#        r"./output-qwen3-0.6-fim-v5"
#    ),
#    (
#        "Qwen3-1.7B 原版",
#        r"./Qwen3-1.7B",
#        None
#    ),
#    (
#        "Qwen3-1.7B + FIM-V5",
#        r"./Qwen3-1.7B",
#        r"./output-qwen3-1.7-fim-v5"
#    ),
    (
        "Qwen3-0.6B-FIM + Persona-V2",
        r"./SpringNote-Qwen3-0.6B-FIM",
        r"./output-qwen3-0.6-persona-v2"
    )

]


# ==========================
# FIM 三段
# ==========================

prefix = """
Graph 微服务的交互流程也跑通了。整体逻
"""


suffix = """
返回需要调用的工具以及是否需要调用的决策。后续节点判断是否中断，如果信息不足，就通过 SSE 把提示信息发给调用方，调用方可以继续补充信息；信息齐全后进入工具执行节点，执行结果以 PCM 语音格式通过 SSE 推送给用户，最后由 finishNode 发送完成信息。整个交互过程中的消息推送都走 SSE 的 PCM 格式。
"""


# 与训练数据严格一致的格式：
# <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>
fim_prompt = (
    "<|fim_prefix|>"
    + prefix.strip()
    + "<|fim_suffix|>"
    + suffix.strip()
    + "<|fim_middle|>"
)



def load_model(base_path, lora_path):

    tokenizer = AutoTokenizer.from_pretrained(
        base_path
    )


    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )


    if lora_path:

        model = PeftModel.from_pretrained(
            model,
            lora_path
        )


    model.eval()

    return tokenizer, model




def inference(tokenizer, model):

    inputs = tokenizer(
        fim_prompt,
        return_tensors="pt"
    ).to(model.device)


    with torch.no_grad():

        output = model.generate(

            **inputs,

            max_new_tokens=128,

            do_sample=False,

            temperature=0.7,

            top_p=0.9,

            eos_token_id=tokenizer.eos_token_id,

        )


    # 只decode新生成的部分，按<|im_end|>截断
    gen_ids = output[0][
        inputs["input_ids"].shape[1]:
    ]


    middle = tokenizer.decode(
        gen_ids,
        skip_special_tokens=False
    ).split("<|im_end|>")[0]


    return middle.strip()



for name, base, lora in MODELS:


    print("\n")
    print("=" * 90)
    print(name)
    print("=" * 90)



    tokenizer, model = load_model(
        base,
        lora
    )


    middle = inference(
        tokenizer,
        model
    )



    print("\n【上文 Prefix】")
    print("-" * 90)
    print(prefix.strip())


    print("\n【模型补全 Middle】")
    print("-" * 90)
    print(middle)


    print("\n【下文 Suffix】")
    print("-" * 90)
    print(suffix.strip())


    print("\n")


    del model
    del tokenizer

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
