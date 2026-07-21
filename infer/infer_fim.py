# =========================
# FIM 模型推理测试
# 位置：infer/infer_fim.py
#
# 加载基础模型（及可选 LoRA），对固定 prefix/suffix 做中间补全。
# =========================

import torch
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel



MODELS = [
#   (
#        "Qwen3-0.6B 原版",
#        ROOT / "models" / "base" / "Qwen3-0.6B",
#        None
#    ),
#    (
#        "Qwen3-0.6B + FIM-V5",
#        ROOT / "models" / "base" / "Qwen3-0.6B",
#        ROOT / "models" / "adapters" / "output-qwen3-0.6-fim-v5"
#    ),
#    (
#        "Qwen3-1.7B 原版",
#        ROOT / "models" / "base" / "Qwen3-1.7B",
#        None
#    ),
#    (
#        "Qwen3-1.7B + FIM-V5",
#        ROOT / "models" / "base" / "Qwen3-1.7B",
#        ROOT / "models" / "adapters" / "output-qwen3-1.7-fim-v5"
#    ),
#    (
#        "Qwen3-0.6B-FIM + Persona-V2",
#        ROOT / "models" / "fim" / "SpringNote-Qwen3-0.6B-FIM",
#        ROOT / "models" / "adapters" / "output-qwen3-0.6-persona-v2"
#    ),
    (
        "Qwen3-1.7B + FIM-V6",
        ROOT / "models" / "fim" / "SpringNote-Qwen3-1.7B-FIM-V7",
        None
    )

]


# ==========================
# FIM 三段
# ==========================

prefix = """
# 2026-07-21 日报

今天主要完成了SpringNote 1.0.3版本的发布工作。前端方面，
"""


suffix = """
将原本基于Cocos的界面迁移到了Vue框架中，并引入了Pixi.js来优化Canvas渲染性能，整体交互流畅度有了明显提升。

与此同时，开始着手Spring AI Alibaba微服务模块的学习和设计实现，目前处于初步调研和架构设计阶段，后续会逐步推进具体模块的开发。
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

            temperature=0.7,

            top_p=0.8,

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
