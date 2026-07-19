import torch

from peft import PeftModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

# =========================
# persona LoRA 缩放实验
#
# 不重训，直接把 adapter 增量乘系数后合并，
# 找一个"FIM 不受污染 + persona 仍然可用"的强度
# =========================

BASE = "./SpringNote-Qwen3-0.6B-FIM"

LORA = "./output-qwen3-0.6-persona-v2"

SCALES = [1.0, 0.7, 0.5, 0.3]

tokenizer = AutoTokenizer.from_pretrained(BASE)

prefix = "Graph 微服务的交互流程也跑通了。整体逻"

suffix = "返回需要调用的工具以及是否需要调用的决策。后续节点判断是否中断，如果信息不足，就通过 SSE 把提示信息发给调用方，调用方可以继续补充信息；信息齐全后进入工具执行节点，执行结果以 PCM 语音格式通过 SSE 推送给用户，最后由 finishNode 发送完成信息。整个交互过程中的消息推送都走 SSE 的 PCM 格式。"

fim_prompt = (
    "<|fim_prefix|>"
    + prefix
    + "<|fim_suffix|>"
    + suffix
    + "<|fim_middle|>"
)

SYS = "你是SpringNote官方AI助手。\n\n你由陈果果基于Qwen3模型微调开发。\n\n你的职责是帮助用户了解SpringNote、\n整理知识、处理笔记相关任务。\n\n回答要求：\n- 准确\n- 简洁\n- 不编造信息\n- 不知道的信息明确说明"


def gen(model, prompt, n=128):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=n,
            do_sample=False
        )

    return tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=False
    )


for scale in SCALES:

    base = AutoModelForCausalLM.from_pretrained(
        BASE,
        dtype=torch.float16,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(
        base,
        LORA
    )

    # ΔW = B @ A，缩 B 即等比缩增量

    for name, p in model.named_parameters():

        if "lora_B" in name:
            p.data.mul_(scale)

    model = model.merge_and_unload()

    model.eval()

    print("=" * 80)

    print(f"SCALE = {scale}")

    middle = gen(model, fim_prompt)
    middle = middle.split("<|im_end|>")[0].strip()

    print("[FIM]", middle[:150].replace("\n", " / "))

    for q in ["SpringNote是谁开发的？", "陈果果毕业于哪里？"]:

        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYS},
                {"role": "user", "content": q}
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        ans = gen(model, prompt)
        ans = ans.split("<|im_end|>")[0].strip()

        print(f"[Q] {q} -> {ans[:80]}")

    del model

    torch.cuda.empty_cache()
