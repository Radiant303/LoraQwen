import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

# =========================
# 合并模型测试:FIM 补全 + persona 聊天
# =========================

MODEL = "./SpringNote-Qwen3-0.6B-FIM-Persona"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()


def gen(prompt, n=128):

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


# ---------- 1. FIM 补全 ----------

prefix = "Graph 微服务的交互流程也跑通了。整体逻辑:"

suffix = "返回需要调用的工具以及是否需要调用的决策。后续节点判断是否中断，如果信息不足，就通过 SSE 把提示信息发给调用方，调用方可以继续补充信息；信息齐全后进入工具执行节点，执行结果以 PCM 语音格式通过 SSE 推送给用户，最后由 finishNode 发送完成信息。整个交互过程中的消息推送都走 SSE 的 PCM 格式。"

fim_prompt = (
    "<|fim_prefix|>\n"
    + prefix
    + "\n\n<|fim_suffix|>\n"
    + suffix
    + "\n\n<|fim_middle|>\n"
)

print("=" * 70)

print("【FIM 补全测试】")
middle = gen(fim_prompt).split("<|im_end|>")[0].strip()
print("=======prefix======")
print(prefix)
print("=======middle======")
print(middle)
print("=======suffix======")
print(suffix)
# ---------- 2. persona 聊天 ----------
SYS = "你是SpringNote官方AI助手。\n\n你由陈果果基于Qwen3模型微调开发。\n\n你的职责是帮助用户了解SpringNote、\n整理知识、处理笔记相关任务。\n\n回答要求：\n- 准确\n- 简洁\n- 不编造信息\n- 不知道的信息明确说明"

questions = [

    "SpringNote是谁开发的？",

    "陈果果毕业于哪里？",

    "SpringNote官方QQ群是多少？",

    "怎么联系SpringNote作者？",

    "SpringNote官网在哪里？",

]

print()

print("=" * 70)

print("【persona 聊天测试】")

for q in questions:

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYS},
            {"role": "user", "content": q}
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    ans = gen(prompt).split("<|im_end|>")[0].strip()

    print()

    print("Q:", q)

    print("A:", ans)
