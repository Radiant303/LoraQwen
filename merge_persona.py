from peft import PeftModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

# =========================
# 合并 persona LoRA（带强度缩放）
#
# SCALE 控制 persona 强度：
#   1.0  persona 最强，但 FIM 会被污染
#   0.7  推荐甜点：FIM 干净，事实问答正确
#   0.3  persona 开始失效
# =========================

BASE = r"./SpringNote-Qwen3-0.6B-FIM-V2"

LORA = r"./output-qwen3-0.6-persona-v3"

OUTPUT = r"./SpringNote-Qwen3-0.6B-FIM-Persona-V2"

SCALE = 0.7


base = AutoModelForCausalLM.from_pretrained(
    BASE,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(
    base,
    LORA
)

# ΔW = B @ A，缩放 lora_B 即等比缩放 persona 增量

for name, p in model.named_parameters():

    if "lora_B" in name:
        p.data.mul_(SCALE)

model = model.merge_and_unload()

model.save_pretrained(
    OUTPUT,
    safe_serialization=True
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE,
    trust_remote_code=True
)

tokenizer.save_pretrained(
    OUTPUT
)

print(f"合并完成 SCALE={SCALE} -> {OUTPUT}")
