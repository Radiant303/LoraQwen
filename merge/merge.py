# =========================
# 合并 FIM LoRA 到基础模型并保存完整模型
# 位置：merge/merge.py
#
# 加载 Qwen3-1.7B 基础模型与 LoRA，合并权重后保存为可独立部署的模型目录。
# =========================

from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "models" / "base" / "Qwen3-1.7B"
LORA = ROOT / "models" / "adapters" / "output-qwen3-1.7-fim-v8"
OUTPUT = ROOT / "models" / "fim" / "SpringNote-Qwen3-1.7B-FIM-V8"



# 加载基础模型
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)


# 加载 LoRA
model = PeftModel.from_pretrained(
    base,
    LORA
)


# 合并 LoRA 权重
model = model.merge_and_unload()


# 保存模型
model.save_pretrained(
    OUTPUT,
    safe_serialization=True
)


# tokenizer也必须保存
tokenizer = AutoTokenizer.from_pretrained(
    BASE,
    trust_remote_code=True
)

tokenizer.save_pretrained(
    OUTPUT
)


print("✅ SpringNote-Qwen3-1.7B-FIM-V8 合并完成")
