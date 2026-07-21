from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE = r"./Qwen3-1.7B"
LORA = r"./output-qwen3-1.7-fim-v7"
OUTPUT = r"./SpringNote-Qwen3-1.7B-FIM-V7"



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


print("✅ SpringNote-Qwen3-0.6B-FIM-V7 合并完成")
