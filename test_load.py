import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL = r"D:\GitHub\LoraQwen\Qwen3-0.6B"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)


print("device:", model.device)


text = "今天学习了"


inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)


out = model.generate(
    **inputs,
    max_new_tokens=50
)


print(
    tokenizer.decode(
        out[0],
        skip_special_tokens=True
    )
)
