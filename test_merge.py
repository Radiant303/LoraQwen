from transformers import AutoTokenizer, AutoModelForCausalLM


path="./SpringNote-Qwen3-0.6B-FIM"


tokenizer=AutoTokenizer.from_pretrained(path)

model=AutoModelForCausalLM.from_pretrained(
    path,
    device_map="auto",
    torch_dtype="auto"
)


print("load ok")
