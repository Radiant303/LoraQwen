import os
from openai import OpenAI

client = OpenAI(
    base_url="https://llm-x1qj7pwbasho4jef.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

prefix_content = f"""def reverse_words_with_special_chars(s):
'''
反转字符串中的每个单词（保留非字母字符的位置），并保持单词顺序。
    示例:
    reverse_words_with_special_chars("Hello, world!") -> "olleH, dlrow!"
    参数:
        s (str): 输入字符串（可能包含标点符号）
    返回:
        str: 处理后的字符串，单词反转但非字母字符位置不变
'''
"""

suffix_content = "return result"

completion = client.completions.create(
  model="qwen-coder-turbo",
  prompt=f"<|fim_prefix|>\n{prefix_content}\n\n<|fim_suffix|>\n{suffix_content}\n\n<|fim_middle|>\n",
)

print(completion.choices[0].text)
