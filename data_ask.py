import json


SYSTEM = """
你是SpringNote官方AI助手。

你由陈果果基于Qwen3模型微调开发。

你的职责是帮助用户了解SpringNote、
整理知识、处理笔记相关任务。

回答要求：
- 准确
- 简洁
- 不编造信息
- 不知道的信息明确说明
""".strip()



def convert():

    with open(
        "persona_train.json",
        encoding="utf-8"
    ) as f:

        data=json.load(f)



    with open(
        "persona_train.jsonl",
        "w",
        encoding="utf-8"
    ) as f:


        for item in data:


            if (
                "instruction" not in item
                or
                "response" not in item
            ):
                continue



            text=(

                "<|im_start|>system\n"
                + SYSTEM
                +
                "\n<|im_end|>\n"

                "<|im_start|>user\n"
                +
                item["instruction"]
                +
                "\n<|im_end|>\n"

                "<|im_start|>assistant\n"
                +
                item["response"]
                +
                "\n<|im_end|>"

            )


            f.write(
                json.dumps(
                    {
                        "text":text
                    },
                    ensure_ascii=False
                )
                +
                "\n"
            )


    print(
        "完成:",
        len(data)
    )



if __name__=="__main__":
    convert()
