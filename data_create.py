import json
import random
import re

from pathlib import Path


FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
EOS = "<|im_end|>"



# ==========================
# 清理文本
# ==========================

def clean_text(text):

    text = text.strip()

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text




# ==========================
# 找自然结束点
# ==========================

def find_middle_end(
        text,
        start,
        target_len,
        allow_expand=True):


    end = min(
        start + target_len,
        len(text)
    )


    # 极短补全不要扩展
    if not allow_expand:

        return end



    search_end=min(
        end+40,
        len(text)
    )


    area=text[
        start:search_end
    ]


    matches=list(
        re.finditer(
            r"[，。；：！？\n]",
            area
        )
    )


    if matches:

        # 找最近标点

        return (
            start
            +
            matches[-1].end()
        )


    return end





# ==========================
# middle长度策略
# ==========================

def random_middle_length():


    r=random.random()


    # 极短补全 25%
    if r < 0.25:

        return (
            random.randint(1,8),
            False
        )


    # 短补全 35%
    elif r < 0.60:

        return (
            random.randint(8,30),
            True
        )


    # 中补全 25%
    elif r < 0.85:

        return (
            random.randint(30,80),
            True
        )


    # 长补全 15%

    else:

        return (
            random.randint(80,160),
            True
        )





# ==========================
# 创建样本
# ==========================

def create_samples(report):


    text=clean_text(report)


    samples=[]


    if len(text)<80:

        return samples



    used=set()



    # 每条生成多个

    for _ in range(8):


        pos=random.randint(
            int(len(text)*0.1),
            int(len(text)*0.9)
        )


        middle_len,expand=random_middle_length()



        end=find_middle_end(
            text,
            pos,
            middle_len,
            expand
        )


        if end>=len(text):

            continue



        middle=text[
            pos:end
        ]



        # 不允许空

        if len(middle)<1:

            continue



        # =====================
        # 上下文窗口
        # =====================

        context_size=300



        prefix=text[
            max(
                0,
                pos-context_size
            ):
            pos
        ]


        suffix=text[
            end:
            min(
                len(text),
                end+context_size
            )
        ]



        if len(prefix)<20:

            continue


        if len(suffix)<10:

            continue



        sample=(
            f"{FIM_PREFIX}\n"
            f"{prefix}\n\n"
            f"{FIM_SUFFIX}\n"
            f"{suffix}\n\n"
            f"{FIM_MIDDLE}\n"
            f"{middle}"
            f"{EOS}"
        )


        if sample in used:

            continue


        used.add(sample)


        samples.append(
            {
                "text":sample
            }
        )


    return samples





# ==========================
# 主程序
# ==========================

def main():


    input_file="input.txt"

    output_file="fim_dataset.jsonl"



    text=Path(
        input_file
    ).read_text(
        encoding="utf-8"
    )



    reports=re.split(
        r"\n\s*\n",
        text
    )



    results=[]



    for report in reports:


        report=report.strip()


        if not report:

            continue


        results.extend(
            create_samples(
                report
            )
        )



    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:


        for item in results:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                +
                "\n"
            )



    print(
        "事项数量:",
        len(reports)
    )

    print(
        "FIM数量:",
        len(results)
    )



if __name__=="__main__":

    main()
