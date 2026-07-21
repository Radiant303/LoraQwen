# =========================
# 训练数据一键构建脚本
# 位置：train/build_all_data.py
#
# 依次执行各个数据生成脚本，从源数据直接生成 data/train/ 下的最终训练文件：
#   - train/build_fim_data.py     -> data/train/train.jsonl
#   - train/build_persona_data.py -> data/train/persona_train_aug_v2.jsonl
#   - train/build_tools_data.py   -> data/train/tools_train_v2.jsonl
#   - train/build_think_data.py   -> data/train/think_train_v2.jsonl
# =========================

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


STEPS = [
    ("FIM 数据", ROOT / "train" / "build_fim_data.py", ROOT / "data" / "train" / "train.jsonl"),
    ("Persona 数据", ROOT / "train" / "build_persona_data.py", ROOT / "data" / "train" / "persona_train_aug_v2.jsonl"),
    ("Tools 数据", ROOT / "train" / "build_tools_data.py", ROOT / "data" / "train" / "tools_train_v2.jsonl"),
    ("Think 数据", ROOT / "train" / "build_think_data.py", ROOT / "data" / "train" / "think_train_v2.jsonl"),
]


def run_step(label, script, output):
    print(f"\n>>> 开始生成: {label} ({script.name}) -> {output}")
    result = subprocess.run(
        [sys.executable, str(script)],
        check=True,
    )
    print(f"    {label} 完成，返回码: {result.returncode}")


def main():
    print("=" * 50)
    print("开始构建全部训练数据")
    print("=" * 50)

    for label, script, output in STEPS:
        run_step(label, script, output)

    print("\n" + "=" * 50)
    print("生成结果汇总")
    print("=" * 50)

    for _, _, output in STEPS:
        if output.exists() and output.stat().st_size > 0:
            print(f"  ✓ {output}: {output.stat().st_size} bytes")
        else:
            print(f"  ✗ {output}: 不存在或为空")


if __name__ == "__main__":
    main()
