# LoraQwen

基于 Qwen3 的 LoRA 微调项目，分两步构建：

1. **FIM**：先让 Qwen3-1.7B 学会中间补全（Fill-in-the-Middle），合并成底座模型
2. **Persona**：再在底座上训练 SpringNote 产品人设（开发者、官网、QQ群等事实 + 隐私问题拒答）

最终产物是单个合并模型 `models/persona/SpringNote-Qwen3-1.7B-FIM-Persona-V5`，可同时做补全和聊天。

## 环境

- Windows + RTX 4060 Laptop (8GB)
- Python 3.11，依赖由 uv 管理（`uv sync`）
- 基础模型：`models/base/Qwen3-1.7B/`（本地目录）

## 整体流水线

![整体流水线图](./images/pipeline.png)

---

# 第一部分：FIM 底座

## 1.1 数据收集与存放

**源文件是 `data/raw/input.txt`**——原始文档（工作日志、技术笔记等纯文本），用空行分段，每段作为一个独立文档。

不需要手工标注，`train/build_fim_data.py` 自动从每段中随机"挖空"生成补全样本，直接输出到 `data/train/train.jsonl`：

- 每段随机取 8 个位置，挖掉中间一段作为 middle，前后各留 300 字符作为 prefix/suffix
- middle 长度按比例混合：25% 极短（1~8 字符）、35% 短、25% 中、15% 长（80~160 字符）
- 优先在标点处截断，保证 middle 语义完整

生成的样本格式（训练/推理必须严格一致，token 之间无换行）：

```
<|fim_prefix|>{上文}<|fim_suffix|>{下文}<|fim_middle|>{要补全的内容}<|im_end|>
```

文件流转：`data/raw/input.txt` →（`train/build_fim_data.py`）→ `data/train/train.jsonl`（训练脚本读这个路径，约 1.5 万条）。

## 1.2 训练

```bash
uv run python train/build_fim_data.py
uv run python train/train_fim.py
```

关键配置（`train/train_fim.py`）：

| 配置 | 值 | 说明 |
|---|---|---|
| base | `models/base/Qwen3-1.7B` | 原版底模 |
| 量化 | 4bit nf4 | 8GB 显存可跑 |
| LoRA | r=32, alpha=64 | attention + MLP 全打 |
| lr | 1e-4 | |
| batch | 4 × 累积 4 | batch 8 时显存占满会卡死桌面，降到 4，等效 batch 16 |
| epochs | 2 | |
| MAX_LENGTH | 1024 | |
| loss | 仅 middle + `<|im_end|>` | 结束符参与训练，否则模型学不到"补全完就停" |

输出目录在脚本顶部 `OUTPUT_DIR` 配置（历史产物：`models/adapters/output-qwen3-1.7-fim-v6/`）。

## 1.3 合并成底座

```bash
uv run python infer/merge.py
```

把 FIM adapter 全量合并进 Qwen3-1.7B，输出 **`models/fim/SpringNote-Qwen3-1.7B-FIM-V2/`**——这是后续 persona 训练的底模，也是纯补全功能的部署模型。

## 1.4 测试

```bash
uv run python infer/infer_fim.py
```

按 1.1 的三段格式构造 prompt 做补全，脚本里的 `MODELS` 列表可对比原版/FIM/persona 各版本效果。

---

# 第二部分：Persona

## 2.1 数据收集与存放

**唯一需要手工维护的源文件是 `data/raw/persona_train.json`**，格式：

```json
[
  {
    "instruction": "SpringNote是谁开发的？",
    "response": "SpringNote由开发者陈果果创建，其GitHub账号是Radiant303。"
  }
]
```

维护规则（重要）：

- 增删改数据**只改 `data/raw/persona_train.json`**，不要直接改 jsonl —— 重新跑 `train/build_persona_data.py` 会覆盖 jsonl
- 同一个事实多写几种问法（"谁开发的"/"作者是谁"/"开发者叫什么"），模型靠重复记忆
- 隐私类问题要教拒答（"没有公开相关信息，无法确认。"），年龄/学历/收入/联系方式等每类至少 5~10 条不同措辞，否则底模会瞎编
- 各文件位置：
  - `data/raw/persona_train.json` — 源数据
  - `data/train/persona_train_aug_v2.jsonl` — system prompt 增强并带空 `<think>` 块的最终训练集

## 2.2 数据预处理

```bash
# json → ChatML jsonl 并做 system prompt 增强：
# 每条数据保留原 prompt + 随机 3 个变体（含通用中文/英文/无 system prompt），
# 同时通过 enable_thinking=False 让 assistant 答案前带空 <think> 块。
uv run python train/build_persona_data.py
```

## 2.3 训练

```bash
uv run python train/train_persona.py
```

关键配置（`train/train_persona.py`）及经验：

| 配置 | 值 | 说明 |
|---|---|---|
| base | `models/fim/SpringNote-Qwen3-1.7B-FIM-V7` | 第一部分产出的 FIM 底座 |
| 量化 | 4bit nf4 | |
| LoRA | r=32, alpha=64 | |
| lr | 1e-4 | **不要低于 1e-4**，2e-5 时 LoRA 学不动 |
| batch | 8 × 累积 2 | 有效 batch 16 |
| epochs | 6 | 411 条约 25 分钟 |
| loss | assistant-only | 只在 assistant 回复上计算损失 |

训练完成后 adapter 保存在 `models/adapters/output-qwen3-1.7-persona-v6/`（`checkpoint-*` 目录是历史残留，确认最终模型没问题后可删除腾空间）。

## 2.4 测试

```bash
# 合并模型完整测试（FIM 补全 + tool 调用 + think 思考 + persona 聊天）
uv run python test/test_merged.py
```

验收标准：

- 事实问答（开发者/GitHub/官网/QQ群）在各种 system prompt 下都答对
- FIM 补全内容不冒出 persona 信息（陈果果/QQ群等）
- 隐私问题答"没有公开相关信息"

## 2.5 合并部署

```bash
uv run python infer/merge_persona.py
```

把 persona adapter 按强度缩放后合并进 FIM 底座，输出 `models/persona/SpringNote-Qwen3-1.7B-FIM-Persona-V5/`。

`SCALE` 的选择（实测结论）：

| SCALE | FIM 补全 | persona 事实 | 拒答 |
|---|---|---|---|
| 1.0 | 被污染 | 正确 | 正确 |
| **0.7（推荐）** | 干净 | 正确 | 失效 |
| ≤0.5 | 干净 | 开始退化 | 失效 |

规律：事实记忆在弱强度也能活，**拒答（对抗底模先验）最先随强度衰减**。要拒答和 FIM 兼得，只能补更多拒答样本重训，没有免费的强度值。

如果产品里补全和聊天是两个入口，更干净的方案是分开部署：补全用纯 `models/fim/SpringNote-Qwen3-1.7B-FIM-V7`，聊天用 SCALE=1.0 合并版。

## 2.6 推荐系统提示词

聊天场景部署时建议使用与训练数据一致的 system prompt（模型在该 prompt 下行为最稳定：事实准确、隐私问题拒答）：

```text
你是SpringNote官方AI助手。

你由陈果果基于Qwen3模型微调开发。

你的职责是帮助用户了解SpringNote、
整理知识、处理笔记相关任务。

回答要求：
- 准确
- 简洁
- 不编造信息
- 不知道的信息明确说明
```

Ollama 可以在 Modelfile 里固化：

```dockerfile
FROM ./springnote-q8_0.gguf
SYSTEM """
你是SpringNote官方AI助手。

你由陈果果基于Qwen3模型微调开发。

你的职责是帮助用户了解SpringNote、
整理知识、处理笔记相关任务。

回答要求：
- 准确
- 简洁
- 不编造信息
- 不知道的信息明确说明
"""
```

注意：补全（FIM）场景**不要**带 system prompt，直接按 1.1 的三段格式构造输入。

## 2.7 转 GGUF

```bash
# 一次性准备
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama.cpp
uv pip install gguf

# 转换（f16 无损；q8_0 体积减半、质量几乎无损）
uv run python llama.cpp/convert_hf_to_gguf.py ./models/persona/SpringNote-Qwen3-1.7B-FIM-Persona-V5 --outfile springnote-f16.gguf --outtype f16
uv run python llama.cpp/convert_hf_to_gguf.py ./models/persona/SpringNote-Qwen3-1.7B-FIM-Persona-V5 --outfile springnote-q8_0.gguf --outtype q8_0
```

更小的量化（Q4_K_M 等）需要从 [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) 下载 Windows 预编译包里的 `llama-quantize.exe`：

```bash
llama-quantize.exe springnote-f16.gguf springnote-Q4_K_M.gguf Q4_K_M
```

部署：

- **Ollama**：写 `FROM ./springnote-q8_0.gguf` 的 Modelfile，然后 `ollama create`
- **LM Studio**：把 gguf 放进 models 目录即可

注意：`llama.cpp/` 转换完建议删除或加进 `.gitignore`。

---

## 文件索引

| 文件 | 作用 |
|---|---|
| `data/raw/input.txt` | FIM 原始文档（空行分段） |
| `data/raw/persona_train.json` | persona 源数据（手工维护） |
| `data/raw/think_train.json` | think 源数据 |
| `train/build_fim_data.py` | FIM 数据生成（随机挖空）→ `data/train/train.jsonl` |
| `train/build_persona_data.py` | persona 数据生成 → `data/train/persona_train_aug_v2.jsonl` |
| `train/build_tools_data.py` | 回忆书工具调用数据生成 → `data/train/tools_train_v2.jsonl` |
| `train/build_think_data.py` | 思考链数据生成 → `data/train/think_train_v2.jsonl` |
| `train/build_all_data.py` | 一键构建上述全部训练数据 |
| `train/train_fim.py` | FIM QLoRA 训练 |
| `train/train_persona.py` | persona / tools / think 多任务 QLoRA 训练 |
| `infer/merge.py` | FIM adapter 合并成底座模型 |
| `infer/merge_persona.py` | persona adapter 缩放合并 |
| `infer/infer_fim.py` | FIM 补全推理/对比测试 |
| `test/test_merged.py` | 合并模型验收测试 |
