# Binder: Binding Language Models in Symbolic Languages

Original paper: [Binding Language Models in Symbolic Languages](https://arxiv.org/abs/2210.02875) (ICLR 2023 Spotlight).

This fork adapts the project to run with **SiliconFlow API (Qwen3-8B)** on **Windows**.

## Quick Start

```bash
# 1. Create environment
conda create -n binder python=3.9
conda activate binder

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key (SiliconFlow)
echo sk-your-key > key.txt

# 4. Run pipeline (3 items each, wiki + tabfact)
python run.py
```

## Commands

### Full Pipeline

```bash
python run.py                                    # wiki + tabfact, 各3条
python run.py --max_items 10                     # wiki + tabfact, 各10条
python run.py --datasets wikitq                  # 只跑 wiki
python run.py --datasets tab_fact --max_items 5  # 只跑 tabfact, 5条
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--max_items` | int | `3` | 每个数据集跑多少条 |
| `--datasets` | list | `wikitq tab_fact` | 选择数据集，支持多个 |

### Evaluation

```bash
# 原始 Binder 评估方法
python eval_original.py --dataset wikitq
python eval_original.py --dataset tab_fact

# DistillTableCoT 评估方法（更宽松的 exact match）
python eval_results.py --dataset wikitq --print_details
python eval_results.py --dataset tab_fact --print_details
```

### Annotation Only

```bash
set PYTHONPATH=D:\CS\srtp\Binder
python scripts/annotate_binder_program.py \
    --dataset wikitq --dataset_split test \
    --prompt_file templates/prompts/wikitq_binder.txt \
    --max_items 3 --sampling_n 1
```

### Execution Only (requires existing annotation results)

```bash
set PYTHONPATH=D:\CS\srtp\Binder
python scripts/execute_binder_program.py \
    --dataset wikitq --dataset_split test \
    --input_program_file binder_program_wikitq_test.json \
    --output_program_execution_file binder_program_wikitq_test_exec.json \
    --max_items 3 --vote_method simple
```

## Output Files

All output goes to `results/`:

| File | Content |
|------|---------|
| `binder_program_<dataset>_test.json` | Generated NSQL programs |
| `binder_program_<dataset>_test_exec.json` | Execution results with pred/gold answers |

Data cache in `output/`:

| File | Content |
|------|---------|
| `wikitq_test.json` | Preprocessed WikiTQ data |
| `tab_fact_test.json` | Preprocessed TabFact data |

## Configuration

### API

- **Provider**: SiliconFlow (`https://api.siliconflow.cn/v1`)
- **Model**: `Qwen/Qwen3-8B`
- **Key file**: `key.txt` (one API key per line)

### Key Modifications from Original

- OpenAI API → SiliconFlow API with Qwen3-8B
- GitHub dataset downloads → HuggingFace `table-benchmark/` → local JSON cache
- Added `--max_items` for quick testing
- Windows compatibility fixes (export → set, encoding, path separators)
- TabFact 0/1 → Entailed/Refuted mapping
- `#`-delimited table format fix for TabFact
- Qwen3 think block (`<think>...</think>`) stripping
- SQLite dialect normalization in post-processing

## Supported Datasets

| Dataset | Tasks | Description |
|---------|-------|-------------|
| WikiTQ | Table QA | 表格问答 |
| TabFact | Fact Verification | 表格事实验证 |
| HybridQA | Table+Text QA | 表格+文本段落问答 |
| MMQA | Multimodal QA | 表格+文本+图片多模态问答 |
