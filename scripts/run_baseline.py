#!/usr/bin/env python3
"""
大模型 Baseline 推理脚本，用于与蒸馏后的学生模型做对照实验。
支持模型: Qwen3-7B, Llama-3.1-8B-Instruct（本地加载）
推理模式: mixed_agent（SQL 优先 + CoT 回退），与学生模型保持一致。

用法示例:
  python3 scripts/run_baseline.py --model qwen3-7b  --task wikitableqa --num_samples 100
  python3 scripts/run_baseline.py --model llama3-8b --task hitab       --num_samples 100
"""
import argparse
import os
import sys
import json
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # scripts/

from datasets import load_dataset

from configs import TASK_CONFIGS, COT_SYSTEM_PROMPT
from utils import format_table

# 复用 test_model.py 里已有的推理函数，避免重复实现
from test_model import (
    load_model_and_tokenizer,
    generate,
    run_sql_agent,
    run_mixed_agent,
)

# ---------------------------------------------------------------------------
# 支持的 baseline 模型配置
# ---------------------------------------------------------------------------
BASELINE_MODELS = {
    "qwen3-8b": {
        "hf_name":    "Qwen/Qwen3-8B",
        "local_path": "./origin/Qwen3-8B",
        "description": "Qwen3-8B (同系列更大模型，对比蒸馏压缩效率)",
    },
    "llama3-8b": {
        "hf_name":    "meta-llama/Llama-3.1-8B-Instruct",
        "local_path": "./origin/Llama-3.1-8B-Instruct",
        "description": "Llama-3.1-8B-Instruct (跨架构对照)",
    },
}


def load_baseline_model(model_key: str):
    """加载 baseline 模型和 tokenizer。优先使用本地路径，否则从 HuggingFace 下载。"""
    cfg = BASELINE_MODELS[model_key]
    if os.path.isdir(cfg["local_path"]):
        load_path = cfg["local_path"]
        print(f"使用本地模型: {load_path}")
    else:
        load_path = cfg["hf_name"]
        print(f"本地路径不存在，从 HuggingFace 加载: {load_path}")

    # 统一以 base 模式加载（非 LoRA，非微调），直接用原始权重
    model, tokenizer = load_model_and_tokenizer(
        model_type="base",
        base_model_name=load_path,
        model_path=load_path,
        is_lora=False,
    )
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="大模型 Baseline 推理（与学生模型对照）")
    parser.add_argument(
        "--model", type=str, required=True, choices=list(BASELINE_MODELS.keys()),
        help="baseline 模型: " + ", ".join(
            f"{k}({v['description']})" for k, v in BASELINE_MODELS.items()
        ),
    )
    parser.add_argument(
        "--task", type=str, default="wikitableqa",
        choices=["wikitableqa", "tabfact", "fetaqa", "hitab"],
    )
    parser.add_argument(
        "--inference_mode", type=str, default="mixed_agent",
        choices=["mixed_agent", "sql_agent"],
        help="推理模式（默认 mixed_agent：SQL 优先 + CoT 回退）",
    )
    parser.add_argument("--split",          type=str, default="train")
    parser.add_argument("--num_samples",    type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_turns",      type=int, default=5)
    parser.add_argument("--max_empty",      type=int, default=2)
    parser.add_argument("--output_dir",     type=str, default="outputs")
    args = parser.parse_args()

    print(f"模型: {args.model}  ({BASELINE_MODELS[args.model]['description']})")
    print(f"任务: {args.task}  推理模式: {args.inference_mode}  样本数: {args.num_samples}")

    # 1. 加载模型
    model, tokenizer = load_baseline_model(args.model)

    # 2. 加载数据集（与 test_model.py 保持一致，使用 TASK_CONFIGS）
    task_config  = TASK_CONFIGS[args.task]
    dataset      = load_dataset(task_config["dataset_name"], split=args.split)
    num_to_test  = min(args.num_samples, len(dataset))
    dataset      = dataset.select(range(num_to_test))

    target_field    = task_config["target_field"]
    postprocess_func = task_config["postprocess_func"]

    # 3. 推理循环
    results = []
    for i, sample in enumerate(tqdm.tqdm(dataset, desc=f"{args.model} → {args.task}")):

        if args.inference_mode == "sql_agent":
            prediction, turn_history, mode = run_sql_agent(
                model, tokenizer, args.task, sample, task_config,
                max_turns=args.max_turns,
                max_empty=args.max_empty,
                max_new_tokens=args.max_new_tokens,
            )
            if not prediction and turn_history:
                prediction = turn_history[-1]["response"]
        else:  # mixed_agent
            prediction, turn_history, mode = run_mixed_agent(
                model, tokenizer, args.task, sample, task_config,
                max_turns=args.max_turns,
                max_empty=args.max_empty,
                max_new_tokens=args.max_new_tokens,
            )

        reference_label = sample[target_field]
        try:
            processed_prediction, _ = postprocess_func(prediction, reference_label)
        except Exception as e:
            print(f"⚠️ 后处理失败 (sample {i}): {e}")
            processed_prediction = prediction

        result = {
            "id":                   i,
            "original_dataset_id":  str(sample.get("original_dataset_id", sample.get("id", "N/A"))),
            "task":                 args.task,
            "model":                args.model,
            "mode":                 mode,
            "prediction":           prediction,
            "processed_prediction": processed_prediction,
            "reference":            reference_label,
        }
        if turn_history:
            result["turn_details"] = turn_history
        results.append(result)

    # 4. 保存结果
    task_output_dir = os.path.join(args.output_dir, args.task)
    os.makedirs(task_output_dir, exist_ok=True)
    out_name    = f"predictions_baseline_{args.model}_{args.task}.json"
    output_path = os.path.join(task_output_dir, out_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n✅ {len(results)} 条结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
