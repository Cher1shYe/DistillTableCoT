# run_qwen3B_evaluate.py
"""
针对小模型 (qwen3 1.7B / 3B 等) 输出的评估脚本。

与 run_evaluate.py 的区别：
- 在计算指标前，先用 configs 中的 extract_*_final_answer 函数
  重新清洗 prediction，覆盖文件里旧的 processed_prediction
  （借用 scripts/reeval.py 的清洗逻辑）。
- 计算完成后按 run_evaluate.py 一致的格式回写文件
  ({"evaluation_results": ..., "predictions": [...]})。

用法:
    python3 run_qwen3B_evaluate.py --task_name wikitableqa \
        --pred_file predictions_qwen3_1.7b_mixed_wikitableqa.json
"""

import os
import json
import argparse
import nltk
import evaluate

from configs import (
    TASK_CONFIGS,
    extract_wiki_final_answer,
    extract_fact_final_answer,
    extract_fetaqa_final_answer,
    extract_hitab_final_answer,
)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("正在下载 NLTK punkt 分词数据...")
    nltk.download('punkt')
    nltk.download('punkt_tab')


# 与 scripts/reeval.py 对齐的清洗函数映射
POSTPROCESS = {
    "wikitableqa": lambda pred, ref: extract_wiki_final_answer(pred),
    "tabfact":     lambda pred, ref: extract_fact_final_answer(pred),
    "fetaqa":      lambda pred, ref: extract_fetaqa_final_answer(pred),
    "hitab":       lambda pred, ref: extract_hitab_final_answer(pred, ref),
}


def reclean_predictions(data, task_name):
    """
    使用 configs 中的清洗函数，在原地覆盖每条记录的 processed_prediction。
    返回 (changed_count, total_count)。
    """
    if task_name not in POSTPROCESS:
        raise ValueError(f"未知任务: {task_name}")

    postfn = POSTPROCESS[task_name]
    changed = 0
    for d in data:
        new_pred = postfn(d.get("prediction", ""), d.get("reference", ""))
        if new_pred != d.get("processed_prediction", ""):
            changed += 1
        d["processed_prediction"] = new_pred
    return changed, len(data)


def evaluate_predictions(task_name, output_dir="outputs", pred_file="predictions.json"):
    """
    针对小模型输出：先重新清洗 processed_prediction，再按任务配置评估并写回。
    """
    if task_name not in TASK_CONFIGS:
        print(f"错误: 任务 '{task_name}' 未在 TASK_CONFIGS 中定义。")
        return

    print(f"--- 开始评估 (小模型) 任务: {task_name} ---")
    print(f"--- 目标文件: {pred_file} ---")

    config = TASK_CONFIGS[task_name]
    predictions_path = os.path.join(output_dir, task_name, pred_file)

    if not os.path.exists(predictions_path):
        print(f"错误: 找不到预测文件: {predictions_path}")
        return

    try:
        with open(predictions_path, 'r', encoding='utf-8') as f:
            raw_content = json.load(f)
        if isinstance(raw_content, dict) and 'predictions' in raw_content:
            print("检测到已存在的评估结果，将使用文件中的预测数据重新评估。")
            data = raw_content['predictions']
        else:
            data = raw_content
    except (json.JSONDecodeError, IOError) as e:
        print(f"错误: 无法读取或解析预测文件: {predictions_path}\n{e}")
        return

    if not data or 'prediction' not in data[0]:
        print(f"错误: 预测文件 '{predictions_path}' 为空或缺少 'prediction' 字段。")
        return

    # --- 关键步骤：用 reeval 的清洗逻辑重写 processed_prediction ---
    print("\n--- 重新清洗 processed_prediction (基于 extract_*_final_answer) ---")
    changed, total = reclean_predictions(data, task_name)
    print(f"  清洗完成：{changed}/{total} 条记录的 processed_prediction 发生变化。")

    processed_preds = [item['processed_prediction'] for item in data]
    raw_references = [item['reference'] for item in data]

    # --- 按 run_evaluate.py 的指标计算逻辑 ---
    results = {}
    print("\n--- 计算评估指标 ---")
    for metric_name in config["metrics"]:
        print(f"Calculating metric: {metric_name}...")

        try:
            metric = evaluate.load(metric_name, trust_remote_code=True)
        except Exception as e:
            print(f"错误: 加载指标 '{metric_name}' 失败: {e}")
            continue

        if metric_name == "exact_match":
            if task_name == "hitab":
                import ast
                correct = 0
                total_n = len(processed_preds)
                for p, r in zip(processed_preds, raw_references):
                    p_str = str(p).strip()
                    r_str = str(r).strip()
                    match = False
                    if p_str == r_str:
                        match = True
                    else:
                        try:
                            p_val = ast.literal_eval(p_str)
                            r_val = ast.literal_eval(r_str)
                            if p_val == r_val:
                                match = True
                            else:
                                p_list = p_val if isinstance(p_val, list) else [p_val]
                                r_list = r_val if isinstance(r_val, list) else [r_val]
                                if len(p_list) == len(r_list):
                                    list_match = True
                                    for p_i, r_i in zip(p_list, r_list):
                                        if float(p_i) != float(r_i):
                                            list_match = False
                                            break
                                    if list_match:
                                        match = True
                        except:
                            pass
                    if match:
                        correct += 1
                results[metric_name] = {"exact_match": correct / total_n if total_n > 0 else 0.0}
            else:
                import re
                import ast

                def normalize_string(s):
                    if not s:
                        return ""
                    s = str(s).lower().strip()
                    s = re.sub(r'(?<=\d),(?=\d)', '', s)
                    s = re.sub(r'\b(a|an|the)\b', ' ', s)
                    s = re.sub(r'[^\w\s-]', ' ', s)
                    if s.endswith('.'):
                        s = s[:-1]
                    return " ".join(s.split())

                correct = 0
                total_n = len(processed_preds)
                for p, r in zip(processed_preds, raw_references):
                    p_norm = normalize_string(p)

                    if isinstance(r, str) and r.startswith('[') and r.endswith(']'):
                        try:
                            r_list = ast.literal_eval(r)
                        except:
                            r_list = [r.strip("[]'\"")]
                    elif isinstance(r, list):
                        r_list = r
                    else:
                        r_list = [str(r)]

                    r_norms = [normalize_string(x) for x in r_list]

                    if p_norm in r_norms:
                        correct += 1
                    elif p_norm == ", ".join(r_norms) or p_norm == ",".join(r_norms):
                        correct += 1
                    elif all(ref in p_norm for ref in r_norms) and len(r_norms) > 1:
                        correct += 1

                results[metric_name] = {"exact_match": correct / total_n if total_n > 0 else 0.0}

        elif metric_name == "accuracy":
            predictions_as_int = [1 if str(p).lower() == 'entailed' else 0 for p in processed_preds]
            references_as_int = [1 if str(r).lower() == 'entailed' else 0 for r in raw_references]
            score = metric.compute(predictions=predictions_as_int, references=references_as_int)
            results[metric_name] = score

        elif metric_name == "rouge":
            predictions_for_rouge = ["\n".join(nltk.sent_tokenize(str(p))) for p in processed_preds]
            references_for_rouge = ["\n".join(nltk.sent_tokenize(str(r))) for r in raw_references]
            score = metric.compute(predictions=predictions_for_rouge, references=references_for_rouge)
            results[metric_name] = score

        elif metric_name == "sacrebleu":
            predictions_for_bleu = [str(p) for p in processed_preds]
            references_for_bleu = [[str(r)] for r in raw_references]
            score = metric.compute(predictions=predictions_for_bleu, references=references_for_bleu)
            results[metric_name] = score

        else:
            print(f"警告: 未知或未明确处理的指标 '{metric_name}'，跳过。")
            continue

    # --- 打印 & 整理保存结果 ---
    print("\n--- 最终评估结果 ---")
    if not results:
        print("没有计算出任何评估结果。")

    results_to_save = {}
    for metric_name, score_dict in results.items():
        print(f"\nMetric: {metric_name}")
        if not score_dict and score_dict != 0.0:
            print("  评估分数为空。")
            continue

        if isinstance(score_dict, dict):
            for k, v in score_dict.items():
                if isinstance(v, (float, int)):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        elif isinstance(score_dict, (float, int)):
            print(f"  Score: {score_dict:.4f}")
        else:
            print(f"  Score: {score_dict}")

        if metric_name == "rouge":
            if isinstance(score_dict, dict):
                for k, v in score_dict.items():
                    if isinstance(v, (float, int)):
                        results_to_save[k] = round(float(v), 4)
        elif metric_name == "sacrebleu":
            if isinstance(score_dict, dict) and 'score' in score_dict:
                results_to_save[metric_name] = round(score_dict['score'], 2)
        elif isinstance(score_dict, dict):
            for k, v in score_dict.items():
                if isinstance(v, (float, int)):
                    results_to_save[k] = round(float(v), 4)
        elif isinstance(score_dict, (float, int)):
            results_to_save[metric_name] = round(float(score_dict), 4)
        else:
            print(f"警告: 无法为 '{metric_name}' 确定要保存的单一数值。")

    print("\n--- 清理后的评估分数 ---")
    print(json.dumps(results_to_save, indent=4))

    print(f"\n--- 更新文件: {predictions_path} ---")
    final_output = {
        "evaluation_results": results_to_save,
        "predictions": data,
    }

    try:
        with open(predictions_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        print("文件已成功更新，包含重清洗后的预测和评估结果。")
    except (IOError, TypeError) as e:
        print(f"错误: 更新结果文件失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对小模型预测结果重清洗并评估。")
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        choices=TASK_CONFIGS.keys(),
        help="要评估的任务名称。",
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        default="predictions.json",
        help="要评估的预测文件名 (例如: predictions_qwen3_1.7b_mixed_wikitableqa.json)。",
    )
    args = parser.parse_args()

    evaluate_predictions(task_name=args.task_name, pred_file=args.pred_file)
