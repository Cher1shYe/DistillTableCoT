# run_evaluation.py

import os
import json
import argparse
import nltk
import evaluate  # 使用新的 evaluate 库
import ast       # 导入 ast 库用于安全地解析字符串

from configs import TASK_CONFIGS

def evaluate_predictions(task_name,k_shot=0, output_dir="outputs"):
    """
    从文件中加载已处理的预测和参考，并根据任务配置计算评估指标。
    """
    if task_name not in TASK_CONFIGS:
        print(f"错误: 任务 '{task_name}' 未在 TASK_CONFIGS 中定义。")
        return
    
    # 构建文件路径（包含 k_shot 信息）
    predictions_path = os.path.join(output_dir, task_name, f"predictions_k{str(k_shot).zfill(2)}.json")
    
    if not os.path.exists(predictions_path):
        print(f"错误: 找不到预测文件: {predictions_path}")
        print(f"请先运行 k_shot={k_shot} 的推理")
        return
    
    print(f"--- 开始评估任务: {task_name} ---")

    # 1. 加载配置文件和预测结果
    config = TASK_CONFIGS[task_name]
    predictions_path = os.path.join(output_dir, task_name, "predictions.json")

    if not os.path.exists(predictions_path):
        print(f"错误: 找不到预测文件: {predictions_path}")
        print(f"请先运行 'python run_inference.py --task_name {task_name}' 来生成预测。")
        return

    try:
        with open(predictions_path, 'r', encoding='utf-8') as f:
            raw_content = json.load(f)
        if isinstance(raw_content, dict) and 'predictions' in raw_content:
            print("检测到已存在的评估结果，将使用文件中的预测数据重新评估。")
            data = raw_content['predictions']
        else:
            # 文件是原始格式 (列表)
            data = raw_content
    except (json.JSONDecodeError, IOError) as e:
        print(f"错误: 无法读取或解析预测文件: {predictions_path}\n{e}")
        return

    # 2. 检查并加载核心数据
    if not data or 'processed_prediction' not in data[0]:
        print(f"错误: 预测文件 '{predictions_path}' 为空或缺少 'processed_prediction' 字段。")
        print("请确保 run_inference.py 脚本正确保存了处理后的预测结果。")
        return
        
    processed_preds = [item['processed_prediction'] for item in data]

    raw_references = [item['reference'] for item in data]

    # 3. 循环计算配置中定义的所有指标
    results = {}
    print("\n--- 计算评估指标 ---")
    for metric_name in config["metrics"]:
        print(f"Calculating metric: {metric_name}...")
        
        try:
            metric = evaluate.load(metric_name, trust_remote_code=True)
        except Exception as e:
            print(f"错误: 加载指标 '{metric_name}' 失败: {e}")
            continue
        
        # --- 根据指标名称选择不同的评估策略 ---
        
        if metric_name == "exact_match":
            # 适用于 wikitableqa
            
            predictions_for_em = [str(p) for p in processed_preds]
            references_for_em = []
            for ref in raw_references:
                ref_cleaned = ref.strip("[]'\"")
                references_for_em.append(ref_cleaned)
            
            score = metric.compute(predictions=predictions_for_em, references=references_for_em)
            results[metric_name] = score

        elif metric_name == "accuracy":
            # 适用于 tabfact
            # 目标格式: predictions=['Entailed'], references=['Refuted']
            predictions_as_int = [1 if p.lower() == 'entailed' else 0 for p in processed_preds]
            references_as_int = [1 if r.lower() == 'entailed' else 0 for r in raw_references]
            
            score = metric.compute(predictions=predictions_as_int, references=references_as_int)
            results[metric_name] = score

        elif metric_name == "rouge":
            # 适用于 fetaqa
            # 目标格式: predictions=['long text...'], references=['long text...']
            predictions_for_rouge = ["\n".join(nltk.sent_tokenize(str(p))) for p in processed_preds]
            references_for_rouge = ["\n".join(nltk.sent_tokenize(str(r))) for r in raw_references]
            
            score = metric.compute(predictions=predictions_for_rouge, references=references_for_rouge)
            results[metric_name] = score

        elif metric_name == "sacrebleu":
            # 适用于 fetaqa
            # 目标格式: predictions=['pred1'], references=[['ref1']]
            predictions_for_bleu = [str(p) for p in processed_preds]
            references_for_bleu = [[str(r)] for r in raw_references]
            
            score = metric.compute(predictions=predictions_for_bleu, references=references_for_bleu)
            results[metric_name] = score
            
        else:
            print(f"警告: 未知或未明确处理的指标 '{metric_name}'，跳过。")
            continue

    # 4. 打印格式化的结果
    print("\n--- 最终评估结果 ---")
    if not results:
        print("没有计算出任何评估结果。")
    
    for metric_name, score_dict in results.items():
        print(f"\nMetric: {metric_name}")
        if not score_dict:
            print("  评估分数为空。")
            continue
            
        if metric_name == "rouge":
            # --- 对 ROUGE 的打印逻辑进行健壮性修改 ---
            for key, value in score_dict.items():
                if isinstance(value, (float, int)):
                    # 如果它是一个简单的数字，就直接打印
                    print(f"  {key}: {value:.4f}")
            # --- 这部分主要面向有多个key value对的sacrebleu ---
        elif isinstance(score_dict, dict):
            for k, v in score_dict.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
            # --- 对EM，accuracy的单评分简单输出 ---
        elif isinstance(score_dict, float):
             print(f"  Score: {score_dict:.4f}")
        else:
            print(f"  Score: {score_dict}")
    results_to_save = {}
    for metric_name, score_dict in results.items():
        if metric_name == "rouge":
            if isinstance(value, (float, int)):
                print(f"  (信息) ROUGE 指标的键 '{key}' 对应的值已是数字，直接保存。")
                results_to_save[key] = round(float(value), 4)   
        elif metric_name == "sacrebleu":
            if 'score' in score_dict:
                # sacrebleu的分数单位是 0-100，保留两位小数即可
                results_to_save[metric_name] = round(score_dict['score'], 2)
        elif isinstance(score_dict, dict) and len(score_dict) == 1:
             key, value = list(score_dict.items())[0]
             results_to_save[key] = round(value, 4)
        else:
             print(f"警告: 无法为 '{metric_name}' 确定要保存的单一数值。")

    print("\n--- 清理后的评估分数 ---")
    print(json.dumps(results_to_save, indent=4))

    print(f"\n--- 更新文件: {predictions_path} ---")
    
    final_output = {
        "evaluation_results": results_to_save,
        "predictions": data  # 'data' 变量持有原始的预测列表
    }

    try:
        with open(predictions_path, 'w', encoding='utf-8') as f:
            # 使用 ensure_ascii=False 以正确处理中文字符
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        print("文件已成功更新，包含评估结果。")
    except (IOError, TypeError) as e:
        print(f"错误: 更新结果文件失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估已生成的模型预测。")
    parser.add_argument(
        "--task_name", 
        type=str, 
        required=True, 
        choices=TASK_CONFIGS.keys(),
        help="要评估的任务名称。"
    )
    args = parser.parse_args()
    
    evaluate_predictions(task_name=args.task_name)