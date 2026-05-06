# run_evaluation.py

import os
import json
import argparse
import nltk
import evaluate
# import ast

from configs import TASK_CONFIGS

# --- 增加 NLTK 依赖的安全检查与自动下载 ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("正在下载 NLTK punkt 分词数据...")
    nltk.download('punkt')
    nltk.download('punkt_tab') # 适配新版 NLTK
# ----------------------------------------

def evaluate_predictions(task_name, output_dir="outputs", pred_file="predictions.json"):
    """
    从文件中加载已处理的预测和参考，并根据任务配置计算评估指标。
    """
    if task_name not in TASK_CONFIGS:
        print(f"错误: 任务 '{task_name}' 未在 TASK_CONFIGS 中定义。")
        return

    print(f"--- 开始评估任务: {task_name} ---")
    print(f"--- 目标文件: {pred_file} ---")

    # 1. 加载配置文件和预测结果
    config = TASK_CONFIGS[task_name]
    predictions_path = os.path.join(output_dir, task_name, pred_file)

    if not os.path.exists(predictions_path):
        print(f"错误: 找不到预测文件: {predictions_path}")
        print(f"请确保该文件存在，或运行推理脚本生成。")
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
            if task_name == "hitab":
                import ast
                correct = 0
                total = len(processed_preds)
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
                results[metric_name] = {"exact_match": correct / total if total > 0 else 0.0}
            else:
                # 适用于 wikitableqa
                import re

                def normalize_string(s):
                    if not s: return ""
                    s = str(s).lower().strip()
                    # 移除数字中的逗号
                    s = re.sub(r'(?<=\d),(?=\d)', '', s)
                    # 移除冠词
                    s = re.sub(r'\b(a|an|the)\b', ' ', s)
                    # 移除大部分标点（保留连字符和内部空格）
                    s = re.sub(r'[^\w\s-]', ' ', s)
                    # 移除末尾句号
                    if s.endswith('.'): s = s[:-1]
                    return " ".join(s.split())

                correct = 0
                total = len(processed_preds)
                for p, r in zip(processed_preds, raw_references):
                    # p 是字符串，r 是列表 (或列表的字符串表示)
                    p_norm = normalize_string(p)
                    
                    # 处理 r 可能被存成字符串 "['ans']" 的情况
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
                
                results[metric_name] = {"exact_match": correct / total if total > 0 else 0.0}

        elif metric_name == "accuracy":
            # 增加 str() 强转，防止 p 或 r 是 None 或非字符串导致 .lower() 报错
            predictions_as_int = [1 if str(p).lower() == 'entailed' else 0 for p in processed_preds]
            references_as_int = [1 if str(r).lower() == 'entailed' else 0 for r in raw_references]
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
    
    results_to_save = {}
    for metric_name, score_dict in results.items():
        print(f"\nMetric: {metric_name}")
        if not score_dict and score_dict != 0.0: # 防止得分为 0.0 时被误判为空
            print("  评估分数为空。")
            continue
            
        # 打印部分
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

        # 提取用于保存的单一数值，避免 NameError
        if metric_name == "rouge":
            if isinstance(score_dict, dict):
                for k, v in score_dict.items():
                    if isinstance(v, (float, int)):
                        results_to_save[k] = round(float(v), 4)
                        
        elif metric_name == "sacrebleu":
            if isinstance(score_dict, dict) and 'score' in score_dict:
                results_to_save[metric_name] = round(score_dict['score'], 2)
                
        elif isinstance(score_dict, dict):
            # 处理 accuracy 或 exact_match 返回单键字典的情况 (如 {'accuracy': 0.85})
            for k, v in score_dict.items():
                if isinstance(v, (float, int)):
                    results_to_save[k] = round(float(v), 4)
                    
        elif isinstance(score_dict, (float, int)):
            # 处理直接返回 float 的情况
             results_to_save[metric_name] = round(float(score_dict), 4)
             
        else:
             print(f"警告: 无法为 '{metric_name}' 确定要保存的单一数值。")

    print("\n--- 清理后的评估分数 ---")
    print(json.dumps(results_to_save, indent=4))

    print(f"\n--- 更新文件: {predictions_path} ---")
    
    final_output = {
        "evaluation_results": results_to_save,
        "predictions": data 
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
    # --- 新增参数：指定预测文件名 ---
    parser.add_argument(
        "--pred_file", 
        type=str, 
        default="predictions.json", 
        help="要评估的预测文件名 (例如: predictions_v2.json)。默认是 predictions.json"
    )
    args = parser.parse_args()
    
    # 传入 pred_file 参数
    evaluate_predictions(task_name=args.task_name, pred_file=args.pred_file)