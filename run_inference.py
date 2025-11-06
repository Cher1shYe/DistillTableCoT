import argparse
import os
import json
import tqdm
from datasets import load_dataset
from configs import TASK_CONFIGS
from utils import format_table, call_deepseek_api

# 添加示例池导入
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ExamplePool.example_pool import ExamplePool

def build_k_shot_prompt(task_name, current_table, current_question, k_shot=0):
    """构建 k-shot prompt"""
    if k_shot == 0:
        # Zero-shot：使用原有模板
        config = TASK_CONFIGS[task_name]
        return config["prompt_template"].format(
            table=format_table(current_table),
            question=current_question
        )
    
    # Few-shot：从示例池获取示例
    pool_manager = ExamplePool(pool_dir="ep_instances")
    shots = pool_manager.get_shots(task_name, current_question, k=k_shot, strategy="random")
    
    return pool_manager.format_few_shot_prompt(task_name, shots, current_table, current_question)

def generate_predictions(task_name, num_samples, k_shot=0, output_dir="outputs"):
    """
    针对指定任务运行推理，支持 k-shot
    """
    if task_name not in TASK_CONFIGS:
        print(f"错误: 任务 '{task_name}' 未在 TASK_CONFIGS 中定义。")
        return

    print(f"--- 开始为任务生成预测: {task_name}, k_shot={k_shot} ---")
    
    # 1. 加载配置和数据
    config = TASK_CONFIGS[task_name]
    try:
        dataset = load_dataset(config["dataset_name"], split=config["dataset_split"])
    except Exception as e:
        print(f"数据集加载失败: {e}")
        return

    # 根据传入参数截取样本
    if num_samples > len(dataset):
        num_samples = len(dataset)
    dataset = dataset.select(range(num_samples))

    results_to_save = []

    # 2. 循环处理每个样本
    for i, sample in enumerate(tqdm.tqdm(dataset, desc=f"Generating for {task_name} (k={k_shot})")):
        # 准备 prompt
        table_str = format_table(sample.get('table') or sample.get('table_text'))
        question = sample.get('question', '') or sample.get('statement', '')
        
        # 使用 k-shot prompt
        prompt = build_k_shot_prompt(task_name, table_str, question, k_shot=k_shot)

        # 调用 API 获取预测
        prediction = call_deepseek_api(prompt)

        # 后处理
        postprocess_func = config["postprocess_func"] 
        target_field = config["target_field"]
        reference_label = sample[target_field]
        processed_prediction, processed_reference = postprocess_func(prediction, reference_label)

        # 准备要保存的数据
        result = {
            "id": i,
            "original_dataset_id": sample.get("original_dataset_id", "N/A"),
            "prompt": prompt,
            "prediction": prediction,
            "processed_prediction": processed_prediction,
            "reference": reference_label,
            "k_shot": k_shot  # 记录使用的 shot 数量
        }
        results_to_save.append(result)

    # 3. 保存结果到文件（按 k_shot 分开保存）
    task_output_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_output_dir, exist_ok=True)
    
    # 文件名包含 k_shot 信息
    output_filename = f"predictions_k{str(k_shot).zfill(2)}.json"
    output_path = os.path.join(task_output_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=4)
        
    print(f"\n成功! {len(results_to_save)} 个预测结果已保存至: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为表格问答任务生成模型预测，支持 k-shot。")
    parser.add_argument(
        "--task_name", 
        type=str, 
        required=True, 
        choices=TASK_CONFIGS.keys(),
        help="要运行的任务名称。"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=5,
        help="要处理的样本数量。"
    )
    parser.add_argument(
        "--k_shot",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help="few-shot 的示例数量 (0=zero-shot)。"
    )
    args = parser.parse_args()
    
    generate_predictions(task_name=args.task_name, num_samples=args.num_samples, k_shot=args.k_shot)