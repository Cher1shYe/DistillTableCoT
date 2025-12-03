# run_inference.py
import os
import json
import tqdm
import argparse
from datasets import load_dataset
from configs import TASK_CONFIGS
from utils import format_table, call_deepseek_api

def generate_predictions(task_name, num_samples, output_dir="outputs"):
    """
    针对指定任务运行推理，并将结果保存到 JSON 文件。
    """
    if task_name not in TASK_CONFIGS:
        print(f"错误: 任务 '{task_name}' 未在 TASK_CONFIGS 中定义。")
        return

    print(f"--- 开始为任务生成预测: {task_name} ---")
    
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
    for i, sample in enumerate(tqdm.tqdm(dataset, desc=f"Generating for {task_name}")):
        # 准备 prompt
        table_str = format_table(sample.get('table') or sample.get('table_text'))
        prompt = config["prompt_template"].format(
            table=table_str,
            question=sample.get('question', ''),
            statement=sample.get('statement', '')
        )

        # 调用 API 获取预测
        prediction = call_deepseek_api(prompt)

        # 加入数据处理的模块
        postprocess_func = config["postprocess_func"] 

        # reference_label为参考答案
        target_field = config["target_field"]
        reference_label = sample[target_field]

        # 这里得到处理后的答案以便evaluator进行处理
        processed_prediction, processed_reference = postprocess_func(prediction, reference_label)

        # 准备要保存的数据
        result = {
            "id": i,
            "original_dataset_id": sample.get("original_dataset_id", "N/A"),
            "prompt": prompt,
            "prediction": prediction,
            "processed_prediction": processed_prediction,
            "reference": reference_label
        }
        results_to_save.append(result)

    # 3. 保存结果到文件
    task_output_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_output_dir, exist_ok=True)
    output_path = os.path.join(task_output_dir, "predictions.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=4)
        
    print(f"\n成功! {len(results_to_save)} 个预测结果已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为表格问答任务生成模型预测。")
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
    args = parser.parse_args()
    
    generate_predictions(task_name=args.task_name, num_samples=args.num_samples)