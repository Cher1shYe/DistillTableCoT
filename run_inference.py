# run_inference.py
import os
import json
import tqdm
import re
import argparse
from datasets import load_dataset
from configs import TASK_CONFIGS
from utils import format_table, call_deepseek_api, table_to_sqlite, _build_sqlite, execute_sql

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
        raw_table = sample.get('table') or sample.get('table_content') or sample.get('table_text')
        table_str = format_table(raw_table, task_name=task_name) 
        
        prompt = config["user_prompt_template"].format(
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
    output_path = os.path.join(task_output_dir, "predictions_v2.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=4)
        
    print(f"\n成功! {len(results_to_save)} 个预测结果已保存至: {output_path}")

def generate_agent_predictions(task_name, num_samples, max_turns=5, output_dir="outputs"):
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

    for i, sample in enumerate(tqdm.tqdm(dataset, desc=f"Agent reasoning for {task_name}")):
        raw_table = sample.get('table') or sample.get('table_content') or sample.get('table_text')
        conn, schema_str = table_to_sqlite(raw_table, task_name=task_name)
        if conn is None:
            # 如果表格解析失败，记录并跳过
            continue
        table_str = format_table(sample.get('table'), task_name=task_name)
        question = sample.get('question') or sample.get('statement', '')
        
    # 基础 User Prompt 包含 Table 内容和数据库 Schema，让模型知道列名
        base_user_content = (
            f"Table Content (Markdown):\n{table_str}\n\n"
            f"Database Schema (SQLite):\n{schema_str}\n\n"
            f"Question: {question}"
        )
        
        last_sql = None
        last_feedback = None
        final_prediction = ""
        turn_history = []

        # --- Agent 多轮循环 ---
        for turn in range(max_turns):
            # 动态构建当前轮次的 Prompt
            if turn == 0:
                # 第一轮：只有原始问题
                current_user_prompt = base_user_content
            else:
                # 后续轮次：原始问题 + 上一轮的反馈
                # 这种结构符合你 AGENT_SYSTEM_PROMPT 中 "IF ERROR/SUCCESS FEEDBACK" 的逻辑
                current_user_prompt = (
                    f"{base_user_content}\n\n"
                    f"--- Previous Attempt ---\n"
                    f"Last SQL: ```sql\n{last_sql}\n```\n"
                    f"Feedback: {last_feedback}"
                    f"Based on this, provide the next SQL or the Final Answer."
                )

            # 组装 Messages 发送给 API
            messages = [
                {"role": "system", "content": config["system_prompt"]},
                {"role": "user", "content": current_user_prompt}
            ]

            # 1. 调用模型
            response = call_deepseek_api(messages)
            turn_history.append({"turn": turn, "prompt": current_user_prompt, "response": response})

            # 2. 检查是否结束
            if "Final Answer:" in response:
                final_prediction = response
                break

            # 3. 提取 SQL 并执行
            sql_query = extract_sql(response)
            if sql_query:
                # [关键修正]：把 table_to_sqlite 返回的 conn 传进去
                success, db_feedback = execute_sql(conn, sql_query)
                
                last_sql = sql_query
                last_feedback = db_feedback
            else:
                last_sql = "None"
                last_feedback = "Error: No SQL query found in your last response. Please provide a SQL query in ```sql ... ```."
        
        conn.close()

        # --- 后处理与保存 ---
        postprocess_func = config["postprocess_func"]
        reference_label = sample[config["target_field"]]
        processed_pred, processed_ref = postprocess_func(final_prediction, reference_label)

        results_to_save.append({
            "id": i,
            "task": task_name,
            "prediction": final_prediction,
            "processed_prediction": processed_pred,
            "reference": processed_ref,
            "turn_details": turn_history # 记录每轮具体发生了什么
        })

    # 保存逻辑...
    task_output_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_output_dir, exist_ok=True)
    output_path = os.path.join(task_output_dir, "predictions_v2.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=4)
        
    print(f"\n成功! {len(results_to_save)} 个预测结果已保存至: {output_path}")

def extract_sql(text):
    """从模型输出中提取 SQL 代码块"""
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None

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
    
    generate_agent_predictions(task_name=args.task_name, num_samples=args.num_samples)