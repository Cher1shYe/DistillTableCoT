# run_inference.py
import os
import json
import tqdm
import re
import argparse
from datasets import load_dataset
from configs import TASK_CONFIGS, COT_SYSTEM_PROMPT
from utils import format_table, call_deepseek_api, table_to_sqlite, execute_sql

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
    output_path = os.path.join(task_output_dir, "predictions_v5.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=4)
        
    print(f"\n成功! {len(results_to_save)} 个预测结果已保存至: {output_path}")

def generate_agent_predictions(task_name, num_samples, max_turns=5, max_empty=2, output_dir="outputs"):
    """
    Agent 推理：优先用 SQL 多轮查询；若 SQL 多次返回空值或无 Final Answer，回退到简单 CoT。
    """
    if task_name not in TASK_CONFIGS:
        print(f"错误: 任务 '{task_name}' 未在 TASK_CONFIGS 中定义。")
        return

    print(f"--- 开始为任务生成预测: {task_name} ---")

    config = TASK_CONFIGS[task_name]
    try:
        dataset = load_dataset(config["dataset_name"], split=config["dataset_split"])
    except Exception as e:
        print(f"数据集加载失败: {e}")
        return

    if num_samples > len(dataset):
        num_samples = len(dataset)
    dataset = dataset.select(range(num_samples))

    results_to_save = []

    for i, sample in enumerate(tqdm.tqdm(dataset, desc=f"Agent reasoning for {task_name}")):
        raw_table = sample.get('table') or sample.get('table_content') or sample.get('table_text')
        conn, schema_str = table_to_sqlite(raw_table, task_name=task_name)
        table_str = format_table(raw_table, task_name=task_name)
        question = sample.get('question') or sample.get('statement', '')

        final_prediction = ""
        mode = "SQL"

        # --- SQL Agent 多轮循环 ---
        if conn is not None:
            base_user_content = (
                f"Table (Markdown):\n{table_str}\n\n"
                f"Schema (SQLite):\n{schema_str}\n\n"
                f"Question: {question}"
            )
            last_sql = None
            last_feedback = None
            final_prediction = ""
            turn_history = []
            empty_count = 0

            for turn in range(max_turns):
                if turn == 0:
                    current_user_prompt = base_user_content
                else:
                    current_user_prompt = (
                        f"{base_user_content}\n\n"
                        f"--- Previous Attempt ---\n"
                        f"Last SQL: ```sql\n{last_sql}\n```\n"
                        f"Feedback: {last_feedback}\n"
                        f"Check if the result answers the question. If not, write new SQL; otherwise output 'Final Answer: <answer>'."
                    )

                messages = [
                    {"role": "system", "content": config["system_prompt"]},
                    {"role": "user", "content": current_user_prompt}
                ]
                response = call_deepseek_api(messages)
                turn_history.append({"turn": turn, "prompt": current_user_prompt, "response": response})

                if "Final Answer:" in response:
                    final_prediction = response
                    break

                sql_query = extract_sql(response)
                if sql_query:
                    success, db_feedback = execute_sql(conn, sql_query)
                    last_sql = sql_query
                    last_feedback = db_feedback
                    # 空结果计数：SQL 成功但无数据
                    if success and "no results" in db_feedback.lower():
                        empty_count += 1
                        if empty_count >= max_empty:
                            break
                else:
                    last_sql = "None"
                    last_feedback = "Error: No SQL code block found. Wrap your query in ```sql ... ```."

            conn.close()

        # --- 回退：表格解析失败、无 Final Answer、或多次空值 → 简单 CoT ---
        if not final_prediction or "Final Answer:" not in final_prediction:
            mode = "CoT"
            cot_prompt = config["cot_user_prompt_template"].format(
                table=table_str,
                question=question,
                statement=question,
            )
            cot_messages = [
                {"role": "system", "content": COT_SYSTEM_PROMPT},
                {"role": "user", "content": cot_prompt}
            ]
            final_prediction = call_deepseek_api(cot_messages)

        postprocess_func = config["postprocess_func"]
        reference_label = sample[config["target_field"]]
        processed_pred, processed_ref = postprocess_func(final_prediction, reference_label)

        results_to_save.append({
            "id": i,
            "task": task_name,
            "mode": mode,
            "prediction": final_prediction,
            "processed_prediction": processed_pred,
            "reference": processed_ref,
            "turn_details": turn_history # 记录每轮具体发生了什么

        })

    # 保存逻辑...
    task_output_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_output_dir, exist_ok=True)
    output_path = os.path.join(task_output_dir, "predictions_v5.json")
    
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