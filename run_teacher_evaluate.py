#!/usr/bin/env python3
"""
教师模型 (DeepSeek API) 评估脚本。

设计原则:
    本脚本的推理逻辑严格对齐 run_inference.py::generate_agent_predictions
    (即生成蒸馏训练数据时使用的同一套逻辑)，唯一差别是把 dataset_split
    和样本数 / 输出文件名等参数化，使 teacher 能在 split=test 上跑评估。

    特别地:
        - 不含 scripts/test_model.py 引入的"强制合成轮"
          (该机制是为小模型避免 SQL 卡死设计的，teacher 无此问题，
           且若加入会破坏 teacher 推理协议与训练数据生成时的一致性)。
        - SQL agent 多轮循环 + (可选) CoT 回退，
          完全镜像 run_inference.py 的实现。

输出:
    outputs/<task>/predictions_teacher_deepseek_<inference_mode>.json
    可被 scripts/batch_eval_qwen3.py 直接识别 (kind=teacher)。

用法示例:
    export DEEPSEEK_API_KEY=sk-xxxx
    python3 run_teacher_evaluate.py --task wikitableqa --num_samples 100
    python3 run_teacher_evaluate.py --task hitab --inference_mode sql_agent
"""
import argparse
import os
import re
import sys
import json
import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from datasets import load_dataset

from configs import TASK_CONFIGS, TASK_TEST_CONFIGS, COT_SYSTEM_PROMPT
from utils import format_table, call_deepseek_api, table_to_sqlite, execute_sql


def extract_sql(text):
    """从模型输出中提取 ```sql ... ``` 代码块 (镜像 run_inference.py 同名函数)。"""
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def teacher_cot_predictions(task_name, split, num_samples,
                            out_name, output_dir="outputs"):
    """
    Teacher (DeepSeek) 标准 CoT baseline。

    与 teacher_agent_predictions 的区别:
      - 不构造 SQLite, 不走 SQL 多轮循环, 不带 schema
      - 用 TASK_TEST_CONFIGS[task]["prompt_template"] (与未训练 1.7B basic v0 同 prompt)
      - 不带 system prompt, 单轮 user message 调用
    设计动机: 给 teacher 加一档不带 SQL 脚手架的标准 CoT 参考线 (ablation),
    证明 mixed_agent 范式对 teacher 自己也有提升, 而非只对学生有用。
    """
    if task_name not in TASK_TEST_CONFIGS:
        print(f"错误: 任务 '{task_name}' 未在 TASK_TEST_CONFIGS 中定义。")
        return

    print(f"--- Teacher CoT baseline 开始: task={task_name}, split={split}, n={num_samples} ---")

    test_config = TASK_TEST_CONFIGS[task_name]
    eval_config = TASK_CONFIGS[task_name]   # 后处理 / target_field / dataset_name 仍走 TASK_CONFIGS

    try:
        dataset = load_dataset(eval_config["dataset_name"], split=split)
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    if num_samples > len(dataset):
        num_samples = len(dataset)
    dataset = dataset.select(range(num_samples))

    results_to_save = []

    for i, sample in enumerate(tqdm.tqdm(dataset, desc=f"teacher-cot → {task_name}")):
        raw_table = sample.get('table') or sample.get('table_content') or sample.get('table_text')
        table_str = format_table(raw_table, task_name=task_name)

        prompt = test_config["prompt_template"].format(
            table=table_str,
            question=sample.get('question', ''),
            statement=sample.get('statement', ''),
        )
        # 与 run_cot_baseline (scripts/test_model.py) 一致: 不带 system, 单轮 user
        messages = [{"role": "user", "content": prompt}]
        final_prediction = call_deepseek_api(messages)

        postprocess_func = eval_config["postprocess_func"]
        reference_label = sample[eval_config["target_field"]]
        try:
            processed_pred, _ = postprocess_func(final_prediction, reference_label)
        except Exception as e:
            print(f"⚠️ 后处理失败 (sample {i}): {e}")
            processed_pred = final_prediction

        results_to_save.append({
            "id": i,
            "original_dataset_id": str(sample.get("original_dataset_id", sample.get("id", "N/A"))),
            "task": task_name,
            "model": "teacher_deepseek",
            "mode": "CoT",
            "prediction": final_prediction,
            "processed_prediction": processed_pred,
            "reference": reference_label,
            # 单轮无 turn_history, 显式留空键以便下游统一访问
            "turn_details": [],
        })

    task_output_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_output_dir, exist_ok=True)
    output_path = os.path.join(task_output_dir, out_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=4)

    print(f"\n✅ {len(results_to_save)} 条 teacher CoT 结果保存至: {output_path}")
    print("   下一步: python3 scripts/batch_eval_qwen3.py")


def teacher_agent_predictions(task_name, split, num_samples,
                              out_name,
                              max_turns=5, max_empty=2,
                              fallback_to_cot=True,
                              output_dir="outputs"):
    """
    Teacher (DeepSeek) 在指定 split 上跑 SQL agent (可选 CoT 回退)。

    与 run_inference.py::generate_agent_predictions 的差异仅在于:
      - dataset_split 参数化 (而非读 config["dataset_split"])
      - num_samples / 输出文件名 / 是否 fallback 都暴露为参数
      - 输出 result 中加 "model": "teacher_deepseek" 标识，
        并保留 reference 原值 (而非 processed_reference) 与 batch_eval 兼容
    其余推理逻辑 (system_prompt / user_prompt 构造 / SQL schema 注入 /
    多轮反馈 prompt / 空结果早停 / CoT 回退 prompt) 逐字对齐。
    """
    if task_name not in TASK_CONFIGS:
        print(f"错误: 任务 '{task_name}' 未在 TASK_CONFIGS 中定义。")
        return

    print(f"--- Teacher 评估开始: task={task_name}, split={split}, n={num_samples} ---")

    config = TASK_CONFIGS[task_name]
    try:
        dataset = load_dataset(config["dataset_name"], split=split)
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    if num_samples > len(dataset):
        num_samples = len(dataset)
    dataset = dataset.select(range(num_samples))

    results_to_save = []

    for i, sample in enumerate(tqdm.tqdm(dataset, desc=f"teacher → {task_name}")):
        raw_table = sample.get('table') or sample.get('table_content') or sample.get('table_text')
        conn, schema_str = table_to_sqlite(raw_table, task_name=task_name)
        table_str = format_table(raw_table, task_name=task_name)
        question = sample.get('question') or sample.get('statement', '')

        final_prediction = ""
        mode = "SQL"
        turn_history = []

        # --- SQL Agent 多轮循环 (镜像 run_inference.py L114-L164) ---
        if conn is not None:
            base_user_content = (
                f"Table (Markdown):\n{table_str}\n\n"
                f"Schema (SQLite):\n{schema_str}\n\n"
                f"Question: {question}"
            )
            last_sql = None
            last_feedback = None
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
                        f"Check if the result answers the question. "
                        f"If not, write new SQL; otherwise output 'Final Answer: <answer>'."
                    )

                messages = [
                    {"role": "system", "content": config["system_prompt"]},
                    {"role": "user", "content": current_user_prompt},
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
                    if success and "no results" in str(db_feedback).lower():
                        empty_count += 1
                        if empty_count >= max_empty:
                            break
                else:
                    last_sql = "None"
                    last_feedback = "Error: No SQL code block found. Wrap your query in ```sql ... ```."

            conn.close()

        # --- CoT 回退 (镜像 run_inference.py L167-L178) ---
        if fallback_to_cot and (not final_prediction or "Final Answer:" not in final_prediction):
            mode = "CoT"
            cot_prompt = config["cot_user_prompt_template"].format(
                table=table_str,
                question=question,
                statement=question,
            )
            cot_messages = [
                {"role": "system", "content": COT_SYSTEM_PROMPT},
                {"role": "user", "content": cot_prompt},
            ]
            final_prediction = call_deepseek_api(cot_messages)

        # --- 后处理 ---
        postprocess_func = config["postprocess_func"]
        reference_label = sample[config["target_field"]]
        try:
            processed_pred, _ = postprocess_func(final_prediction, reference_label)
        except Exception as e:
            print(f"⚠️ 后处理失败 (sample {i}): {e}")
            processed_pred = final_prediction

        results_to_save.append({
            "id": i,
            "original_dataset_id": str(sample.get("original_dataset_id", sample.get("id", "N/A"))),
            "task": task_name,
            "model": "teacher_deepseek",
            "mode": mode,
            "prediction": final_prediction,
            "processed_prediction": processed_pred,
            "reference": reference_label,   # 保留原始 reference 以便 batch_eval 重清洗
            "turn_details": turn_history,
        })

    task_output_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_output_dir, exist_ok=True)
    output_path = os.path.join(task_output_dir, out_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=4)

    print(f"\n✅ {len(results_to_save)} 条 teacher 结果保存至: {output_path}")
    print("   下一步: python3 scripts/batch_eval_qwen3.py")


def main():
    parser = argparse.ArgumentParser(
        description="Teacher (DeepSeek) baseline 评估，逻辑对齐 run_inference.py"
    )
    parser.add_argument("--task", type=str, required=True,
                        choices=["wikitableqa", "tabfact", "fetaqa", "hitab"])
    parser.add_argument("--inference_mode", type=str, default="mixed_agent",
                        choices=["mixed_agent", "sql_agent", "cot"],
                        help="mixed_agent = SQL+CoT 回退 (= run_inference.py 默认行为); "
                             "sql_agent  = 关闭 CoT 回退 (= run_inference.py --pure_sql); "
                             "cot        = 标准 CoT baseline, 用 TASK_TEST_CONFIGS 的 prompt、"
                             "无 system, 单轮调用 (与 run_baseline.py --inference_mode cot 同款)")
    parser.add_argument("--split", type=str, default="test",
                        help="数据切片，默认 test (与学生模型评估一致)")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--max_empty", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--out_name", type=str, default=None,
                        help="自定义输出文件名；默认 predictions_teacher_deepseek_<mode>.json")
    args = parser.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("⚠️  环境变量 DEEPSEEK_API_KEY 未设置；utils 里的 client 会返回 [API_CLIENT_NOT_INITIALIZED]。")
        print("    请先 export DEEPSEEK_API_KEY=sk-xxx 再跑。")
        return

    out_name = args.out_name or f"predictions_teacher_deepseek_{args.inference_mode}.json"
    if not out_name.endswith(".json"):
        out_name += ".json"

    if args.inference_mode == "cot":
        # max_turns / max_empty 对 CoT 模式无意义, 静默忽略 (argparse 仍允许传)
        teacher_cot_predictions(
            task_name=args.task,
            split=args.split,
            num_samples=args.num_samples,
            out_name=out_name,
            output_dir=args.output_dir,
        )
    else:
        fallback_to_cot = (args.inference_mode == "mixed_agent")
        teacher_agent_predictions(
            task_name=args.task,
            split=args.split,
            num_samples=args.num_samples,
            out_name=out_name,
            max_turns=args.max_turns,
            max_empty=args.max_empty,
            fallback_to_cot=fallback_to_cot,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
