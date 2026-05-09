import argparse
import re
import torch
import os
import sys
import json
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 添加根目录到 path 以便引入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import TASK_TEST_CONFIGS, TASK_CONFIGS, COT_SYSTEM_PROMPT, strip_think_block
from utils import format_table


def load_model_and_tokenizer(model_type, base_model_name, model_path, is_lora):
    """
    统一的模型/分词器加载逻辑，与根目录 test_model.py 的 call_local_model 保持一致。
    - base:    直接加载基础模型
    - trained: 加载微调后的模型
        * is_lora=True  -> 在基础模型上挂载 LoRA 适配器
        * is_lora=False -> 直接从 model_path 加载全量微调后的权重
    """
    if model_type == "trained" and not is_lora:
        # 全量微调：直接从保存目录加载（与根 test_model.py 一致）
        load_path = model_path
    else:
        # 基础模型 或 LoRA 底座
        load_path = base_model_name

    print(f"🔍 加载分词器: {load_path}")
    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if "<|endoftext|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|endoftext|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token

    print(f"🔍 加载模型: {load_path}")
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        # 显式开启 PyTorch 内置的加速 (Scaled Dot Product Attention)
        attn_implementation="sdpa",
    )

    if model_type == "trained" and is_lora:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LoRA 权重未找到: {model_path}")
        from peft import PeftModel
        print(f"🔍 加载 LoRA 权重: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        print("✅ Trained LoRA 模型加载完成")
    elif model_type == "trained":
        print("✅ 全量微调模型加载完成")
    else:
        print("✅ Base 模型加载完成")

    print(f"✅ 模型类型: {type(model)}")
    model.eval()
    return model, tokenizer


def build_messages_and_prompt(task, prompt_mode, sample):
    """
    根据 prompt_mode 组装对话消息以及用于保存的 prompt 字符串：
    - agent: 使用 TASK_CONFIGS（含系统提示词，适合带纠错轨迹训练的 Agent 模型）
    - basic: 使用 TASK_TEST_CONFIGS（仅用户提示，与根 test_model.py 对齐，
             适合基础数据训练的小模型 以及 未微调的 base 模型）
    返回: (messages, prompt_for_save, config)
    """
    table_str = format_table(sample.get('table') or sample.get('table_text'), task_name=task)

    if prompt_mode == "agent":
        config = TASK_CONFIGS[task]
        # agent 模式下 tabfact 的 user_prompt_template 用 {question} 占位 statement
        question = sample.get('question', '') or sample.get('statement', '')
        system_prompt = config["system_prompt"]
        user_prompt = config["user_prompt_template"].format(
            table=table_str,
            question=question,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # 保存时把 system + user 拼一起，便于后续评估时溯源完整输入
        prompt_for_save = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"
    else:
        config = TASK_TEST_CONFIGS[task]
        # 完全对齐根 test_model.py：同时把 question / statement 传入
        prompt = config["prompt_template"].format(
            table=table_str,
            question=sample.get('question', ''),
            statement=sample.get('statement', ''),
        )
        messages = [{"role": "user", "content": prompt}]
        prompt_for_save = prompt

    return messages, prompt_for_save, config


def generate(model, tokenizer, messages, max_new_tokens=1024):
    """
    生成逻辑严格对齐根目录 test_model.py 的 call_local_model。
    """
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True,
    )
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    generation_config = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'max_new_tokens': max_new_tokens,
        'temperature': 0.6,
        'do_sample': True,
        'top_p': 0.95,
        'top_k': 20,
        'repetition_penalty': 1.05,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }

    with torch.no_grad():
        outputs = model.generate(**generation_config)

    generated_tokens = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()


# ---------------------------------------------------------------------------
# SQL agent helpers  (mirrors run_inference.py::generate_agent_predictions)
# ---------------------------------------------------------------------------

def _extract_sql(text: str):
    """Extract the SQL query from a ```sql ... ``` code block."""
    m = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def run_sql_agent(model, tokenizer, task, sample, config,
                  max_turns=5, max_empty=2, max_new_tokens=512):
    """Multi-turn SQL agent inference with actual SQL execution.

    Replicates the teacher's generate_agent_predictions loop so the student
    model is evaluated under the exact same conditions it was distilled from.

    Returns (final_prediction, turn_history, mode).
    mode is always "SQL" from this function.
    """
    from utils import table_to_sqlite, execute_sql

    raw_table = sample.get('table') or sample.get('table_content') or sample.get('table_text')
    table_str = format_table(raw_table, task_name=task)
    question = sample.get('question', '') or sample.get('statement', '')

    conn, schema_str = table_to_sqlite(raw_table, task_name=task)

    final_prediction = ""
    turn_history = []

    if conn is None:
        return final_prediction, turn_history, "SQL"

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
        response = generate(model, tokenizer, messages, max_new_tokens)
        turn_history.append({"turn": turn, "prompt": current_user_prompt, "response": response})

        if "Final Answer:" in response:
            final_prediction = response
            break

        sql_query = _extract_sql(response)
        if sql_query:
            success, db_feedback = execute_sql(conn, sql_query)
            last_sql = sql_query
            last_feedback = db_feedback
            if success and "no results" in str(db_feedback).lower():
                empty_count += 1
                if empty_count >= max_empty:
                    break
        else:
            last_sql = None
            last_feedback = "Error: No SQL code block found. Wrap your query in ```sql ... ```."

    # If the loop ended without a Final Answer but we have a successful SQL result,
    # add one forced synthesis turn: ask the model to convert the SQL result into an answer.
    # This stays within the AGENT_SYSTEM_PROMPT distribution the model was trained on.
    if (not final_prediction
            and last_sql
            and last_feedback
            and "SQL Error" not in str(last_feedback)
            and "no results" not in str(last_feedback).lower()):
        force_prompt = (
            f"{base_user_content}\n\n"
            f"--- SQL Result ---\n"
            f"SQL: ```sql\n{last_sql}\n```\n"
            f"Result:\n{last_feedback}\n\n"
            f"The SQL result above contains the information needed to answer the question. "
            f"Do NOT write any more SQL. Output exactly 'Final Answer: <your answer>'."
        )
        messages = [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": force_prompt},
        ]
        response = generate(model, tokenizer, messages, max_new_tokens)
        turn_history.append({"turn": "force", "prompt": force_prompt, "response": response})
        if "Final Answer:" in response:
            final_prediction = response

    conn.close()
    return final_prediction, turn_history, "SQL"


def run_mixed_agent(model, tokenizer, task, sample, config,
                    max_turns=5, max_empty=2, max_new_tokens=512):
    """SQL-first with CoT fallback — mirrors the v5 teacher inference logic.

    Tries the SQL agent loop first; if no Final Answer is produced, falls back
    to a single-turn CoT call using COT_SYSTEM_PROMPT and cot_user_prompt_template.

    Returns (final_prediction, turn_history, mode).
    mode is "SQL" on success, "CoT" on fallback.
    """
    final_prediction, turn_history, mode = run_sql_agent(
        model, tokenizer, task, sample, config, max_turns, max_empty, max_new_tokens
    )

    if not final_prediction or "Final Answer:" not in final_prediction:
        mode = "CoT"
        raw_table = sample.get('table') or sample.get('table_content') or sample.get('table_text')
        table_str = format_table(raw_table, task_name=task)
        question = sample.get('question', '') or sample.get('statement', '')
        cot_prompt = config["cot_user_prompt_template"].format(
            table=table_str, question=question, statement=question
        )
        cot_messages = [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": cot_prompt},
        ]
        final_prediction = generate(model, tokenizer, cot_messages, max_new_tokens)

    return final_prediction, turn_history, mode


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3 model (Base vs Trained)")
    parser.add_argument("--model_type", type=str, choices=["base", "trained"], required=True,
                        help="'base' 加载原始模型; 'trained' 加载微调模型")
    parser.add_argument("--task", type=str, default="wikitableqa",
                        choices=["wikitableqa", "tabfact", "fetaqa", "hitab"],
                        help="要测试的任务")
    parser.add_argument("--prompt_mode", type=str, choices=["basic", "agent"], default=None,
                        help="提示词模式: 'basic'(单用户提示) 或 'agent'(含系统提示)。"
                             "single/sql_agent/mixed_agent 模式会自动选择; 其他模式 base 默认 basic, trained 默认 agent。")
    parser.add_argument("--inference_mode", type=str,
                        choices=["single", "sql_agent", "mixed_agent"], default="single",
                        help="推理模式:\n"
                             "  single      — 单轮生成 (v1 CoT 训练模型)\n"
                             "  sql_agent   — SQL 多轮执行循环 (v2 SQL 训练模型)\n"
                             "  mixed_agent — SQL 优先+CoT 回退 (v5 混合训练模型)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B",
                        help="基础模型名称/路径 (作为 LoRA 底座 或 base 模式加载对象)")
    parser.add_argument("--model_path", type=str, default="./outputs/models/Qwen3-1.7B-agent/",
                        help="微调后模型路径 (LoRA 适配器目录 或 全量微调保存目录)")
    parser.add_argument("--is_lora", action="store_true",
                        help="指定 --model_path 是 LoRA 适配器 (否则按全量微调加载)")
    parser.add_argument("--split", type=str, default="train",
                        help="数据切片 (test/validation/train)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="生成的最大新 token 数 (sql_agent/mixed_agent 每轮均用此值)")
    parser.add_argument("--max_turns", type=int, default=5,
                        help="sql_agent / mixed_agent 的最大 SQL 轮数")
    parser.add_argument("--max_empty", type=int, default=2,
                        help="sql_agent / mixed_agent 连续空结果超过此值后停止 SQL 循环")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="测试的数据条数")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="预测结果保存根目录 (会自动按任务名再建子目录)")
    parser.add_argument("--out_name", type=str, default=None,
                        help="输出 JSON 文件名。不传则按 model_type/inference_mode 自动生成")
    args = parser.parse_args()

    # --- 推断 prompt_mode (single → basic/agent, sql_agent/mixed_agent → agent) ---
    if args.inference_mode in ("sql_agent", "mixed_agent"):
        args.prompt_mode = "agent"
    elif args.prompt_mode is None:
        args.prompt_mode = "agent" if args.model_type == "trained" else "basic"

    # 默认输出文件名
    if args.out_name is None:
        args.out_name = f"predictions_{args.model_type}_{args.inference_mode}.json"
    if not args.out_name.endswith(".json"):
        args.out_name += ".json"

    print(f"🧩 model_type={args.model_type}  inference_mode={args.inference_mode}  "
          f"prompt_mode={args.prompt_mode}  is_lora={args.is_lora}  "
          f"task={args.task}  num_samples={args.num_samples}")
    print(f"📝 输出文件: {os.path.join(args.output_dir, args.task, args.out_name)}")

    # 1. 加载模型
    model, tokenizer = load_model_and_tokenizer(
        model_type=args.model_type,
        base_model_name=args.base_model,
        model_path=args.model_path,
        is_lora=args.is_lora,
    )

    # 2. 加载数据集
    cfg_dict = TASK_CONFIGS if args.prompt_mode == "agent" else TASK_TEST_CONFIGS
    task_config = cfg_dict[args.task]
    dataset_name = task_config["dataset_name"]
    target_field = task_config["target_field"]
    postprocess_func = task_config["postprocess_func"]

    print(f"📚 加载数据集 {dataset_name} [{args.split}] ...")
    dataset = load_dataset(dataset_name, split=args.split)
    num_to_test = min(args.num_samples, len(dataset))
    dataset = dataset.select(range(num_to_test))

    # 3. 循环推理
    results_to_save = []
    for i, sample in enumerate(tqdm.tqdm(dataset, desc=f"Generating for {args.task}")):

        turn_history = []
        mode = "single"

        if args.inference_mode == "sql_agent":
            # --- v2 训练模型：纯 SQL 多轮执行 ---
            prediction, turn_history, mode = run_sql_agent(
                model, tokenizer, args.task, sample, task_config,
                max_turns=args.max_turns, max_empty=args.max_empty,
                max_new_tokens=args.max_new_tokens,
            )
            # If the loop ends without Final Answer, fall back to the last response
            if not prediction:
                prediction = turn_history[-1]["response"] if turn_history else ""

        elif args.inference_mode == "mixed_agent":
            # --- v5 训练模型：SQL 优先 + CoT 回退 ---
            prediction, turn_history, mode = run_mixed_agent(
                model, tokenizer, args.task, sample, task_config,
                max_turns=args.max_turns, max_empty=args.max_empty,
                max_new_tokens=args.max_new_tokens,
            )

        else:
            # --- single：单轮生成 (v1 CoT 训练模型) ---
            messages, _, _ = build_messages_and_prompt(args.task, args.prompt_mode, sample)
            prediction = generate(model, tokenizer, messages, max_new_tokens=args.max_new_tokens)

        # 后处理 & 参考答案
        reference_label = sample[target_field]
        try:
            # single 模式下模型输出 <think>...</think>\n{answer}，需先剥掉 think 块
            # 否则 postprocess 函数会在 think 块内误匹配 "the answer is X" 等短语
            if args.inference_mode == "single":
                stripped = strip_think_block(prediction).strip()
                pred_for_postprocess = stripped if stripped else prediction
            else:
                pred_for_postprocess = prediction
            processed_prediction, _ = postprocess_func(pred_for_postprocess, reference_label)
        except Exception as e:
            print(f"⚠️ 后处理失败 (sample {i}): {e}")
            processed_prediction = prediction

        result = {
            "id": i,
            "original_dataset_id": str(sample.get("original_dataset_id", sample.get("id", "N/A"))),
            "task": args.task,
            "mode": mode,
            "prediction": prediction,
            "processed_prediction": processed_prediction,
            "reference": reference_label,
        }
        if turn_history:
            result["turn_details"] = turn_history
        results_to_save.append(result)

    # 4. 保存
    task_output_dir = os.path.join(args.output_dir, args.task)
    os.makedirs(task_output_dir, exist_ok=True)
    output_path = os.path.join(task_output_dir, args.out_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=4)

    print(f"\n✅ {len(results_to_save)} 个预测结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
