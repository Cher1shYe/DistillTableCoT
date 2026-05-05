import argparse
import torch
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 添加根目录到 path 以便引入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import TASK_TEST_CONFIGS, TASK_CONFIGS
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
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
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


def build_messages(task, prompt_mode, sample):
    """
    根据 prompt_mode 组装对话消息：
    - agent: 使用 TASK_CONFIGS（含系统提示词，适合带纠错轨迹训练的 Agent 模型）
    - basic: 使用 TASK_TEST_CONFIGS（仅用户提示，与根 test_model.py 对齐，
             适合基础数据训练的小模型 以及 未微调的 base 模型）
    """
    table_str = format_table(sample.get('table') or sample.get('table_text'))
    question = sample.get('question', '') or sample.get('statement', '')

    if prompt_mode == "agent":
        config = TASK_CONFIGS[task]
        system_prompt = config["system_prompt"]
        user_prompt = config["user_prompt_template"].format(
            table=table_str,
            question=question,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        dataset_name = config["dataset_name"]
    else:
        config = TASK_TEST_CONFIGS[task]
        prompt = config["prompt_template"].format(
            table=table_str,
            question=sample.get('question', ''),
            statement=sample.get('statement', ''),
        ) if "{statement}" in config["prompt_template"] else config["prompt_template"].format(
            table=table_str,
            question=question,
        )
        messages = [{"role": "user", "content": prompt}]
        dataset_name = config["dataset_name"]

    return messages, dataset_name


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
    response = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3 model (Base vs Trained)")
    parser.add_argument("--model_type", type=str, choices=["base", "trained"], required=True,
                        help="'base' 加载原始模型; 'trained' 加载微调模型")
    parser.add_argument("--task", type=str, default="wikitableqa",
                        choices=["wikitableqa", "tabfact", "fetaqa", "hitab"],
                        help="要测试的任务")
    parser.add_argument("--prompt_mode", type=str, choices=["basic", "agent"], default=None,
                        help="提示词模式: 'basic'(单用户提示, 对应基础数据训练的小模型) "
                             "或 'agent'(系统+用户提示, 对应带纠错轨迹训练的 Agent 模型)。"
                             "base 模型默认 basic; trained 模型默认 agent")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B",
                        help="基础模型名称/路径 (作为 LoRA 底座 或 base 模式加载对象)")
    parser.add_argument("--model_path", type=str, default="./outputs/models/Qwen3-1.7B-agent/",
                        help="微调后模型路径 (LoRA 适配器目录 或 全量微调保存目录)")
    parser.add_argument("--is_lora", action="store_true",
                        help="指定 --model_path 是 LoRA 适配器 (否则按全量微调加载)")
    parser.add_argument("--split", type=str, default="train",
                        help="数据切片 (test/validation/train)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="生成的最大新 token 数")
    args = parser.parse_args()

    # 默认提示词模式
    if args.prompt_mode is None:
        args.prompt_mode = "agent" if args.model_type == "trained" else "basic"

    print(f"🧩 model_type={args.model_type}  prompt_mode={args.prompt_mode}  "
          f"is_lora={args.is_lora}  task={args.task}")

    # 1. 加载模型
    model, tokenizer = load_model_and_tokenizer(
        model_type=args.model_type,
        base_model_name=args.base_model,
        model_path=args.model_path,
        is_lora=args.is_lora,
    )

    # 2. 组装消息 (确定数据集)
    # 先取一条 dummy 样本构造 dataset_name
    cfg_dict = TASK_CONFIGS if args.prompt_mode == "agent" else TASK_TEST_CONFIGS
    dataset_name = cfg_dict[args.task]["dataset_name"]

    print(f"📚 加载数据集 {dataset_name} [{args.split}] ...")
    dataset = load_dataset(dataset_name, split=args.split)
    sample = dataset[0]

    messages, _ = build_messages(args.task, args.prompt_mode, sample)

    print("\n" + "=" * 50)
    print("Formatted Input Messages:")
    print("=" * 50)
    for msg in messages:
        print(f"[{msg['role'].upper()}]:\n{msg['content']}\n")
    print("=" * 50 + "\n")

    # 3. 生成
    print("Generating answer... (this may take a few seconds)")
    response = generate(model, tokenizer, messages, max_new_tokens=args.max_new_tokens)

    print("\n" + "=" * 50)
    print(f"Model Response ({args.model_type.upper()} MODEL, {args.prompt_mode.upper()} PROMPT):")
    print("=" * 50)
    print(response)
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()