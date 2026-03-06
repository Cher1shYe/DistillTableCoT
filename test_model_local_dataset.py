# test_model_local_dataset.py
import os
import json
import tqdm
import argparse
import torch
from datasets import load_dataset, load_from_disk
from configs import TASK_TEST_CONFIGS
from utils import format_table
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_DATA_DIR = "local_datasets"
# 添加示例池导入
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def call_local_model(prompt, model_path, max_length=1024, temperature=0.7):
    """
    调用本地训练好的小模型进行推理
    """
    try:
        # 加载模型和tokenizer（单例模式，避免重复加载）
        if not hasattr(call_local_model, 'model'):
            print(f"🔍 加载本地模型: {model_path}")

            import os
            print("目录中的文件:", os.listdir(model_path))
            
            # 加载tokenizer
            call_local_model.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            if call_local_model.tokenizer.pad_token is None:
                # 兼容我们在训练时加入的 pad_token 或 endoftext
                if "<|endoftext|>" in call_local_model.tokenizer.get_vocab():
                    call_local_model.tokenizer.pad_token = "<|endoftext|>"
                else:
                    call_local_model.tokenizer.pad_token = call_local_model.tokenizer.eos_token
            
            # 加载模型
            call_local_model.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            call_local_model.model_path = model_path
            print(f"✅ 模型加载完成，类型: {type(call_local_model.model)}")

            print("✅ 模型加载完成")

        # DEBUG 1：将输入组装成标准对话格式 (Messages)
        messages = [
            {"role": "user", "content": prompt}
        ]

        # DEBUG 2：应用 Chat Template（与训练时完全对齐！）
        formatted_prompt = call_local_model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True # 极其重要：它会自动在末尾加上 <|im_start|>assistant\n
        )

        # 编码输入
        inputs = call_local_model.tokenizer(
            formatted_prompt,  # 🔥 使用渲染后的 formatted_prompt，而不是原始的 prompt 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048,
            padding=True
        )

        # # 🔥 关键修复：使用字典键访问，不是对象属性
        # print(f"📊 输入类型: {type(inputs)}")  # 应该是dict
        # print(f"📊 输入键: {list(inputs.keys())}")  # 查看字典键
        
        # 移动到模型设备
        device = call_local_model.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 🔥 关键修复：使用字典键访问input_ids
        input_ids = inputs['input_ids']  # 不是 inputs.input_ids
        input_len = input_ids.shape[1]   # 获取序列长度
        attention_mask = inputs['attention_mask']
        # print(input_ids)
        # print(input_len)
        # print(type(inputs))

        # 🔥 Qwen专用生成配置
        generation_config = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_new_tokens': max_length,
            'temperature': 0.6,             # 推荐为0.6，增加确定性
            'do_sample': True,
            'top_p': 0.95,                  # 限制长尾 Token
            'top_k': 20,
            'repetition_penalty': 1.05,      # 加入重复惩罚
            'pad_token_id': call_local_model.tokenizer.pad_token_id,
            'eos_token_id': call_local_model.tokenizer.eos_token_id,
        }
        
        

        # 生成文本
        with torch.no_grad():
            outputs = call_local_model.model.generate(**generation_config)

        generated_tokens = outputs[0][input_ids.shape[-1]:] 
        response = call_local_model.tokenizer.decode(generated_tokens, skip_special_tokens=False) # 这里不屏蔽特殊字符
        
        return response.strip()
        
    except Exception as e:
        print(f"❌ 本地模型调用失败: {e}")
        return f"模型错误: {str(e)}"

def generate_predictions(task_name, num_samples, model_path, split="test", output_dir="outputs", out_name="predictions_qwen3.json"):
    """
    针对指定任务运行推理，并将结果保存到 JSON 文件。
    """
    if task_name not in TASK_TEST_CONFIGS:
        print(f"错误: 任务 '{task_name}' 未在 TASK_TEST_CONFIGS 中定义。")
        return

    print(f"--- 开始为任务生成预测: {task_name} ---")
    
    # 1. 加载配置和数据
    config = TASK_TEST_CONFIGS[task_name]
    try:
        hf_dataset_name = config["dataset_name"]
        local_folder_name = hf_dataset_name
        local_dataset_path = os.path.join(LOCAL_DATA_DIR, local_folder_name)
        print(f"从本地路径加载数据集字典: {local_dataset_path}")
        dataset_dict = load_from_disk(local_dataset_path)
        split_name = config["dataset_split"]
        if split_name not in dataset_dict:
             # 有些数据集用 'dev'，有些用 'validation'，这里做个兼容
             if split_name == 'validation' and 'dev' in dataset_dict:
                 print(f"警告: split '{split_name}' 不存在，尝试使用 'dev' 代替。")
                 split_name = 'dev'
             elif split_name == 'train' and 'test' not in dataset_dict: # 兼容只有一个split的情况
                 # 如果只有一个 split，直接用它
                 first_split_key = list(dataset_dict.keys())[0]
                 print(f"警告: split '{split_name}' 不存在, 使用找到的唯一 split '{first_split_key}'。")
                 split_name = first_split_key
             else:
                # 如果还是找不到，就抛出错误
                raise KeyError(f"在数据集 {local_dataset_path} 中找不到指定的 split: '{split_name}'. 可用的 splits: {list(dataset_dict.keys())}")


        dataset = dataset_dict[split_name]
        
        print(f"成功加载 split '{split_name}'，包含 {len(dataset)} 个样本。")

    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        print(f"请确认 '{local_dataset_path}' 路径存在，并且是使用 'download_data.py' 完整下载的数据集目录。")
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
        # prediction = call_deepseek_api(prompt)
        # model_path = "./outputs/models/qwen3-1.7b"
        prediction = call_local_model(prompt, model_path)

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
    output_path = os.path.join(task_output_dir, out_name)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=4)
        
    print(f"\n成功! {len(results_to_save)} 个预测结果已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为表格问答任务生成模型预测。")
    parser.add_argument(
        "--task_name", 
        type=str, 
        required=True, 
        choices=TASK_TEST_CONFIGS.keys(),
        help="要运行的任务名称。"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="test",
        help="指定推理使用的数据切片 (例如: test, validation, train)"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=5,
        help="要处理的样本数量。"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="待测试模型路径"
    )
    parser.add_argument(
        "--out_name",
        type=str,
        help="输出json的文件名"
    )
    args = parser.parse_args()
    
    generate_predictions(task_name=args.task_name, num_samples=args.num_samples, model_path=args.model_path, split=args.split, out_name=args.out_name)
