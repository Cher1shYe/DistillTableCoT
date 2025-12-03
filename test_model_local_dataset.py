# run_inference.py
import os
import json
import tqdm
import argparse
import torch
from datasets import load_dataset, load_from_disk
from configs import TASK_CONFIGS
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
            
            # 加载tokenizer
            call_local_model.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            if call_local_model.tokenizer.pad_token is None:
                call_local_model.tokenizer.pad_token = call_local_model.tokenizer.eos_token
            
            # 加载模型
            call_local_model.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                # return_dict=False
            )

            # # 验证加载的是模型实例
            # if isinstance(call_local_model.model, dict):
            #     print("❌ 错误：加载的是配置字典，不是模型")
            #     # 尝试重新加载
            #     call_local_model.model = AutoModelForCausalLM.from_pretrained(
            #         model_path,
            #         torch_dtype=torch.float16,
            #         device_map="auto",
            #         trust_remote_code=True
            #     )
            
            call_local_model.model_path = model_path
            print(f"✅ 模型加载完成，类型: {type(call_local_model.model)}")

            print("✅ 模型加载完成")
        
        # # 验证模型类型
        # if isinstance(call_local_model.model, dict):
        #     raise ValueError("模型加载失败：返回的是配置字典")

        # 编码输入
        inputs = call_local_model.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024,
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
            'max_new_tokens': max_length,  # 🔥 使用max_new_tokens而不是max_length
            'temperature': 0.7,
            'do_sample': True,
            # 'pad_token_id': call_local_model.tokenizer.pad_token_id,
            # 'eos_token_id': call_local_model.tokenizer.eos_token_id,
            # 'return_dict_in_generate': True,  # 🔥 强制返回字典
        }
        
        

        # 生成文本
        with torch.no_grad():
            # outputs = call_local_model.model.generate(
            #     **inputs,
            #     max_length=input_len + max_length,
            #     temperature=temperature,
            #     do_sample=True,
            #     pad_token_id=call_local_model.tokenizer.pad_token_id,
            #     eos_token_id=call_local_model.tokenizer.eos_token_id,
            #     num_return_sequences=1,
            #     return_dict_in_generate=True,
            #     output_hidden_states=False,
            #     output_attentions=False
            # )

            # 先测试前向传播
            # test_output = call_local_model.model(**inputs)
            # print(f"📊 模型输出类型: {type(test_output)}")
            # print(test_output)
            
            outputs = call_local_model.model.generate(**generation_config)

        
        # 解码输出（只取新生成的部分）
        #if isinstance(outputs, tuple):
            # print("⚠️  generate返回元组")
        generated_tokens = outputs[0][input_len:]
        response = call_local_model.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
        
    except Exception as e:
        print(f"❌ 本地模型调用失败: {e}")
        return f"模型错误: {str(e)}"

def generate_predictions(task_name, num_samples, model_path, output_dir="outputs", out_name="predictions_qwen3.json"):
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
    
    generate_predictions(task_name=args.task_name, num_samples=args.num_samples, model_path=args.model_path, out_name=args.out_name)
