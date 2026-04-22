import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Dict
import os

# ==================== 数据预处理模块 ====================

def load_and_process_data(json_path: str, include_context: bool = False) -> List[Dict]:
    """
    加载并处理JSON数据，转换为训练样本
    
    Args:
        json_path: JSON文件路径
        include_context: 是否在后续turn中包含历史对话上下文
    
    Returns:
        处理后的训练样本列表
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    training_samples = []
    
    for item in raw_data:
        task_id = item['id']
        turns = item['turn_details']
        
        # 用于累积历史对话
        conversation_history = []
        
        for turn_idx, turn in enumerate(turns):
            prompt = turn['prompt']
            response = turn['response']
            
            # 构建当前turn的完整输入
            if include_context and turn_idx > 0:
                # 包含历史对话上下文
                full_prompt = "\n\n".join(conversation_history) + "\n\n" + prompt
            else:
                full_prompt = prompt
            
            training_samples.append({
                'id': f"{task_id}_turn_{turn_idx}",
                'instruction': full_prompt,
                'output': response,
                'task': item.get('task', 'unknown')
            })
            
            # 更新历史记录
            conversation_history.append(f"User: {prompt}")
            conversation_history.append(f"Assistant: {response}")
    
    print(f"✅ 成功处理 {len(training_samples)} 个训练样本")
    return training_samples


def format_qwen_prompt(instruction: str, output: str = None) -> str:
    """
    格式化为Qwen模型的标准对话格式
    
    Qwen3使用的是ChatML格式
    """
    if output is None:
        # 推理时使用
        return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    else:
        # 训练时使用
        return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"


def preprocess_function(examples, tokenizer, max_length=2048):
    """
    将样本转换为模型输入格式
    """
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    
    for instruction, output in zip(examples['instruction'], examples['output']):
        # 格式化完整对话
        full_text = format_qwen_prompt(instruction, output)
        
        # Tokenize
        tokenized = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        # 创建labels（只计算assistant回复部分的loss）
        prompt_text = format_qwen_prompt(instruction)
        prompt_length = len(tokenizer(prompt_text, return_tensors=None)['input_ids'])
        
        labels = tokenized['input_ids'].copy()
        # 将prompt部分的labels设为-100（不计算loss）
        labels[:prompt_length] = [-100] * prompt_length
        
        model_inputs["input_ids"].append(tokenized['input_ids'])
        model_inputs["attention_mask"].append(tokenized['attention_mask'])
        model_inputs["labels"].append(labels)
    
    return model_inputs


# ==================== 模型训练模块 ====================

def setup_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-4B-Instruct"):
    """
    初始化模型和tokenizer
    """
    print(f"🔧 加载模型: {model_name}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='right'  # 重要：训练时使用right padding
    )
    
    # 确保有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # 使用bf16节省显存
        device_map="auto",
        trust_remote_code=True
    )
    
    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,  # LoRA秩，可以根据显存调整（8-64）
        lora_alpha=16,  # LoRA缩放参数
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Qwen3的attention和MLP层
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def train_model(
    model,
    tokenizer,
    train_dataset,
    output_dir: str = "./qwen3_finetuned",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 4
):
    """
    执行模型训练
    """
    # 训练参数配置
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        bf16=True,  # 使用bf16混合精度训练
        gradient_checkpointing=True,  # 节省显存
        optim="adamw_torch",
        report_to="tensorboard",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
    )
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("🚀 开始训练...")
    trainer.train()
    
    # 保存最终模型
    print(f"💾 保存模型到 {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return trainer


# ==================== 主函数 ====================

def main():
    """
    完整的训练流程
    """
    # 配置参数
    JSON_PATH = "./outputs/wikitableqa/predictions_v2.json"  # 你的JSON文件路径
    MODEL_NAME = "Qwen/Qwen3-4B"  # 或者本地路径
    OUTPUT_DIR = "./qwen3_wikitableqa_finetuned"
    MAX_LENGTH = 2048
    
    # 1. 加载和处理数据
    print("=" * 50)
    print("📊 步骤1: 数据预处理")
    print("=" * 50)
    training_samples = load_and_process_data(JSON_PATH, include_context=True)
    
    # 转换为Dataset对象
    dataset = Dataset.from_list(training_samples)
    print(f"数据集大小: {len(dataset)}")
    print(f"样本示例:\n{dataset[0]}\n")
    
    # 2. 初始化模型和tokenizer
    print("=" * 50)
    print("🤖 步骤2: 模型初始化")
    print("=" * 50)
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)
    
    # 3. 预处理数据集
    print("=" * 50)
    print("🔄 步骤3: Tokenization")
    print("=" * 50)
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    # 4. 训练模型
    print("=" * 50)
    print("🎓 步骤4: 模型训练")
    print("=" * 50)
    trainer = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset,
        output_dir=OUTPUT_DIR,
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4,
        gradient_accumulation_steps=4
    )
    
    print("\n" + "=" * 50)
    print("✨ 训练完成！")
    print("=" * 50)
    print(f"模型保存位置: {OUTPUT_DIR}")
    print(f"可以使用以下代码加载模型:")
    print(f"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("{MODEL_NAME}")
model = PeftModel.from_pretrained(base_model, "{OUTPUT_DIR}")
tokenizer = AutoTokenizer.from_pretrained("{OUTPUT_DIR}")
    """)


if __name__ == "__main__":
    main()