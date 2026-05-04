import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, Any, List
import os
from data_loader.cot_dataset import CoTDataset

class QwenDistillTrainer:
    """Qwen模型蒸馏训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self):
        """初始化模型和tokenizer"""
        print("Initializing model and tokenizer...")
        
        # local or remote
        if 'local_path' in self.config['model'] and os.path.exists(self.config['model']['local_path']):
            model_path = self.config['model']['local_path']
            tokenizer_path = self.config['model']['local_path']
            print(f"Loading local model: {model_path}")
        else:
            model_path = self.config['model']['model_name']
            tokenizer_path = self.config['model']['tokenizer_name']
            print(f"Loading remote model: {tokenizer_path}")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        # Qwen 的 tokenizer 包含:
        # 151643 -> <|endoftext|>
        # 151645 -> <|im_end|> (通常作为 eos_token)
        
        if self.tokenizer.pad_token is None:
            # 优先尝试使用 <|endoftext|> 作为 pad_token，它通常在 Qwen 词表中存在但很少用到
            if "<|endoftext|>" in self.tokenizer.get_vocab():
                self.tokenizer.pad_token = "<|endoftext|>"
                print(f"Using <|endoftext|> (id: {self.tokenizer.pad_token_id}) as pad_token.")
            else:
                # 如果没有，就回退使用 eos_token 作为 pad_token
                # 只要 attention_mask 设置正确，这完全没有问题
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"Using eos_token (id: {self.tokenizer.pad_token_id}) as pad_token.")
        need_resize_embeddings = False
        
        # 加载模型
        torch_dtype = getattr(torch, self.config['model'].get('torch_dtype', 'bfloat16'))
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=self.config['model'].get('device_map', 'auto'),
            low_cpu_mem_usage=self.config['model'].get('low_cpu_mem_usage', True),
            attn_implementation="sdpa"
        )
        
        # 增加 LoRA 支持 (业界标准微调 Baseline)
        if self.config['training'].get('use_lora', False):
            print("🚀 Configuring LoRA for the model...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config['training'].get('lora_r', 64),
                lora_alpha=self.config['training'].get('lora_alpha', 16),
                lora_dropout=self.config['training'].get('lora_dropout', 0.1),
                target_modules=self.config['training'].get('lora_target_modules', ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
                bias="none"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        print(f"✅ Model loaded: {self.config['model']['model_name']}")
        print(f"✅ Model dtype: {self.model.dtype}")
        print(f"✅ Model device: {self.model.device}")
        
    def prepare_data(self):
        """准备训练数据"""
        print("Preparing training data...")
        
        dataset_type = self.config['data'].get('dataset_type', 'cot')
        
        if dataset_type == 'agent':
            print("=> Using AgentDataset for Multi-Turn Agent Training")
            from data_loader.agent_dataset import AgentDataset
            DatasetClass = AgentDataset
        else:
            print("=> Using CoTDataset for Single-Turn CoT Training")
            DatasetClass = CoTDataset
        
        # 训练集
        self.train_dataset = DatasetClass(
            data_paths=self.config['data']['data_paths'],
            tokenizer=self.tokenizer,
            max_input_length=self.config['training']['max_input_length'],
            max_target_length=self.config['training']['max_target_length'],
            split="train"
        )
        
        # 验证集
        self.val_dataset = DatasetClass(
            data_paths=self.config['data']['data_paths'],
            tokenizer=self.tokenizer,
            max_input_length=self.config['training']['max_input_length'],
            max_target_length=self.config['training']['max_target_length'],
            split="val"
        )
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
    
    def train(self):
        """执行训练"""
        if self.model is None or self.tokenizer is None:
            self.setup()
        
        if self.train_dataset is None:
            self.prepare_data()

        use_bf16 = self.config['training'].get('bf16', False)
        use_fp16 = self.config['training'].get('fp16', False)

        # 逻辑互斥：如果开启了 bf16，必须强制关闭 fp16，否则 Trainer 会报错
        if use_bf16:
            use_fp16 = False
        
        lr_value = self.config['training']['learning_rate']
        if isinstance(lr_value, str):
            lr_value = float(lr_value.strip())          # "3e-5" 或 "0.00003" → 3e-5
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config['training']['output_dir'],
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['eval_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=lr_value,
            warmup_ratio=self.config['training'].get('warmup_ratio', 0.1),
            logging_dir=os.path.join(self.config['training']['output_dir'], 'logs'),
            logging_steps=self.config['training']['logging_steps'],
            eval_steps=self.config['training']['eval_steps'],
            save_steps=self.config['training']['save_steps'],
            save_total_limit=self.config['training']['save_total_limit'],
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,


            # --- 修复bug ---
            fp16=use_fp16,  # 使用处理后的变量
            bf16=use_bf16,  # 使用处理后的变量


            dataloader_pin_memory=True,
            remove_unused_columns=False,
            dataloader_num_workers=self.config['training'].get('num_workers', 4),
            report_to=self.config['training'].get('report_to', "none"), 
            seed=self.config['training'].get('seed', 42),
            gradient_checkpointing=self.config['training'].get('gradient_checkpointing', True),
            optim=self.config['training'].get('optim', "adamw_torch"), # 推荐 adamw_torch
            max_steps=self.config['training'].get('max_steps', -1) # -1 表示按 epoch 走
        )

        print("=== TrainingArguments 参数类型检查 ===")
        print(f"learning_rate 值: {training_args.learning_rate}")
        print(f"lr_value 类型: {type(lr_value).__name__}")
        print(f"learning_rate 类型: {type(training_args.learning_rate).__name__}")
        print(f"num_train_epochs: {training_args.num_train_epochs}")
        print(f"per_device_train_batch_size: {training_args.per_device_train_batch_size}")
        print(f"per_device_eval_batch_size: {training_args.per_device_eval_batch_size}")
        print(f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
        print(f"warmup_ratio: {training_args.warmup_ratio}")
        print(f"logging_dir: {training_args.logging_dir}")
        print(f"logging_steps: {training_args.logging_steps}")
        print(f"eval_steps: {training_args.eval_steps}")
        print(f"save_steps: {training_args.save_steps}")
        print(f"save_total_limit: {training_args.save_total_limit}")
        print(f"eval_strategy: {training_args.eval_strategy}")
        print(f"save_strategy: {training_args.save_strategy}")
        print(f"load_best_model_at_end: {training_args.load_best_model_at_end}")
        print(f"metric_for_best_model: {training_args.metric_for_best_model}")
        print(f"greater_is_better: {training_args.greater_is_better}")
        print(f"fp16: {training_args.fp16}")
        print(f"dataloader_pin_memory: {training_args.dataloader_pin_memory}")
        print(f"remove_unused_columns: {training_args.remove_unused_columns}")
        print(f"report_to: {training_args.report_to}")
        print(f"seed: {training_args.seed}")
        
        # 数据整理器

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding="longest",
            label_pad_token_id=-100  # 确保 padding 部分不参与 loss 计算
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer,
        )
        
        self.is_model_available()

        # self.simplest_test()

        # 开始训练
        print("Starting training...")
        trainer.train()
        
        # 保存最终模型
        final_output_dir = os.path.join(self.config['training']['output_dir'], "final_model")
        trainer.save_model(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)
        
        print(f"Training completed! Model saved to: {final_output_dir}")
        
        return trainer
    
    def simplest_test(self):
        """最简单的测试"""
        sample = self.train_dataset[0]
        self.model.eval()
        
        # ✅ 获取模型所在的设备 (比如 cuda:0)
        device = self.model.device
        print(f"将测试数据推送到设备: {device}")
        
        # ✅ 把所有的 tensor 加上 .to(device)
        inputs = {
            'input_ids': sample['input_ids'].unsqueeze(0).long().to(device),
            'attention_mask': sample['attention_mask'].unsqueeze(0).long().to(device),
            'labels': sample['labels'].unsqueeze(0).long().to(device)
        }

        with torch.no_grad():
            outputs = self.model(**inputs)
            print(f"单个样本前向传播成功，损失: {outputs.loss}")
    
    def is_model_available(self):
        """检查模型是否可用"""
        if self.model is None:
            print("❌ 模型为None")
            return False

        # 检查是否是有效的transformers模型
        try:
            from transformers import PreTrainedModel
            if not isinstance(self.model, PreTrainedModel):
                print("❌ 模型不是有效的PreTrainedModel")
                return False
        except:
            print("❌ 无法导入PreTrainedModel")
            return False

        # 检查模型参数
        try:
            param_count = sum(p.numel() for p in self.model.parameters())
            if param_count == 0:
                print("❌ 模型参数数量为0")
                return False
            print(f"✅ 模型参数数量: {param_count:,}")
        except Exception as e:
            print(f"❌ 检查模型参数失败: {e}")
            return False

        return True
    
def create_minimal_test():
    import torch

    """创建最小复现案例"""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建两个长度差异大的样本（使用torch tensor）
    # 正确的测试数据
    sample1 = {
        # 'input_ids': [1, 2, 3, 4, 5],  # 长度5
        # 'attention_mask': [1, 1, 1, 1, 1],  # 全部是数字1
        'labels': [-100, -100, -100, 6, 7]  # 前3个是输入
    }
    
    sample2 = {
        # 'input_ids': [1, 2, 3, 4, 5, 6, 7],  # 长度7
        # 'attention_mask': [1, 1, 1, 1, 1, 1, 1],  # 全部是数字1
        'labels': [-100, -100, 8, 9, 10, 11, 12]  # 前2个是输入
    }

    print("测试数据检查:")
    for i, sample in enumerate([sample1, sample2]):
        print(f"样本 {i}:")
        for key, value in sample.items():
            print(f"  {key}: 长度={len(value)}, 类型={type(value[0]).__name__}")
    
    from transformers import DataCollatorWithPadding
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        max_length=512,
        padding='max_length'
    )
    
    try:
        batch = collator([sample1, sample2])
        print("✅ 最小测试通过")
    except Exception as e:
        print(f"❌ 最小测试失败: {e}")
        import traceback
        traceback.print_exc()


