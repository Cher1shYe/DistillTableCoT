import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from typing import Dict, Any, List
import os
from data_loader.cot_dataset import CoTDataset, DataCollatorForCoT

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
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        torch_dtype = getattr(torch, self.config['model'].get('torch_dtype', 'float16'))
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto",

            torch_dtype='float32'
        )
        
        print(f"Model loaded: {self.config['model']['model_name']}")
        print(f"Model device: {self.model.device}")
        
    def prepare_data(self):
        """准备训练数据"""
        print("Preparing training data...")
        
        # 训练集
        self.train_dataset = CoTDataset(
            data_paths=self.config['data']['data_paths'],
            tokenizer=self.tokenizer,
            max_input_length=self.config['training']['max_input_length'],
            max_target_length=self.config['training']['max_target_length'],
            split="train"
        )
        
        # 验证集
        self.val_dataset = CoTDataset(
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


            dataloader_pin_memory=False,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            disable_tqdm=False,
            report_to="none",  # 禁用wandb等记录
            seed=self.config['training']['seed'],

            gradient_checkpointing=True,       # 梯度检查点 (时间换空间)
            optim="adafactor",

            max_steps=100
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

        # from data_loader.cot_dataset import DataCollatorForCoT
    
        # data_collator = DataCollatorForCoT(tokenizer=self.tokenizer)

        # from transformers import DataCollatorWithPadding
        # data_collator = DataCollatorWithPadding(
        #     tokenizer=self.tokenizer,
        #     padding='longest',
        #     max_length=None,
        #     pad_to_multiple_of=8,
        # )
        
        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            # data_collator=data_collator,
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
        # 只测试数据加载，不训练
        sample = self.train_dataset[0]
        print(f"样本形状检查:")
        for key in ['input_ids', 'attention_mask', 'labels']:
            value = sample[key]
            print(f"{key}: 形状={value.shape}, 维度={value.dim()}")

        self.model.eval()  # 关闭 Dropout 等训练专用层

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(f"使用设备: {device}")

        # self.model = self.model.to(device)
        
        # 测试单个样本前向传播
        inputs = {
            'input_ids': sample['input_ids'].unsqueeze(0).long(),
            'attention_mask': sample['attention_mask'].unsqueeze(0).long(),
            'labels': sample['labels'].unsqueeze(0).long()
        }

        print("inputs is:\n")
        print(inputs)

        # print(f"模型设备: {next(self.model.parameters()).device}")
        # print(f"输入数据设备: {inputs['input_ids'].device}")
        
        # print(f"input_ids 形状: {inputs['input_ids'].shape}")  # 应为 (1, seq_len)

        # print(f"可用显存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        # print(f"模型权重类型: {next(self.model.parameters()).dtype}")
        # print(f"输入数据类型: {inputs['input_ids'].dtype}")

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


