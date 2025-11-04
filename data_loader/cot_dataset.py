import json
import os
from glob import glob
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class CoTDataset(Dataset):
    """思维链数据集加载器"""
    
    def __init__(self, data_paths: List[str], tokenizer: Any, 
                 max_input_length: int = 1024, max_target_length: int = 512,
                 split: str = "train"):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.split = split
        
        # 加载所有数据
        self.data = self._load_data(data_paths)
        
        # 数据集划分
        self.data = self._split_data(self.data, split)
        
        print(f"Loaded {len(self.data)} samples for {split} split")
    
    def _load_data(self, data_paths: List[str]) -> List[Dict]:
        """加载所有预测文件数据"""
        all_data = []
        
        for data_path in data_paths:
            # 支持通配符路径
            if '*' in data_path:
                files = glob(data_path)
            else:
                files = [data_path]
            
            for file_path in files:
                if not os.path.exists(file_path):
                    print(f"Warning: File {file_path} not found, skipping")
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 处理数据格式
                    if isinstance(data, list):
                        # 直接是列表格式
                        for item in data:
                            standardized_item = self._standardize_item(item, os.path.basename(file_path))
                            all_data.append(standardized_item)
                    else:
                        # 可能是包含predictions字段的字典格式
                        data_list = data.get('predictions', [])
                        print(f"格式2: predictions字段，包含 {len(data_list)} 条数据")
                        for item in data_list:
                            standardized_item = self._standardize_item(item, os.path.basename(file_path))
                            all_data.append(standardized_item)
                            
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
        
        return all_data
    
    def _standardize_item(self, item: Dict, source_file: str) -> Dict:
        """标准化数据项格式"""
        # 提取思维链和答案
        prediction = item.get('prediction', '')
        processed_prediction = item.get('processed_prediction', '')
        reference = item.get('reference', '')
        
        # 从prediction中提取思维链和最终答案
        if 'Final Answer:' in prediction:
            # 分离思维链和最终答案
            parts = prediction.split('Final Answer:')
            cot = parts[0].strip()
            final_answer = parts[1].strip() if len(parts) > 1 else processed_prediction
        else:
            # 如果没有明确分隔，使用processed_prediction作为答案
            cot = prediction
            final_answer = processed_prediction or reference

        # to_return = {
        #     'id': item.get('id', ''),
        #     'original_dataset_id': item.get('original_dataset_id', ''),
        #     'prompt': item.get('prompt', ''),
        #     'question': self._extract_question(item.get('prompt', '')),
        #     'table': self._extract_table(item.get('prompt', '')),
        #     'cot': cot,
        #     'answer': final_answer,
        #     'reference': reference,
        #     'source_file': source_file
        # }
        # print(to_return)
        
        return {
            'id': item.get('id', ''),
            'original_dataset_id': item.get('original_dataset_id', ''),
            'prompt': item.get('prompt', ''),
            'question': self._extract_question(item.get('prompt', '')),
            'table': self._extract_table(item.get('prompt', '')),
            'cot': cot,
            'answer': final_answer,
            'reference': reference,
            'source_file': source_file
        }
    
    def _extract_question(self, prompt: str) -> str:
        """从prompt中提取问题"""
        if 'Question:' in prompt:
            parts = prompt.split('Question:')
            if len(parts) > 1:
                question_part = parts[1].split('Detailed Answer:')[0]
                return question_part.strip()
        return prompt[:200] + "..." if len(prompt) > 200 else prompt
    
    def _extract_table(self, prompt: str) -> str:
        """从prompt中提取表格数据"""
        if 'Table:' in prompt:
            parts = prompt.split('Table:')
            if len(parts) > 1:
                table_part = parts[1].split('Question:')[0]
                return table_part.strip()
        return ""
    
    def _split_data(self, data: List[Dict], split: str) -> List[Dict]:
        """数据集划分"""
        if not data:
            return []
            
        # 固定随机种子确保可重复性
        import random
        random.seed(42)
        random.shuffle(data)
        
        n = len(data)
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)
        
        if split == "train":
            return data[:train_size]
        elif split == "val":
            return data[train_size:train_size + val_size]
        elif split == "test":
            return data[train_size + val_size:]
        else:
            return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """返回单个训练样本 - 修复版本"""
        item = self.data[idx]
        
        input_text = f"问题：{item['question']}\n表格：{item['table']}\n推理："
        
        # 构建目标文本（模型要生成的内容）
        target_text = f"{item['cot']} 答案：{item['answer']}"
        
        # 合并为完整序列：输入 + 目标
        full_text = input_text + target_text

        # 使用固定的最大长度
        fixed_max_length = 1024  # 统一填充到这个长度
        
        # Tokenize完整序列
        encoding = self.tokenizer(
            full_text,
            max_length=fixed_max_length,  # 总长度限制
            padding='max_length',  # 不在这里填充，由DataCollator统一处理
            truncation=True,
            return_tensors="pt"
        )
        print(idx)
        # 计算输入文本的长度（用于创建标签掩码）
        input_encoding = self.tokenizer(
            input_text,
            add_special_tokens=False,
            max_length=fixed_max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        input_len = min(input_encoding['input_ids'].shape[1], fixed_max_length)
        
        # 创建标签：输入部分设为-100（忽略损失），目标部分保留
        labels = encoding['input_ids'].clone().squeeze(0)
        labels[:input_len] = -100
        # labels[:input_len] = -100  # 输入部分不计算损失

        # 确保填充位置的标签为-100
        pad_mask = encoding['attention_mask'].squeeze(0) == 0
        labels[pad_mask] = -100

        # 验证数据格式
        assert encoding['input_ids'].shape[1] == fixed_max_length, "input_ids长度不正确"
        assert labels.shape[0] == fixed_max_length, "labels长度不正确"
        assert encoding['attention_mask'].shape[1] == fixed_max_length, "attention_mask长度不正确"
        
        # 调试：检查长度是否一致
        if idx in [5542, 1762, 3954, 5968]:  # 打印出错批次的索引
            print(f"调试样本 {idx}:")
            print(f"  input_ids长度: {len(encoding['input_ids'].squeeze(0))}")
            print(f"  labels长度: {len(labels)}")
            print(f"  input_ids类型: {type(encoding['input_ids'])}")
            print(f"  labels类型: {type(labels)}")
            # 检查labels中是否有嵌套
            if isinstance(labels[0], list):
                print("  ❌ labels是嵌套列表！")
            else:
                print("  ✅ labels是1D列表")
        
        # to_return = {
        #     'input_ids': encoding['input_ids'].squeeze(0),
        #     'attention_mask': encoding['attention_mask'].squeeze(0),
        #     'labels': labels,
        #     'question': item['question'],
        #     'answer': item['answer'],
        #     'cot': item['cot']
        # }
        # print(to_return)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
            # 'question': item['question'],
            # 'answer': item['answer'],
            # 'cot': item['cot']
        }

# class DataCollatorForCoT:
#     """思维链数据整理器"""
    
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
#         self.tokenizer.pad_token = self.tokenizer.eos_token
    
#     def __call__(self, batch):
#         """处理批次数据"""
#         # 提取所有字段
#         input_ids = [item['input_ids'] for item in batch]
#         attention_mask = [item['attention_mask'] for item in batch]
#         labels = [item['labels'] for item in batch]
#         questions = [item['question'] for item in batch]
#         answers = [item['answer'] for item in batch]
#         cots = [item.get('cot', '') for item in batch]
        
#         # 统一填充到相同长度
#         padded_batch = self.tokenizer.pad(
#             {
#                 'input_ids': input_ids,
#                 'attention_mask': attention_mask,
#                 'labels': labels  # 重要：labels也参与填充
#             },
#             padding=True,
#             return_tensors="pt"
#         )
        
#         # 确保填充后的labels中，填充位置设为-100
#         padded_batch['labels'][padded_batch['labels'] == self.tokenizer.pad_token_id] = -100
        
#         # 添加元数据
#         padded_batch['questions'] = questions
#         padded_batch['answers'] = answers
#         padded_batch['cots'] = cots

#         return padded_batch

class DataCollatorForCoT:
    """思维链数据整理器 - 修复版本"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(self, batch):
        """处理批次数据 - 修复版本"""
        print(f"🔧 DataCollator处理批次，样本数: {len(batch)}")
        
        # 提取所有字段
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # === 关键修复1：检查数据类型 ===
        print("数据类型检查:")
        for i, (inp, lbl) in enumerate(zip(input_ids, labels)):
            print(f"  样本 {i}: input_ids类型={type(inp).__name__}, labels类型={type(lbl).__name__}")
            
            # 确保是列表类型
            if not isinstance(inp, list):
                print(f"  ⚠️ 样本 {i} input_ids不是列表，尝试转换")
                input_ids[i] = inp.tolist() if hasattr(inp, 'tolist') else list(inp)
            if not isinstance(lbl, list):
                print(f"  ⚠️ 样本 {i} labels不是列表，尝试转换")
                labels[i] = lbl.tolist() if hasattr(lbl, 'tolist') else list(lbl)
        
        # === 关键修复2：检查长度一致性 ===
        print("长度检查:")
        for i, (inp, lbl) in enumerate(zip(input_ids, labels)):
            inp_len = len(inp)
            lbl_len = len(lbl)
            print(f"  样本 {i}: input_ids长度={inp_len}, labels长度={lbl_len}")
            
            if inp_len != lbl_len:
                print(f"  ⚠️ 样本 {i} 长度不一致，进行对齐")
                # 对齐到较短的长度
                min_len = min(inp_len, lbl_len)
                input_ids[i] = inp[:min_len]
                labels[i] = lbl[:min_len]
        
        # === 关键修复3：安全填充 ===
        try:
            print("尝试使用tokenizer.pad进行填充...")
            padded_batch = self.tokenizer.pad(
                {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                },
                padding=True,
                return_tensors="pt"
            )
            print("✅ tokenizer.pad成功")
        except Exception as e:
            print(f"❌ tokenizer.pad失败: {e}")
            print("使用手动填充回退...")
            padded_batch = self._manual_pad(input_ids, attention_mask, labels)
        
        # === 关键修复4：确保填充位置labels设为-100 ===
        # 找到填充位置（attention_mask为0的位置）
        pad_mask = padded_batch['attention_mask'] == 0
        padded_batch['labels'][pad_mask] = -100
        
        # 添加元数据
        padded_batch['questions'] = [item.get('question', '') for item in batch]
        padded_batch['answers'] = [item.get('answer', '') for item in batch]
        padded_batch['cots'] = [item.get('cot', '') for item in batch]
        
        print(f"✅ 批次处理完成: input_ids形状={padded_batch['input_ids'].shape}")
        return padded_batch
    
    def _manual_pad(self, input_ids, attention_mask, labels):
        """手动填充回退方案"""
        print("使用手动填充...")
        
        # 找到最大长度
        max_len = max(
            max(len(seq) for seq in input_ids),
            max(len(seq) for seq in attention_mask),
            max(len(seq) for seq in labels)
        )
        
        print(f"手动填充到最大长度: {max_len}")
        
        # 手动填充
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for i, (inp, attn, lbl) in enumerate(zip(input_ids, attention_mask, labels)):
            # 填充input_ids
            if len(inp) < max_len:
                padded_inp = inp + [self.tokenizer.pad_token_id] * (max_len - len(inp))
            else:
                padded_inp = inp[:max_len]
            padded_input_ids.append(padded_inp)
            
            # 填充attention_mask
            if len(attn) < max_len:
                padded_attn = attn + [0] * (max_len - len(attn))
            else:
                padded_attn = attn[:max_len]
            padded_attention_mask.append(padded_attn)
            
            # 填充labels（填充位置设为-100）
            if len(lbl) < max_len:
                padded_lbl = lbl + [-100] * (max_len - len(lbl))
            else:
                padded_lbl = lbl[:max_len]
            padded_labels.append(padded_lbl)
            
            print(f"  样本 {i}: 原始长度={len(inp)}, 填充后长度={len(padded_inp)}")
        
        # 转换为张量
        import torch
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long)
        }