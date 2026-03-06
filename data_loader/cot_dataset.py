import json
import os
from glob import glob
from typing import List, Dict, Any
import re
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class CoTDataset(Dataset):
    """思维链数据集加载器"""
    
    def __init__(self, data_paths: List[str], tokenizer: Any, 
                 max_input_length: int = 1024, max_target_length: int = 1024,
                 split: str = "train"):
        self.tokenizer = tokenizer
        # 总最大长度 = prompt最大长度 + 答案最大长度
        self.max_length = max_input_length + max_target_length
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
                            all_data.append(self._standardize_item(item, file_path))
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
        """解析 Teacher 的预测结果，拆分出推理过程和最终答案"""
        prediction = item.get('prediction', '')
        
        # 优化后的正则：
        # \**\s* 兼容前面的 Markdown 加粗符号 (比如 **Answer:)
        # (?:...) 匹配你提供的各种可能的答案前缀
        # \s*\**\s* 兼容后面的冒号、空格和 Markdown 符号 (比如 Answer:** )
        # (.+) 贪婪匹配后面所有的内容，作为最终答案
        pattern = r'\**\s*(?:the final answer is|the answer is|so the answer is|final answer:|answer:)\s*\**\s*(.+)'
        
        match = re.search(pattern, prediction, re.IGNORECASE | re.DOTALL)
        
        if match:
            # match.start() 拿到的是 "Answer:" 这个词开始的索引位置
            # 因此，索引位置之前的【所有内容】，都是它的思考过程 (CoT)
            reasoning = prediction[:match.start()].strip()
            
            # match.group(1) 拿到的则是剥离了 "Answer:" 前缀后，纯粹的答案结果
            final_answer = match.group(1).strip()
            
            # 安全校验：如果模型直接输出了 "Answer: xxx" 而没有前面的推理过程
            # 那就不加 <think> 标签，避免生成空的 <think>\n</think>
            if reasoning:
                assistant_content = f"<think>\n{reasoning}\n</think>\n{final_answer}"
            else:
                assistant_content = final_answer
        else:
            # 如果正则完全没有命中（极其罕见的异常格式），把全部文本作为答案
            assistant_content = prediction

        return {
            'id': item.get('id', ''),
            'original_dataset_id': item.get('original_dataset_id', ''),
            'prompt': item.get('prompt', ''),
            'assistant_content': assistant_content
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
        if not data: return []
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
        item = self.data[idx]
        
        # 1. 组装对话体 (Messages)
        prompt_messages = [
            {"role": "user", "content": item['prompt']}
        ]
        full_messages = [
            {"role": "user", "content": item['prompt']},
            {"role": "assistant", "content": item['assistant_content']}
        ]

        # 2. 使用 apply_chat_template 渲染文本
        # 渲染 Prompt 部分 (add_generation_prompt=True 会加上 "<|im_start|>assistant\n")
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        # 渲染完整的 对话 (开启 enable_thinking=True 激活 Qwen3 的专属模版逻辑)
        full_text = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, enable_thinking=True
        )

        # 3. 进行分词 (注意：不加 padding！返回普通的 1D List)
        # 先对 prompt 切词，获取其准确的 Token 长度
        prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_tokens["input_ids"])

        # 再对完整的文本切词
        full_tokens = self.tokenizer(
            full_text, 
            add_special_tokens=False, 
            truncation=True, 
            max_length=self.max_length
        )
        
        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]

        # 4. 构造 Labels (将 Prompt 部分全部替换为 -100)
        labels = input_ids.copy()
        
        # 如果 prompt 长度超过了截断后的总长度（极端情况），做个安全限制
        actual_prompt_len = min(prompt_len, len(labels))
        
        # 将 input_ids 中属于 User 提问的部分，在 labels 里设为 -100
        for i in range(actual_prompt_len):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# 不使用原来的 DataCollatorForCoT 类
# 在 qwen_trainer.py 中直接使用官方的 DataCollatorForSeq2Seq。
