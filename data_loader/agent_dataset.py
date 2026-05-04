import json
import os
from glob import glob
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
import random


# 推理时使用的 System Prompt（与 configs.py 中 AGENT_SYSTEM_PROMPT 保持一致）
# 训练时必须注入相同的 System Prompt，否则模型推理时的行为分布会与训练时不一致
AGENT_SYSTEM_PROMPT = """You are an expert data analyst interacting with a SQLite database.
Act based on the input provided:

1. IF NO FEEDBACK (First Turn): Reason BRIEFLY in <think>...</think> (MAX 3 SENTENCES), You MUST start with `Columns: [exact names from Schema]`. Then output your query in ```sql ... ```.
2. IF ERROR FEEDBACK: Reflect BRIEFLY on the error in <think>...</think> (MAX 3 SENTENCES), then output a corrected SQLite query.
3. IF SUCCESS FEEDBACK: Based on the result, output EXACTLY "Final Answer: <answer>".

Crucial Notes for SQLite:
- NEVER output "Final Answer:" without seeing a successful query result first!
- Keep your <think> process extremely concise and direct.
- Do NOT use DISTINCT for counting unless the question explicitly asks for unique items.
"""


class AgentDataset(Dataset):
    """多轮智能体反馈数据集加载器 (用于错误修正逻辑蒸馏)
    
    核心设计：
    1. 从 predictions_vX.json 中读取 turn_details（多轮纠错轨迹）
    2. 对每个轮次进行"对话切片"，生成独立的训练样本
    3. 注入与推理时一致的 System Prompt（确保 train/inference 分布一致）
    4. 使用 tokenizer.apply_chat_template 构建标准 ChatML 格式
    5. 精准遮蔽 Loss：只在 assistant 最后一轮回复上计算梯度
    """
    
    def __init__(self, data_paths: List[str], tokenizer: Any, 
                 max_input_length: int = 1024, max_target_length: int = 1024,
                 split: str = "train", system_prompt: str = None,
                 only_correct: bool = False):
        """
        Args:
            data_paths: 数据文件路径列表
            tokenizer: Qwen tokenizer
            max_input_length: 最大输入长度
            max_target_length: 最大目标长度
            split: 数据集划分 (train/val/test)
            system_prompt: 自定义 system prompt，默认使用 AGENT_SYSTEM_PROMPT
            only_correct: 是否只保留最终答案正确的样本
        """
        self.tokenizer = tokenizer
        self.max_length = max_input_length + max_target_length
        self.split = split
        self.system_prompt = system_prompt or AGENT_SYSTEM_PROMPT
        self.only_correct = only_correct
        
        # 加载所有数据
        self.data = self._load_data(data_paths)
        
        # 预过滤：移除 prompt 过长导致 target 完全被截断的无效样本
        before_filter = len(self.data)
        self.data = self._filter_overlong(self.data)
        filtered_count = before_filter - len(self.data)
        if filtered_count > 0:
            print(f"  ⚠ Filtered {filtered_count} samples where prompt exceeds max_length={self.max_length}")
        
        # 数据集划分
        self.data = self._split_data(self.data, split)
        
        print(f"Loaded {len(self.data)} Agent multi-turn samples for {split} split")
    
    def _load_data(self, data_paths: List[str]) -> List[Dict]:
        """加载所有预测文件数据并切片为多轮训练样本"""
        all_data = []
        
        for data_path in data_paths:
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

                    data_list = data if isinstance(data, list) else data.get('predictions', [])
                    
                    for item in data_list:
                        # 可选：跳过最终答案错误的样本（避免让小模型学到错误推理）
                        if self.only_correct:
                            pred = str(item.get('processed_prediction', ''))
                            ref = str(item.get('reference', ''))
                            if pred != ref:
                                continue
                        
                        task_id = item.get('id', 'unknown')
                        turn_details = item.get('turn_details', [])
                        
                        if not turn_details:
                            continue
                            
                        for turn_idx, current_turn in enumerate(turn_details):
                            messages_history = []
                            
                            # 1. 重建此轮之前的历史对话
                            for prev_i in range(turn_idx):
                                prev_turn = turn_details[prev_i]
                                messages_history.append({
                                    "role": "user",
                                    "content": prev_turn.get('prompt', '')
                                })
                                messages_history.append({
                                    "role": "assistant",
                                    "content": prev_turn.get('response', '')
                                })
                                
                            # 2. 加入本轮的 User Prompt
                            messages_history.append({
                                "role": "user",
                                "content": current_turn.get('prompt', '')
                            })
                            
                            # 3. 目标是本轮的 Response
                            target_response = current_turn.get('response', '')
                            
                            if not target_response:
                                continue
                                
                            all_data.append({
                                'id': f"{task_id}_turn_{turn_idx}",
                                'messages_history': messages_history,
                                'target_response': target_response
                            })
                            
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print(f"  Total raw samples loaded: {len(all_data)}")
        return all_data
    
    def _filter_overlong(self, data: List[Dict]) -> List[Dict]:
        """预过滤掉 prompt 本身就超过 max_length 的样本（这些样本的 labels 全是 -100，无法训练）"""
        filtered = []
        for item in data:
            system_msg = [{"role": "system", "content": self.system_prompt}]
            prompt_messages = system_msg + item['messages_history']
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_len = len(self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            # 至少留 32 个 token 给 target
            if prompt_len < self.max_length - 32:
                filtered.append(item)
        return filtered

    def _split_data(self, data: List[Dict], split: str) -> List[Dict]:
        """数据集划分（以原始 task_id 为单位，避免同一道题的不同 turn 分散到 train/val）"""
        if not data: return []
        
        # 按原始 task_id 分组（去掉 _turn_X 后缀）
        groups = {}
        for item in data:
            base_id = item['id'].rsplit('_turn_', 1)[0]
            groups.setdefault(base_id, []).append(item)
        
        group_keys = sorted(groups.keys())
        random.seed(42)
        random.shuffle(group_keys)
        
        n = len(group_keys)
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)
        
        if split == "train":
            selected_keys = group_keys[:train_size]
        elif split == "val":
            selected_keys = group_keys[train_size:train_size + val_size]
        elif split == "test":
            selected_keys = group_keys[train_size + val_size:]
        else:
            return data
            
        result = []
        for k in selected_keys:
            result.extend(groups[k])
        return result
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # === 1. 构建 Messages ===
        # 注入 System Prompt（与推理时保持一致）
        system_msg = [{"role": "system", "content": self.system_prompt}]
        
        prompt_messages = system_msg + item['messages_history']
        full_messages = system_msg + item['messages_history'] + [
            {"role": "assistant", "content": item['target_response']}
        ]

        # === 2. 使用 apply_chat_template 渲染 ===
        # Prompt 部分 (add_generation_prompt=True 会追加 "<|im_start|>assistant\n")
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        # 完整对话
        full_text = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        # === 3. Tokenize ===
        prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_tokens["input_ids"])

        full_tokens = self.tokenizer(
            full_text, 
            add_special_tokens=False, 
            truncation=True, 
            max_length=self.max_length
        )
        
        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]

        # === 4. 构造 Labels ===
        labels = input_ids.copy()
        actual_prompt_len = min(prompt_len, len(labels))
        for i in range(actual_prompt_len):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
