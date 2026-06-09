"""
route-aware SFT 数据集 (对应 cost_aware.pdf 阶段三 / §6.1)。

吃 scripts/build_route_sft.py 产出的 jsonl (每行 {input, target, ...})：
    input  = Table + Schema + Question
    target = <ROUTE>X</ROUTE> + 轨迹 + <ANSWER>...</ANSWER>

与 CoTDataset 的差别：
    - 数据是 jsonl (逐行)，且已 oracle 过滤好，无需 only_correct/再解析。
    - 显式 enable_thinking=False：让 assistant 直接从 <ROUTE> 开始生成，
      不让 Qwen3 模板插入 <think>，保证"先路由"是输出的第一个动作 (训练/推理一致)。
其余 (apply_chat_template 渲染、mask prompt 只算 target loss) 与 CoTDataset 一致。
"""
import json
import os
from glob import glob
from typing import List, Dict, Any

from torch.utils.data import Dataset


class RouteSFTDataset(Dataset):
    """route-aware SFT 数据集加载器 (单轮 user -> assistant)。"""

    def __init__(self, data_paths: List[str], tokenizer: Any,
                 max_input_length: int = 2048, max_target_length: int = 2048,
                 split: str = "train", only_correct: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_input_length + max_target_length
        self.split = split
        # only_correct 对 route 数据无意义 (已 oracle 过滤)，仅为与 CoTDataset 签名兼容

        self.data = self._load_data(data_paths)
        self.data = self._split_data(self.data, split)
        print(f"Loaded {len(self.data)} route-SFT samples for {split} split")

    def _load_data(self, data_paths: List[str]) -> List[Dict]:
        all_data = []
        for data_path in data_paths:
            files = glob(data_path) if '*' in data_path else [data_path]
            for file_path in files:
                if not os.path.exists(file_path):
                    print(f"Warning: File {file_path} not found, skipping")
                    continue
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        if item.get("input") and item.get("target"):
                            all_data.append({
                                "prompt": item["input"],
                                "assistant_content": item["target"],
                            })
        return all_data

    def _split_data(self, data: List[Dict], split: str) -> List[Dict]:
        if not data:
            return []
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
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        prompt_messages = [{"role": "user", "content": item['prompt']}]
        full_messages = [
            {"role": "user", "content": item['prompt']},
            {"role": "assistant", "content": item['assistant_content']},
        ]

        # enable_thinking=False：assistant 直接从 <ROUTE> 开始，不插入 <think>，
        # 训练/推理一致。若 tokenizer 不认该参数则回退到默认渲染。
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            full_text = self.tokenizer.apply_chat_template(
                full_messages, tokenize=False, enable_thinking=False,
            )
        except TypeError:
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
            )
            full_text = self.tokenizer.apply_chat_template(
                full_messages, tokenize=False,
            )

        prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_tokens["input_ids"])

        full_tokens = self.tokenizer(
            full_text, add_special_tokens=False,
            truncation=True, max_length=self.max_length,
        )
        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]

        # mask 掉 prompt 部分，只对 target (含 <ROUTE>...<ANSWER>) 算 loss
        labels = input_ids.copy()
        actual_prompt_len = min(prompt_len, len(labels))
        for i in range(actual_prompt_len):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
