import json
import os
import re
from glob import glob
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
import random
from utils_train.eval_utils import is_match
# Import COT_SYSTEM_PROMPT from configs so the string stays in one place.
# Guarded import: utils_train tests may run without the full project on PYTHONPATH.
try:
    from configs import COT_SYSTEM_PROMPT as _COT_SYSTEM_PROMPT, AGENT_SYSTEM_PROMPT as _AGENT_SYSTEM_PROMPT
    COT_SYSTEM_PROMPT = _COT_SYSTEM_PROMPT
    AGENT_SYSTEM_PROMPT = _AGENT_SYSTEM_PROMPT
except ImportError:
    COT_SYSTEM_PROMPT = (
        "You read tables and answer questions. "
        "Think step-by-step then output exactly 'Final Answer: <answer>'."
    )
    AGENT_SYSTEM_PROMPT = """You are a data analyst using a SQLite database.

Steps:
1. First turn: In your thinking, first perform "Table Positioning" by identifying and listing the relevant column names. Then, think step-by-step about the SQL logic and write one ```sql ... ``` query.
2. After feedback: Check the SQL result. If the result is non-empty then actually answers the question. If wrong or empty, write a new ```sql ... ```. If correct, output exactly 'Final Answer: <answer>'.

Rules:
- During positioning, clearly state which columns are relevant to the question to ensure accuracy.
- Keep the original table's units/format in the final answer.
- For entities (like cities or names), prefer the full text as it appears in the table cell.
- When counting ("how many"), do not use DISTINCT unless the question specifically asks for "unique" or "different" items.
- Never output 'Final Answer:' without a valid non-empty SQL result.
"""


def _extract_cot_prompt(sql_turn_prompt: str) -> str:
    """Derive a clean CoT user prompt from a v5 SQL-turn prompt.

    The first SQL turn prompt has the form:
        Table (Markdown):\n{table_md}\n\nSchema (SQLite):\n{schema}\n\nQuestion: {question}

    We keep only the Markdown table and the question so the CoT prompt matches the
    cot_user_prompt_template used at teacher inference time (no schema needed for CoT).
    """
    # Extract Markdown table section
    table_match = re.search(r'Table \(Markdown\):\n(.*?)\n\nSchema', sql_turn_prompt, re.DOTALL)
    table_md = table_match.group(1).strip() if table_match else ""

    # Extract question (last "Question:" line; ignore "Previous Attempt" sections)
    question_match = re.search(r'^Question:\s*(.+)$', sql_turn_prompt, re.MULTILINE)
    question = question_match.group(1).strip() if question_match else ""

    if table_md and question:
        return f"Table:\n{table_md}\n\nQuestion: {question}\n\nThink step-by-step then output 'Final Answer: <answer>'."
    # Fallback: return the original prompt as-is
    return sql_turn_prompt


def _format_cot_fallback_response(prediction: str) -> str:
    """Wrap a v5 CoT-fallback prediction in <think>...</think> structure.

    The teacher's CoT response follows the pattern:
        {reasoning paragraphs}\n\nFinal Answer: {answer}

    We split at the last 'Final Answer:' occurrence so the reasoning block goes
    into <think> and the answer line follows outside.
    """
    prediction = prediction.strip()
    # Already has <think> — leave untouched
    if re.search(r'<think>.*?</think>', prediction, re.DOTALL):
        return prediction

    idx = prediction.rfind('Final Answer:')
    if idx == -1:
        return f"<think>\n{prediction}\n</think>"

    reasoning = prediction[:idx].strip()
    answer_line = prediction[idx:].strip()  # "Final Answer: ..."

    if reasoning:
        return f"<think>\n{reasoning}\n</think>\n{answer_line}"
    return answer_line


def _format_target_response(response: str) -> str:
    """Wrap a teacher SQL-agent response with proper <think>...</think> structure.

    The teacher model produces raw text that may or may not contain explicit thinking
    markers.  This function normalises every turn into the format the student is expected
    to generate at inference time:

    * SQL-generation turns  →  <think>\\n{reasoning}\\n</think>\\n```sql\\n{query}\\n```
    * Final-Answer turns    →  Final Answer: {answer}   (no think block)
    * Anything else         →  <think>\\n{text}\\n</think>
    """
    response = response.strip()

    # Strip unclosed <think> tags: if the response has <think> but no </think>, the model
    # output was truncated mid-thought.  Remove the dangling tag so the code below can
    # re-wrap the content correctly without producing double-nested <think><think> blocks.
    if '<think>' in response and '</think>' not in response:
        response = response.replace('<think>', '').strip()

    # If the response already contains properly closed <think>...</think>, leave it alone.
    if re.search(r'<think>.*?</think>', response, re.DOTALL):
        return response

    # Final-Answer turns: the model should output the answer directly, no thinking needed.
    if re.match(r'Final Answer:', response, re.IGNORECASE):
        return response

    # SQL-generation turns: split the reasoning prose from the ```sql ... ``` block.
    sql_match = re.search(r'(```sql\b.*?```)', response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        reasoning = response[:sql_match.start()].strip()
        sql_block = sql_match.group(1).strip()
        tail = response[sql_match.end():].strip()
        result = f"<think>\n{reasoning}\n</think>\n{sql_block}"
        if tail:
            result += f"\n{tail}"
        return result

    # Fallback: pure reasoning with no SQL — wrap everything in think.
    return f"<think>\n{response}\n</think>"


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

                    # FeTaQA uses free-form sentences → use ROUGE-L; others use exact match.
                    rouge_threshold = 0.3 if 'fetaqa' in file_path.lower() else None

                    for item in data_list:
                        # 可选：跳过最终答案错误的样本（避免让小模型学到错误推理）
                        if self.only_correct:
                            pred = item.get('processed_prediction', '')
                            ref = item.get('reference', '')
                            if not is_match(pred, ref, rouge_threshold=rouge_threshold):
                                continue

                        task_id = item.get('id', 'unknown')
                        turn_details = item.get('turn_details', [])

                        if not turn_details:
                            continue

                        # --- SQL agent turns (present in v2 and v5 SQL-mode samples) ---
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
                                    "content": _format_target_response(prev_turn.get('response', ''))
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

                            # Wrap with <think>...</think> so the student learns to produce
                            # the correct thinking structure at inference time.
                            target_response = _format_target_response(target_response)

                            all_data.append({
                                'id': f"{task_id}_turn_{turn_idx}",
                                'messages_history': messages_history,
                                'target_response': target_response,
                                'system_prompt': self.system_prompt,
                            })

                        # --- v5 CoT-fallback turn (mode="CoT" samples only) ---
                        # In v5, when all SQL attempts fail the teacher falls back to a plain
                        # CoT call whose response is stored only in the top-level `prediction`
                        # field (not in turn_details).  We inject it as an extra training
                        # sample so the student also learns the CoT fallback behaviour.
                        if item.get('mode') == 'CoT':
                            cot_response = item.get('prediction', '').strip()
                            # Must contain a Final Answer to be useful
                            if cot_response and 'Final Answer:' in cot_response:
                                # Reconstruct a minimal CoT user prompt from the first SQL
                                # turn's prompt: strip the Schema block and Previous Attempt
                                # section to get a clean table + question context.
                                raw_prompt = turn_details[0].get('prompt', '')
                                cot_user_prompt = _extract_cot_prompt(raw_prompt)
                                cot_target = _format_cot_fallback_response(cot_response)
                                all_data.append({
                                    'id': f"{task_id}_cot_fallback",
                                    'messages_history': [{"role": "user", "content": cot_user_prompt}],
                                    'target_response': cot_target,
                                    # Use the lighter COT_SYSTEM_PROMPT so train matches inference
                                    'system_prompt': COT_SYSTEM_PROMPT,
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
            item_system_prompt = item.get('system_prompt', self.system_prompt)
            system_msg = [{"role": "system", "content": item_system_prompt}]
            prompt_messages = system_msg + item['messages_history']
            # Use the same template settings as __getitem__ so the length estimate is accurate.
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
        
        # Group by original task_id so all turns of the same example (SQL turns AND the
        # optional CoT-fallback turn) always land in the same split.
        groups = {}
        for item in data:
            # Strip either '_turn_N' or '_cot_fallback' suffixes
            base_id = re.split(r'_turn_\d+$|_cot_fallback$', item['id'])[0]
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
        # Each sample stores its own system_prompt (either AGENT_SYSTEM_PROMPT for SQL turns
        # or COT_SYSTEM_PROMPT for v5 CoT-fallback turns) so train/inference prompts match.
        item_system_prompt = item.get('system_prompt', self.system_prompt)
        system_msg = [{"role": "system", "content": item_system_prompt}]

        prompt_messages = system_msg + item['messages_history']
        full_messages = system_msg + item['messages_history'] + [
            {"role": "assistant", "content": item['target_response']}
        ]

        # === 2. 使用 apply_chat_template 渲染 ===
        # target_response already contains explicit <think>...</think> tags from
        # _format_target_response.  Do NOT pass enable_thinking=True: Qwen3's template with
        # that flag prepends an extra "<think>\n" token to every assistant turn, which would
        # (a) double/misplace the opening tag and (b) cause prompt_len to miscount the
        # assistant-prefix boundary — both corrupting labels.
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
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
