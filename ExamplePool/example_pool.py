import json
import os
import random
from typing import List, Dict, Any
from datasets import load_dataset

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import TASK_CONFIGS
from utils import format_table, call_deepseek_api

class ExamplePool:
    def __init__(self, pool_dir="example_pools"):
        self.pool_dir = pool_dir
        os.makedirs(pool_dir, exist_ok=True)
        self.pools = {}
        
    def build_cot_prompt(self, table: str, question: str) -> str:
        """构建生成 CoT 的专用 prompt"""
        return f"""请仔细分析下面的表格并回答问题。请按照以下步骤进行推理：

表格：
{table}

问题：{question}

请按照以下格式回答：
1. 首先，理解表格的结构和内容...
2. 然后，分析问题的要求...
3. 接着，在表格中查找相关信息...
4. 最后，基于找到的信息得出结论...

推理过程：
"""

    def generate_cot_example(self, task_name: str, sample: Dict) -> Dict[str, Any]:
        """为单个样本生成 CoT 示例"""
        table_str = format_table(sample.get('table') or sample.get('table_text'))
        question = sample.get('question', '') or sample.get('statement', '')
        
        # 使用专门的 CoT prompt
        cot_prompt = self.build_cot_prompt(table_str, question)
        
        # 调用 API 生成 CoT
        cot_response = call_deepseek_api(cot_prompt)
        
        # 获取标准答案
        target_field = TASK_CONFIGS[task_name]["target_field"]
        answer = sample[target_field]
        
        return {
            "table": sample.get('table') or sample.get('table_text'),
            "question": question,
            "answer": answer,
            "cot_reasoning": cot_response,
            "task": task_name,
            "original_id": sample.get("original_dataset_id", "unknown")
        }
    
    def build_pool_for_task(self, task_name: str, num_examples: int = 50) -> str:
        """为特定任务构建示例池"""
        if task_name not in TASK_CONFIGS:
            raise ValueError(f"未知任务: {task_name}")
        
        config = TASK_CONFIGS[task_name]
        
        # 加载数据集
        try:
            dataset = load_dataset(config["dataset_name"], split=config["dataset_split"])
        except Exception as e:
            print(f"数据集加载失败: {e}")
            return None
        
        # 随机选择样本
        if num_examples > len(dataset):
            num_examples = len(dataset)
        
        selected_indices = random.sample(range(len(dataset)), num_examples)
        selected_samples = [dataset[i] for i in selected_indices]
        
        pool = []
        print(f"开始为 {task_name} 生成 {num_examples} 个 CoT 示例...")
        
        for i, sample in enumerate(selected_samples):
            print(f"生成示例 {i+1}/{num_examples}")
            try:
                example = self.generate_cot_example(task_name, sample)
                pool.append(example)
            except Exception as e:
                print(f"生成示例 {i} 时出错: {e}")
                continue
        
        # 保存示例池
        pool_file = os.path.join(self.pool_dir, f"{task_name}_pool.json")
        with open(pool_file, 'w', encoding='utf-8') as f:
            json.dump(pool, f, ensure_ascii=False, indent=2)
        
        print(f"示例池已保存: {pool_file} (包含 {len(pool)} 个示例)")
        return pool_file
    
    def load_pool(self, task_name: str) -> List[Dict]:
        """加载特定任务的示例池"""
        pool_file = os.path.join(self.pool_dir, f"{task_name}_pool.json")
        
        if not os.path.exists(pool_file):
            print(f"示例池文件不存在: {pool_file}")
            return []
        
        try:
            with open(pool_file, 'r', encoding='utf-8') as f:
                pool = json.load(f)
            print(f"已加载 {task_name} 示例池: {len(pool)} 个示例")
            return pool
        except Exception as e:
            print(f"加载示例池失败: {e}")
            return []
    
    def get_shots(self, task_name: str, current_question: str = "", k: int = 1, 
                 strategy: str = "random") -> List[Dict]:
        """从示例池中抽取 k 个示例"""
        if task_name not in self.pools:
            self.pools[task_name] = self.load_pool(task_name)
        
        pool = self.pools[task_name]
        
        if len(pool) < k:
            print(f"警告: 示例池只有 {len(pool)} 个示例，但请求了 {k} 个")
            k = len(pool)
        
        if strategy == "random":
            # 随机选择
            shots = random.sample(pool, k)
        elif strategy == "similarity":
            # 简单的基于关键词的相似性选择
            shots = self._select_by_similarity(pool, current_question, k)
        else:
            # 默认随机选择
            shots = random.sample(pool, k)
        
        return shots
    
    def _select_by_similarity(self, pool: List[Dict], question: str, k: int) -> List[Dict]:
        """基于问题相似性选择示例（简单实现）"""
        if not question:
            return random.sample(pool, k)
        
        # 简单的关键词匹配
        question_words = set(question.lower().split())
        scored_examples = []
        
        for example in pool:
            example_question = example["question"].lower()
            example_words = set(example_question.split())
            
            # 计算 Jaccard 相似度
            intersection = len(question_words.intersection(example_words))
            union = len(question_words.union(example_words))
            similarity = intersection / union if union > 0 else 0
            
            scored_examples.append((similarity, example))
        
        # 按相似度排序并选择前 k 个
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [example for _, example in scored_examples[:k]]
    
    def get_pool_stats(self, task_name: str) -> Dict:
        """获取示例池统计信息"""
        pool = self.load_pool(task_name)
        
        if not pool:
            return {"error": "池为空或加载失败"}
        
        # 计算平均 CoT 长度
        cot_lengths = [len(example["cot_reasoning"]) for example in pool]
        
        return {
            "task": task_name,
            "total_examples": len(pool),
            "avg_cot_length": sum(cot_lengths) / len(cot_lengths),
            "min_cot_length": min(cot_lengths),
            "max_cot_length": max(cot_lengths)
        }
    def format_few_shot_prompt(self, task_name: str, shots: List[Dict], 
                          current_table: str, current_question: str) -> str:
        """构建 few-shot prompt"""
        
        # 构建示例部分
        examples_text = ""
        for i, shot in enumerate(shots, 1):
            example_table = format_table(shot["table"])
            examples_text += f"\n示例{i}：\n"
            examples_text += f"表格：{example_table}\n"
            examples_text += f"问题：{shot['question']}\n"
            examples_text += f"推理：{shot['cot_reasoning']}\n" 
            examples_text += f"答案：{shot['answer']}\n"
        
        # 根据任务类型构建不同的提示词
        if task_name == "wikitableqa":
            prompt = f"""请参考以下示例来回答表格问题：{examples_text}

    现在请回答当前问题：

    表格：{format_table(current_table)}

    问题：{current_question}

    请先推理再给出答案："""
        
        elif task_name == "tabfact":
            prompt = f"""请参考以下示例来判断表格陈述：{examples_text}

    现在请判断当前陈述：

    表格：{format_table(current_table)}

    陈述：{current_question}

    请先推理再判断是 Entailed 还是 Refuted："""
        
        elif task_name == "fetaqa":
            prompt = f"""请参考以下示例来回答表格问题：{examples_text}

    现在请回答当前问题：

    表格：{format_table(current_table)}

    问题：{current_question}

    请先推理再给出详细答案："""
        
        else:
            # 默认回退到 zero-shot
            config = TASK_CONFIGS[task_name]
            prompt = config["prompt_template"].format(
                table=format_table(current_table),
                question=current_question
            )
        
        return prompt


# 全局实例
example_pool_manager = ExamplePool()