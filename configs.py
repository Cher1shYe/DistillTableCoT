# configs.py
import nltk
import re
import ast
# 确保 NLTK 的 punkt 分词器已下载
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def extract_wiki_final_answer(prediction_text):
    """
    该函数主要通过查找模型输出的特殊语句来清洗结果，此方法对大模型输出具有局限性
    wikitableqa评价指标是EM(exact_match)，大模型输出具有多样性，难以将数据全部清洗出来
    该函数已经基本考虑所有输出情况，但测试100个样本发现模型仍可能会输出特殊答案，清洗效率约98%
    """
    # 清理函数，用于最后净化答案
    def clean_answer(answer_str):
        if not answer_str:
            return ""
        # 1. 移除 LaTeX 相关的剩余符号
        answer_str = re.sub(r'\\text\{|\}|\\boxed\{', '', answer_str)
        # 移除所有 Markdown 标记 (*, _, `)
        answer_str = re.sub(r'[*_`]', '', answer_str)
        # 移除首尾的非字母数字字符 (比如冒号、句号、空格、换行符)

        # WikiTableQA 答案包含逗号（如列表），绝对不能按逗号 split。
        # 通常只需要移除括号内的备注，例如 "100 (estimated)" -> "100"
        parts = re.split(r'\s*\(', answer_str, 1) 
        answer_str = parts[0]

        answer_str = re.sub(r'^[^\w\d]+|[^\w\d]+$', '', answer_str)

        # 5. 移除首尾标点和空白
        # 移除句尾的句号，但要小心缩写（不过 WikiTableQA 主要是短语，移除末尾句号通常是安全的）
        answer_str = answer_str.strip()
        if answer_str.endswith('.'):
            answer_str = answer_str[:-1]
            
        return answer_str.strip()

    text = prediction_text.strip()

    # 首先查找 "Final Answer:" (Agent 模式最常用)
    final_answer_match = re.search(r'Final Answer:\s*(.*)', text, re.IGNORECASE)
    if final_answer_match:
        return clean_answer(final_answer_match.group(1))

    # 其次查找所有 "Answer:" 的匹配
    answer_matches = list(re.finditer(r'\bAnswer:\s*(.+?)(?:\.|\n|$)', text, re.IGNORECASE | re.DOTALL))
    if answer_matches:
        # 如果找到了，取最后一个匹配项
        last_match = answer_matches[-1]
        answer = last_match.group(1)
        return clean_answer(answer)

    # 尝试用正则表达式匹配常见的答案指示词
    match = re.search(
        r'(?:the final answer is|the answer is|so the answer is)\**\s*(.+)',
        text,
        re.IGNORECASE | re.DOTALL
    )
    
    if match:
        answer = match.group(1)
        return clean_answer(answer)

    # 2. 阅读prediction结果发现可能出现无指示词情况，此时直接对** **中的结果进行提取即可
    bold_matches = re.findall(r'\*\*(.*?)\*\*', text)
    if len(bold_matches) == 1:
        return clean_answer(bold_matches[0])
    # 3. 提取 LaTeX \boxed{...} (R1 模型最爱用) ---
    boxed_matches = re.findall(r'\\boxed\{(.*?)\}', text, re.DOTALL)
    if boxed_matches:
        return clean_answer(boxed_matches[-1])
    # 4. 如果以上匹配都失败，尝试一个更简单的回退逻辑：取最后一个冒号 ":" 后面的内容。
    if ":" in text:
        parts = text.rsplit(":", 1)
        if len(parts) > 1 and parts[1].strip():
            return clean_answer(parts[1])
    
    # 5. 如果还是失败，取最后一行非空文本
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    if lines:
        last_line = lines[-1] 
        return clean_answer(last_line)

    # 6. 如果以上全部失败，返回原始文本的清理版
    return clean_answer(text)
def extract_fact_final_answer(prediction_text):
    """
    简单逻辑就是从最后一句提取答案直接找到Refuted或者Entailed字符串
    """
    text = prediction_text.strip()
    # 将文本分割成句子
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        # 如果分句失败，就按换行符分割
        sentences = text.split('\n')
    # 从后往前遍历句子
    for sentence in reversed(sentences):
        lower_sentence = sentence.lower()
        is_entailed = 'entailed' in lower_sentence
        is_refuted = 'refuted' in lower_sentence
        
        # 如果一句话里同时包含或都不包含，则跳过，因为它可能不是明确的结论句
        if is_entailed and is_refuted:
            continue
        if not is_entailed and not is_refuted:
            continue
            
        # 找到了一个只包含其中一个关键词的句子，这很可能就是结论
        if is_refuted:
            return 'Refuted'
        if is_entailed:
            return 'Entailed'
    #    如果句子分析失败，回退到查找最后一个出现的关键词
    #    找到 'refuted' 和 'entailed' 在文本中最后出现的位置
    last_entailed_pos = text.lower().rfind('entailed')
    last_refuted_pos = text.lower().rfind('refuted')
        # 比较哪个词最后出现
    if last_refuted_pos > last_entailed_pos:
        return 'Refuted'
    if last_entailed_pos > last_refuted_pos:
        return 'Entailed'
    
    # 如果只有一个词出现过（另一个是-1），也能正确处理
    if last_refuted_pos == -1:
        return 'Entailed'
    if last_entailed_pos == -1:
        return 'Refuted'
    
    # 3. 如果两个关键词都完全没出现，返回无法判断
    return 'N/A'

def extract_fetaqa_final_answer(prediction_text):
    """
    专门从 FETAQA 的输出中提取 "Final Answer:" 后面的内容。
    """
    if not prediction_text:
        return ""
    text = re.sub(r'\*\*', '', prediction_text)
    # 优先寻找 "Final Answer:" 标记
    # re.IGNORECASE: 忽略大小写, re.DOTALL: . 匹配换行
    stop_pattern = r'(?:\n\s*\n|Final\s*Answer:|Answer:|Check:|---|$)'
    
    pattern = r'Final Answer:\s*([\s\S]*?)(?=' + stop_pattern + r')'
    # 搜索匹配
    matches = re.findall(pattern, text, re.IGNORECASE)

    if matches:
        # FETAQA 的特点是长句回答。
        # 在复读机模式下，通常会有多个匹配项。
        # 我们优先取第一个非空的、长度合理的匹配项。
        # 往往第一个 Final Answer 是最准确的，后面的可能是幻觉循环。
        for match in matches:
            cleaned_candidate = match.strip()
            # 简单的过滤器：答案至少应该有几个字符，且不纯是标点
            if len(cleaned_candidate) > 1:
                # 再次清理可能残留的 markdown 符号
                return re.sub(r'[*_`]', '', cleaned_candidate).strip()

    # 如果没找到 "Final Answer:" 标记（或者只有空的），
    # 尝试回退策略：直接取最后一行（适用于某些只有结果没有标记的情况）
    lines = text.strip().split('\n')
    if lines:
        return re.sub(r'[*_`]', '', lines[-1]).strip()

        
    return ""

def extract_hitab_final_answer(prediction_text, reference_label):
    """
    专门为 HiTab 定制的结果提取：
    1. 拿取“最后一个” answer 后的内容，解决多重前缀嵌套问题。
    2. 探测 Reference 格式，智能处理单字符串大小写、多字符串列表分割。
    3. 修复小数点截断。
    4. 智能识别百分比，自动进行 /100 转换。
    """
    text = prediction_text.strip()
    
    # === 1. 安全提取大模型的原始回答 (获取最后一个 answer 之后的内容) ===
    # 使用 split 切割所有的 trigger 词，确保我们只拿最后一段，完美避开 " \n Answer:" 这种嵌套
    pattern = re.compile(
        r'(?:the final answer is|the answer is|so the answer is|final answer:|answer:)\**\s*', 
        re.IGNORECASE
    )
    splits = pattern.split(text)
    
    if len(splits) > 1:
        ans_raw = splits[-1].strip()
    else:
        # 兜底：如果没有 trigger 词，取最后一行
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        ans_raw = lines[-1] if lines else text.strip()

    # 去除 Markdown 和句尾的句号（保留中间的小数点）
    ans_raw = re.sub(r'[*_`]', '', ans_raw)
    if ans_raw.endswith('.'):
        ans_raw = ans_raw[:-1]
    ans_raw = ans_raw.strip()

    # === 2. 智能迎合 Reference 的格式 ===
    ref_str = str(reference_label).strip()
    
    if ref_str.startswith('[') and ref_str.endswith(']'):
        try:
            ref_list = ast.literal_eval(ref_str)
        except:
            ref_list = None
            
        if isinstance(ref_list, list) and len(ref_list) > 0:
            ref_val_first = ref_list[0]
            
            # --- 场景 A: 答案是字符串 (对应你的 Case 1 和 Case 2) ---
            if isinstance(ref_val_first, str):
                ans_clean = ans_raw.lower()
                
                # 多字符串列表：如 ['ethiopia', 'somalia', ...]
                if len(ref_list) > 1:
                    # 兼容逗号、分号或换行符分割，并去除两端空白
                    parts = [p.strip() for p in re.split(r',|\n|;', ans_clean) if p.strip()]
                    # 清理部分大模型喜欢在最后一个词前加的 "and " 
                    parts = [re.sub(r'^and\s+', '', p) for p in parts]
                    return str(parts)
                
                # 单字符串：如 ['flora']
                else:
                    return f"['{ans_clean}']"
                    
            # --- 场景 B: 答案是数字类型 ---
            elif isinstance(ref_val_first, (int, float)):
                # 抹除千位分隔符逗号
                ans_raw_clean = re.sub(r'(?<=\d),(?=\d)', '', ans_raw)
                # 提取出文本里的所有数字（含小数和负数）
                numbers_str = re.findall(r'-?\d+(?:\.\d+)?', ans_raw_clean)
                
                if numbers_str:
                    formatted_nums = []
                    for i, num_str in enumerate(numbers_str):
                        val = float(num_str) if '.' in num_str else int(num_str)
                        
                        # 百分比转换逻辑
                        if '%' in ans_raw and isinstance(ref_val_first, float) and abs(ref_val_first) <= 1.0:
                            val = float(num_str) / 100.0
                            
                        # 补全 .0 逻辑
                        elif i < len(ref_list) and isinstance(ref_list[i], float) and str(ref_list[i]).endswith('.0') and '.' not in num_str:
                            val = float(num_str)
                            
                        formatted_nums.append(val)
                    
                    # 截取跟 reference 相同数量的数字
                    formatted_nums = formatted_nums[:len(ref_list)]
                    return str(formatted_nums)
                    
    # 如果 Reference 解析失败，兜底加上方括号返回
    return f"[{ans_raw}]" if ref_str.startswith('[') else ans_raw.strip()

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

# 多次SQL查询仍返回空/失败时使用的简单 CoT 回退提示
COT_SYSTEM_PROMPT = "You read tables and answer questions. Think step-by-step then output exactly 'Final Answer: <answer>'."

# 初版提示词和configs
# TASK_CONFIGS = {
#     "wikitableqa": {
#         "dataset_name": "table-benchmark/wikiqa",
#         "dataset_split": "train",
#         "prompt_template": "Read the table below, and answer it with your reasoning. After your reasoning, give a precise answer with:'Answer:' prefix.\n\nTable:\n{table}\n\nQuestion: {question}\nAnswer:",
#         "input_fields": ["question", "table"],
#         "target_field": "answer",
#         "metrics": ["exact_match"],
#         "postprocess_func": lambda pred, label: (
#             extract_wiki_final_answer(pred), 
#             label.strip()
#         ),
#     },
#     "tabfact": {
#         "dataset_name": "table-benchmark/tabfact",
#         "dataset_split": "train",
#         "prompt_template": "Read the table below and determine if the statement is entailed or refuted.\n\nTable:\n{table}\n\nStatement: {question}\nIs the statement entailed or refuted? Answer with your reasoning, and state whether the content is correct or incorrect with only Entailed or Refuted.\nAnswer:",
#         "input_fields": ["question", "table"],
#         "target_field": "answer",
#         "metrics": ["accuracy"],
#         "postprocess_func": lambda pred, label: (
#             extract_fact_final_answer(pred), 
#             label.strip()
#         ),
#     },
#     "fetaqa": {
#         "dataset_name": "table-benchmark/fetaqa",
#         "dataset_split": "train",
#         "prompt_template": "Read the table below and provide a detailed, free-form answer to the question.First, think step by step to lay out your reasoning. After your reasoning, use one sentence to provide a final, concise answer prefixed with 'Final Answer:'.\n\nTable:\n{table}\n\nQuestion: {question}\nDetailed Answer:",
#         "input_fields": ["question", "table", "table_title"],
#         "target_field": "answer",
#         "metrics": ["rouge", "sacrebleu"],
#         "postprocess_func": lambda pred, label: (
#             extract_fetaqa_final_answer(pred), 
#             label.strip()
#         ),
#     },
#     "hitab": {
#         "dataset_name": "kasnerz/hitab",
#         "dataset_split": "train",
#         "prompt_template": "Read the table below, and answer it with your reasoning. After your reasoning, give a precise answer with:'Answer:' prefix.\n\nTable:\n{table}\n\nQuestion: {question}\nAnswer:",
#         "input_fields": ["question", "table"],
#         "target_field": "answer",        
#         "metrics": ["exact_match"],
#         # 【修改这里】：传入 extract_hitab_final_answer，并确保同时传递 pred 和 label
#         "postprocess_func": lambda pred, label: (
#             extract_hitab_final_answer(pred, label), 
#             label.strip()
#         ),
#     }
# }
TASK_CONFIGS = {
    "wikitableqa": {
        "dataset_name": "table-benchmark/wikiqa",
        "dataset_split": "train",
        "system_prompt": AGENT_SYSTEM_PROMPT + "\nTask: Final answer is a precise entity, number, or short phrase.",
        "user_prompt_template": "Table:\n{table}\n\nQuestion: {question}\n\nOutput 'Final Answer: <precise answer>'.",
        "cot_user_prompt_template": "Table:\n{table}\n\nQuestion: {question}\n\nAnswer with a precise entity, number, or short phrase. End with 'Final Answer: <answer>'.",
        "input_fields": ["question", "table"],
        "target_field": "answer",
        "metrics": ["exact_match"],
        "postprocess_func": lambda pred, label: (
            extract_wiki_final_answer(pred) if pred else "",
            label.strip()
        ),
    },
    "tabfact": {
        "dataset_name": "table-benchmark/tabfact",
        "dataset_split": "train",
        "system_prompt": AGENT_SYSTEM_PROMPT + "\nTask: Verify whether the statement is Entailed or Refuted by the table.",
        "user_prompt_template": "Table:\n{table}\n\nStatement: {question}\n\nOutput exactly 'Final Answer: Entailed' or 'Final Answer: Refuted'.",
        "cot_user_prompt_template": "Table:\n{table}\n\nStatement: {question}\n\nDecide if the statement is supported. End with exactly 'Final Answer: Entailed' or 'Final Answer: Refuted'.",
        "input_fields": ["question", "table"],
        "target_field": "answer",
        "metrics": ["accuracy"],
        "postprocess_func": lambda pred, label: (
            extract_fact_final_answer(pred) if pred else "",
            label.strip()
        ),
    },
    "fetaqa": {
        "dataset_name": "table-benchmark/fetaqa",
        "dataset_split": "train",
        "system_prompt": AGENT_SYSTEM_PROMPT + "\nTask: Final answer is one coherent sentence based on the query result.",
        "user_prompt_template": "Table:\n{table}\n\nQuestion: {question}\n\nOutput 'Final Answer: <one-sentence answer>'.",
        "cot_user_prompt_template": "Table:\n{table}\n\nQuestion: {question}\n\nAnswer in one sentence. End with 'Final Answer: <sentence>'.",
        "input_fields": ["question", "table", "table_title"],
        "target_field": "answer",
        "metrics": ["rouge", "sacrebleu"],
        "postprocess_func": lambda pred, label: (
            extract_fetaqa_final_answer(pred) if pred else "",
            label.strip()
        ),
    },
    "hitab": {
        "dataset_name": "kasnerz/hitab",
        "dataset_split": "train",
        "system_prompt": AGENT_SYSTEM_PROMPT + "\nTask: Final answer is a precise number, entity, or short phrase.",
        "user_prompt_template": "Table:\n{table}\n\nQuestion: {question}\n\nOutput 'Final Answer: <precise answer>'.",
        "cot_user_prompt_template": "Table:\n{table}\n\nQuestion: {question}\n\nAnswer with a precise number, entity, or short phrase. End with 'Final Answer: <answer>'.",
        "input_fields": ["question", "table"],
        "target_field": "answer",
        "metrics": ["exact_match"],
        "postprocess_func": lambda pred, label: (
            extract_hitab_final_answer(pred, label) if pred else "",
            label.strip()
        ),
    }
}


# 初版给小模型用的configs
TASK_TEST_CONFIGS = {
    "wikitableqa": {
        "dataset_name": "table-benchmark/wikiqa",
        "dataset_split": "train",
        "prompt_template": "Read the table below, and answer it with your reasoning. After your reasoning, give a precise answer with:'Answer:' prefix.\n\nTable:\n{table}\n\nQuestion: {question}",
        "input_fields": ["question", "table"],
        "target_field": "answer",
        "metrics": ["exact_match"],
        "postprocess_func": lambda pred, label: (
            extract_wiki_final_answer(pred), 
            label.strip()
        ),
    },
    "tabfact": {
        "dataset_name": "table-benchmark/tabfact",
        "dataset_split": "train",
        "prompt_template": "Read the table below and determine if the statement is entailed or refuted.\n\nTable:\n{table}\n\nStatement: {question}\nIs the statement entailed or refuted? Answer with your reasoning, and state whether the content is correct or incorrect with only Entailed or Refuted.",
        "input_fields": ["question", "table"],
        "target_field": "answer",
        "metrics": ["accuracy"],
        "postprocess_func": lambda pred, label: (
            extract_fact_final_answer(pred), 
            label.strip()
        ),
    },
    "fetaqa": {
        "dataset_name": "table-benchmark/fetaqa",
        "dataset_split": "train",
        "prompt_template": "Read the table below and provide a detailed, free-form answer to the question.First, think step by step to lay out your reasoning. After your reasoning, use one sentence to provide a final, concise answer prefixed with 'Final Answer:'.\n\nTable:\n{table}\n\nQuestion: {question}",
        "input_fields": ["question", "table", "table_title"],
        "target_field": "answer",
        "metrics": ["rouge", "sacrebleu"],
        "postprocess_func": lambda pred, label: (
            extract_fetaqa_final_answer(pred), 
            label.strip()
        ),
    },
        "hitab": {
        "dataset_name": "kasnerz/hitab",  # HF上最通用的 HiTab 源
        "dataset_split": "train",         # 下载训练集来看看
        "prompt_template": "Read the table below, and answer it with your reasoning. After your reasoning, give a precise answer with:'Answer:' prefix.\n\nTable:\n{table}\n\nQuestion: {question}",
        "input_fields": ["question", "table"],
        "target_field": "answer",         
        "metrics": ["exact_match"],
        "postprocess_func": lambda pred, label: (
            extract_hitab_final_answer(pred, label), 
            label.strip()
        ),
    }
}