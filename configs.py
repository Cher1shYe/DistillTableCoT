# configs.py
import nltk
import re
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
        # 移除所有 Markdown 标记 (*, _, `)
        answer_str = re.sub(r'[*_`]', '', answer_str)
        # 移除首尾的非字母数字字符 (比如冒号、句号、空格、换行符)
        answer_str = re.sub(r'^[^\w\d]+|[^\w\d]+$', '', answer_str)
        return answer_str.strip()
    
    text = prediction_text.strip()

    # 1. 尝试用正则表达式匹配常见的答案指示词
    #    - re.IGNORECASE: 忽略大小写
    #    - re.DOTALL: 让 . 能匹配换行符
    #    - (?:...): 非捕获组，只用于匹配，不作为结果
    #    - \s*: 匹配任意数量的空格
    #    - \**\s*: 匹配粗体标记和空格
    #    - (.+): 捕获我们想要的答案
    match = re.search(
        r'(?:the final answer is|the answer is|so the answer is|answer:)\**\s*(.+)',
        text,
        re.IGNORECASE | re.DOTALL
    )
    
    if match:
        # 如果匹配成功，获取第一个捕获组的内容并对数据进行清洗
        answer = match.group(1)
        return clean_answer(answer)

    # 阅读prediction结果发现可能出现无指示词情况，此时直接对** **中的结果进行提取即可
    bold_matches = re.findall(r'\*\*(.*?)\*\*', text)
    if len(bold_matches) == 1:
        return clean_answer(bold_matches[0])
    
    # 3. 如果以上匹配都失败，尝试一个更简单的回退逻辑：取最后一个冒号 ":" 后面的内容。
    if ":" in text:
        parts = text.rsplit(":", 1)
        if len(parts) > 1 and parts[1].strip():
            return clean_answer(parts[1])
    
    # 4. 如果还是失败，取最后一行非空文本
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    if lines:
        last_line = lines[-1] 
        return clean_answer(last_line)

    # 5. 如果以上全部失败，返回原始文本的清理版
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
    text = prediction_text.strip()
    
    # 优先寻找 "Final Answer:" 标记
    # re.IGNORECASE: 忽略大小写, re.DOTALL: . 匹配换行
    match = re.search(r'Final Answer:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
    
    if match:
        # 如果找到，返回标记后面的所有内容，并清理
        final_answer = match.group(1).strip()
        # 移除可能的 Markdown 标记
        return re.sub(r'[*_`]', '', final_answer)

    # 如果因为某种原因模型没有遵循指令，我们回退到之前的“温和清理”逻辑
    # (这里可以调用之前的 clean_freeform_answer 函数，或者直接返回原始文本的清理版)
    # 为了简单，我们直接返回清理过的原始文本
    return re.sub(r'[*_`]', '', text).strip()
TASK_CONFIGS = {
    "wikitableqa": {
        "dataset_name": "table-benchmark/wikiqa",
        "dataset_split": "train",
        "prompt_template": "Read the table below and answer the question.\n\nTable:\n{table}\n\nQuestion: {question}\nAnswer:",
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
        "prompt_template": "Read the table below and determine if the statement is entailed or refuted.\n\nTable:\n{table}\n\nStatement: {question}\nIs the statement entailed or refuted? Answer with your reasoning, and state whether the content is correct or incorrect with only Entailed or Refuted.\nAnswer:.\nAnswer:",
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
        "prompt_template": "Read the table below and provide a detailed, free-form answer to the question.First, think step by step to lay out your reasoning. After your reasoning, use one sentence to provide a final, concise answer prefixed with 'Final Answer:'.\n\nTable:\n{table}\n\nQuestion: {question}\nDetailed Answer:",
        "input_fields": ["question", "table", "table_title"],
        "target_field": "answer",
        "metrics": ["rouge", "sacrebleu"],
        "postprocess_func": lambda pred, label: (
            extract_fetaqa_final_answer(pred), 
            label.strip()
        ),
    }
}
