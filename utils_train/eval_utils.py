import re
import ast

def normalize_string(s):
    if not s: return ""
    s = str(s).lower().strip()
    # 移除数字中的逗号
    s = re.sub(r'(?<=\d),(?=\d)', '', s)
    # 移除冠词
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # 移除大部分标点（保留连字符和内部空格）
    s = re.sub(r'[^\w\s-]', ' ', s)
    # 移除末尾句号
    if s.endswith('.'): s = s[:-1]
    return " ".join(s.split())


def _lcs_length(x, y):
    """Token-level LCS length via DP."""
    m, n = len(x), len(y)
    # Use two rows to save memory
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l_f1(pred: str, ref: str) -> float:
    """Compute token-level ROUGE-L F1 after normalisation."""
    p_toks = normalize_string(pred).split()
    r_toks = normalize_string(ref).split()
    if not p_toks or not r_toks:
        return 0.0
    lcs = _lcs_length(p_toks, r_toks)
    precision = lcs / len(p_toks)
    recall = lcs / len(r_toks)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def is_match(pred, ref, rouge_threshold: float = None):
    """Return True if pred matches ref.

    For free-form tasks (e.g. FeTaQA) pass rouge_threshold (e.g. 0.3) to use
    ROUGE-L F1 instead of exact matching.
    """
    if pred is None or ref is None:
        return False

    if rouge_threshold is not None:
        return rouge_l_f1(str(pred), str(ref)) >= rouge_threshold

    p_norm = normalize_string(str(pred))

    # 处理 WikiTableQA 的列表格式参考答案
    ref_str = str(ref)
    if ref_str.startswith('[') and ref_str.endswith(']'):
        try:
            r_list = ast.literal_eval(ref_str)
        except:
            r_list = [ref_str.strip("[]'\"")]
    elif isinstance(ref, list):
        r_list = ref
    else:
        r_list = [ref_str]

    r_norms = [normalize_string(str(x)) for x in r_list]

    # 1. 完全匹配（其中一个即可）
    if p_norm in r_norms:
        return True

    # 2. 预测包含了所有参考项（针对连接后的列表）
    if p_norm == ", ".join(r_norms) or p_norm == ",".join(r_norms):
        return True

    # 3. 针对 WikiTableQA：如果参考答案是列表，且预测命中其中之一
    if any(p_norm == rn for rn in r_norms):
        return True

    # 4. 宽松匹配：如果预测包含所有参考项且参考项多于1个
    if all(ref_n in p_norm for ref_n in r_norms) and len(r_norms) > 1:
        return True

    return False
