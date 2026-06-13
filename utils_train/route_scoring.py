"""
route-aware 评估的逐样本判分，口径与 scripts/batch_eval_qwen3.py 完全一致：
  1. 用 configs.TASK_CONFIGS[task]["postprocess_func"] 把原始答案清洗成
     processed_prediction (hitab 做数字归一化/补 .0，wikitableqa/fetaqa 各自清洗)；
  2. 按任务套用 batch_eval 的匹配逻辑：
       hitab        : ast.literal_eval 后逐元素 float 比较
       wikitableqa  : normalize_string 后成员 / 拼接 / 子串匹配
       tabfact      : Entailed/Refuted 映射成 0/1 比较
       fetaqa       : 逐样本 ROUGE-L >= 阈值 (batch 用语料级 ROUGE，这里取逐样本近似)

这样 route-SFT 学生的数字才与历史 baseline (batch_eval) 同口径可比。
"""
import ast
import re
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from configs import TASK_CONFIGS

FETAQA_ROUGE_THRESHOLD = 0.3
_rouge_scorer = None


def _normalize_string(s):
    """与 batch_eval 内联的 normalize_string 逐字对齐 (wikitableqa 用)。"""
    if not s:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r'(?<=\d),(?=\d)', '', s)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s-]', ' ', s)
    if s.endswith('.'):
        s = s[:-1]
    return " ".join(s.split())


def _match_hitab(p, r):
    p_str, r_str = str(p).strip(), str(r).strip()
    if p_str == r_str:
        return True
    try:
        p_val = ast.literal_eval(p_str)
        r_val = ast.literal_eval(r_str)
        if p_val == r_val:
            return True
        p_list = p_val if isinstance(p_val, list) else [p_val]
        r_list = r_val if isinstance(r_val, list) else [r_val]
        if len(p_list) == len(r_list):
            return all(float(a) == float(b) for a, b in zip(p_list, r_list))
    except Exception:
        pass
    return False


def _match_wiki(p, r):
    p_norm = _normalize_string(p)
    if isinstance(r, str) and r.startswith('[') and r.endswith(']'):
        try:
            r_list = ast.literal_eval(r)
        except Exception:
            r_list = [r.strip("[]'\"")]
    elif isinstance(r, list):
        r_list = r
    else:
        r_list = [str(r)]
    r_norms = [_normalize_string(x) for x in r_list]
    if p_norm in r_norms:
        return True
    if p_norm == ", ".join(r_norms) or p_norm == ",".join(r_norms):
        return True
    if len(r_norms) > 1 and all(ref in p_norm for ref in r_norms):
        return True
    return False


def _match_tabfact(p, r):
    return (1 if str(p).lower() == 'entailed' else 0) == \
           (1 if str(r).lower() == 'entailed' else 0)


def _match_fetaqa(p, r):
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer
        _rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = _rouge_scorer.score(str(r), str(p))['rougeL'].fmeasure
    return score >= FETAQA_ROUGE_THRESHOLD


_MATCHERS = {
    "hitab": _match_hitab,
    "wikitableqa": _match_wiki,
    "tabfact": _match_tabfact,
    "fetaqa": _match_fetaqa,
}


def match_processed(task, processed_pred, reference):
    """对 **已 postprocess** 的预测判对错 (baseline 的 processed_prediction 用)。"""
    if task not in TASK_CONFIGS:
        return str(processed_pred).strip().lower() == str(reference).strip().lower()
    return bool(_MATCHERS[task](processed_pred, str(reference).strip()))


def score_answer(task, raw_answer, reference):
    """对 **原始** 答案判对错，与 batch_eval 同口径。raw_answer = 学生 <ANSWER> 文本。

    先走 configs 的 postprocess_func 清洗，再套任务匹配逻辑。
    """
    if task not in TASK_CONFIGS:
        return str(raw_answer).strip().lower() == str(reference).strip().lower()
    processed_pred, processed_label = TASK_CONFIGS[task]["postprocess_func"](
        str(raw_answer or ""), str(reference))
    return bool(_MATCHERS[task](processed_pred, processed_label))
