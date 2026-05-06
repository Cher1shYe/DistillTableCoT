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

def is_match(pred, ref):
    if pred is None or ref is None:
        return False
    
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
