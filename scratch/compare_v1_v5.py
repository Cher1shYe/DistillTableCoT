import json
import re
import ast

def normalize_string(s):
    if not s: return ""
    s = str(s).lower().strip()
    s = re.sub(r'(?<=\d),(?=\d)', '', s)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s-]', ' ', s)
    if s.endswith('.'): s = s[:-1]
    return " ".join(s.split())

def is_match(pred, ref):
    p_norm = normalize_string(pred)
    if isinstance(ref, str) and ref.startswith('[') and ref.endswith(']'):
        try:
            r_list = ast.literal_eval(ref)
        except:
            r_list = [ref.strip("[]'\"")]
    elif isinstance(ref, list):
        r_list = ref
    else:
        r_list = [str(ref)]
    r_norms = [normalize_string(x) for x in r_list]
    if p_norm in r_norms: return True
    if p_norm == ", ".join(r_norms) or p_norm == ",".join(r_norms): return True
    if all(ref_n in p_norm for ref_n in r_norms) and len(r_norms) > 1: return True
    return False

def compare_files(file1, file2, limit=200):
    with open(file1, 'r') as f:
        data1_raw = json.load(f)
    with open(file2, 'r') as f:
        data2_raw = json.load(f)
        
    data1 = data1_raw.get('predictions', data1_raw)[:limit]
    data2 = data2_raw.get('predictions', data2_raw)[:limit]
    
    v1_correct_v5_wrong = []
    v5_correct_v1_wrong = []
    
    for i in range(min(len(data1), len(data2))):
        item1 = data1[i]
        item2 = data2[i]
        
        c1 = is_match(item1.get('processed_prediction', ''), item1.get('reference', ''))
        c2 = is_match(item2.get('processed_prediction', ''), item2.get('reference', ''))
        
        if c1 and not c2:
            v1_correct_v5_wrong.append(i)
        elif c2 and not c1:
            v5_correct_v1_wrong.append(i)
            
    print(f"V1 samples analyzed: {len(data1)}")
    print(f"V1 correct, V5 wrong: {len(v1_correct_v5_wrong)}")
    print(f"V5 correct, V1 wrong: {len(v5_correct_v1_wrong)}")
    
    print("\n--- Examples: V1 correct, V5 wrong ---")
    for idx in v1_correct_v5_wrong[:3]:
        it1 = data1[idx]
        it2 = data2[idx]
        print(f"ID {idx}")
        print(f"Q: {it1.get('prompt', '').split('Question: ')[-1].split('Answer:')[0].strip()}")
        print(f"Ref: {it1.get('reference')}")
        print(f"V1 Pred: {it1.get('processed_prediction')}")
        print(f"V5 Pred: {it2.get('processed_prediction')}")

    print("\n--- Examples: V5 correct, V1 wrong ---")
    for idx in v5_correct_v1_wrong[:3]:
        it1 = data1[idx]
        it2 = data2[idx]
        print(f"ID {idx}")
        print(f"Q: {it1.get('prompt', '').split('Question: ')[-1].split('Answer:')[0].strip()}")
        print(f"Ref: {it1.get('reference')}")
        print(f"V1 Pred: {it1.get('processed_prediction')}")
        print(f"V5 Pred: {it2.get('processed_prediction')}")

if __name__ == "__main__":
    compare_files('outputs/wikitableqa/predictions_v1.json', 'outputs/wikitableqa/predictions_v5.json')
