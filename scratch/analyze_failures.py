import json

def check_failures(file_path):
    with open(file_path, 'r') as f:
        full_data = json.load(f)
    
    data = full_data.get('predictions', [])
    if not data and isinstance(full_data, list):
        data = full_data
    
    import re
    import ast

    def normalize_string(s):
        if not s: return ""
        s = str(s).lower().strip()
        s = re.sub(r'(?<=\d),(?=\d)', '', s)
        if s.endswith('.'): s = s[:-1]
        return s.strip()

    failures = []
    for item in data:
        pred = item.get('processed_prediction', '')
        ref = item.get('reference', '')
        
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
        
        is_correct = False
        if p_norm in r_norms:
            is_correct = True
        elif p_norm == ", ".join(r_norms) or p_norm == ",".join(r_norms):
            is_correct = True
        elif all(ref_n in p_norm for ref_n in r_norms) and len(r_norms) > 1:
            is_correct = True
        
        if not is_correct:
            failures.append({
                "id": item['id'],
                "question": item.get('prompt', '').split('Question: ')[-1] if 'prompt' in item else 'N/A',
                "prediction": item.get('prediction', ''),
                "processed_prediction": pred,
                "reference": ref,
                "mode": item.get('mode', '')
            })
            
    print(f"Total failures: {len(failures)} / {len(data)}")
    for f in failures[:5]:
        print(f"--- ID {f['id']} ({f['mode']}) ---")
        print(f"Q: {f['question']}")
        print(f"Pred: {f['processed_prediction']}")
        print(f"Ref: {f['reference']}")
        # print(f"Raw Prediction: {f['prediction']}")

if __name__ == "__main__":
    check_failures('outputs/wikitableqa/predictions_v5.json')
