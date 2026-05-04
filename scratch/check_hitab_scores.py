import json
import ast

def lenient_match(pred, ref):
    if pred == ref: return True
    # Try ast eval
    try:
        p_val = ast.literal_eval(pred)
        r_val = ast.literal_eval(ref)
        if p_val == r_val: return True
    except:
        pass
    
    # Try float parsing
    try:
        p_list = ast.literal_eval(pred) if pred.startswith('[') else [pred]
        r_list = ast.literal_eval(ref) if ref.startswith('[') else [ref]
        if len(p_list) == len(r_list):
            match = True
            for p, r in zip(p_list, r_list):
                if float(p) != float(r):
                    match = False
            if match: return True
    except:
        pass
    return False

def check_hitab():
    for ver in ['v1', 'v2', 'v4']:
        path = f"/Users/yebohou/zju/SRTP/DTC/DistillTableCoT/outputs/hitab/predictions_{ver}.json"
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data = data.get('predictions', [])
            
            correct = 0
            for item in data:
                pred = item.get('processed_prediction', '')
                ref = item.get('reference', '')
                if lenient_match(pred, ref):
                    correct += 1
            print(f"{ver} Lenient Accuracy: {correct}/{len(data)} = {correct/len(data) if data else 0:.4f}")
        except Exception as e:
            print(f"Error {ver}: {e}")

if __name__ == "__main__":
    check_hitab()
