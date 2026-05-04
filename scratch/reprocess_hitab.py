import json
import os
import sys

# Ensure we can import configs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import extract_hitab_final_answer

def lenient_match(pred, ref):
    if str(pred).strip() == str(ref).strip(): return True
    import ast
    try:
        p_val = ast.literal_eval(pred)
        r_val = ast.literal_eval(ref)
        if p_val == r_val: return True
    except:
        pass
    
    try:
        p_list = ast.literal_eval(pred) if str(pred).startswith('[') else [pred]
        r_list = ast.literal_eval(ref) if str(ref).startswith('[') else [ref]
        if len(p_list) == len(r_list):
            match = True
            for p, r in zip(p_list, r_list):
                if float(p) != float(r):
                    match = False
            if match: return True
    except:
        pass
    return False

def reprocess_file(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, dict):
        predictions = data.get('predictions', [])
    else:
        predictions = data
        
    correct = 0
    total = len(predictions)
    
    for item in predictions:
        pred_text = item.get('prediction', '')
        ref_text = item.get('reference', '')
        
        # Apply the fixed extraction logic
        if pred_text:
            new_processed = extract_hitab_final_answer(pred_text, ref_text)
        else:
            new_processed = ""
            
        item['processed_prediction'] = new_processed
        
        # Check correctness
        if lenient_match(new_processed, ref_text):
            correct += 1

    print(f"{os.path.basename(file_path)} Score: {correct}/{total} = {correct/total:.4f}")
    
    # Save back
    new_data = {"predictions": predictions} if isinstance(data, dict) else predictions
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
        
    print(f"Successfully reprocessed {file_path}")

if __name__ == "__main__":
    base_dir = '/Users/yebohou/zju/SRTP/DTC/DistillTableCoT/outputs/hitab'
    for ver in ['v1', 'v2', 'v4']:
        reprocess_file(os.path.join(base_dir, f'predictions_{ver}.json'))
