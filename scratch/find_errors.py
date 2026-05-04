import json
import os

def find_errors(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if isinstance(data, dict) and 'predictions' in data:
        predictions = data['predictions']
    else:
        predictions = data
        
    error_cases = []
    
    for item in predictions:
        pred = str(item.get('processed_prediction', '')).strip()
        ref = str(item.get('reference', '')).strip("[]'\"")
        
        # Simple exact match logic similar to run_evaluate.py
        if pred != ref:
            error_cases.append({
                'id': item.get('id'),
                'question': item.get('prompt', '').split('Question:')[-1].strip(),
                'pred': pred,
                'ref': ref,
                'raw_prediction': item.get('prediction', '')
            })
            
    print(f"Total evaluated: {len(predictions)}")
    print(f"Total errors found: {len(error_cases)}")
    print("-" * 50)
    
    # Print the first 10 errors for analysis
    for err in error_cases[:10]:
        print(f"ID: {err['id']}")
        print(f"Question: {err['question']}")
        print(f"Prediction: {err['pred']}")
        print(f"Reference: {err['ref']}")
        print(f"Raw Model Output (Snippet): {err['raw_prediction'][:200]}...")
        print("-" * 50)

if __name__ == "__main__":
    file_path = "/Users/yebohou/zju/SRTP/DTC/DistillTableCoT/outputs/wikitableqa/predictions_v1.json"
    find_errors(file_path)
