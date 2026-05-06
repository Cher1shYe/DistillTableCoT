import json
import re

def compare_tabfact(file1, file2, limit=200):
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
        
        p1 = str(item1.get('processed_prediction', '')).lower().strip()
        r1 = str(item1.get('reference', '')).lower().strip()
        p2 = str(item2.get('processed_prediction', '')).lower().strip()
        r2 = str(item2.get('reference', '')).lower().strip()
        
        c1 = p1 == r1
        c2 = p2 == r2
        
        if c1 and not c2:
            v1_correct_v5_wrong.append(i)
        elif c2 and not c1:
            v5_correct_v1_wrong.append(i)
            
    print(f"TabFact Samples analyzed: {len(data2)}")
    print(f"V1 correct, V5 wrong: {len(v1_correct_v5_wrong)}")
    print(f"V5 correct, V1 wrong: {len(v5_correct_v1_wrong)}")
    
    print("\n--- Examples: V1 correct, V5 wrong ---")
    for idx in v1_correct_v5_wrong[:3]:
        it1 = data1[idx]
        it2 = data2[idx]
        print(f"ID {idx}")
        print(f"Statement: {it1.get('prompt', '').split('Statement: ')[-1].split('Is the statement')[0].strip()}")
        print(f"Ref: {it1.get('reference')}")
        print(f"V1 Pred: {it1.get('processed_prediction')}")
        print(f"V5 Pred: {it2.get('processed_prediction')}")
        # print(f"V5 Prediction full: {it2.get('prediction')}")

if __name__ == "__main__":
    compare_tabfact('outputs/tabfact/predictions_v1.json', 'outputs/tabfact/predictions_v5.json')
