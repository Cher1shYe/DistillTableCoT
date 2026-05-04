import json

def analyze_hitab_v4():
    with open('/Users/yebohou/zju/SRTP/DTC/DistillTableCoT/outputs/hitab/predictions_v1.json', 'r') as f:
        v1_data = json.load(f)['predictions']
    with open('/Users/yebohou/zju/SRTP/DTC/DistillTableCoT/outputs/hitab/predictions_v4.json', 'r') as f:
        v4_data = json.load(f)
        
    v1_dict = {item['id']: item for item in v1_data}
    v4_dict = {item['id']: item for item in v4_data}
    
    from check_hitab_scores import lenient_match
    
    v1_correct_v4_wrong = []
    
    for _id, v4_item in v4_dict.items():
        if _id not in v1_dict: continue
        v1_item = v1_dict[_id]
        
        v1_pred = str(v1_item.get('processed_prediction'))
        v4_pred = str(v4_item.get('processed_prediction'))
        ref = str(v4_item.get('reference'))
        
        v1_correct = lenient_match(v1_pred, ref)
        v4_correct = lenient_match(v4_pred, ref)
        
        if v1_correct and not v4_correct:
            v1_correct_v4_wrong.append((v1_item, v4_item))
            
    print(f"Total V1 correct but V4 wrong (out of {len(v4_dict)}): {len(v1_correct_v4_wrong)}")
    
    # Print a couple of examples
    for v1_item, v4_item in v1_correct_v4_wrong[:3]:
        print("="*50)
        print("ID:", v1_item['id'])
        print("Question:", v1_item['prompt'].split('Question:')[-1].strip())
        print("Reference:", v1_item['reference'])
        print("\n--- V1 (Correct) ---")
        print(v1_item.get('prediction', '')[:300] + '...')
        print("\n--- V4 (Wrong) ---")
        turn_details = v4_item.get('turn_details', [])
        for t in turn_details:
            print(f"Turn {t['turn']}:")
            print("SQL:", t['response'].split('```sql')[-1].split('```')[0] if '```sql' in t['response'] else 'No SQL')
            if 'Feedback:' in t.get('prompt', ''):
                feedback = t['prompt'].split('Feedback:')[-1].split('Based on this')[0]
                print(f"Feedback from previous: {feedback.strip()}")
        print("Final V4 Prediction:", v4_item.get('prediction', ''))

if __name__ == "__main__":
    analyze_hitab_v4()
