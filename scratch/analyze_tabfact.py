import json

def analyze_tabfact_difference():
    with open('/Users/yebohou/zju/SRTP/DTC/DistillTableCoT/outputs/tabfact/predictions_v1.json', 'r') as f:
        v1_data = json.load(f)['predictions']
    with open('/Users/yebohou/zju/SRTP/DTC/DistillTableCoT/outputs/tabfact/predictions_v2.json', 'r') as f:
        v2_data = json.load(f)['predictions']
        
    v1_dict = {item['id']: item for item in v1_data}
    v2_dict = {item['id']: item for item in v2_data}
    
    v1_correct_v2_wrong = []
    
    for _id, v2_item in v2_dict.items():
        if _id not in v1_dict: continue
        v1_item = v1_dict[_id]
        
        v1_pred = str(v1_item.get('processed_prediction')).lower()
        v2_pred = str(v2_item.get('processed_prediction')).lower()
        ref = str(v2_item.get('reference')).lower()
        
        v1_correct = (v1_pred == ref)
        v2_correct = (v2_pred == ref)
        
        if v1_correct and not v2_correct:
            v1_correct_v2_wrong.append((v1_item, v2_item))
            
    print(f"Total V1 correct but V2 wrong: {len(v1_correct_v2_wrong)}")
    
    # Print a couple of examples
    for v1_item, v2_item in v1_correct_v2_wrong[:3]:
        print("="*50)
        print("ID:", v1_item['id'])
        print("Question (Statement):", v1_item['prompt'].split('Statement:')[-1].split('Is the')[0].strip())
        print("Reference:", v1_item['reference'])
        print("\n--- V1 (Correct) ---")
        print(v1_item.get('prediction', '')[:300] + '...')
        print("\n--- V2 (Wrong) ---")
        turn_details = v2_item.get('turn_details', [])
        for t in turn_details:
            print(f"Turn {t['turn']}:")
            print("SQL:", t['response'].split('```sql')[-1].split('```')[0] if '```sql' in t['response'] else 'No SQL')
            if 'Feedback:' in t.get('prompt', ''):
                feedback = t['prompt'].split('Feedback:')[-1].split('Based on this')[0]
                print(f"Feedback from previous: {feedback.strip()}")
        print("Final V2 Prediction:", v2_item.get('prediction', ''))

if __name__ == "__main__":
    analyze_tabfact_difference()
