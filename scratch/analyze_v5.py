import json
import ast

def clean_str(s):
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and len(parsed) == 1:
            return str(parsed[0]).strip().lower()
    except:
        pass
    return s.strip().lower()

with open("outputs/wikitableqa/predictions_v5.json", "r") as f:
    v5_data = json.load(f).get("predictions", [])

with open("outputs/wikitableqa/predictions_v2.json", "r") as f:
    v2_data = json.load(f).get("predictions", [])

v2_dict = {item["id"]: item for item in v2_data}

pure_cot_count = 0
sql_count = 0
pure_cot_correct = 0
sql_correct = 0

v5_worse = []
v5_better = []

for item in v5_data:
    pred = clean_str(str(item.get("processed_prediction", "")))
    ref = clean_str(str(item.get("reference", "")))
    is_correct = (pred == ref)
    
    turns = item.get("turn_details", [])
    mode = "SQL"
    if len(turns) > 0 and turns[0].get("mode") == "Pure CoT":
        mode = "Pure CoT"
        pure_cot_count += 1
        if is_correct: pure_cot_correct += 1
    else:
        sql_count += 1
        if is_correct: sql_correct += 1
        
    v2_item = v2_dict.get(item["id"])
    if v2_item:
        v2_pred = clean_str(str(v2_item.get("processed_prediction", "")))
        v2_ref = clean_str(str(v2_item.get("reference", "")))
        v2_correct = (v2_pred == v2_ref)
        
        if is_correct and not v2_correct:
            v5_better.append((item, v2_item, mode))
        elif not is_correct and v2_correct:
            v5_worse.append((item, v2_item, mode))

print(f"Total samples: {len(v5_data)}")
print(f"Pure CoT used: {pure_cot_count} ({pure_cot_correct}/{pure_cot_count} correct) -> {pure_cot_correct/max(1, pure_cot_count):.1%}")
print(f"SQL Agent used: {sql_count} ({sql_correct}/{sql_count} correct) -> {sql_correct/max(1, sql_count):.1%}")

print(f"\nv5 better than v2 (Gains): {len(v5_better)}")
print(f"v5 worse than v2 (Losses): {len(v5_worse)}")

print("\n--- Why did v5 lose to v2? (Top 5 examples) ---")
for v5_item, v2_item, mode in v5_worse[:5]:
    print(f"ID: {v5_item['id']} | Mode: {mode}")
    prompt_str = v2_item.get('prompt', '')
    if 'Question: ' in prompt_str:
        q_str = prompt_str.split('Question: ')[-1].split('Answer:')[0].strip()
    else:
        q_str = "N/A"
    print(f"Question: {q_str}")
    print(f"v2 (CoT) Output: {v2_item.get('processed_prediction')}")
    print(f"v5 ({mode}) Output: {v5_item.get('processed_prediction')}")
    print(f"Reference: {v5_item.get('reference')}")
    print("-" * 50)
