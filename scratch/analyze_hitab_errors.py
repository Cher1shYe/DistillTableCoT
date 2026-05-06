import json

with open("outputs/hitab/predictions_v2.json", "r") as f:
    data = json.load(f)

errors = []
if isinstance(data, dict) and "predictions" in data:
    data = data["predictions"]

for item in data:
    if not isinstance(item, dict):
        continue
    pred = str(item.get("processed_prediction", "")).strip()
    ref = str(item.get("reference", "")).strip()
    if pred != ref and item.get("prediction", "").strip() != "":
        errors.append(item)

print(f"Total errors with non-empty prediction: {len(errors)}")

for i in range(min(5, len(errors))):
    item = errors[i]
    print(f"ID: {item.get('id')}")
    print(f"Prediction: {item.get('prediction')}")
    print(f"Processed: {item.get('processed_prediction', '')}")
    print(f"Reference: {item.get('reference', '')}")
    print("-" * 50)
