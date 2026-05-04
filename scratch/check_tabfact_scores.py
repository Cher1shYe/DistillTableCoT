import json

for ver in ['v1', 'v2']:
    path = f"/Users/yebohou/zju/SRTP/DTC/DistillTableCoT/outputs/tabfact/predictions_{ver}.json"
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"{ver} accuracy:", data.get('evaluation_results', 'Not found'))
    except Exception as e:
        print(f"Error loading {ver}: {e}")
