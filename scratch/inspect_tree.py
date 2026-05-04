from datasets import load_dataset
import ast, json

ds = load_dataset("kasnerz/hitab", split="test")

# First, just print first 10 questions 
for idx in range(10):
    item = ds[idx]
    print(f"ID {idx}: {item['question'][:100]}")
    print(f"  Answer: {item['answer']}")
    
    table = ast.literal_eval(item["table_content"]) if isinstance(item["table_content"], str) else item["table_content"]
    texts = table["texts"]
    left_root = table.get("left_root", {})
    
    # Check for hierarchy markers
    has_empty_rows = False
    for row in texts:
        data_cols = row[1:] if len(row) > 1 else []
        if data_cols and all(str(v).strip() in ('', 'None', 'none') for v in data_cols):
            has_empty_rows = True
            break
    
    def tree_depth(node):
        children = node.get("children", [])
        if not children: return 0
        return 1 + max(tree_depth(c) for c in children)
    
    d = tree_depth(left_root)
    print(f"  Tree depth: {d}, Has empty rows: {has_empty_rows}")
    print()
