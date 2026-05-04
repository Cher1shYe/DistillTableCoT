import json
from datasets import load_dataset
from utils import _parse_table_universal

def inspect_hitab_samples():
    print("Loading hitab dataset...")
    dataset = load_dataset("kasnerz/hitab", split="test")
    
    for idx, item in enumerate(dataset):
        if idx not in [2, 3]: continue
        
        print(f"========== ID {idx} ==========")
        print("Question:", item['question'])
        headers, data_rows = _parse_table_universal(item['table_content'], task_name="hitab")
        print("\nHEADERS:", headers)
        print("ROWS:")
        for row in data_rows:
            print("  ", row)
        print("="*50)

if __name__ == "__main__":
    inspect_hitab_samples()
