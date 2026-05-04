from datasets import load_dataset
import json

def inspect_sample_12():
    dataset = load_dataset("allennlp/wikitableqa", split="test")
    sample = dataset[12]
    print("Sample 12 keys:", sample.keys())
    print("Question:", sample['question'])
    print("Reference:", sample['answer_text'])
    
    table = sample['table']
    print("\nTable Header:", table['header'])
    print("\nTable Rows (first 2):", table['rows'][:2])
    
    # Check if any row has metadata or something like that
    print("\nRow 0 details:")
    for i, cell in enumerate(table['rows'][0]):
        print(f"Cell {i}: {cell} (type: {type(cell)})")

if __name__ == "__main__":
    inspect_sample_12()
