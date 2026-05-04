"""Verify the specific table from predictions IDs 5-7 (the one with black population/immigrant women)."""
import sys
sys.path.insert(0, '.')
from datasets import load_dataset
from utils import _parse_table_universal, format_table
import ast

ds = load_dataset("kasnerz/hitab", split="test")

# Search for the table that has "immigrant women" AND yearly columns
for idx in range(len(ds)):
    item = ds[idx]
    table = ast.literal_eval(item['table_content']) if isinstance(item['table_content'], str) else item['table_content']
    texts = table['texts']
    
    # Look for the specific table: has "immigrant women" and year columns
    has_immigrant_women = any("immigrant women" in str(row).lower() for row in texts)
    has_year_cols = any("2001" in str(row) and "2016" in str(row) for row in texts)
    
    if has_immigrant_women and has_year_cols:
        print(f"Found at HF idx {idx}")
        print(f"Q: {item['question']}")
        print(f"A: {item['answer']}")
        print()
        
        # Show raw grid
        print("Raw grid:")
        for i, row in enumerate(texts):
            dc = row[1:]
            empty = all(str(v).strip() in ('', 'None', 'none', 'percent') for v in dc) if dc else False
            tag = " ← GROUP HEADER (all empty)" if empty else ""
            print(f"  Row {i}: {row}{tag}")
        
        print()
        
        # Show flattened result
        headers, rows = _parse_table_universal(table, task_name='hitab')
        print(f"After flattening ({len(rows)} rows):")
        print(f"Headers: {headers}")
        for row in rows:
            none_check = any(str(v) in ('None', '') for v in row[:2])
            tag = " ← group_level has None!" if none_check else ""
            print(f"  {row}{tag}")
        
        # Check: is "immigrant women" still present?
        has_iw = any("immigrant women" in str(row) for row in rows)
        print(f"\n'immigrant women' row still present after cleanup: {has_iw}")
        
        has_cb = any("canadian-born women" in str(row) for row in rows)
        print(f"'canadian-born women' row still present after cleanup: {has_cb}")
        break
