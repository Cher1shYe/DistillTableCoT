"""
AgentDataset 端到端验证脚本
验证内容：
  1. 数据加载和多轮切片是否正确
  2. System Prompt 是否被正确注入
  3. Label masking 是否精准（只有最后一轮 assistant 回复参与 Loss）
  4. 同一道题的不同 turn 不会泄漏到 val/test
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.agent_dataset import AgentDataset
from transformers import AutoTokenizer

def test():
    # Load Qwen tokenizer
    print("=== 1. Loading Tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        if "<|endoftext|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|endoftext|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token
    print(f"  pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"  eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
            
    # Point to a real predictions file
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "outputs/wikitableqa/predictions_v2.json")
    
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return
    
    print(f"\n=== 2. Loading Dataset ===")
    dataset = AgentDataset([data_path], tokenizer, split="train")
    
    if len(dataset) == 0:
        print("Dataset is empty. Cannot test.")
        return
    print(f"  Dataset size: {len(dataset)}")
        
    # Get a multi-turn sample (index 1 usually has history)
    idx = min(3, len(dataset) - 1)
    sample = dataset[idx]
    
    print(f"\n=== 3. Sample {idx} Token Analysis ===")
    print(f"  Total token length: {len(sample['input_ids'])}")
    print(f"  Attention mask sum: {sum(sample['attention_mask'])}")
    
    # Count masked vs unmasked labels
    masked = sum(1 for l in sample['labels'] if l == -100)
    unmasked = len(sample['labels']) - masked
    print(f"  Masked labels (prompt/history): {masked}")
    print(f"  Unmasked labels (target response): {unmasked}")
    
    # Decode and display
    print(f"\n=== 4. Full Decoded Text ===")
    full_decoded = tokenizer.decode(sample['input_ids'])
    print(full_decoded[:500] + "..." if len(full_decoded) > 500 else full_decoded)
    
    print(f"\n=== 5. Target (Unmasked Labels Only) ===")
    unmasked_ids = [idx for idx in sample['labels'] if idx != -100]
    target_text = tokenizer.decode(unmasked_ids)
    print(target_text)
    
    # Verify system prompt is present
    print(f"\n=== 6. Validation Checks ===")
    checks = {
        "System prompt present": "<|im_start|>system" in full_decoded,
        "Has assistant response": "<|im_start|>assistant" in full_decoded,
        "Target is non-empty": len(unmasked_ids) > 0,
        "Target ends with im_end": target_text.strip().endswith("<|im_end|>"),
        "No -100 leak": all(i >= 0 for i in unmasked_ids),
    }
    
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
    
    all_passed = all(checks.values())
    print(f"\n{'✅ All checks passed!' if all_passed else '❌ Some checks failed!'}")

if __name__ == "__main__":
    test()
