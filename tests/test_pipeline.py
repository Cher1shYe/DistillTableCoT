import sys
import os
import json
import sqlite3
import pandas as pd
from unittest.mock import MagicMock
import pytest

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import table_to_sqlite, execute_sql, format_table
from data_loader.agent_dataset import AgentDataset, _format_target_response

def test_sql_execution():
    """测试 SQLite 转换与执行逻辑"""
    print("\n--- Testing SQL Execution ---")
    table_data = {
        "header": ["Name", "Age", "City"],
        "rows": [
            ["Alice", 20, "New York"],
            ["Bob", 25, "Los Angeles"],
            ["Charlie", 30, "Chicago"]
        ]
    }
    
    conn, schema = table_to_sqlite(table_data, task_name="wikitableqa")
    assert schema is not None
    
    # 测试有效 SQL
    query = "SELECT Name FROM my_table WHERE Age > 22"
    success, result = execute_sql(conn, query)
    assert success == True
    assert "Bob" in result
    assert "Charlie" in result
    
    # 测试无效 SQL
    query = "SELECT Name FROM non_existent_table"
    success, result = execute_sql(conn, query)
    assert success == False
    assert "SQL Error" in result
    
    conn.close()

def test_data_processing():
    """测试 AgentDataset 的多轮对话切片与格式化逻辑"""
    print("\n--- Testing Data Processing (AgentDataset) ---")
    
    # 创建临时 mock 数据
    mock_data = [
        {
            "id": "test_1",
            "task": "wikitableqa",
            "table": {"header": ["Name"], "rows": [["Alice"]]},
            "question": "Who?",
            "turn_details": [
                {
                    "turn": 0,
                    "prompt": "Question: Who?",
                    "response": "<think>Thinking...</think> ```sql\nSELECT * FROM table\n```"
                },
                {
                    "turn": 1,
                    "prompt": "Feedback: Alice",
                    "response": "Final Answer: Alice"
                }
            ]
        }
    ]
    mock_file = "tests/temp_mock_predictions.json"
    with open(mock_file, 'w') as f:
        json.dump(mock_data, f)
    
    try:
        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: str(messages)
        tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        
        dataset = AgentDataset(
            data_paths=[mock_file],
            tokenizer=tokenizer,
            split="all"
        )
        
        # 验证是否正确切片为 2 个样本 (Turn 0 and Turn 1)
        assert len(dataset) >= 2
        
        # 验证 Turn 0 格式
        sample0 = next(d for d in dataset.data if d['id'] == "test_1_turn_0")
        assert "<think>" in sample0['target_response']
        assert "```sql" in sample0['target_response']
        
        # 验证 Turn 1 格式
        sample1 = next(d for d in dataset.data if d['id'] == "test_1_turn_1")
        assert "Final Answer: Alice" in sample1['target_response']
        
    finally:
        if os.path.exists(mock_file):
            os.remove(mock_file)

def test_agent_loop_logic():
    """测试 scripts/test_model.py 中的 Agent 闭环逻辑"""
    print("\n--- Testing Agent Loop Logic ---")
    from scripts.test_model import _extract_sql, run_sql_agent
    
    table_data = {
        "header": ["Name", "Age"],
        "rows": [["Alice", "20"], ["Bob", "25"]]
    }
    task = "wikitableqa"
    sample = {"table": table_data, "question": "Who is 25?"}
    
    # Mock model and tokenizer
    model = MagicMock()
    tokenizer = MagicMock()
    
    # Mock 生成响应：第一轮 SQL，第二轮 Final Answer
    responses = [
        "<think>Finding 25.</think>\n```sql\nSELECT Name FROM my_table WHERE Age = 25\n```",
        "Final Answer: Bob"
    ]
    
    def mock_generate(model, tokenizer, messages, max_new_tokens):
        if not hasattr(mock_generate, "call_count"):
            mock_generate.call_count = 0
        resp = responses[mock_generate.call_count]
        mock_generate.call_count += 1
        return resp

    # 替换生成函数进行测试
    import scripts.test_model as test_model_script
    original_generate = test_model_script.generate
    test_model_script.generate = mock_generate
    
    try:
        config = {
            "system_prompt": "You are an agent.",
            "user_prompt_template": "Table: {table}\nQuestion: {question}"
        }
        
        prediction, turn_history, mode = run_sql_agent(
            model, tokenizer, task, sample, config, max_turns=5
        )
        
        assert "Final Answer: Bob" in prediction
        assert len(turn_history) == 2
        assert "SELECT Name" in turn_history[0]["response"]
        assert mode == "SQL"
        
    finally:
        test_model_script.generate = original_generate

if __name__ == "__main__":
    # 直接运行
    test_sql_execution()
    test_data_processing()
    test_agent_loop_logic()
    print("\n✨ All verification tests passed! The pipeline is ready for production. ✨")
