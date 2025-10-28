# utils.py
import os
from openai import OpenAI

# --- API 客户端初始化 ---
try:
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
except Exception as e:
    print(f"API 客户端初始化失败，请检查环境变量 DEEPSEEK_API_KEY: {e}")
    client = None

def format_table(table_data):
    """将结构化的表格字典转换为字符串"""
    if isinstance(table_data, str):
        return table_data
    
    try:
        header = " | ".join(table_data['header'])
        rows_str = []
        for row in table_data['rows'][:15]:
            truncated_row = [str(cell)[:30] + '...' if len(str(cell)) > 30 else str(cell) for cell in row]
            rows_str.append(" | ".join(truncated_row))
        return header + "\n" + "\n".join(rows_str)
    except Exception as e:
        print(f"警告: 格式化表格时出错: {e}")
        return "[Error formatting table]"

def call_deepseek_api(prompt):
    """调用 DeepSeek API 并返回结果"""
    if client is None:
        return "[API_CLIENT_NOT_INITIALIZED]"
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[API_ERROR: {e}]"
