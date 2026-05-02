import os
import json
import sqlite3
# 引入我们新写的 table_to_sqlite
from utils import table_to_sqlite 

def test_visualization():
    base_dir = "local_datasets"
    tasks = ["wikitableqa", "tabfact", "fetaqa", "hitab"]
    
    print("="*60)
    print("🌟 SQLite 转换效果预览 (Schema + Query Test) 🌟")
    print("="*60)

    for task in tasks:
        json_path = os.path.join(base_dir, task, "sample_1.json")
        
        if not os.path.exists(json_path):
            print(f"跳过 {task}: 未找到 sample_1.json")
            continue
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取表格字段
        raw_table = data.get('table') or data.get('table_content') or data.get('table_text')
        
        if raw_table is None:
            print(f"警告: {task} 的 sample_1.json 中未找到识别的表格字段。")
            continue

        print(f"\n[任务名称]: {task.upper()}")
        print("-" * 40)
        
        # 【核心修改】：调用 table_to_sqlite 获取连接和 Schema
        table_name = "my_table"
        conn, schema = table_to_sqlite(raw_table, task_name=task, table_name=table_name)
        
        if conn is None:
            print("转换失败：未能识别的表格格式。")
            continue
            
        # 1. 打印给大模型看的 Schema
        print(">>> 传给大模型的 Schema (DDL):")
        print(schema)
        print("\n>>> 数据库实际查询测试 (前3行数据):")
        
        # 2. 模拟大模型执行 SQL 查询
        try:
            cursor = conn.cursor()
            # 查一下前三行看看数据是不是成功进去了
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
            results = cursor.fetchall()
            
            for i, row in enumerate(results):
                print(f"Row {i+1}: {row}")
                
        except Exception as e:
            print(f"查询出错: {e}")
            
        finally:
            # 测试完毕，关闭内存数据库连接
            conn.close()
            
        print("-" * 60)

if __name__ == "__main__":
    test_visualization()