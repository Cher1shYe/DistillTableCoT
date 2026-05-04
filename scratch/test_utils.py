import sqlite3
import pandas as pd
from datetime import datetime
from utils import _apply_date_conversion, _clean_headers_for_sql, table_to_sqlite, execute_sql, parse_date_string

def test_apply_date_conversion():
    print("Testing _apply_date_conversion...")
    test_cases = [
        (["$1,200.50", "10%", "2023-01-01", "normal string", None, "-", "N/A"], 
         [1200.5, 0.1, datetime(2023, 1, 1), "normal string", None, None, None])
    ]
    for inputs, expected in test_cases:
        results = _apply_date_conversion(inputs)
        print(f"Input: {inputs}")
        print(f"Result: {results}")
        # Note: datetime objects comparison works fine
        assert results == expected
    print("Test passed!\n")

def test_clean_headers_for_sql():
    print("Testing _clean_headers_for_sql...")
    headers = ["Name", "Select", "Order By", "Date", "Name"]
    expected = ["Name", "Select_col", "Order_By", "Date", "Name_1"]
    results = _clean_headers_for_sql(headers)
    print(f"Input: {headers}")
    print(f"Result: {results}")
    assert results == expected
    print("Test passed!\n")

def test_table_to_sqlite_and_query():
    print("Testing table_to_sqlite and execute_sql...")
    table_data = {
        "header": ["Player", "Score", "Date"],
        "rows": [
            ["Alice", "1,000", "2023-01-01"],
            ["Bob", "500.5", "2023-02-15"],
            ["Charlie", "20%", "2023-03-10"]
        ]
    }
    conn, schema = table_to_sqlite(table_data, task_name="wikitableqa")
    print(f"Schema: {schema}")
    
    success, res = execute_sql(conn, "SELECT * FROM my_table WHERE Score > 100")
    print(f"Query Result:\n{res}")
    assert "Alice" in res
    assert "Bob" in res
    assert "Charlie" not in res # 20% is 0.2
    
    success, res = execute_sql(conn, "SELECT * FROM my_table ORDER BY Date DESC")
    print(f"Ordered Result:\n{res}")
    
    print("Test passed!\n")

if __name__ == "__main__":
    try:
        test_apply_date_conversion()
        test_clean_headers_for_sql()
        test_table_to_sqlite_and_query()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
