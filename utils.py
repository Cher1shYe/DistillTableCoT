# utils.py
import os
import ast
import re
import sqlite3
import pandas as pd
from datetime import datetime
from openai import OpenAI

# --- API 客户端初始化 (保持原样) ---
try:
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
except Exception as e:
    print(f"API 客户端初始化失败: {e}")
    client = None

def _parse_table_universal(table_data, max_rows=None, task_name=None):
    """
    统一解析入口：将任何格式的表格数据解析为标准的 (headers: list[str], data_rows: list[list[str]])
    """
    if not table_data:
        return [], []

    parsed_data = table_data
    if isinstance(table_data, str):
        table_data = table_data.strip()
        # 处理 TabFact
        if (task_name == "tabfact" or 
            ('#' in table_data and '\n' in table_data 
             and not table_data.startswith(('{', '[')))):
            lines = [l for l in table_data.split('\n') if l.strip()]
            header = lines[0].split('#')
            rows = [line.split('#') for line in lines[1:]]
            if max_rows:
                rows = rows[:max_rows]
            return header, rows
        try:
            parsed_data = ast.literal_eval(table_data)
        except:
            parsed_data = table_data

    # 情况 A: HiTab
    if task_name == "hitab" or (isinstance(parsed_data, dict) and "texts" in parsed_data):
        grid = [list(row) for row in parsed_data["texts"]]
        merged = parsed_data.get("merged_regions", [])

        for m in merged:
            val = ""
            for r in range(m['first_row'], m['last_row'] + 1):
                if r >= len(grid): continue
                for c in range(m['first_column'], m['last_column'] + 1):
                    if c >= len(grid[r]): continue
                    if grid[r][c].strip():
                        val = grid[r][c]
                        break
                if val: break
            for r in range(m['first_row'], m['last_row'] + 1):
                if r >= len(grid): continue
                for c in range(m['first_column'], m['last_column'] + 1):
                    if c >= len(grid[r]): continue
                    grid[r][c] = val

        start_row = 1
        try:
            if 'left_root' in parsed_data and 'children' in parsed_data['left_root']:
                start_row = parsed_data['left_root']['children'][0]['row_index']
            else:
                start_row = 2
        except:
            start_row = 2
        start_row = min(start_row, len(grid))

        header_rows = grid[:start_row]
        rows = grid[start_row:]
        if max_rows:
            rows = rows[:max_rows]

        cols = len(grid[0]) if grid else 0
        headers = []
        for c in range(cols):
            h_list = []
            for r in range(start_row):
                h = str(header_rows[r][c]).strip()
                if h and h not in h_list:
                    h_list.append(h)
            # ✅ 统一用 " - " 拼接（对人类友好）
            headers.append(" - ".join(h_list) if h_list else f"Col_{c}")
        return headers, rows

    # 情况 B: WikiTableQA
    if task_name == "wikitableqa" or (isinstance(parsed_data, dict) and 'header' in parsed_data):
        headers = [str(x) for x in parsed_data['header']]
        rows = parsed_data['rows']
        if max_rows:
            rows = rows[:max_rows]
        return headers, rows

    # 情况 C: FeTaQA
    if task_name == "fetaqa" or (isinstance(parsed_data, list) 
                                  and len(parsed_data) > 0 
                                  and isinstance(parsed_data[0], list)):
        headers = [str(x).strip() for x in parsed_data[0]]
        rows = parsed_data[1:]
        if max_rows:
            rows = rows[:max_rows]
        return headers, rows

    return [], []


def _clean_headers_for_sql(headers):
    """
    将人类可读的表头转换为 SQL 安全的列名，并返回映射关系。
    返回: clean_headers: list[str]
    """
    clean_headers = []
    seen = set()
    for i, h in enumerate(headers):
        h = str(h).strip().replace('\n', ' ').replace('"', '').replace("'", "")
        # 把 " - " 替换为 "_"，使其成为合法 SQL 标识符
        h = h.replace(' - ', '_').replace(' ', '_')
        if not h:
            h = f"col_{i}"
        orig_h = h
        counter = 1
        while h in seen:
            h = f"{orig_h}_{counter}"
            counter += 1
        seen.add(h)
        clean_headers.append(h)
    return clean_headers


def _apply_date_conversion(cells):
    """统一的日期转换：逐格转换，转换失败保留原值"""
    result = []
    for val in cells:
        val_str = str(val).strip()
        parsed = parse_date_string(val_str)
        result.append(parsed if parsed else val_str)
    return result


def format_table(table_data, max_rows=None, task_name=None):
    """统一转 Markdown"""
    headers, data_rows = _parse_table_universal(table_data, max_rows, task_name)
    if not headers:
        return str(table_data) if table_data else ""

    md = "| " + " | ".join(headers) + " |\n"
    md += "|" + "|".join(["---"] * len(headers)) + "|\n"
    for row in data_rows:
        cells = _apply_date_conversion(row)
        md += "| " + " | ".join(cells) + " |\n"
    return md


def table_to_sqlite(table_data, max_rows=None, task_name=None, table_name="my_table"):
    """统一转 SQLite"""
    headers, data_rows = _parse_table_universal(table_data, max_rows, task_name)
    if not headers:
        return None, ""

    # ✅ 日期转换：与 Markdown 完全一致，逐格转换
    converted_rows = [_apply_date_conversion(row) for row in data_rows]

    # SQL 安全列名
    clean_headers = _clean_headers_for_sql(headers)

    df = pd.DataFrame(converted_rows, columns=clean_headers)
    conn = sqlite3.connect(':memory:')
    df.to_sql(table_name, conn, index=False, if_exists='replace')

    cursor = conn.cursor()
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
    schema = cursor.fetchone()[0]

    return conn, schema
def execute_sql(conn, sql_query):
    """
    在给定的 sqlite 连接上执行 SQL，并返回 (是否成功, 结果字符串)
    不依赖 SQLAlchemy，完全使用原生 sqlite3 游标执行。
    """
    try:
        # [关键点]: 这里调用的是 conn.cursor()
        cursor = conn.cursor()
        cursor.execute(sql_query)
        
        # 获取列名
        columns = [description[0] for description in cursor.description]
        
        # 获取前 10 行数据 (避免 Token 爆炸)
        rows = cursor.fetchmany(10)
        
        if not rows:
            return True, "SQL executed successfully, but returned no results (Empty Table)."
        
        # 仅使用 Pandas 将结果排版为漂亮的字符串
        df_res = pd.DataFrame(rows, columns=columns)
        res_str = df_res.to_string(index=False)
        
        return True, res_str
        
    except Exception as e:
        # 捕获真实的 SQL 语法错误
        return False, f"SQL Error: {str(e)}"

# 时间安全解析函数（原封不动）
def parse_date_string(date_str):
    if not isinstance(date_str, str) or not str(date_str).strip():
        return None
    date_str = str(date_str).strip()
    
    # 【新增逻辑】：处理一个格子里写两个日期的情况 (比如 "25.01.1943/03.03.1943" 或 "1943-01-02 / 1943-05-06")
    # 我们默认取第一个日期进行判断和排序
    if '/' in date_str and len(date_str) > 10:
        # 按 / 分割，取第一部分
        date_str = date_str.split('/')[0].strip()
        
    # 【新增格式】：加上中间是小数点的格式 "%d.%m.%Y" 和 "%m.%d.%Y"
    date_formats = [
        "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y",
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d",
        "%d.%m.%Y", "%m.%d.%Y"  # <- 就是差了这两个！
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
            
    return None

# def call_deepseek_api(prompt):
#     """调用 DeepSeek API 并返回结果"""
#     if client is None: return "[API_CLIENT_NOT_INITIALIZED]"
#     try:
#         response = client.chat.completions.create(
#             model="deepseek-chat",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.0,
#             max_tokens=1500
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"[API_ERROR: {e}]"
    
def call_deepseek_api(prompt_or_messages):
    """
    调用 DeepSeek API。
    支持传入字符串 (Prompt) 或 消息列表 (Messages)。
    """
    if client is None: 
        return "[API_CLIENT_NOT_INITIALIZED]"
    
    # 自动转换格式：如果传入的是字符串，则包装成标准 messages 格式
    if isinstance(prompt_or_messages, str):
        messages = [{"role": "user", "content": prompt_or_messages}]
    else:
        messages = prompt_or_messages

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.0, # Agent 任务通常需要高确定性
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[API_ERROR: {e}]"