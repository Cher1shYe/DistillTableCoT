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

def format_table(table_data, max_rows=None, task_name=None):
    """
    万能表格解析器：
    将 WikiTableQA (dict-string), TabFact (sep-string), 
    FeTaQA (list-string), HiTab (complex-dict) 统一转为标准 Markdown
    """
    if not table_data:
        return ""
    # --- 小助手：尝试转换日期，转换失败则返回原字符串 ---
    def _format_cell(val):
        val_str = str(val).strip()
        parsed = parse_date_string(val_str)
        return parsed if parsed else val_str
    # --- 1. 数据解析 (保持之前的万能解析逻辑) ---
    parsed_data = table_data
    if isinstance(table_data, str):
        table_data = table_data.strip()
        # 处理 TabFact
        if task_name == "tabfact" or '#' in table_data and '\n' in table_data and not table_data.startswith(('{', '[')):
            lines = [l for l in table_data.split('\n') if l.strip()]
            header = lines[0].split('#')
            rows = lines[1:]
            if max_rows: rows = rows[:max_rows] # 只有传了参数才截断
            md = "| " + " | ".join(header) + " |\n"
            md += "|" + "|".join(["---"] * len(header)) + "|\n"
            for line in rows:
                # 🌟 逐个单元格进行日期转换
                cells = [_format_cell(c) for c in line.split('#')]
                md += "| " + " | ".join(cells) + " |\n"
            return md
        
        try:
            parsed_data = ast.literal_eval(table_data)
        except:
            parsed_data = table_data

    # --- 2. 开始分情况转 Markdown ---

    # 情况 A: HiTab
    if task_name == "hitab" or isinstance(parsed_data, dict) and "texts" in parsed_data:
        grid = [list(row) for row in parsed_data["texts"]]
        merged = parsed_data.get("merged_regions", [])
        
        # 填充合并单元格 (加入越界保护机制)
        for m in merged:
            val = ""
            # 第一步：在这个合并区域里找非空值
            for r in range(m['first_row'], m['last_row']+1):
                if r >= len(grid): continue  # 【保护】如果行超出了真实的网格，跳过
                for c in range(m['first_column'], m['last_column']+1):
                    if c >= len(grid[r]): continue  # 【保护】如果列超出了真实的网格，跳过
                    
                    if grid[r][c].strip(): 
                        val = grid[r][c]
                        break
                if val: break
                
            # 第二步：将找到的 val 填满整个合并区域
            for r in range(m['first_row'], m['last_row']+1):
                if r >= len(grid): continue  # 【保护】如果行超出了真实的网格，跳过
                for c in range(m['first_column'], m['last_column']+1): 
                    if c >= len(grid[r]): continue  # 【保护】如果列超出了真实的网格，跳过
                    
                    grid[r][c] = val
        
        start_row = 1
        try:
            if 'left_root' in parsed_data and 'children' in parsed_data['left_root']:
                start_row = parsed_data['left_root']['children'][0]['row_index']
            else: start_row = 2
        except: start_row = 2
            
        start_row = min(start_row, len(grid))
        
        header_rows = grid[:start_row]
        data_rows = grid[start_row:]
        if max_rows: data_rows = data_rows[:max_rows] # 自适应截断
        
        cols = len(grid[0]) if grid else 0
        final_headers = []
        for c in range(cols):
            h_list = []
            for r in range(start_row):
                h = str(header_rows[r][c]).strip()
                if h and h not in h_list: h_list.append(h)
            final_headers.append(" - ".join(h_list) if h_list else f"Col_{c}")
            
        md = "| " + " | ".join(final_headers) + " |\n|" + "|".join(["---"]*cols) + "|\n"
        for row in data_rows:
            # 🌟 逐个单元格进行日期转换
            md += "| " + " | ".join(_format_cell(x) for x in row) + " |\n"
        return md

    # 情况 B: WikiTableQA
    if task_name == "wikitableqa" or isinstance(parsed_data, dict) and 'header' in parsed_data:
        header = parsed_data['header']
        rows = parsed_data['rows']
        if max_rows: rows = rows[:max_rows] # 自适应截断
        md = "| " + " | ".join(str(x) for x in header) + " |\n"
        md += "|" + "|".join(["---"] * len(header)) + "|\n"
        for row in rows:
            # 🌟 逐个单元格进行日期转换
            md += "| " + " | ".join(_format_cell(x) for x in row) + " |\n"
        return md

    # 情况 C: FeTaQA
    if task_name == "fetaqa" or isinstance(parsed_data, list) and len(parsed_data) > 0 and isinstance(parsed_data[0], list):
        header = parsed_data[0]
        rows = parsed_data[1:]
        if max_rows: rows = rows[:max_rows] # 自适应截断
        md = "| " + " | ".join(str(x).strip() for x in header) + " |\n"
        md += "|" + "|".join(["---"] * len(header)) + "|\n"
        for row in rows:
            md += "| " + " | ".join(_format_cell(x) for x in row) + " |\n"
        return md

    return str(table_data)

def table_to_sqlite(table_data, max_rows=None, task_name=None, table_name="my_table"):
    """
    万能表格数据库转换器：
    将 WikiTableQA, TabFact, FeTaQA, HiTab 统一解析并存入 SQLite 内存数据库。
    返回: (conn, schema_str)
    """
    if not table_data:
        return None, ""

    # --- 1. 数据解析 (完全保持之前的万能解析逻辑) ---
    parsed_data = table_data
    if isinstance(table_data, str):
        table_data = table_data.strip()
        # 处理 TabFact
        if task_name == "tabfact" or '#' in table_data and '\n' in table_data and not table_data.startswith(('{', '[')):
            lines = [l for l in table_data.split('\n') if l.strip()]
            header = lines[0].split('#')
            rows = lines[1:]
            if max_rows: rows = rows[:max_rows] # 截断
            
            data_rows = [line.split('#') for line in rows]
            return _build_sqlite(header, data_rows, table_name)
        
        try:
            parsed_data = ast.literal_eval(table_data)
        except:
            parsed_data = table_data

    # 定义统一的变量来接收解析结果
    headers = []
    data_rows = []

    # --- 2. 开始分情况提取数据 ---

    # 情况 A: HiTab
    if task_name == "hitab" or isinstance(parsed_data, dict) and "texts" in parsed_data:
        grid = [list(row) for row in parsed_data["texts"]]
        merged = parsed_data.get("merged_regions", [])
        
        # 填充合并单元格 (保持原有的越界保护机制)
        for m in merged:
            val = ""
            for r in range(m['first_row'], m['last_row']+1):
                if r >= len(grid): continue 
                for c in range(m['first_column'], m['last_column']+1):
                    if c >= len(grid[r]): continue 
                    if grid[r][c].strip(): 
                        val = grid[r][c]
                        break
                if val: break
                
            for r in range(m['first_row'], m['last_row']+1):
                if r >= len(grid): continue 
                for c in range(m['first_column'], m['last_column']+1): 
                    if c >= len(grid[r]): continue 
                    grid[r][c] = val
        
        start_row = 1
        try:
            if 'left_root' in parsed_data and 'children' in parsed_data['left_root']:
                start_row = parsed_data['left_root']['children'][0]['row_index']
            else: start_row = 2
        except: start_row = 2
            
        start_row = min(start_row, len(grid))
        
        header_rows = grid[:start_row]
        rows = grid[start_row:]
        if max_rows: rows = rows[:max_rows] # 截断
        
        cols = len(grid[0]) if grid else 0
        for c in range(cols):
            h_list = []
            for r in range(start_row):
                h = str(header_rows[r][c]).strip()
                if h and h not in h_list: h_list.append(h)
            # 【修改点】：为了 SQL 语法的安全性，把多级表头的 " - " 换成 "_"
            headers.append("_".join(h_list) if h_list else f"Col_{c}")
        
        data_rows = rows

    # 情况 B: WikiTableQA
    elif task_name == "wikitableqa" or isinstance(parsed_data, dict) and 'header' in parsed_data:
        headers = parsed_data['header']
        rows = parsed_data['rows']
        if max_rows: rows = rows[:max_rows] # 截断
        data_rows = rows

    # 情况 C: FeTaQA
    elif task_name == "fetaqa" or isinstance(parsed_data, list) and len(parsed_data) > 0 and isinstance(parsed_data[0], list):
        headers = parsed_data[0]
        rows = parsed_data[1:]
        if max_rows: rows = rows[:max_rows] # 截断
        data_rows = rows
        
    else:
        # 如果都不是，无法转成数据库
        return None, ""

    # --- 3. 统一写入 SQLite ---
    return _build_sqlite(headers, data_rows, table_name)


def _build_sqlite(headers, data_rows, table_name):
    """
    负责将清洗好的 headers 和 data_rows 写入内存数据库，并返回连接和 Schema
    """
    # 处理 SQL 敏感问题：列名不能为空、不能重复、去掉换行符
    clean_headers = []
    seen = set()
    for i, h in enumerate(headers):
        h = str(h).strip().replace('\n', ' ').replace('"', '').replace("'", "")
        if not h: 
            h = f"col_{i}"
        
        # 处理重名列 (比如出现两个 Year 列，会变成 Year, Year_1)
        orig_h = h
        counter = 1
        while h in seen:
            h = f"{orig_h}_{counter}"
            counter += 1
        seen.add(h)
        clean_headers.append(h)

    # 用 Pandas 建库
    df = pd.DataFrame(data_rows, columns=clean_headers)

    # === 🌟 绝对安全的列级日期清洗 🌟 ===
    for col in df.columns:
        # 1. 用你的安全函数尝试解析这一列的所有数据
        parsed_series = df[col].apply(parse_date_string)
        
        # 2. 统计解析成功的数量 (非 None 的数量)
        valid_date_count = parsed_series.notna().sum()
        total_count = len(df[col])
        
        # 3. 【双重保险】：只有当这列有超过 50% 的数据能严格匹配你的日期格式时，才转换它
        if total_count > 0 and (valid_date_count / total_count) > 0.5:
            # 遇到解析不出来的（比如个别缺失值），保留原样或者置空
            df[col] = parsed_series.fillna(df[col])
    # =====================================

    conn = sqlite3.connect(':memory:')
    df.to_sql(table_name, conn, index=False, if_exists='replace')

    # 获取建表语句 (Schema)
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