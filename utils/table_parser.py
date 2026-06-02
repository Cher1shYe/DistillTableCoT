"""
Convert various table formats to (headers, rows) for Binder compatibility.
"""

from typing import Tuple, List, Any
import pandas as pd
import re
import io


def _parse_table_universal(table_data: Any, task_name: str = None) -> Tuple[List[str], List[List[str]]]:
    """
    Convert various table formats into (headers, rows) tuple.

    Supported formats:
    - pandas DataFrame
    - dict with 'header' and 'rows' keys
    - list of dicts (columnar: each dict is a column)
    - list of lists (first row as header)
    - markdown table string
    - CSV/TSV string
    """

    # DataFrame: most common from HF table-benchmark datasets
    if isinstance(table_data, pd.DataFrame):
        headers = [str(c) for c in table_data.columns.tolist()]
        rows = [[str(v) for v in row] for row in table_data.values.tolist()]
        return headers, rows

    # Dict with explicit header/rows
    if isinstance(table_data, dict):
        if 'header' in table_data and 'rows' in table_data:
            headers = [str(h) for h in table_data['header']]
            rows = [[str(c) for c in row] for row in table_data['rows']]
            return headers, rows

    # String: try Python dict repr first (table-benchmark stores tables as repr strings)
    if isinstance(table_data, str):
        if table_data.strip().startswith('{'):
            import ast
            try:
                parsed = ast.literal_eval(table_data)
                if isinstance(parsed, dict) and 'header' in parsed and 'rows' in parsed:
                    headers = [str(h).replace('\xa0', ' ') for h in parsed['header']]
                    rows = [[str(c).replace('\xa0', ' ') for c in row] for row in parsed['rows']]
                    return headers, rows
            except (ValueError, SyntaxError):
                pass

        # Try markdown table format: | header1 | header2 |\n|-----|-----|\n| val1 | val2 |
        if '|' in table_data and '\n' in table_data:
            headers, rows = _parse_markdown_table(table_data)
            if headers and rows:
                return headers, rows

        # Try CSV/TSV
        try:
            delimiter = '\t' if '\t' in table_data else ','
            reader = pd.read_csv(io.StringIO(table_data), sep=delimiter)
            headers = [str(c) for c in reader.columns.tolist()]
            rows = [[str(v) for v in row] for row in reader.values.tolist()]
            if headers:
                return headers, rows
        except Exception:
            pass

        # Fallback: single column
        return ["content"], [[line] for line in table_data.strip().split('\n') if line.strip()]

    # List of dicts (columnar format: [{"col1": val1, "col2": val2}, ...])
    if isinstance(table_data, list) and len(table_data) > 0:
        if isinstance(table_data[0], dict):
            headers = list(table_data[0].keys())
            rows = [[str(row.get(h, '')) for h in headers] for row in table_data]
            return [str(h) for h in headers], rows
        # List of lists (first row as header)
        elif isinstance(table_data[0], (list, tuple)):
            headers = [str(h) for h in table_data[0]]
            rows = [[str(c) for c in row] for row in table_data[1:]]
            return headers, rows

    print(f"[WARNING] _parse_table_universal: unknown table format, type={type(table_data)}")
    return [], []


def _parse_markdown_table(text: str) -> Tuple[List[str], List[List[str]]]:
    """Parse a markdown pipe table into headers and rows."""
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if len(lines) < 2:
        return [], []

    # First line is header, second is separator (|---|---|), rest are data
    header_line = lines[0]
    headers = [h.strip() for h in header_line.split('|') if h.strip()]

    # Skip separator line if present
    start = 1
    if start < len(lines) and re.match(r'^[\|\-\s:]+$', lines[start]):
        start = 2

    rows = []
    for line in lines[start:]:
        cells = [c.strip() for c in line.split('|') if c.strip()]
        if cells and len(cells) == len(headers):
            rows.append(cells)

    return headers, rows
