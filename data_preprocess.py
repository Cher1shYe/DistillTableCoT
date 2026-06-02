"""
data_preprocess.py —— Binder 基线第一步：数据提取与格式转换

从 HuggingFace 加载四个数据集（WikiTableQA, TabFact, FeTaQA, HiTab），
将每条数据转成 Binder 兼容的 JSON 格式，保存到 output/ 目录。

Binder 目标格式:
{
    "id": str,
    "question": str,
    "table": {
        "page_title": str,
        "header": [str, ...],
        "rows": [[str, ...], ...]
    },
    "answer_text": [str, ...],
    "answer_raw": <原始答案，保留原始类型>
}
"""

import ast
import json
import os
import sys
import traceback
from collections import Counter

from datasets import load_dataset, get_dataset_split_names
from tqdm import tqdm

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.table_parser import _parse_table_universal


# ─── 数据集配置 ───────────────────────────────────────────────

DATASET_CONFIGS = {
    "wikitableqa": {
        "hf_path": "table-benchmark/wikiqa",
        "question_field": "question",
        "table_field": "table",
        "title_field": None,           # wikiqa 没有 table_title
        "task_name": "wikitableqa",
    },
    "tabfact": {
        "hf_path": "table-benchmark/tabfact",
        "question_field": "question",   # table-benchmark/tabfact 用 question
        "table_field": "table",
        "title_field": "table_title",
        "task_name": "tabfact",
    },
    "fetaqa": {
        "hf_path": "table-benchmark/fetaqa",
        "question_field": "question",
        "table_field": "table",
        "title_field": "table_title",
        "task_name": "fetaqa",
    },
    "hitab": {
        "hf_path": "kasnerz/hitab",
        "question_field": "question",
        "table_field": "table_content",  # HiTab 用 table_content
        "title_field": None,
        "task_name": "hitab",
    },
}


# ─── 答案规范化 ───────────────────────────────────────────────

def normalize_answer(answer, dataset_name):
    """
    将各数据集的不同答案格式统一为:
        answer_text: list[str]  —— Binder 兼容格式
        answer_raw : 原始值      —— 保留原始类型

    返回 (answer_text, answer_raw)
    """
    if answer is None:
        return [""], None

    if dataset_name == "wikitableqa":
        # HF 中 answer 是形如 "['2004']" 的字符串
        if isinstance(answer, str):
            try:
                parsed = ast.literal_eval(answer)
            except (ValueError, SyntaxError):
                parsed = [answer]
            if isinstance(parsed, list):
                return [str(x) for x in parsed], answer
            return [str(parsed)], answer
        elif isinstance(answer, list):
            return [str(x) for x in answer], answer
        else:
            return [str(answer)], answer

    elif dataset_name == "tabfact":
        # HF 中 answer 是 "entailed" 或 "refuted" (小写)
        if isinstance(answer, (int, float)):
            label = "Entailed" if int(answer) == 1 else "Refuted"
            return [label], answer
        ans_str = str(answer).strip().lower()
        if ans_str == "entailed":
            return ["Entailed"], answer
        elif ans_str == "refuted":
            return ["Refuted"], answer
        else:
            return [str(answer)], answer

    elif dataset_name == "fetaqa":
        # 自由文本回答
        if isinstance(answer, list):
            return [str(x) for x in answer], answer
        return [str(answer)], answer

    elif dataset_name == "hitab":
        # 可能是列表、数字、字符串、嵌套列表
        if isinstance(answer, list):
            return [str(x) for x in answer], answer
        elif isinstance(answer, (int, float)):
            return [str(answer)], answer
        else:
            return [str(answer)], answer

    # fallback
    return [str(answer)], answer


# ─── 数据集加载 ───────────────────────────────────────────────

def load_split_safe(hf_path, split):
    """
    安全加载指定分集。如果 split 不存在，尝试 test → validation → train。
    """
    try:
        available = get_dataset_split_names(hf_path)
    except Exception:
        available = None

    if available:
        print(f"  {hf_path} 可用分集: {available}")

    # 按优先级尝试
    candidates = [split]
    if split != "test":
        candidates.append("test")
    if split != "validation":
        candidates.append("validation")
    if split != "train":
        candidates.append("train")

    for cand in candidates:
        try:
            ds = load_dataset(hf_path, split=cand)
            print(f"  → 加载 '{cand}' 分集成功, 共 {len(ds)} 条")
            return ds, cand
        except Exception as e:
            continue

    raise ValueError(f"无法加载 {hf_path} 的任何分集，尝试了: {candidates}")


# ─── 主处理逻辑 ───────────────────────────────────────────────

def process_dataset(dataset_name, split, output_dir):
    """
    加载并处理一个数据集。

    返回 (items: list[dict], stats: dict)
    """
    config = DATASET_CONFIGS[dataset_name]
    print(f"\n{'='*60}")
    print(f"处理 {dataset_name} (split={split})")
    print(f"{'='*60}")

    # 加载数据
    dataset, actual_split = load_split_safe(config["hf_path"], split)

    items = []
    parse_errors = 0
    empty_tables = 0
    row_counts = []
    col_counts = []

    for idx, sample in enumerate(tqdm(dataset, desc=f"  {dataset_name}/{actual_split}")):
        try:
            # 1. 提取问题
            question = sample.get(config["question_field"], "")
            if question is None:
                question = ""

            # 2. 提取表格原始数据
            table_data = sample.get(config["table_field"], None)

            # 3. 解析表格为 (headers, rows)
            headers, rows = _parse_table_universal(
                table_data,
                task_name=config["task_name"]
            )

            if not headers and not rows:
                empty_tables += 1

            # 4. 提取表格标题
            if config["title_field"]:
                page_title = sample.get(config["title_field"], "") or ""
            else:
                page_title = ""

            # 5. 提取答案
            raw_answer = sample.get("answer", sample.get("label", None))
            answer_text, answer_raw = normalize_answer(raw_answer, dataset_name)

            # 6. 生成 ID（保证唯一性：优先用已有 id 字段，否则用 dataset_name+idx）
            item_id = sample.get("id", None)
            if item_id is None or item_id == "":
                item_id = f"{dataset_name}_{idx}"

            # 7. 构建 Binder 格式
            item = {
                "id": str(item_id),
                "question": str(question),
                "table": {
                    "page_title": str(page_title),
                    "header": [str(h) for h in headers],
                    "rows": [[str(c) for c in row] for row in rows],
                },
                "answer_text": answer_text,
                "answer_raw": answer_raw,
            }
            items.append(item)

            # 统计
            row_counts.append(len(rows))
            col_counts.append(len(headers))

        except Exception as e:
            parse_errors += 1
            print(f"\n  [ERROR] idx={idx}: {e}")
            traceback.print_exc()

    # 汇总统计
    stats = {
        "dataset": dataset_name,
        "split": actual_split,
        "total": len(items),
        "parse_errors": parse_errors,
        "empty_tables": empty_tables,
        "avg_rows": sum(row_counts) / len(row_counts) if row_counts else 0,
        "avg_cols": sum(col_counts) / len(col_counts) if col_counts else 0,
        "max_rows": max(row_counts) if row_counts else 0,
        "max_cols": max(col_counts) if col_counts else 0,
    }

    # 保存
    dataset_out_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_out_dir, exist_ok=True)
    out_path = os.path.join(dataset_out_dir, f"{actual_split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"  → 保存到 {out_path}")
    print(f"  → 统计: total={stats['total']}, errors={parse_errors}, "
          f"empty_tables={empty_tables}, avg_rows={stats['avg_rows']:.1f}, "
          f"avg_cols={stats['avg_cols']:.1f}")

    return items, stats


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "output")

    # 要处理的数据集和分集
    tasks = [
        ("wikitableqa", "test"),
        ("tabfact", "test"),
        ("fetaqa", "test"),
        ("hitab", "test"),
    ]

    all_stats = []

    for dataset_name, split in tasks:
        try:
            _, stats = process_dataset(dataset_name, split, output_dir)
            all_stats.append(stats)
        except Exception as e:
            print(f"\n[FATAL] {dataset_name}/{split} 处理失败: {e}")
            traceback.print_exc()
            all_stats.append({
                "dataset": dataset_name,
                "split": split,
                "total": 0,
                "error": str(e),
            })

    # 保存汇总
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")
    for s in all_stats:
        if "error" in s:
            print(f"  {s['dataset']}/{s['split']}: FAILED - {s['error']}")
        else:
            print(f"  {s['dataset']}/{s['split']}: {s['total']} items "
                  f"(avg {s['avg_rows']:.0f} rows x {s['avg_cols']:.0f} cols)")
    print(f"\n汇总保存到 {summary_path}")


if __name__ == "__main__":
    main()
