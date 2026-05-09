# scripts/batch_eval_qwen3.py
"""
批量重清洗 + 评估学生模型 4 个核心变体 (basic / cot / agent / mixed) 以及
所有 baseline 大模型 (predictions_baseline_*.json) 的预测文件，
打印一张横向对比表。每个文件会按 run_qwen3B_evaluate 的逻辑被覆盖回写。

只纳入用于横向对比的变体；distilled / origin 等历史文件会被忽略。

用法:
    python3 scripts/batch_eval_qwen3.py                # 全跑+回写
    python3 scripts/batch_eval_qwen3.py --dry_run      # 仅打印，不覆盖
    python3 scripts/batch_eval_qwen3.py --include_legacy  # 同时评估 distilled / origin
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nltk
import evaluate

from configs import (
    TASK_CONFIGS,
    extract_wiki_final_answer,
    extract_fact_final_answer,
    extract_fetaqa_final_answer,
    extract_hitab_final_answer,
)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

POSTPROCESS = {
    "wikitableqa": lambda pred, ref: extract_wiki_final_answer(pred),
    "tabfact":     lambda pred, ref: extract_fact_final_answer(pred),
    "fetaqa":      lambda pred, ref: extract_fetaqa_final_answer(pred),
    "hitab":       lambda pred, ref: extract_hitab_final_answer(pred, ref),
}

# 缓存 metric 实例，避免重复加载
_METRIC_CACHE = {}


def _load_metric(name):
    if name not in _METRIC_CACHE:
        _METRIC_CACHE[name] = evaluate.load(name, trust_remote_code=True)
    return _METRIC_CACHE[name]


def reclean(data, task):
    fn = POSTPROCESS[task]
    changed = 0
    for d in data:
        new_pred = fn(d.get("prediction", ""), d.get("reference", ""))
        if new_pred != d.get("processed_prediction", ""):
            changed += 1
        d["processed_prediction"] = new_pred
    return changed


def compute_metrics(data, task):
    config = TASK_CONFIGS[task]
    processed_preds = [item['processed_prediction'] for item in data]
    raw_references = [item['reference'] for item in data]
    results_to_save = {}

    for metric_name in config["metrics"]:
        if metric_name == "exact_match":
            if task == "hitab":
                import ast
                correct, total = 0, len(processed_preds)
                for p, r in zip(processed_preds, raw_references):
                    p_str, r_str = str(p).strip(), str(r).strip()
                    match = False
                    if p_str == r_str:
                        match = True
                    else:
                        try:
                            p_val = ast.literal_eval(p_str)
                            r_val = ast.literal_eval(r_str)
                            if p_val == r_val:
                                match = True
                            else:
                                p_list = p_val if isinstance(p_val, list) else [p_val]
                                r_list = r_val if isinstance(r_val, list) else [r_val]
                                if len(p_list) == len(r_list):
                                    list_match = all(
                                        float(p_i) == float(r_i)
                                        for p_i, r_i in zip(p_list, r_list)
                                    )
                                    if list_match:
                                        match = True
                        except Exception:
                            pass
                    if match:
                        correct += 1
                results_to_save["exact_match"] = round(correct / total, 4) if total else 0.0
            else:
                import re
                import ast

                def normalize_string(s):
                    if not s:
                        return ""
                    s = str(s).lower().strip()
                    s = re.sub(r'(?<=\d),(?=\d)', '', s)
                    s = re.sub(r'\b(a|an|the)\b', ' ', s)
                    s = re.sub(r'[^\w\s-]', ' ', s)
                    if s.endswith('.'):
                        s = s[:-1]
                    return " ".join(s.split())

                correct, total = 0, len(processed_preds)
                for p, r in zip(processed_preds, raw_references):
                    p_norm = normalize_string(p)
                    if isinstance(r, str) and r.startswith('[') and r.endswith(']'):
                        try:
                            r_list = ast.literal_eval(r)
                        except Exception:
                            r_list = [r.strip("[]'\"")]
                    elif isinstance(r, list):
                        r_list = r
                    else:
                        r_list = [str(r)]
                    r_norms = [normalize_string(x) for x in r_list]
                    if p_norm in r_norms:
                        correct += 1
                    elif p_norm == ", ".join(r_norms) or p_norm == ",".join(r_norms):
                        correct += 1
                    elif all(ref in p_norm for ref in r_norms) and len(r_norms) > 1:
                        correct += 1
                results_to_save["exact_match"] = round(correct / total, 4) if total else 0.0

        elif metric_name == "accuracy":
            metric = _load_metric("accuracy")
            preds_int = [1 if str(p).lower() == 'entailed' else 0 for p in processed_preds]
            refs_int  = [1 if str(r).lower() == 'entailed' else 0 for r in raw_references]
            score = metric.compute(predictions=preds_int, references=refs_int)
            results_to_save["accuracy"] = round(float(score["accuracy"]), 4)

        elif metric_name == "rouge":
            metric = _load_metric("rouge")
            preds_r = ["\n".join(nltk.sent_tokenize(str(p))) for p in processed_preds]
            refs_r  = ["\n".join(nltk.sent_tokenize(str(r))) for r in raw_references]
            score = metric.compute(predictions=preds_r, references=refs_r)
            for k, v in score.items():
                if isinstance(v, (float, int)):
                    results_to_save[k] = round(float(v), 4)

        elif metric_name == "sacrebleu":
            metric = _load_metric("sacrebleu")
            preds_b = [str(p) for p in processed_preds]
            refs_b  = [[str(r)] for r in raw_references]
            score = metric.compute(predictions=preds_b, references=refs_b)
            results_to_save["sacrebleu"] = round(score["score"], 2)
    return results_to_save


# 学生模型纳入横向对比的 4 个变体（顺序就是输出表里的列顺序）
STUDENT_VARIANTS = ("basic", "cot", "agent", "mixed")
# 这些 token 出现在文件名里就视作该变体（兼容历史命名）
STUDENT_VARIANT_KEYS = {
    "basic": ("basic_model",),       # predictions_qwen3_1.7b_basic_model_v0.json
    "cot":   ("_cot_",),             # predictions_qwen3_1.7b_cot_<task>.json
    "agent": ("_agent_",),           # predictions_qwen3_1.7b_agent_<task>.json
    "mixed": ("_mixed_",),           # predictions_qwen3_1.7b_mixed_<task>.json
}
LEGACY_VARIANT_KEYS = ("distilled", "origin")


def detect_variant(fname: str):
    """根据文件名返回 (kind, variant_label)。
    kind ∈ {"student", "teacher", "baseline", "legacy", None}
    """
    # 教师模型: predictions_teacher_<model>_<mode>.json
    if fname.startswith("predictions_teacher_"):
        label = fname[len("predictions_teacher_"):].rsplit(".json", 1)[0]
        return "teacher", label
    # baseline 大模型: predictions_baseline_<model>.json
    if fname.startswith("predictions_baseline_"):
        label = fname[len("predictions_baseline_"):].rsplit(".json", 1)[0]
        return "baseline", label
    # 学生模型 4 个核心变体
    for variant, keys in STUDENT_VARIANT_KEYS.items():
        if any(k in fname for k in keys):
            return "student", variant
    # 历史包袱
    if any(k in fname for k in LEGACY_VARIANT_KEYS):
        return "legacy", "legacy"
    return None, None


def find_files(include_legacy: bool = False):
    files = []
    for task in ("wikitableqa", "tabfact", "fetaqa", "hitab"):
        d = os.path.join("outputs", task)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json"):
                continue
            kind, label = detect_variant(f)
            if kind is None:
                continue
            if kind == "legacy" and not include_legacy:
                continue
            files.append((task, os.path.join(d, f), kind, label))
    return files


def _main_metric(task, metrics):
    """返回每个任务用于横向对比的主指标 (display_str, sortable_value)。"""
    if task == "fetaqa":
        rl = metrics.get("rougeL")
        bleu = metrics.get("sacrebleu")
        return f"rougeL={rl}  bleu={bleu}", rl if rl is not None else -1
    if task == "tabfact":
        acc = metrics.get("accuracy")
        return f"acc={acc}", acc if acc is not None else -1
    em = metrics.get("exact_match")
    return f"em={em}", em if em is not None else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="只打印分数，不回写文件")
    parser.add_argument("--include_legacy", action="store_true",
                        help="同时评估 distilled / origin 等历史文件 (默认排除)")
    args = parser.parse_args()

    files = find_files(include_legacy=args.include_legacy)
    print(f"发现 {len(files)} 个预测文件 (include_legacy={args.include_legacy})\n")

    summary = []
    for task, path, kind, label in files:
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and "predictions" in raw:
                data = raw["predictions"]
            else:
                data = raw
            n = len(data)
            changed = reclean(data, task)
            metrics = compute_metrics(data, task)
            summary.append({
                "task": task,
                "kind": kind,
                "variant": label,
                "file": os.path.basename(path),
                "n": n,
                "changed": changed,
                "metrics": metrics,
            })
            fname = os.path.basename(path)
            metric_str = "  ".join(f"{k}={v}" for k, v in metrics.items())
            tag = f"{kind}:{label}"
            print(f"[{task:12s}] [{tag:22s}] {fname:55s} n={n:4d} changed={changed:4d}  {metric_str}")

            if not args.dry_run:
                final_output = {"evaluation_results": metrics, "predictions": data}
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(final_output, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[{task:12s}] {path}  ERROR: {e}")

    # ---- 横向对比表 ----
    print("\n" + "=" * 78)
    print("横向对比表 (学生 4 变体 + baseline 大模型)")
    print("=" * 78)
    by_task = {}
    for row in summary:
        by_task.setdefault(row["task"], []).append(row)

    for task in ("wikitableqa", "tabfact", "fetaqa", "hitab"):
        if task not in by_task:
            continue
        rows = by_task[task]
        students = {r["variant"]: r for r in rows if r["kind"] == "student"}
        teachers = [r for r in rows if r["kind"] == "teacher"]
        baselines = [r for r in rows if r["kind"] == "baseline"]
        legacies = [r for r in rows if r["kind"] == "legacy"]

        print(f"\n## {task}")
        # 学生 4 变体按固定顺序打印
        for v in STUDENT_VARIANTS:
            if v in students:
                metric_str, _ = _main_metric(task, students[v]["metrics"])
                print(f"  student   {v:<14s}  n={students[v]['n']:<4d}  {metric_str}  ({students[v]['file']})")
            else:
                print(f"  student   {v:<14s}  -- 缺失 --")
        # teacher / baseline 按指标排序
        for kind_label, rows_g in (("teacher", teachers), ("baseline", baselines)):
            if not rows_g:
                continue
            rows_sorted = sorted(rows_g, key=lambda r: -_main_metric(task, r["metrics"])[1])
            for r in rows_sorted:
                metric_str, _ = _main_metric(task, r["metrics"])
                print(f"  {kind_label:<9s} {r['variant']:<14s}  n={r['n']:<4d}  {metric_str}  ({r['file']})")
        if args.include_legacy and legacies:
            for r in legacies:
                metric_str, _ = _main_metric(task, r["metrics"])
                print(f"  legacy    {r['file']:<22s}  n={r['n']:<4d}  {metric_str}")

    # 保存 JSON 汇总
    if not args.dry_run:
        out_path = "outputs/qwen3_eval_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n汇总已写入 {out_path}")


if __name__ == "__main__":
    main()
