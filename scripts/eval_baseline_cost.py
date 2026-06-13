#!/usr/bin/env python3
"""
给旧 baseline 模型(无路由的 1.7b basic/cot/agent/mixed SFT)补 cost-aware 指标，
和 eval_route.py 的 route-SFT 结果同表对比。无需 GPU、无需重跑模型：

  - 评估子集 : 与 route-SFT 完全相同 —— route_sft_v1.jsonl 里的 id (oracle 可解子集)
  - accuracy : 对 processed_prediction 用同一个 is_match (fetaqa 用 ROUGE-L>=0.3)
  - 成本     : 用 Qwen3 tokenizer 把已保存的生成文本重新 tokenize 计数 (近似生成 token 数)；
               多轮模式 (agent/mixed) 按 turn_details 逐轮累加
  - tool_calls: 含 SQL 的轮数 (单轮模式恒为 0)

用法:
    python3 scripts/eval_baseline_cost.py                       # 全部 4 任务 × 4 baseline
    python3 scripts/eval_baseline_cost.py --tasks hitab tabfact
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils_train.route_scoring import match_processed

TASKS = ["hitab", "fetaqa", "tabfact", "wikitableqa"]
# mode -> 预测文件名模板
BASELINE_FILES = {
    "basic": "predictions_qwen3_1.7b_basic_model_v0.json",
    "cot": "predictions_qwen3_1.7b_cot_{task}.json",
    "agent": "predictions_qwen3_1.7b_agent_{task}.json",
    "mixed": "predictions_qwen3_1.7b_mixed_{task}.json",
}
SQL_PAT = re.compile(r"```sql|\bSELECT\b", re.IGNORECASE)


def load_tokenizer():
    from transformers import AutoTokenizer
    local = os.path.join(ROOT, "origin", "Qwen3-1.7B_origin")
    path = local if os.path.exists(local) else "Qwen/Qwen3-1.7B"
    print(f"tokenizer: {path}")
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def count_tokens(tok, text):
    if not text:
        return 0
    return len(tok(text, add_special_tokens=False)["input_ids"])


def eval_one(tok, preds, eval_ids, task):
    """返回该 baseline 在评估子集上的 (rows, summary)。"""
    rows = []
    for p in preds:
        if p["id"] not in eval_ids:
            continue
        turns = p.get("turn_details")
        if turns:
            n_tok = sum(count_tokens(tok, t.get("response", "")) for t in turns)
            tool_calls = sum(1 for t in turns if SQL_PAT.search(t.get("response", "")))
        else:
            n_tok = count_tokens(tok, p.get("prediction", ""))
            tool_calls = 0
        correct = match_processed(task, p.get("processed_prediction", ""), p.get("reference"))
        rows.append({"id": p["id"], "correct": bool(correct),
                     "tokens": n_tok, "tool_calls": tool_calls})
    n = len(rows)
    if n == 0:
        return rows, None
    return rows, {
        "n": n,
        "task_accuracy": sum(r["correct"] for r in rows) / n,
        "avg_output_tokens": sum(r["tokens"] for r in rows) / n,
        "avg_tool_calls": sum(r["tool_calls"] for r in rows) / n,
    }


def main():
    ap = argparse.ArgumentParser(description="旧 baseline 补 cost 指标 (本地 CPU)")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--out", default="outputs/baseline_cost_summary.json")
    args = ap.parse_args()

    tok = load_tokenizer()
    summary = {}
    for task in args.tasks:
        v1_path = os.path.join("outputs", task, "route_sft_v1.jsonl")
        if not os.path.exists(v1_path):
            print(f"⚠️ {v1_path} 不存在，跳过 {task}")
            continue
        eval_ids = set()
        with open(v1_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    eval_ids.add(json.loads(line)["id"])

        print(f"\n===== {task}  (评估子集 n={len(eval_ids)}, 与 route-SFT 相同) =====")
        print(f"  {'baseline':8s}  {'acc':>7s}  {'avg_tok':>8s}  {'tool':>5s}")
        summary[task] = {}
        for mode, tmpl in BASELINE_FILES.items():
            fp = os.path.join("outputs", task, tmpl.format(task=task))
            if not os.path.exists(fp):
                print(f"  {mode:8s}  (文件缺失: {os.path.basename(fp)})")
                continue
            with open(fp, encoding="utf-8") as f:
                preds = json.load(f)["predictions"]
            _, s = eval_one(tok, preds, eval_ids, task)
            if s is None:
                print(f"  {mode:8s}  (无 id 交集)")
                continue
            summary[task][mode] = s
            print(f"  {mode:8s}  {s['task_accuracy']:6.1%}  {s['avg_output_tokens']:8.0f}  {s['avg_tool_calls']:5.2f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 汇总已保存: {args.out}")


if __name__ == "__main__":
    main()
