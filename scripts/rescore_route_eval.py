#!/usr/bin/env python3
"""
用 batch_eval 同口径 (utils_train.route_scoring) 对已有的 route_eval*.json 重新判分，
无需重跑模型 —— 直接拿每条 sample 里存好的 pred_answer 重算 correct，再重建汇总。

解决：旧 eval_route 用通用 is_match，没走 configs 的 postprocess (extract_*_final_answer)，
导致 '374685' vs '[374685.0]'、大小写、数字补 .0 等被误判 (尤其坑 SQL 路径的裸数字答案)。

用法:
    python3 scripts/rescore_route_eval.py outputs/route_sft_v3/route_eval_sample_exec.json
    python3 scripts/rescore_route_eval.py outputs/route_sft_v3/*.json     # 批量
输出: 同目录下 <原名>.rescored.json，并打印新旧对比。
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils_train.route_scoring import score_answer

ROUTES = ("direct", "cot", "sql")


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def build_summary(rows, title):
    n = len(rows)
    if n == 0:
        return None
    acc = mean([r["correct"] for r in rows])
    oracle_acc = mean([r["oracle_solvable"] for r in rows])
    route_acc = mean([r["pred_route"] == r["oracle_path"] for r in rows])
    avg_tok = mean([r["output_tokens"] for r in rows])
    avg_tool = mean([r["tool_calls"] for r in rows])
    dist = {rt: sum(1 for r in rows if r["pred_route"] == rt) for rt in ROUTES}
    over = [r for r in rows if r["oracle_path"] == "direct"]
    overthink = mean([r["pred_route"] not in (None, "direct") for r in over]) if over else 0.0
    under = [r for r in rows if r["oracle_path"] in ("cot", "sql")]
    underthink = mean([(r["pred_route"] == "direct" and not r["correct"]) for r in under]) if under else 0.0
    return {
        "title": title, "n": n,
        "task_accuracy": acc, "oracle_accuracy": oracle_acc,
        "oracle_gap": oracle_acc - acc, "route_accuracy": route_acc,
        "avg_output_tokens": avg_tok, "avg_tool_calls": avg_tool,
        "route_distribution": dist,
        "n_unparsed": sum(1 for r in rows if r["pred_route"] is None),
        "overthinking_rate": overthink, "underthinking_rate": underthink,
    }


def rescore_file(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data["samples"]

    old_acc = mean([r["correct"] for r in rows])
    by_route_delta = defaultdict(lambda: [0, 0])  # (task,route) -> [新对, 旧对]
    for r in rows:
        new_correct = score_answer(r["task"], r.get("pred_answer"), r.get("reference"))
        key = (r["task"], r["pred_route"])
        by_route_delta[key][0] += int(new_correct)
        by_route_delta[key][1] += int(r["correct"])
        r["correct"] = bool(new_correct)
    new_acc = mean([r["correct"] for r in rows])

    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    summaries = [build_summary(v, f"task={t}") for t, v in sorted(by_task.items())]
    if len(by_task) > 1:
        summaries.append(build_summary(rows, "ALL TASKS"))
    data["summaries"] = summaries
    data["rescored"] = True

    out_path = path.replace(".json", ".rescored.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n===== {os.path.basename(path)} =====")
    print(f"  总 accuracy: {old_acc:.1%} → {new_acc:.1%}  ({new_acc - old_acc:+.1%})")
    print(f"  按 (任务,路径) 的 acc 变化 (新对/总, 旧对/总):")
    for (task, route), (new_c, old_c) in sorted(by_route_delta.items()):
        n = sum(1 for r in rows if r["task"] == task and r["pred_route"] == route)
        flag = "  ⬆" if new_c > old_c else ""
        print(f"    {task:12s} {str(route):7s} {new_c}/{n}={new_c/n:.0%}  (旧 {old_c}/{n}={old_c/n:.0%}){flag}")
    print(f"  ✅ {out_path}")
    return summaries


def main():
    ap = argparse.ArgumentParser(description="route_eval json 重新判分 (batch_eval 口径)")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    for fp in args.files:
        rescore_file(fp)


if __name__ == "__main__":
    main()
