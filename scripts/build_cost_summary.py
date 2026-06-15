#!/usr/bin/env python3
"""
统一 cost 汇总：把 teacher / 旧 baseline / route-SFT 三方的
(task accuracy, avg output tokens, avg tool_calls) 放到同一张表里，
口径全部对齐 (同 oracle 评估子集 n≈90、同 batch_eval 判分)。

  - teacher  : R1_prediction_<path>_v1.json 的 usage.completion_tokens / tool_calls
  - baseline : 复用 eval_baseline_cost (predictions_qwen3_1.7b_<mode>_<task>.json)
  - route    : route_eval*.rescored.json (free 模式总体 + sample 模式各路径)

输出: outputs/cost_summary.json + 终端表格。

用法:
    python3 scripts/build_cost_summary.py
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils_train.route_scoring import match_processed

TASKS = ["hitab", "fetaqa", "tabfact", "wikitableqa"]
TEACHER_FILES = {"direct": "R1_prediction_direct_v1.json",
                 "cot": "R1_prediction_cot_v1.json",
                 "sql": "R1_prediction_sql_agent_v1.json"}
V3 = "outputs/route_sft_v3"


def eval_ids(task):
    ids = set()
    with open(f"outputs/{task}/route_sft_v1.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def teacher_cost(task, ids):
    out = {}
    for path, fn in TEACHER_FILES.items():
        recs = {r["id"]: r for r in json.load(open(f"outputs/{task}/{fn}"))}
        c = t = k = n = 0
        for i in ids:
            if i not in recs:
                continue
            n += 1
            r = recs[i]
            u = r.get("usage") or {}
            t += u.get("completion_tokens") or 0
            k += r.get("tool_calls") or 0
            if match_processed(task, r.get("processed_prediction", ""), r.get("reference")):
                c += 1
        if n:
            out[path] = {"acc": c / n, "tokens": t / n, "tool_calls": k / n, "n": n}
    return out


def route_cost():
    free = json.load(open(f"{V3}/route_eval.rescored.json"))
    samp = json.load(open(f"{V3}/route_eval_sample_exec.rescored.json"))
    # free 模式总体 (实际部署形态)
    free_t = defaultdict(lambda: {"c": 0, "n": 0, "tok": 0, "tool": 0})
    for s in free["samples"]:
        x = free_t[s["task"]]
        x["c"] += s["correct"]; x["n"] += 1
        x["tok"] += s["output_tokens"]; x["tool"] += s["tool_calls"]
    # sample 模式按所选路径
    bypath = defaultdict(lambda: {"c": 0, "n": 0, "tok": 0, "tool": 0})
    for s in samp["samples"]:
        x = bypath[(s["task"], s["pred_route"])]
        x["c"] += s["correct"]; x["n"] += 1
        x["tok"] += s["output_tokens"]; x["tool"] += s["tool_calls"]
    return free_t, bypath


def main():
    summary = {}
    free_t, bypath = route_cost()

    for task in TASKS:
        ids = eval_ids(task)
        tc = teacher_cost(task, ids)
        ft = free_t[task]
        summary[task] = {
            "n": len(ids),
            "teacher": tc,
            "route_free": {"acc": ft["c"] / ft["n"], "tokens": ft["tok"] / ft["n"],
                           "tool_calls": ft["tool"] / ft["n"]},
            "route_by_path": {
                r: {"acc": bypath[(task, r)]["c"] / bypath[(task, r)]["n"],
                    "tokens": bypath[(task, r)]["tok"] / bypath[(task, r)]["n"],
                    "tool_calls": bypath[(task, r)]["tool"] / bypath[(task, r)]["n"],
                    "n": bypath[(task, r)]["n"]}
                for r in ("direct", "cot", "sql") if bypath[(task, r)]["n"]
            },
        }
        print(f"\n===== {task} (n={len(ids)}) =====")
        print("  [teacher]   " + "  ".join(
            f"{p}:{v['acc']:.0%}/{v['tokens']:.0f}tok/{v['tool_calls']:.1f}tool"
            for p, v in tc.items()))
        print(f"  [route free] acc={ft['c']/ft['n']:.0%} tok={ft['tok']/ft['n']:.0f} tool={ft['tool']/ft['n']:.2f}")
        print("  [route/路径] " + "  ".join(
            f"{r}:{summary[task]['route_by_path'][r]['acc']:.0%}/"
            f"{summary[task]['route_by_path'][r]['tokens']:.0f}tok"
            for r in summary[task]["route_by_path"]))

    with open("outputs/cost_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n✅ outputs/cost_summary.json")


if __name__ == "__main__":
    main()
