#!/usr/bin/env python3
"""学生 oracle 聚合 —— 把 eval_route 的三次强制路由结果合成 oracle 分析。

用法 (先用 eval_route 的 --force_route 跑三次,再聚合):
    python3 scripts/eval_route.py --model_path <SFT模型> --eval_files <v1 jsonl...> \
        --force_route direct --out_name force_direct.json
    python3 scripts/eval_route.py ... --force_route cot --out_name force_cot.json
    python3 scripts/eval_route.py ... --force_route sql --exec_sql --out_name force_sql.json
    python3 scripts/build_student_oracle.py \
        --direct outputs/hitab/force_direct.json \
        --cot    outputs/hitab/force_cot.json \
        --sql    outputs/hitab/force_sql.json

每个输入文件是 eval_route 产出的 {"summaries":..., "samples":[{id,task,correct,output_tokens,...}]}。
oracle = 逐题挑"最便宜的答对路"(direct<cot<sql);判分/ token 口径已由 eval_route 统一
(route_scoring 判分;output_tokens 只数模型生成,sql 真执行时注入结果不计)。
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

TIER = {"direct": 0, "cot": 1, "sql": 2}


def load_samples(path):
    """eval_route 输出 → {(task,id): {"correct":bool, "tok":int}}。"""
    d = json.load(open(path, encoding="utf-8"))
    samples = d.get("samples") if isinstance(d, dict) else d
    out = {}
    for s in samples:
        out[(s["task"], s["id"])] = {"correct": bool(s.get("correct")),
                                     "tok": s.get("output_tokens") or 0}
    return out


def main():
    ap = argparse.ArgumentParser(description="学生 oracle 聚合 (三次强制路由 → oracle)")
    ap.add_argument("--direct", required=True, help="--force_route direct 的 eval 输出")
    ap.add_argument("--cot", required=True, help="--force_route cot 的 eval 输出")
    ap.add_argument("--sql", required=True, help="--force_route sql 的 eval 输出")
    ap.add_argument("--out", default="outputs/student_oracle.json", help="汇总 JSON 输出路径")
    args = ap.parse_args()

    paths = {"direct": load_samples(args.direct), "cot": load_samples(args.cot),
             "sql": load_samples(args.sql)}
    # 三路按 (task,id) 取交集
    keys = set(paths["direct"]) & set(paths["cot"]) & set(paths["sql"])
    tasks = sorted({t for t, _ in keys})

    summary = {}
    print(f"{'task':<13}{'n':>5}{'oracle_acc':>12}{'oracle_tok':>12}   oracle路径分布(可解题上)  | 各单路 acc")
    for task in tasks:
        ids = sorted(i for (t, i) in keys if t == task)
        n = len(ids)
        solvable = 0
        oracle_toks, oracle_routes = [], Counter()
        per_path = {p: {"c": 0, "tok": 0} for p in TIER}
        for i in ids:
            correct = {p: paths[p][(task, i)]["correct"] for p in TIER}
            tok = {p: paths[p][(task, i)]["tok"] for p in TIER}
            for p in TIER:
                per_path[p]["c"] += correct[p]; per_path[p]["tok"] += tok[p]
            ok = [p for p in ("direct", "cot", "sql") if correct[p]]
            if ok:
                solvable += 1
                op = min(ok, key=lambda p: TIER[p])
                oracle_routes[op] += 1
                oracle_toks.append(tok[op])
        oacc = solvable / n if n else 0.0
        otok = sum(oracle_toks) / len(oracle_toks) if oracle_toks else 0.0
        summary[task] = {
            "n": n, "oracle_accuracy": oacc, "oracle_avg_token": otok,
            "oracle_route_dist": {p: oracle_routes.get(p, 0) for p in TIER},
            "per_path": {p: {"acc": per_path[p]["c"] / n if n else 0.0,
                             "avg_token": per_path[p]["tok"] / n if n else 0.0} for p in TIER},
        }
        dist = "  ".join(f"{p}:{oracle_routes.get(p,0)}({oracle_routes.get(p,0)/solvable:.0%})"
                         if solvable else f"{p}:0" for p in ("direct", "cot", "sql"))
        pp = "  ".join(f"{p}:{per_path[p]['c']/n:.0%}" for p in ("direct", "cot", "sql")) if n else ""
        print(f"{task:<13}{n:>5}{oacc:>11.1%}{otok:>11.0f}   {dist}  | {pp}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(summary, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n✅ 汇总已保存: {args.out}")


if __name__ == "__main__":
    main()
