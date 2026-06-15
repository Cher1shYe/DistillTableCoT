#!/usr/bin/env python3
"""一次性脚本：从教师 direct/cot/sql_agent 三文件重算 teacher oracle 表格。

oracle = 逐题挑"最便宜的答对路径"(tier: direct<cot<sql)。
  - oracle accuracy : P(≥1 路答对)   —— 分母 = 全部教师评过的题(含不可解)
  - oracle avg token: 可解题上, oracle 路径的 completion_tokens 均值
  - oracle route 分布: 可解题上 oracle_path 的 direct/cot/sql 占比
判分复用 match_processed(与 build_cost_summary / batch_eval 同口径)。
"""
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils_train.route_scoring import match_processed

TASKS = ["hitab", "fetaqa", "tabfact", "wikitableqa"]
TIER = {"direct": 0, "cot": 1, "sql": 2}
FILES = {
    "v1": {"direct": "R1_prediction_direct_v1.json", "cot": "R1_prediction_cot_v1.json",
           "sql": "R1_prediction_sql_agent_v1.json"},
    "v2": {"direct": "R1_prediction_direct_v2.json", "cot": "R1_prediction_cot_v2.json",
           "sql": "R1_prediction_sql_agent_v2.json"},
}


def load(task, fn):
    p = f"outputs/{task}/{fn}"
    return {r["id"]: r for r in json.load(open(p))} if os.path.exists(p) else None


def tokens(rec):
    return (rec.get("usage") or {}).get("completion_tokens") or 0


def analyze(task, version):
    recs = {p: load(task, fn) for p, fn in FILES[version].items()}
    if any(v is None for v in recs.values()):
        return None
    ids = set(recs["direct"]) & set(recs["cot"]) & set(recs["sql"])

    n = len(ids)
    solvable = 0
    oracle_toks, oracle_routes = [], Counter()
    per_path = {p: {"c": 0, "tok": 0} for p in TIER}  # 各路 acc / avg token (上下文)

    for i in ids:
        correct, tok = {}, {}
        for p in TIER:
            r = recs[p][i]
            correct[p] = bool(match_processed(task, r.get("processed_prediction", ""),
                                              r.get("reference")))
            tok[p] = tokens(r)
            per_path[p]["c"] += correct[p]
            per_path[p]["tok"] += tok[p]
        ok = [p for p in ("direct", "cot", "sql") if correct[p]]
        if ok:
            solvable += 1
            op = min(ok, key=lambda p: TIER[p])   # 最便宜的答对路
            oracle_routes[op] += 1
            oracle_toks.append(tok[op])

    return {
        "n": n, "solvable": solvable,
        "oracle_acc": solvable / n if n else 0.0,
        "oracle_avg_tok": sum(oracle_toks) / len(oracle_toks) if oracle_toks else 0.0,
        "routes": oracle_routes,
        "per_path": {p: {"acc": per_path[p]["c"] / n, "tok": per_path[p]["tok"] / n}
                     for p in TIER},
    }


def fmt_dist(routes, solvable):
    return "  ".join(
        f"{p}:{routes.get(p,0)}({routes.get(p,0)/solvable:.0%})" if solvable else f"{p}:0"
        for p in ("direct", "cot", "sql"))


def main():
    for version, label in [("v1", "TEST (v1)"), ("v2", "TRAIN (v2)")]:
        print(f"\n{'='*78}\n {label}  —— teacher oracle 表格\n{'='*78}")
        print(f"{'task':<13}{'n':>5}{'oracle_acc':>12}{'oracle_tok':>12}   oracle路径分布(可解题上)")
        rows = []
        for t in TASKS:
            a = analyze(t, version)
            if not a:
                print(f"{t:<13}  (缺文件)")
                continue
            rows.append((t, a))
            print(f"{t:<13}{a['n']:>5}{a['oracle_acc']:>11.1%}{a['oracle_avg_tok']:>11.0f}   "
                  f"{fmt_dist(a['routes'], a['solvable'])}")
        # 各路单独 acc / avg token (帮助解读 oracle)
        print(f"\n  [各单路 acc / avg token 对照]")
        for t, a in rows:
            pp = a["per_path"]
            print(f"  {t:<11} " + "  ".join(
                f"{p}:{pp[p]['acc']:.0%}/{pp[p]['tok']:.0f}tok" for p in ("direct", "cot", "sql")))


if __name__ == "__main__":
    main()
