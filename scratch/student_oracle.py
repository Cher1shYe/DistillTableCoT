#!/usr/bin/env python3
"""学生(小模型)oracle —— 对同一题看三条路结果,逐题挑"最便宜的答对路"。

三路来源(都在测试集上,按 (task,id) 取交集对齐):
  direct = outputs/route_sft_v3/route_eval.rescored.json   (塌缩 route-SFT free 输出,~98%走direct)
           已有 correct / output_tokens,子集 n≈90(=teacher 可解子集)。
  cot    = outputs/<task>/predictions_qwen3_1.7b_cot_<task>.json   (cot 蒸馏学生,单轮)
  sql    = outputs/<task>/predictions_qwen3_1.7b_agent_<task>.json (agent 蒸馏学生,多轮)
判分全程 route_scoring(与 teacher oracle 同口径):
  direct 用其自带 correct;cot/agent 用 match_processed(processed_prediction)。
token(模型生成口径):
  direct 用 output_tokens;cot=tokenize(prediction);agent=Σ tokenize(各轮 response)。
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

print("加载 Qwen tokenizer ...")
from transformers import AutoTokenizer
TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)


def ntok(text):
    return len(TOK(str(text or ""), add_special_tokens=False)["input_ids"])


def load_direct():
    """route_eval → {(task,id): (correct, tokens)}。"""
    d = json.load(open("outputs/route_sft_v3/route_eval.rescored.json"))
    samples = d.get("samples") or d
    out = {}
    for s in samples:
        out[(s["task"], s["id"])] = (bool(s["correct"]), s.get("output_tokens") or 0)
    return out


def load_pred(variant, task, agent=False):
    """cot/agent 预测 → {id: (correct, tokens)}。"""
    p = f"outputs/{task}/predictions_qwen3_1.7b_{variant}_{task}.json"
    d = json.loads(open(p).read())
    preds = d["predictions"] if isinstance(d, dict) else d
    out = {}
    for r in preds:
        correct = bool(match_processed(task, r.get("processed_prediction", ""), r.get("reference")))
        if agent:
            tds = r.get("turn_details") or []
            tok = sum(ntok(t.get("response")) for t in tds) if tds else ntok(r.get("prediction"))
        else:
            tok = ntok(r.get("prediction"))
        out[r["id"]] = (correct, tok)
    return out


def main():
    direct = load_direct()
    print(f"{'task':<13}{'n':>5}{'oracle_acc':>12}{'oracle_tok':>12}   oracle路径分布(可解题上)")
    rows = []
    for task in TASKS:
        cot = load_pred("cot", task)
        sql = load_pred("agent", task, agent=True)
        ids = sorted(i for (t, i) in direct if t == task)  # route_eval 子集

        n = 0
        solvable = 0
        oracle_toks, oracle_routes = [], Counter()
        per_path = {p: {"c": 0, "tok": 0, "n": 0} for p in TIER}
        for i in ids:
            if i not in cot or i not in sql:
                continue
            n += 1
            correct = {"direct": direct[(task, i)][0], "cot": cot[i][0], "sql": sql[i][0]}
            tok = {"direct": direct[(task, i)][1], "cot": cot[i][1], "sql": sql[i][1]}
            for p in TIER:
                per_path[p]["c"] += correct[p]; per_path[p]["tok"] += tok[p]; per_path[p]["n"] += 1
            ok = [p for p in ("direct", "cot", "sql") if correct[p]]
            if ok:
                solvable += 1
                op = min(ok, key=lambda p: TIER[p])
                oracle_routes[op] += 1
                oracle_toks.append(tok[op])
        rows.append((task, n, solvable, oracle_routes, oracle_toks, per_path))
        oacc = solvable / n if n else 0
        otok = sum(oracle_toks) / len(oracle_toks) if oracle_toks else 0
        dist = "  ".join(f"{p}:{oracle_routes.get(p,0)}({oracle_routes.get(p,0)/solvable:.0%})"
                         if solvable else f"{p}:0" for p in ("direct", "cot", "sql"))
        print(f"{task:<13}{n:>5}{oacc:>11.1%}{otok:>11.0f}   {dist}")

    print(f"\n  [各单路 学生 acc / avg token 对照]")
    for task, n, solvable, oroutes, otoks, pp in rows:
        print(f"  {task:<11} " + "  ".join(
            f"{p}:{pp[p]['c']/pp[p]['n']:.0%}/{pp[p]['tok']/pp[p]['n']:.0f}tok"
            for p in ("direct", "cot", "sql")))


if __name__ == "__main__":
    main()
