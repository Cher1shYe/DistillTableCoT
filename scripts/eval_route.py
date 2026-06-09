#!/usr/bin/env python3
"""
(e) route-aware 学生模型评估 (对应 cost_aware.pdf §7 评价指标 / §15 实验三)。

让训练后的学生在评估集上自己生成 <ROUTE>X</ROUTE> + 轨迹 + <ANSWER>...</ANSWER>，
解析它选的路径与答案，对照 oracle 算一整套 cost-aware 指标：

  基础性能:
    - task accuracy        : 最终答案是否正确 (is_match；fetaqa 用 ROUGE-L>=0.3)
  成本指标:
    - avg output tokens    : 平均生成 token 数 (学会克制 → 远低于 always-CoT/SQL)
    - avg tool_calls       : = P(route=SQL)  (MVP 不真执行 SQL，按是否选 SQL 路径计成本)
  路由指标:
    - route accuracy       : 选的路径 == oracle_path 的比例
    - oracle gap           : oracle 上界 (至少一条路径对) − 学生 task accuracy
    - overthinking rate    : oracle=direct 却选了 cot/sql 的比例 (该省没省)
    - underthinking rate   : oracle∈{cot,sql} 却选了 direct 且答错的比例 (该花没花→错)

输入评估集 = scripts/build_route_sft.py 产出的 jsonl (含 input/oracle_path/reference/paths_correct)，
通常用 test 版 (v1)。

⚠️ MVP 简化: 不真执行 SQL (学生 SQL 能力弱、且真执行需 agent loop)，
   SQL 路径直接取学生生成的 <ANSWER>。真执行版留作后续 (--exec_sql TODO)。

用法:
    # LoRA adapter (训练产物 final_model 里有 adapter_config.json → 自动识别)
    python3 scripts/eval_route.py \
        --model_path outputs/models/Qwen3-1.7B-route-sft/final_model \
        --eval_files outputs/hitab/route_sft_v1.jsonl
    # 一次评多个任务
    python3 scripts/eval_route.py --model_path ... \
        --eval_files outputs/hitab/route_sft_v1.jsonl outputs/tabfact/route_sft_v1.jsonl
"""
import argparse
import json
import os
import re
import sys

import torch
import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils_train.eval_utils import is_match

ROUTES = ("direct", "cot", "sql")


def load_model(model_path, base_model, dtype):
    """加载训练后的学生：自动识别 LoRA adapter vs 全量模型。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if is_lora:
        from peft import PeftModel
        print(f"🔗 LoRA adapter → 基座 {base_model} + adapter {model_path}")
        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch_dtype, device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
    else:
        print(f"📦 全量模型 → {model_path}")
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map="auto",
            trust_remote_code=True,
        )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok


def build_prompt_text(tokenizer, user_content):
    """与训练 (RouteSFTDataset) 一致：enable_thinking=False + add_generation_prompt。"""
    msgs = [{"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def parse_output(text):
    """从学生输出里解析 (route, answer)。"""
    route = None
    m = re.search(r"<ROUTE>\s*([A-Za-z_]+)\s*</ROUTE>", text, re.IGNORECASE)
    if m:
        route = m.group(1).strip().lower()
        if route not in ROUTES:
            route = None

    ms = list(re.finditer(r"<ANSWER>\s*(.*?)\s*</ANSWER>", text, re.DOTALL | re.IGNORECASE))
    if ms:
        ans = ms[-1].group(1).strip()
    elif "Final Answer:" in text:
        ans = text.split("Final Answer:")[-1].strip()
    else:
        ans = text.strip()
    return route, ans


@torch.no_grad()
def generate(model, tokenizer, prompt_text, max_new_tokens):
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False,                          # 贪心，评估可复现
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    n_out = int((gen_ids != tokenizer.pad_token_id).sum().item())
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, n_out


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def evaluate(model, tokenizer, samples, max_new_tokens):
    """逐样本生成 + 解析 + 判对，返回 per-sample 明细。"""
    rows = []
    for s in tqdm.tqdm(samples, desc="eval-route"):
        task = s.get("task", "")
        rouge_threshold = 0.3 if task == "fetaqa" else None

        prompt_text = build_prompt_text(tokenizer, s["input"])
        gen_text, n_out = generate(model, tokenizer, prompt_text, max_new_tokens)
        pred_route, pred_answer = parse_output(gen_text)

        correct = is_match(pred_answer, s.get("reference"), rouge_threshold=rouge_threshold)
        oracle_path = s.get("oracle_path")
        pc = s.get("paths_correct", {})
        oracle_solvable = any(pc.get(r) for r in ROUTES)

        rows.append({
            "id": s.get("id"),
            "task": task,
            "pred_route": pred_route,
            "pred_answer": pred_answer,
            "reference": s.get("reference"),
            "correct": bool(correct),
            "oracle_path": oracle_path,
            "oracle_solvable": bool(oracle_solvable),
            "output_tokens": n_out,
            "tool_calls": 1 if pred_route == "sql" else 0,
            "raw_output": gen_text[:1000],
        })
    return rows


def report(rows, title):
    n = len(rows)
    if n == 0:
        return None
    acc = mean([r["correct"] for r in rows])
    oracle_acc = mean([r["oracle_solvable"] for r in rows])
    route_acc = mean([r["pred_route"] == r["oracle_path"] for r in rows])
    avg_tok = mean([r["output_tokens"] for r in rows])
    avg_tool = mean([r["tool_calls"] for r in rows])

    dist = {rt: sum(1 for r in rows if r["pred_route"] == rt) for rt in ROUTES}
    n_unparsed = sum(1 for r in rows if r["pred_route"] is None)

    over = [r for r in rows if r["oracle_path"] == "direct"]
    overthink = mean([r["pred_route"] not in (None, "direct") for r in over]) if over else 0.0
    under = [r for r in rows if r["oracle_path"] in ("cot", "sql")]
    underthink = mean([(r["pred_route"] == "direct" and not r["correct"]) for r in under]) if under else 0.0

    print(f"\n===== {title}  (n={n}) =====")
    print(f"  task accuracy        : {acc:.1%}")
    print(f"  oracle 上界 (≥1路对)  : {oracle_acc:.1%}   → oracle gap = {oracle_acc - acc:+.1%}")
    print(f"  route accuracy       : {route_acc:.1%}")
    print(f"  avg output tokens    : {avg_tok:.0f}")
    print(f"  avg tool_calls(=SQL) : {avg_tool:.2f}")
    print(f"  路由分布(预测)        : direct={dist['direct']} cot={dist['cot']} sql={dist['sql']} 未解析={n_unparsed}")
    print(f"  overthinking rate    : {overthink:.1%}   (oracle=direct 却走了贵路径)")
    print(f"  underthinking rate   : {underthink:.1%}   (oracle贵路径 却走 direct 且答错)")
    return {
        "title": title, "n": n,
        "task_accuracy": acc, "oracle_accuracy": oracle_acc,
        "oracle_gap": oracle_acc - acc, "route_accuracy": route_acc,
        "avg_output_tokens": avg_tok, "avg_tool_calls": avg_tool,
        "route_distribution": dist, "n_unparsed": n_unparsed,
        "overthinking_rate": overthink, "underthinking_rate": underthink,
    }


def main():
    ap = argparse.ArgumentParser(description="(e) route-aware 学生模型评估")
    ap.add_argument("--model_path", required=True,
                    help="训练产物目录 (LoRA adapter 或全量模型)")
    ap.add_argument("--base_model", default="Qwen/Qwen3-1.7B",
                    help="LoRA 时的基座模型 (全量模型忽略)")
    ap.add_argument("--eval_files", nargs="+", required=True,
                    help="评估用 route_sft jsonl (通常 test 版 v1)，可多个任务")
    ap.add_argument("--max_new_tokens", type=int, default=1536)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 条 (调试用)，0=全部")
    ap.add_argument("--out_name", default="route_eval.json")
    args = ap.parse_args()

    # 加载评估样本 (按 task 分组)
    by_task = {}
    for fp in args.eval_files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                s = json.loads(line)
                by_task.setdefault(s.get("task", "unknown"), []).append(s)

    if args.limit:
        for t in by_task:
            by_task[t] = by_task[t][:args.limit]

    model, tokenizer = load_model(args.model_path, args.base_model, args.dtype)

    all_rows = []
    summaries = []
    for task, samples in by_task.items():
        rows = evaluate(model, tokenizer, samples, args.max_new_tokens)
        summaries.append(report(rows, f"task={task}"))
        all_rows.extend(rows)

    if len(by_task) > 1:
        summaries.append(report(all_rows, "ALL TASKS"))

    out_path = os.path.join(os.path.dirname(args.eval_files[0]), args.out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summaries": summaries, "samples": all_rows}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n✅ 明细 + 汇总已保存: {out_path}")


if __name__ == "__main__":
    main()
