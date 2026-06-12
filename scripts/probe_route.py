#!/usr/bin/env python3
"""
路由信号探针：诊断 route-SFT 模型"全选 direct"是
  (a) 有判别信号但被多数类先验压住(贪心 argmax 永远是 DIRECT)，还是
  (b) 压根没学到逐题信号。

做法：对每道评估题，把 prompt 渲染成与训练一致的格式(enable_thinking=False)，
分别强制接上 "<ROUTE>DIRECT" / "<ROUTE>COT" / "<ROUTE>SQL"，
算三个路由词的条件 log 概率(一次 forward 拿全部 logits，无需生成)。

输出：
  - 每题三路 logprob + 贪心选择
  - 按 oracle 路径分组的平均 logprob 差(信号强弱)
  - logit 调整(减去训练集先验 log 频率)后的路由选择与 route accuracy
    —— 若调整后 route accuracy 明显高于全选 direct 的基线，说明是情形(a)，
       不用重训，推理时做先验校正即可。

用法 (Mac MPS / CUDA / CPU 自适应):
    python3 scripts/probe_route.py \
        --model_path outputs/models/route_sft_v2_final \
        --eval_files outputs/hitab/route_sft_v1.jsonl outputs/fetaqa/route_sft_v1.jsonl \
                     outputs/tabfact/route_sft_v1.jsonl outputs/wikitableqa/route_sft_v1.jsonl
"""
import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

import torch
import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ROUTES = ["direct", "cot", "sql"]
ROUTE_TEXT = {"direct": "<ROUTE>DIRECT", "cot": "<ROUTE>COT", "sql": "<ROUTE>SQL"}


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_model(model_path, base_model, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    if is_lora:
        from peft import PeftModel
        print(f"🔗 LoRA: 基座 {base_model} + adapter {model_path}")
        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    else:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True)
    model.to(device).eval()
    return model, tok


def build_prompt_ids(tok, user_content):
    msgs = [{"role": "user", "content": user_content}]
    try:
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def route_logprobs(model, tok, prompt_ids, device):
    """对三个候选路由，算 '<ROUTE>X' 这段 token 的总 logprob (一次 forward / 候选)。"""
    out = {}
    for r in ROUTES:
        cand_ids = tok(ROUTE_TEXT[r], add_special_tokens=False)["input_ids"]
        ids = torch.tensor([prompt_ids + cand_ids], device=device)
        logits = model(ids).logits[0]
        lp = 0.0
        for i, tid in enumerate(cand_ids):
            pos = len(prompt_ids) + i - 1  # 预测第 pos+1 个 token 用 pos 处 logits
            lp += torch.log_softmax(logits[pos].float(), dim=-1)[tid].item()
        out[r] = lp
    return out


def main():
    ap = argparse.ArgumentParser(description="route 信号探针")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--eval_files", nargs="+", required=True)
    ap.add_argument("--train_files", nargs="+",
                    default=[f"outputs/{t}/route_sft_v2.jsonl"
                             for t in ("hitab", "fetaqa", "tabfact", "wikitableqa")],
                    help="用于估计训练先验(各路由的 log 频率)做 logit 调整")
    ap.add_argument("--limit", type=int, default=0, help="每任务只探前 N 条，0=全部")
    ap.add_argument("--out", default="outputs/route_probe.json")
    args = ap.parse_args()

    device, dtype = pick_device()
    print(f"device={device} dtype={dtype}")
    model, tok = load_model(args.model_path, args.base_model, device, dtype)

    # 训练先验 (按任务分别估计：模型可能学的是任务条件先验)
    prior = defaultdict(Counter)
    for fp in args.train_files:
        if not os.path.exists(fp):
            continue
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    s = json.loads(line)
                    prior[s.get("task", "?")][s["oracle_path"]] += 1
    log_prior = {}
    for task, c in prior.items():
        n = sum(c.values())
        log_prior[task] = {r: math.log(max(c.get(r, 1), 1) / n) for r in ROUTES}
        print(f"训练先验 {task}: " + " ".join(f"{r}={c.get(r,0)/n:.0%}" for r in ROUTES))

    # 评估样本
    samples = []
    by_task_count = Counter()
    for fp in args.eval_files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    s = json.loads(line)
                    if args.limit and by_task_count[s.get("task")] >= args.limit:
                        continue
                    by_task_count[s.get("task")] += 1
                    samples.append(s)

    rows = []
    for s in tqdm.tqdm(samples, desc="probe"):
        prompt_ids = build_prompt_ids(tok, s["input"])
        lps = route_logprobs(model, tok, prompt_ids, device)
        task = s.get("task", "?")
        lp_adj = {r: lps[r] - log_prior.get(task, {}).get(r, 0.0) for r in ROUTES}
        rows.append({
            "id": s.get("id"), "task": task, "oracle_path": s.get("oracle_path"),
            "logprob": lps,
            "greedy_route": max(lps, key=lps.get),
            "adjusted_route": max(lp_adj, key=lp_adj.get),
        })

    # ===== 报告 =====
    for task in sorted({r["task"] for r in rows}):
        sub = [r for r in rows if r["task"] == task]
        n = len(sub)
        greedy_acc = sum(r["greedy_route"] == r["oracle_path"] for r in sub) / n
        adj_acc = sum(r["adjusted_route"] == r["oracle_path"] for r in sub) / n
        gdist = Counter(r["greedy_route"] for r in sub)
        adist = Counter(r["adjusted_route"] for r in sub)
        print(f"\n===== {task} (n={n}) =====")
        print(f"  贪心:   route_acc={greedy_acc:.1%}  分布={dict(gdist)}")
        print(f"  先验校正: route_acc={adj_acc:.1%}  分布={dict(adist)}")
        # 信号强弱：oracle 组间的 (lp[cot]-lp[direct]) 均值差
        for alt in ("cot", "sql"):
            d_or = [r["logprob"][alt] - r["logprob"]["direct"]
                    for r in sub if r["oracle_path"] == alt]
            d_not = [r["logprob"][alt] - r["logprob"]["direct"]
                     for r in sub if r["oracle_path"] != alt]
            if d_or and d_not:
                m1, m2 = sum(d_or) / len(d_or), sum(d_not) / len(d_not)
                print(f"  信号[{alt}]: oracle={alt} 时 Δlp={m1:+.2f} | 其他题 Δlp={m2:+.2f}"
                      f"  → 组间差 {m1 - m2:+.2f} ({'有信号' if m1 - m2 > 0.3 else '弱/无'})")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"\n✅ 明细已保存: {args.out}")


if __name__ == "__main__":
    main()
