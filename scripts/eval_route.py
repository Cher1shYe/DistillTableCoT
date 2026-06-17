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

SQL 真执行 (--exec_sql)：
    默认不执行 (沿用 MVP 简化，SQL 路径直接取学生生成的 <ANSWER>，模型在脑内编执行结果)。
    加 --exec_sql 后改为 **多轮 ReAct 执行，与教师 sql_agent 完全对齐**：
    每轮生成到 </SQL> 截停 → 在该题 SQLite 表上真执行 (复用 utils.table_to_sqlite/
    execute_sql，与 teacher 数据生成同一套) → 注入真实 <EXECUTION_RESULT> → 模型续写；
    若续写又出现新 <SQL> 则再执行一轮，直到模型写出 <ANSWER> 或到 --max_sql_turns
    (默认 5，连续空结果 2 次早停，均镜像教师 max_turns/max_empty)。
    表数据按 task+id 从原数据集加载 (--exec_split 对应评估集的 split，v1=test)。
    token 记账与教师一致：output_tokens 只累加**模型各轮生成**的 token，注入的执行结果
    属于下一轮输入不计；tool_calls = 整条轨迹实际执行 SQL 的次数。

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
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

import torch
import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils_train.route_scoring import score_answer

ROUTES = ("direct", "cot", "sql")
ROUTE_NAME = {"direct": "DIRECT", "cot": "COT", "sql": "SQL"}
ROUTE_TEXT = {r: f"<ROUTE>{ROUTE_NAME[r]}" for r in ROUTES}


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


# 未微调基座的 direct 对照：用与 teacher direct 同样的指令，让基座"直接作答不推理"，
# 输入用 route-SFT 完全相同的 input (表格+schema+问题)，保证只有"是否蒸馏"这一个变量。
BASE_DIRECT_SYSTEM = ("You read tables and answer questions directly, "
                      "with no explanation or reasoning. Output exactly 'Final Answer: <answer>'.")


def build_base_direct_prompt(tokenizer, user_content):
    msgs = [{"role": "system", "content": BASE_DIRECT_SYSTEM},
            {"role": "user", "content": user_content}]
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
def route_logprobs(model, tokenizer, prompt_text):
    """对三个候选路由各做一次 forward，算 '<ROUTE>X' 这段 token 的总 logprob。"""
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    out = {}
    for r in ROUTES:
        cand_ids = tokenizer(ROUTE_TEXT[r], add_special_tokens=False)["input_ids"]
        ids = torch.tensor([prompt_ids + cand_ids], device=model.device)
        logits = model(ids).logits[0]
        lp = 0.0
        for i, tid in enumerate(cand_ids):
            lp += torch.log_softmax(logits[len(prompt_ids) + i - 1].float(), dim=-1)[tid].item()
        out[r] = lp
    return out


def choose_route(lps, mode, temperature=1.0, log_prior=None):
    """根据三路 logprob 选路由。

    greedy   : argmax (模型有逐题判别力时才会分化)
    sample   : 按概率采样 (温度可调；只随机化路由，执行仍贪心)
    adjusted : 减去训练先验的 log 频率再 argmax (类不平衡的 logit 校正)
    """
    if mode == "adjusted" and log_prior:
        adj = {r: lps[r] - log_prior.get(r, 0.0) for r in ROUTES}
        return max(adj, key=adj.get)
    if mode == "sample":
        vals = [lps[r] / temperature for r in ROUTES]
        m = max(vals)
        ps = [math.exp(v - m) for v in vals]
        x = random.random() * sum(ps)
        cum = 0.0
        for r, p in zip(ROUTES, ps):
            cum += p
            if x <= cum:
                return r
        return ROUTES[-1]
    return max(lps, key=lps.get)


def load_train_priors(train_files):
    """从训练 jsonl 估计各任务的路由 log 频率 (adjusted 模式用)。"""
    cnt = defaultdict(Counter)
    for fp in train_files:
        if not os.path.exists(fp):
            continue
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    s = json.loads(line)
                    route = s.get("train_route") or s.get("oracle_path")
                    cnt[s.get("task", "?")][route] += 1
    priors = {}
    for task, c in cnt.items():
        n = sum(c.values())
        priors[task] = {r: math.log(max(c.get(r, 1), 1) / n) for r in ROUTES}
    return priors


@torch.no_grad()
def generate(model, tokenizer, prompt_text, max_new_tokens, stop_strings=None):
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    extra = {"stop_strings": stop_strings, "tokenizer": tokenizer} if stop_strings else {}
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False,                          # 贪心，评估可复现
        pad_token_id=tokenizer.pad_token_id,
        **extra,
    )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    n_out = int((gen_ids != tokenizer.pad_token_id).sum().item())
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, n_out


def load_tables(task, split, ids):
    """按 run_teacher_evaluate 同样的方式加载原数据集，返回 id->raw_table (id=样本序号)。"""
    from datasets import load_dataset
    from configs import TASK_CONFIGS
    ds = load_dataset(TASK_CONFIGS[task]["dataset_name"], split=split)
    max_id = max(ids)
    tables = {}
    for i, sample in enumerate(ds):
        if i in ids:
            tables[i] = (sample.get("table") or sample.get("table_content")
                         or sample.get("table_text"))
        if i >= max_id:
            break
    return tables


def exec_sql_and_continue(model, tokenizer, base_prompt, gen_head,
                          raw_table, task, max_new_tokens,
                          max_turns=5, max_empty=2):
    """多轮 SQL 执行，镜像教师 sql_agent：每轮停在 </SQL> → 真执行 → 注入真实
    <EXECUTION_RESULT> → 续写；续写又冒出新 <SQL> 就再执行一轮，直到写出 <ANSWER>
    或到 max_turns (连续空结果 max_empty 次也早停)。

    gen_head 是首段 (已停在第一个 </SQL>)。整条轨迹用同一个 SQLite 连接。
    返回 (assembled_text, n_continue_tokens, tool_calls)：
      - assembled_text   : 首段之后的全部轨迹 (注入的执行结果 + 各轮续写)；
      - n_continue_tokens: 只累加**模型续写**的 token (注入结果不计)，与教师 completion 口径一致；
                           gen_head 自身 token 已在调用处计入，这里不重复。
      - tool_calls       : 整条轨迹实际执行 SQL 的次数。
    """
    from utils import table_to_sqlite, execute_sql

    conn = None
    if raw_table is not None:
        conn, _ = table_to_sqlite(raw_table, task_name=task)

    assembled = gen_head      # 首段已停在 </SQL>，无需再截断
    n_cont_total = 0
    tool_calls = 0
    empty_count = 0

    for _ in range(max_turns):
        # 取轨迹里最后一个 (=最新、尚未执行的) SQL
        sqls = re.findall(r"<SQL>\s*(.*?)\s*</SQL>", assembled, re.DOTALL | re.IGNORECASE)
        sql = sqls[-1].strip() if sqls else None
        if not sql:
            feedback = "SQL Error: no <SQL> block found."
        elif conn is None:
            feedback = "SQL Error: table unavailable."
        else:
            tool_calls += 1
            _, feedback = execute_sql(conn, sql)

        assembled += f"\n<EXECUTION_RESULT>\n{feedback}\n</EXECUTION_RESULT>\n"
        if "no results" in str(feedback).lower():
            empty_count += 1

        # 连续空结果到上限 → 不再停在 </SQL>，让模型基于已有结果收尾出 <ANSWER>
        last_round = empty_count >= max_empty
        cont, n_cont = generate(model, tokenizer, base_prompt + assembled, max_new_tokens,
                                stop_strings=None if last_round else ["</SQL>"])
        n_cont_total += n_cont
        assembled += cont
        if last_round or not re.search(r"<SQL>", cont, re.IGNORECASE):
            break             # 续写没有新 SQL = 已收尾 (<ANSWER>)

    if conn is not None:
        conn.close()
    return assembled, n_cont_total, tool_calls


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def evaluate(model, tokenizer, samples, max_new_tokens,
             route_mode="free", route_temperature=1.0, priors=None,
             exec_sql=False, tables=None, max_sql_turns=5, force_route=None):
    """逐样本生成 + 解析 + 判对，返回 per-sample 明细。

    route_mode != free 时：先用三次 forward 给路由打分并按模式选路，
    再把 '<ROUTE>X</ROUTE>' 强制作为前缀让模型贪心生成后续轨迹。
    force_route 设为 direct/cot/sql 时：跳过路由打分，强制所有题走该路 (oracle 分析用)。
    exec_sql 时：生成在 </SQL> 截停，真执行后注入结果再续写 (见模块 docstring)。
    """
    stops = ["</SQL>"] if exec_sql else None
    rows = []
    for s in tqdm.tqdm(samples, desc="eval-route"):
        task = s.get("task", "")
        raw_table = (tables or {}).get(task, {}).get(s.get("id"))
        real_tc = None  # exec_sql 时的真实执行次数

        if route_mode == "base_direct":
            # 未微调基座 + direct 指令，强制 direct 路径 (零蒸馏对照)
            prompt_text = build_base_direct_prompt(tokenizer, s["input"])
            gen_text, n_out = generate(model, tokenizer, prompt_text, max_new_tokens)
            pred_route = "direct"
            _, pred_answer = parse_output(gen_text)
            correct = score_answer(task, pred_answer, s.get("reference"))
            oracle_path = s.get("oracle_path")
            pc = s.get("paths_correct", {})
            rows.append({
                "id": s.get("id"), "task": task, "pred_route": pred_route,
                "pred_answer": pred_answer, "reference": s.get("reference"),
                "correct": bool(correct), "oracle_path": oracle_path,
                "oracle_solvable": any(pc.get(r) for r in ROUTES),
                "output_tokens": n_out, "tool_calls": 0,
                "raw_output": gen_text[:1000],
            })
            continue

        prompt_text = build_prompt_text(tokenizer, s["input"])
        if route_mode == "free" and force_route is None:
            gen_text, n_out = generate(model, tokenizer, prompt_text, max_new_tokens,
                                       stop_strings=stops)
            if exec_sql and re.search(r"<SQL>", gen_text, re.IGNORECASE):
                gen_text, n_cont, real_tc = exec_sql_and_continue(
                    model, tokenizer, prompt_text, gen_text,
                    raw_table, task, max_new_tokens, max_turns=max_sql_turns)
                n_out += n_cont
            elif exec_sql:
                real_tc = 0
            pred_route, pred_answer = parse_output(gen_text)
        else:
            if force_route is not None:
                pred_route = force_route          # 强制走指定路径 (oracle 分析)
            else:
                lps = route_logprobs(model, tokenizer, prompt_text)
                log_prior = (priors or {}).get(task)
                pred_route = choose_route(lps, route_mode, route_temperature, log_prior)
            prefix = f"<ROUTE>{ROUTE_NAME[pred_route]}</ROUTE>\n"
            gen_text, n_gen = generate(model, tokenizer, prompt_text + prefix,
                                       max_new_tokens, stop_strings=stops)
            n_out = n_gen + len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
            if exec_sql and pred_route == "sql":
                gen_text, n_cont, real_tc = exec_sql_and_continue(
                    model, tokenizer, prompt_text + prefix, gen_text,
                    raw_table, task, max_new_tokens, max_turns=max_sql_turns)
                n_out += n_cont
            elif exec_sql:
                real_tc = 0
            gen_text = prefix + gen_text
            _, pred_answer = parse_output(gen_text)

        correct = score_answer(task, pred_answer, s.get("reference"))
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
            "tool_calls": real_tc if real_tc is not None else (1 if pred_route == "sql" else 0),
            "raw_output": gen_text[:1000],
        })
    return rows


def report(rows, title):
    n = len(rows)
    if n == 0:
        return None
    acc = mean([r["correct"] for r in rows])
    oracle_acc = mean([r["oracle_solvable"] for r in rows])
    # route accuracy 只在"有 oracle 路"的可解题上统计 (no-solution 题 oracle_path=None,
    # 无正确路可对，纳入会系统性压低)
    solvable_rows = [r for r in rows if r["oracle_path"] is not None]
    route_acc = mean([r["pred_route"] == r["oracle_path"] for r in solvable_rows]) if solvable_rows else 0.0
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
    print(f"  route accuracy       : {route_acc:.1%}   (仅可解题 n={len(solvable_rows)})")
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
    ap.add_argument("--route_mode", default="free",
                    choices=["free", "greedy", "sample", "adjusted", "base_direct"],
                    help="free=模型自由生成(默认)；greedy/sample/adjusted=先给三路打分选路由再强制前缀执行；"
                         "base_direct=未微调基座+direct指令强制走direct (零蒸馏对照，--model_path 传基座)")
    ap.add_argument("--route_temperature", type=float, default=1.0,
                    help="sample 模式的路由采样温度")
    ap.add_argument("--force_route", choices=["direct", "cot", "sql"], default=None,
                    help="强制所有题走指定路径 (oracle 分析):跳过路由打分，直接以 <ROUTE>X> 为前缀生成。"
                         "配合 --exec_sql 时 sql 路真执行。三条路各跑一次 → scripts/build_student_oracle.py 聚合")
    ap.add_argument("--train_files", nargs="*",
                    default=[f"outputs/{t}/route_sft_v2.jsonl"
                             for t in ("hitab", "fetaqa", "tabfact", "wikitableqa")],
                    help="adjusted 模式用来估计训练先验的 jsonl")
    ap.add_argument("--seed", type=int, default=42, help="sample 模式的随机种子")
    ap.add_argument("--exec_sql", action="store_true",
                    help="SQL 路径真执行：多轮 </SQL> 截停 → SQLite 执行 → 注入结果 → 续写 (镜像教师)")
    ap.add_argument("--exec_split", default="test",
                    help="exec_sql 加载表数据用的 split (评估 v1 文件 → test)")
    ap.add_argument("--max_sql_turns", type=int, default=5,
                    help="exec_sql 多轮执行的最大轮数 (镜像教师 max_turns，默认 5)")
    args = ap.parse_args()
    random.seed(args.seed)

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

    priors = None
    if args.route_mode == "adjusted":
        priors = load_train_priors(args.train_files)
        for t, p in priors.items():
            print(f"训练先验 {t}: " + " ".join(f"{r}={math.exp(v):.0%}" for r, v in p.items()))

    tables = None
    if args.exec_sql:
        tables = {}
        for task, samples in by_task.items():
            ids = {s.get("id") for s in samples}
            print(f"📊 加载表数据 {task} (split={args.exec_split}, n={len(ids)}) ...")
            tables[task] = load_tables(task, args.exec_split, ids)

    all_rows = []
    summaries = []
    for task, samples in by_task.items():
        rows = evaluate(model, tokenizer, samples, args.max_new_tokens,
                        route_mode=args.route_mode,
                        route_temperature=args.route_temperature, priors=priors,
                        exec_sql=args.exec_sql, tables=tables,
                        max_sql_turns=args.max_sql_turns, force_route=args.force_route)
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
