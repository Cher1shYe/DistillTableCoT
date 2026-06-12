#!/usr/bin/env python3
"""
(d) route-aware SFT 数据构造 (对应 cost_aware.pdf 阶段三 / §6.1 / §10)。

把同一批样本 (同 task/split/num_samples → 同 id) 上 direct/cot/sql 三条教师路径 join 起来：
  对每个样本：
    - 算每条路径是否正确 (is_match) + 成本 (completion_tokens / tool_calls)
    - 选 oracle = 答对路径中"成本最低"的那条
                  (主序按路径档位 tier: direct<cot<sql, 次序按 completion_tokens)
    - 按 oracle_path 把 "输入 → 输出" 拼成一条 SFT 样本：
          input  = Table(Markdown) + Schema(SQLite) + Question   (统一带 schema，
                   这样模型选了 SQL 路径才有信息写查询；取自 sql 路径 turn0 的 prompt)
          target = <ROUTE>X</ROUTE> + 该路径轨迹 + <ANSWER>...</ANSWER>

这是 **teacher-oracle** 版：路由标签与轨迹都来自教师 (DeepSeek)。
student-oracle 版 (用学生自己跑的三路径) 复用同一脚本，把输入文件换成学生预测即可。

为什么只保留"至少一条路径答对"的样本：
    没有任何路径答对 → 没有可信的 oracle 路由标签，拿它训练只会教模型瞎选路。

防塌缩 (--direct_keep_ratio)：
    简单数据集里 direct 可能占 70%+，全喂进去会让 router 退化成"无脑 DIRECT"。
    --direct_keep_ratio 0.5 表示只保留 50% 的 direct 题继续用 direct 监督；
    超出部分的题 **不删除**，改用该题其他答对路径的轨迹做监督 (优先 sql，最稀缺；
    其次 cot)；若该题只有 direct 答对则无法改挂、仍留 direct。
    总数据量不变。jsonl 里 oracle_path 仍是真 oracle (评估口径不变)，
    train_route 是实际监督路径。

输出: outputs/<task>/route_sft_v<version>.jsonl   (每行一个训练样本)

用法:
    python3 scripts/build_route_sft.py --task hitab --version 1                 # test-100 (验证管线)
    python3 scripts/build_route_sft.py --task hitab --version 2                 # train (正式训练数据)
    python3 scripts/build_route_sft.py --task hitab --version 2 --direct_keep_ratio 0.5
    python3 scripts/build_route_sft.py --task hitab --files direct=... cot=... sql=...
"""
import argparse
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils_train.eval_utils import is_match


# 成本阶梯：数字越小越便宜，作为 oracle 选择的主排序键
PATH_TIER = {"direct": 0, "cot": 1, "sql": 2}
# label -> 默认文件名后缀 (sql 的教师文件叫 sql_agent)
DEFAULT_FILENAME = {
    "direct": "R1_prediction_direct_v{v}.json",
    "cot": "R1_prediction_cot_v{v}.json",
    "sql": "R1_prediction_sql_agent_v{v}.json",
}
# target 里 ROUTE 标签用的大写名
ROUTE_NAME = {"direct": "DIRECT", "cot": "COT", "sql": "SQL"}


def load_path_file(path):
    """加载一个预测文件，兼容 list 与 {'predictions': [...]} 两种格式，返回 id->rec。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("predictions", [])
    return {rec["id"]: rec for rec in data}


def completion_tokens(rec):
    u = rec.get("usage") or {}
    return u.get("completion_tokens") or 0


def extract_final_answer(rec):
    """从一条预测记录里抽出"最终答案"文本 (给 <ANSWER> 用)。

    direct/sql 有显式 'Final Answer:' 标记 → 取其后；
    cot 是 R1 散文回答无标记 → 回退到 processed_prediction (去掉 list 括号)。
    """
    pred = str(rec.get("prediction", "") or "")
    if "Final Answer:" in pred:
        return pred.split("Final Answer:")[-1].strip()
    pp = str(rec.get("processed_prediction", "") or "").strip()
    if pp.startswith("[") and pp.endswith("]"):
        pp = pp[1:-1].strip()
    return pp


def extract_last_sql(turns):
    """从多轮 turn_details 里取最后一个 ```sql ... ``` 代码块 (最接近最终答案那条)。"""
    sqls = []
    for t in turns:
        m = re.search(r"```sql\s*(.*?)\s*```", t.get("response", "") or "",
                      re.DOTALL | re.IGNORECASE)
        if m:
            sqls.append(m.group(1).strip())
    return sqls[-1] if sqls else None


def extract_last_feedback(turns):
    """SQL 执行结果存在"下一轮 prompt 的 Feedback: 段"里，取最后一个。"""
    for t in reversed(turns):
        p = t.get("prompt", "") or ""
        if "Feedback:" in p:
            seg = p.split("Feedback:")[-1]
            seg = seg.split("Check if")[0].strip()
            return seg
    return None


def build_input(sql_rec):
    """统一输入 = sql 路径 turn0 的 prompt (已是 Table+Schema+Question 的完整拼接)。

    取自 sql 路径是因为只有它带 SQLite schema；direct/cot 的教师调用没存 prompt。
    若 sql 路径没有 turn_details (极少数建库失败) → 返回 None，该样本跳过。
    """
    turns = sql_rec.get("turn_details") or []
    if not turns:
        return None
    return turns[0].get("prompt") or None


def build_target(oracle_path, recs, max_reasoning_chars=0):
    """按 oracle_path 拼出训练目标文本。

    max_reasoning_chars > 0 时截断 cot 的 R1 推理链 (R1 对 1.7B 太长太绕，
    且 max_target_length 开太大训练慢)。截断只影响 <REASONING> 段，<ANSWER> 另拼，不丢答案。
    """
    name = ROUTE_NAME[oracle_path]
    if oracle_path == "direct":
        ans = extract_final_answer(recs["direct"])
        return f"<ROUTE>{name}</ROUTE>\n<ANSWER>{ans}</ANSWER>"

    if oracle_path == "cot":
        c = recs["cot"]
        reasoning = (c.get("reasoning") or "").strip()
        if max_reasoning_chars and len(reasoning) > max_reasoning_chars:
            reasoning = reasoning[:max_reasoning_chars].rstrip() + " ..."
        ans = extract_final_answer(c)
        return (f"<ROUTE>{name}</ROUTE>\n"
                f"<REASONING>{reasoning}</REASONING>\n"
                f"<ANSWER>{ans}</ANSWER>")

    # sql
    s = recs["sql"]
    turns = s.get("turn_details") or []
    sql = extract_last_sql(turns) or ""
    feedback = extract_last_feedback(turns)
    ans = extract_final_answer(s)
    parts = [f"<ROUTE>{name}</ROUTE>", f"<SQL>\n{sql}\n</SQL>"]
    if feedback:
        parts.append(f"<EXECUTION_RESULT>\n{feedback}\n</EXECUTION_RESULT>")
    parts.append(f"<ANSWER>{ans}</ANSWER>")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description="(d) route-aware SFT 数据构造")
    ap.add_argument("--task", required=True,
                    choices=["wikitableqa", "tabfact", "fetaqa", "hitab"])
    ap.add_argument("--version", type=int, default=1,
                    help="读 outputs/<task>/R1_prediction_*_v<version>.json (v1=test, v2=train 约定)")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--files", nargs="*", default=None,
                    help="显式 label=path 覆盖默认文件名，如 direct=... cot=... sql=...")
    ap.add_argument("--direct_keep_ratio", type=float, default=1.0,
                    help="保留多少比例的 direct 题继续用 direct 监督；超出部分改挂到"
                         "该题其他答对的路径 (优先 sql 其次 cot)，不删数据。默认 1.0 不调整")
    ap.add_argument("--max_reasoning_chars", type=int, default=4000,
                    help="截断 cot 的 R1 推理链字符数 (≈1000 token)，0=不截断")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_name", default=None,
                    help="自定义输出文件名，默认 route_sft_v<version>.jsonl")
    args = ap.parse_args()

    random.seed(args.seed)
    rouge_threshold = 0.3 if args.task == "fetaqa" else None
    task_dir = os.path.join(args.output_dir, args.task)

    # ---- 收集三路径文件 ----
    path_files = {}
    if args.files:
        for spec in args.files:
            label, fp = spec.split("=", 1)
            path_files[label] = fp
    else:
        for label, tmpl in DEFAULT_FILENAME.items():
            path_files[label] = os.path.join(task_dir, tmpl.format(v=args.version))

    missing = [f"{l}={p}" for l, p in path_files.items() if not os.path.exists(p)]
    if missing:
        print("❌ 缺少路径文件：\n  " + "\n  ".join(missing))
        return

    print(f"==== build route-SFT | task={args.task} | version={args.version} ====")
    for label in ("direct", "cot", "sql"):
        print(f"  [{label:<6}] {path_files[label]}")

    path_data = {l: load_path_file(fp) for l, fp in path_files.items()}
    common_ids = sorted(set.intersection(*[set(d.keys()) for d in path_data.values()]))
    n = len(common_ids)
    if n == 0:
        print("❌ 三路径无共同 id，检查是否同 split/num_samples")
        return

    # ---- 逐样本：算 oracle + 构造 SFT 样本 ----
    samples = []
    no_solution = 0          # 三路径全错，无 oracle
    no_input = 0             # sql 路径缺 turn_details，取不到统一输入
    route_dist = {"direct": 0, "cot": 0, "sql": 0}

    for sid in common_ids:
        recs = {l: path_data[l][sid] for l in ("direct", "cot", "sql")}

        correct = []
        for l in ("direct", "cot", "sql"):
            r = recs[l]
            if is_match(r.get("processed_prediction", ""), r.get("reference"),
                        rouge_threshold=rouge_threshold):
                correct.append(l)
        if not correct:
            no_solution += 1
            continue

        oracle_path = min(correct, key=lambda l: (PATH_TIER[l], completion_tokens(recs[l])))

        inp = build_input(recs["sql"])
        if inp is None:
            no_input += 1
            continue

        samples.append({
            "id": sid,
            "task": args.task,
            "oracle_path": oracle_path,
            "train_route": oracle_path,     # 实际监督路径，下面可能被改挂
            "input": inp,
            "target": None,                 # 路由确定后统一构建
            # 调试/评估用的元信息（训练时只取 input/target；评估 (e) 用 reference/paths_correct）
            "reference": recs[oracle_path].get("reference"),
            "paths_correct": {l: (l in correct) for l in ("direct", "cot", "sql")},
            "oracle_cost": {
                "completion_tokens": completion_tokens(recs[oracle_path]),
                "tool_calls": recs[oracle_path].get("tool_calls") or 0,
            },
            "_recs": recs,                  # 临时字段，落盘前删除
        })
        route_dist[oracle_path] += 1

    # ---- 可选：改挂超额 direct 防塌缩 (不删数据) ----
    if args.direct_keep_ratio < 1.0:
        directs = [s for s in samples if s["oracle_path"] == "direct"]
        keep_k = int(round(len(directs) * args.direct_keep_ratio))
        # 可改挂 = 该题还有别的路径答对；挂到当前更稀缺的那条 (动态平衡 sql/cot)
        movable = [s for s in directs
                   if s["paths_correct"]["sql"] or s["paths_correct"]["cot"]]
        random.shuffle(movable)
        cnt = {r: sum(1 for s in samples if s["oracle_path"] == r) for r in ("sql", "cot")}
        moved = {"sql": 0, "cot": 0}
        for s in movable[:max(0, len(directs) - keep_k)]:
            options = [r for r in ("sql", "cot") if s["paths_correct"][r]]
            new_route = min(options, key=lambda r: cnt[r])
            s["train_route"] = new_route
            cnt[new_route] += 1
            moved[new_route] += 1
        stuck = (len(directs) - keep_k) - sum(moved.values())
        print(f"\n⚖️  改挂超额 direct (ratio={args.direct_keep_ratio}): "
              f"→sql {moved['sql']} 条, →cot {moved['cot']} 条"
              + (f", {stuck} 条仅 direct 可解无法改挂" if stuck > 0 else ""))

    # ---- 按最终监督路径构建 target ----
    route_dist = {"direct": 0, "cot": 0, "sql": 0}
    for s in samples:
        s["target"] = build_target(s["train_route"], s.pop("_recs"),
                                   args.max_reasoning_chars)
        route_dist[s["train_route"]] += 1

    # ---- 落盘 jsonl ----
    out_name = args.out_name or f"route_sft_v{args.version}.jsonl"
    out_path = os.path.join(task_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ---- 统计 ----
    total = len(samples)
    print(f"\n样本数 (共同 id): {n}")
    print(f"  丢弃 (三路全错, 无 oracle): {no_solution}")
    print(f"  丢弃 (sql 缺输入):          {no_input}")
    print(f"  最终训练样本:               {total}")
    print("\n路径(ROUTE)分布：")
    for l in ("direct", "cot", "sql"):
        c = route_dist[l]
        print(f"  {l:<6}: {c:>5}  ({c/total:.1%})" if total else f"  {l:<6}: 0")
    avg_len = sum(len(s["target"]) for s in samples) / total if total else 0
    print(f"\ntarget 平均字符长度: {avg_len:.0f}")
    print(f"✅ 已保存: {out_path}")


if __name__ == "__main__":
    main()
