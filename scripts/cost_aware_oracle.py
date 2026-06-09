#!/usr/bin/env python3
"""
cost-aware oracle + 成本-准确率对比 (对应 cost_aware.pdf 阶段二 + 第7节评价指标)。

把同一批样本 (同 task/split/num_samples → 同 id) 上多条推理路径的预测文件 join 起来：
  对每个样本：
    - 算每条路径是否正确          (is_match)
    - 算每条路径的成本            (路径档位 tier + token + tool_calls + latency)
    - 选 oracle = 所有正确路径中"成本最低"的那条
                  (主序按路径档位 tier，次序按 completion_tokens，再按 latency)

为什么按 tier 而不是直接按 R1 的原始 token 排序：
    R1 对什么题都过度推理，原始 token 是个糟糕的成本代理。我们要优化的是最终
    部署的"学生侧成本"，它由所选路径的类型决定 (Direct<CoT<SQL<SQL-Repair<Fallback)，
    所以用路径档位作主序，token/latency 只做同档位内的 tiebreak。

输出：
  - 每条固定路径 baseline 的 accuracy / avg completion / reasoning / tool_calls / latency
  - oracle 的 accuracy (= 至少一条路径答对) / avg 成本 / 路径分布
  - overthinking_rate：最贵路径 (mixed) 答对、但其实存在更便宜路径也对的比例
  - 相对"永远走 mixed"这一现状系统的成本节省
  - join 明细落盘 outputs/<task>/cost_aware_join_v<version>.json (后续建 SFT/DPO 数据用)

用法：
    python3 scripts/cost_aware_oracle.py --task hitab
    python3 scripts/cost_aware_oracle.py --task hitab --version 1
    python3 scripts/cost_aware_oracle.py --task hitab --files direct=outputs/hitab/R1_prediction_direct_v1.json mixed=...
"""
import argparse
import json
import os
import sys
from glob import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils_train.eval_utils import is_match


# 成本阶梯：数字越小越便宜，作为 oracle 选择的主排序键
PATH_TIER = {"direct": 0, "cot": 1, "sql": 2, "sql_repair": 3, "mixed": 4}


def infer_path_label(fname):
    """从文件名推断路径标签。"""
    f = os.path.basename(fname).lower()
    if "direct" in f:
        return "direct"
    if "mixed_agent" in f or "mixed" in f:
        return "mixed"
    if "sql_agent" in f or "sql" in f:
        return "sql"
    if "cot" in f:
        return "cot"
    return None


def load_path_file(path):
    """加载一个预测文件，兼容 list 与 {'predictions': [...]} 两种格式。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("predictions", [])
    return {rec["id"]: rec for rec in data}


def path_cost(rec):
    """从一条预测记录里抽出成本字段。"""
    u = rec.get("usage") or {}
    return {
        "completion_tokens": u.get("completion_tokens") or 0,
        "reasoning_tokens": u.get("reasoning_tokens") or 0,
        "prompt_tokens": u.get("prompt_tokens") or 0,
        "tool_calls": rec.get("tool_calls") or 0,
        "latency": rec.get("latency") or 0.0,
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser(description="cost-aware oracle + 成本-准确率对比")
    ap.add_argument("--task", required=True,
                    choices=["wikitableqa", "tabfact", "fetaqa", "hitab"])
    ap.add_argument("--version", type=int, default=1,
                    help="自动发现 outputs/<task>/R1_prediction_*_v<version>.json")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--files", nargs="*", default=None,
                    help="显式指定 label=path（覆盖自动发现），如 direct=outputs/hitab/R1_prediction_direct_v1.json")
    args = ap.parse_args()

    rouge_threshold = 0.3 if args.task == "fetaqa" else None
    task_dir = os.path.join(args.output_dir, args.task)

    # ---- 收集各路径文件 ----
    path_files = {}   # label -> filepath
    if args.files:
        for spec in args.files:
            label, fp = spec.split("=", 1)
            path_files[label] = fp
    else:
        pattern = os.path.join(task_dir, f"R1_prediction_*_v{args.version}.json")
        for fp in sorted(glob(pattern)):
            label = infer_path_label(fp)
            if label is None:
                print(f"⚠️  跳过无法识别路径的文件: {fp}")
                continue
            path_files[label] = fp

    if not path_files:
        print(f"❌ 在 {task_dir} 没找到任何路径文件 (pattern: R1_prediction_*_v{args.version}.json)")
        return

    print(f"==== cost-aware oracle | task={args.task} | version={args.version} ====")
    print("发现路径文件：")
    for label, fp in path_files.items():
        print(f"  [{label:<6}] {fp}")

    # ---- 加载并按 id join ----
    path_data = {label: load_path_file(fp) for label, fp in path_files.items()}
    common_ids = set.intersection(*[set(d.keys()) for d in path_data.values()])
    common_ids = sorted(common_ids)
    n = len(common_ids)
    if n == 0:
        print("❌ 各路径文件没有共同 id，无法 join（检查是否同 split/num_samples）")
        return
    # 提示 id 数不一致
    for label, d in path_data.items():
        if len(d) != n:
            print(f"⚠️  [{label}] 有 {len(d)} 条，共同 id 只有 {n} 条（按交集对齐）")

    labels = sorted(path_files.keys(), key=lambda l: PATH_TIER.get(l, 99))

    # ---- 逐样本算 correct + cost，并选 oracle ----
    per_path_correct = {l: 0 for l in labels}
    per_path_cost = {l: {"completion_tokens": [], "reasoning_tokens": [],
                         "tool_calls": [], "latency": []} for l in labels}
    oracle_solved = 0
    oracle_path_dist = {l: 0 for l in labels}
    oracle_cost = {"completion_tokens": [], "tool_calls": [], "latency": []}
    overthink_mixed = 0          # mixed 答对 且 有更便宜路径也对
    mixed_correct_total = 0
    # 同样本对比：仅在 mixed 答对的样本上，对齐统计 mixed 与 oracle 的成本
    # （避免 oracle 只在"解出样本"上平均、baseline 在全集上平均的口径不一致）
    same = {"m_compl": [], "o_compl": [], "m_tool": [], "o_tool": [], "m_lat": [], "o_lat": []}
    join_rows = []

    for sid in common_ids:
        row = {"id": sid, "paths": {}}
        ref = None
        correct_paths = []   # (tier, completion_tokens, latency, label)
        for l in labels:
            rec = path_data[l][sid]
            ref = rec.get("reference")
            pred = rec.get("processed_prediction", "")
            ok = is_match(pred, ref, rouge_threshold=rouge_threshold)
            c = path_cost(rec)
            if ok:
                per_path_correct[l] += 1
                correct_paths.append((PATH_TIER.get(l, 99),
                                      c["completion_tokens"], c["latency"], l))
            per_path_cost[l]["completion_tokens"].append(c["completion_tokens"])
            per_path_cost[l]["reasoning_tokens"].append(c["reasoning_tokens"])
            per_path_cost[l]["tool_calls"].append(c["tool_calls"])
            per_path_cost[l]["latency"].append(c["latency"])
            row["paths"][l] = {
                "correct": ok,
                "processed_prediction": pred,
                "tier": PATH_TIER.get(l, 99),
                **c,
            }
        row["reference"] = ref

        # oracle = 正确路径里 (tier, completion, latency) 最小的
        bc = None
        if correct_paths:
            correct_paths.sort()
            best = correct_paths[0]
            best_label = best[3]
            oracle_solved += 1
            oracle_path_dist[best_label] += 1
            bc = row["paths"][best_label]
            oracle_cost["completion_tokens"].append(bc["completion_tokens"])
            oracle_cost["tool_calls"].append(bc["tool_calls"])
            oracle_cost["latency"].append(bc["latency"])
            row["oracle_path"] = best_label
            row["oracle_correct"] = True
        else:
            row["oracle_path"] = None
            row["oracle_correct"] = False

        # overthinking + 同样本成本对比：均在 mixed 答对的样本上
        if "mixed" in row["paths"] and row["paths"]["mixed"]["correct"]:
            mixed_correct_total += 1
            mc = row["paths"]["mixed"]
            # mixed 对 → 必有正确路径 → oracle 已解出，bc 一定非空
            same["m_compl"].append(mc["completion_tokens"])
            same["o_compl"].append(bc["completion_tokens"])
            same["m_tool"].append(mc["tool_calls"])
            same["o_tool"].append(bc["tool_calls"])
            same["m_lat"].append(mc["latency"])
            same["o_lat"].append(bc["latency"])
            cheaper_ok = any(
                row["paths"][l]["correct"] and PATH_TIER.get(l, 99) < PATH_TIER["mixed"]
                for l in labels if l != "mixed"
            )
            if cheaper_ok:
                overthink_mixed += 1

        join_rows.append(row)

    # ---- 打印固定路径 baseline ----
    print(f"\n样本数 (共同 id): {n}\n")
    hdr = f"{'path':<8}{'acc':>8}{'compl_tok':>11}{'reason_tok':>12}{'tool':>7}{'lat(s)':>9}"
    print("== 固定路径 baseline ==")
    print(hdr)
    print("-" * len(hdr))
    for l in labels:
        acc = per_path_correct[l] / n
        pc = per_path_cost[l]
        print(f"{l:<8}{acc:>8.1%}"
              f"{mean(pc['completion_tokens']):>11.0f}"
              f"{mean(pc['reasoning_tokens']):>12.0f}"
              f"{mean(pc['tool_calls']):>7.2f}"
              f"{mean(pc['latency']):>9.1f}")

    # ---- oracle ----
    print("\n== cost-aware oracle (正确路径中成本最低) ==")
    print(f"oracle accuracy (至少一条路径对): {oracle_solved}/{n} = {oracle_solved/n:.1%}")
    print(f"oracle avg completion_tokens (解出的样本上): {mean(oracle_cost['completion_tokens']):.0f}")
    print(f"oracle avg tool_calls: {mean(oracle_cost['tool_calls']):.2f}")
    print(f"oracle avg latency(s): {mean(oracle_cost['latency']):.1f}")
    print("oracle 路径分布:")
    for l in labels:
        if oracle_solved:
            print(f"  {l:<8}: {oracle_path_dist[l]:>4}  ({oracle_path_dist[l]/oracle_solved:.1%})")

    # ---- 对比 / 故事指标（同样本口径：仅在 mixed 答对的样本上）----
    print("\n== 成本节省 vs '永远走 mixed' (同样本：mixed 答对的子集) ==")
    if "mixed" in labels and mixed_correct_total:
        m_compl, o_compl = mean(same["m_compl"]), mean(same["o_compl"])
        m_tool, o_tool = mean(same["m_tool"]), mean(same["o_tool"])
        m_lat, o_lat = mean(same["m_lat"]), mean(same["o_lat"])
        print(f"  子集大小 (mixed 答对): {mixed_correct_total} 条，二者准确率均为 100%")
        dc = f"  (↓{(1 - o_compl/m_compl):.0%})" if m_compl else ""
        dl = f"  (↓{(1 - o_lat/m_lat):.0%})" if m_lat else ""
        print(f"  compl_tok: mixed {m_compl:.0f}  →  oracle {o_compl:.0f}{dc}")
        print(f"  tool_call: mixed {m_tool:.2f}  →  oracle {o_tool:.2f}")
        print(f"  latency  : mixed {m_lat:.1f}s  →  oracle {o_lat:.1f}s{dl}")
        print(f"\n  overthinking_rate (mixed 答对但有更便宜路径也对): "
              f"{overthink_mixed}/{mixed_correct_total} = {overthink_mixed/mixed_correct_total:.1%}")
    elif "mixed" not in labels:
        print("  (无 mixed 路径文件，跳过)")

    # ---- 落盘 join 明细 ----
    out_path = os.path.join(task_dir, f"cost_aware_join_v{args.version}.json")
    summary = {
        "task": args.task,
        "version": args.version,
        "n": n,
        "path_files": path_files,
        "baseline": {
            l: {
                "accuracy": per_path_correct[l] / n,
                "avg_completion_tokens": mean(per_path_cost[l]["completion_tokens"]),
                "avg_reasoning_tokens": mean(per_path_cost[l]["reasoning_tokens"]),
                "avg_tool_calls": mean(per_path_cost[l]["tool_calls"]),
                "avg_latency": mean(per_path_cost[l]["latency"]),
            } for l in labels
        },
        "oracle": {
            "accuracy": oracle_solved / n,
            "avg_completion_tokens": mean(oracle_cost["completion_tokens"]),
            "avg_tool_calls": mean(oracle_cost["tool_calls"]),
            "avg_latency": mean(oracle_cost["latency"]),
            "path_distribution": oracle_path_dist,
        },
        "overthinking_rate_mixed": (overthink_mixed / mixed_correct_total) if mixed_correct_total else None,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": join_rows}, f, ensure_ascii=False, indent=2)
    print(f"\n✅ join 明细 + 汇总已保存: {out_path}")
    print("   (含每样本各路径 correct/cost 与 oracle_path，可直接用于阶段三 SFT/DPO 数据构造)")


if __name__ == "__main__":
    main()
