#!/usr/bin/env python3
"""
Re-evaluate existing prediction JSON files with the current (fixed) postprocess functions.
Updates processed_prediction in-place and prints new vs old accuracy.

Usage:
    python3 scripts/reeval.py                      # re-eval all known files
    python3 scripts/reeval.py --dry_run            # print scores only, don't overwrite
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import (
    extract_wiki_final_answer,
    extract_fact_final_answer,
    extract_fetaqa_final_answer,
    extract_hitab_final_answer,
)
from utils_train.eval_utils import is_match

POSTPROCESS = {
    "wikitableqa": lambda pred, ref: extract_wiki_final_answer(pred),
    "tabfact":     lambda pred, ref: extract_fact_final_answer(pred),
    "fetaqa":      lambda pred, ref: extract_fetaqa_final_answer(pred),
    "hitab":       lambda pred, ref: extract_hitab_final_answer(pred, ref),
}

ROUGE_TASKS = {"fetaqa"}

FILES = {
    "wikitableqa": [
        "outputs/wikitableqa/predictions_qwen3_1.7b_basic_model_v0.json",
        "outputs/wikitableqa/predictions_qwen3_1.7b_cot_wikitableqa.json",
        "outputs/wikitableqa/predictions_qwen3_1.7b_agent_wikitableqa.json",
    ],
    "tabfact": [
        "outputs/tabfact/predictions_qwen3_1.7b_basic_model_v0.json",
        "outputs/tabfact/predictions_qwen3_1.7b_cot_tabfact.json",
        "outputs/tabfact/predictions_qwen3_1.7b_agent_tabfact.json",
    ],
    "fetaqa": [
        "outputs/fetaqa/predictions_qwen3_1.7b_basic_model_v0.json",
        "outputs/fetaqa/predictions_qwen3_1.7b_cot_fetaqa.json",
        "outputs/fetaqa/predictions_qwen3_1.7b_agent_fetaqa.json",
    ],
    "hitab": [
        "outputs/hitab/predictions_qwen3_1.7b_basic_model_v0.json",
        "outputs/hitab/predictions_qwen3_1.7b_cot_hitab.json",
        "outputs/hitab/predictions_qwen3_1.7b_agent_hitab.json",
    ],
}


def load_json(path):
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "predictions" in raw:
        return raw, raw["predictions"]
    return None, raw


def score(data, task):
    rouge_thr = 0.3 if task in ROUGE_TASKS else None
    return sum(
        1 for d in data
        if is_match(str(d.get("processed_prediction", "")), str(d.get("reference", "")), rouge_threshold=rouge_thr)
    ) / len(data) * 100


def reeval_file(path, task, dry_run):
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return

    raw_obj, data = load_json(path)
    postfn = POSTPROCESS[task]

    old_acc = score(data, task)
    changed = 0

    for d in data:
        new_pred = postfn(d.get("prediction", ""), d.get("reference", ""))
        if new_pred != d.get("processed_prediction", ""):
            changed += 1
        d["processed_prediction"] = new_pred

    new_acc = score(data, task)
    label = os.path.basename(path)
    print(f"  {label}: {old_acc:5.1f}% → {new_acc:5.1f}%  (changed {changed}/{len(data)})")

    if not dry_run:
        if raw_obj is not None:
            raw_obj["predictions"] = data
            save_obj = raw_obj
        else:
            save_obj = data
        with open(path, "w", encoding="utf-8") as f:
            json.dump(save_obj, f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Print scores without overwriting files")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for task, paths in FILES.items():
        print(f"\n=== {task} ===")
        for rel_path in paths:
            reeval_file(os.path.join(base, rel_path), task, args.dry_run)

    if args.dry_run:
        print("\n(dry_run mode — files not modified)")
    else:
        print("\nFiles updated.")


if __name__ == "__main__":
    main()
