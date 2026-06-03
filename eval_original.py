"""
Original Binder evaluation — reads results/ exec JSON and computes accuracy
using the exact same Evaluator from utils/evaluator.py.

Usage:
    python eval_original.py --dataset wikitq
    python eval_original.py --dataset tab_fact
    python eval_original.py --dataset wikitq --exec_file my_exec.json
"""
import os
import json
import argparse

ROOT_DIR = os.path.dirname(__file__)

from utils.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Original Binder evaluation")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["wikitq", "tab_fact"],
                        help="Dataset to evaluate")
    parser.add_argument("--exec_file", type=str, default=None,
                        help="Execution result file (default: results/binder_program_<dataset>_test_exec.json)")
    args = parser.parse_args()

    if args.exec_file is None:
        args.exec_file = f"binder_program_{args.dataset}_test_exec.json"

    filepath = os.path.join(ROOT_DIR, "results", args.exec_file)
    if not os.path.exists(filepath):
        filepath = args.exec_file

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    evaluator = Evaluator()
    correct = 0
    total = 0

    for eid in sorted(data.keys(), key=int):
        item = data[eid]
        pred = item.get("pred_answer", "")
        gold = item.get("gold_answer", "")
        question = item.get("question", "")

        score = evaluator.evaluate(
            pred_answer=pred,
            gold_answer=gold,
            dataset=args.dataset,
            question=question,
        )
        correct += score
        total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"\n{'=' * 40}")
    print(f"Dataset: {args.dataset}")
    print(f"Total:   {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
