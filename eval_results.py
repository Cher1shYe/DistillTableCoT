"""
Evaluate Binder execution results using the same methods as DistillTableCoT.
Reads *_exec.json from results/ and computes exact_match (WikiTQ) or accuracy (TabFact).

Usage:
    python eval_results.py --dataset wikitq
    python eval_results.py --dataset tab_fact
    python eval_results.py --dataset wikitq --exec_file binder_program_wikitq_test_exec.json
"""

import os
import re
import ast
import json
import argparse
import nltk

# Ensure NLTK punkt is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

ROOT_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')


def normalize_string(s):
    """Normalize string for exact match (same as DistillTableCoT)."""
    if not s:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r'(?<=\d),(?=\d)', '', s)           # remove commas in numbers
    s = re.sub(r'\b(a|an|the)\b', ' ', s)          # remove articles
    s = re.sub(r'[^\w\s-]', ' ', s)                # remove most punctuation
    if s.endswith('.'):
        s = s[:-1]
    return " ".join(s.split())


def compute_wikitq_em(preds, refs):
    """Exact match for WikiTQ — identical logic to DistillTableCoT."""
    correct = 0
    total = len(preds)
    for p, r in zip(preds, refs):
        p_norm = normalize_string(p)

        if isinstance(r, str) and r.startswith('[') and r.endswith(']'):
            try:
                r_list = ast.literal_eval(r)
            except Exception:
                r_list = [r.strip("[]'\"")]
        elif isinstance(r, list):
            r_list = r
        else:
            r_list = [str(r)]

        r_norms = [normalize_string(x) for x in r_list]

        if p_norm in r_norms:
            correct += 1
        elif p_norm == ", ".join(r_norms) or p_norm == ",".join(r_norms):
            correct += 1
        elif all(ref in p_norm for ref in r_norms) and len(r_norms) > 1:
            correct += 1

    return correct, total


def compute_tabfact_accuracy(preds, refs):
    """Accuracy for TabFact — identical logic to DistillTableCoT."""
    preds_int = [1 if str(p).lower().strip() == 'entailed' else 0 for p in preds]
    refs_int = [1 if str(r).lower().strip() == 'entailed' else 0 for r in refs]

    correct = sum(1 for p, r in zip(preds_int, refs_int) if p == r)
    return correct, len(preds)


def load_binder_results(exec_file):
    """Load Binder execution results and convert to (preds, refs) lists."""
    filepath = os.path.join(RESULTS_DIR, exec_file)
    if not os.path.exists(filepath):
        filepath = exec_file  # try as full path

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    preds = []
    refs = []
    questions = []

    for eid in sorted(data.keys(), key=int):
        item = data[eid]
        # pred_answer: e.g. ['100000'] or [0] or ['Entailed']
        # gold_answer: e.g. ['100,000'] or ['Entailed']
        pred = item.get('pred_answer', '')
        gold = item.get('gold_answer', '')
        question = item.get('question', '')

        # Flatten list answers to string
        if isinstance(pred, list):
            pred = pred[0] if pred else ''
        if isinstance(gold, list):
            gold = gold[0] if gold else ''

        preds.append(str(pred))
        refs.append(str(gold))
        questions.append(question)

    return preds, refs, questions


def main():
    parser = argparse.ArgumentParser(description='Evaluate Binder execution results')
    parser.add_argument('--dataset', type=str, required=True, choices=['wikitq', 'tab_fact'],
                        help='Dataset to evaluate')
    parser.add_argument('--exec_file', type=str, default=None,
                        help='Execution result file (default: binder_program_<dataset>_test_exec.json)')
    parser.add_argument('--print_details', action='store_true',
                        help='Print per-example pred vs gold')
    args = parser.parse_args()

    if args.exec_file is None:
        args.exec_file = f'binder_program_{args.dataset}_test_exec.json'

    preds, refs, questions = load_binder_results(args.exec_file)

    if args.dataset == 'wikitq':
        correct, total = compute_wikitq_em(preds, refs)
        metric_name = 'exact_match'
    else:
        correct, total = compute_tabfact_accuracy(preds, refs)
        metric_name = 'accuracy'

    score = correct / total if total > 0 else 0.0

    print(f"\n{'=' * 60}")
    print(f"Evaluation: {args.dataset}")
    print(f"File: {args.exec_file}")
    print(f"{'=' * 60}")
    print(f"Total examples: {total}")
    print(f"Correct: {correct}")
    print(f"{metric_name}: {score:.4f} ({correct}/{total})")

    if args.print_details:
        print(f"\n--- Per-example details ---")
        for i, (q, p, r) in enumerate(zip(questions, preds, refs)):
            status = '✓' if (
                (args.dataset == 'wikitq' and compute_wikitq_em([p], [r])[0] == 1) or
                (args.dataset == 'tab_fact' and compute_tabfact_accuracy([p], [r])[0] == 1)
            ) else '✗'
            print(f"  [{i}] {status} pred='{p[:60]}' | gold='{str(r)[:60]}'")


if __name__ == '__main__':
    main()
