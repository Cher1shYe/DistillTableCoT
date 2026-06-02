"""
General utilities.
"""
import json
import os
from typing import List, Union, Dict
from functools import cmp_to_key
import math
from collections.abc import Iterable

# Use HF mirror for China access (must be set before importing datasets)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from datasets import load_dataset

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")

def _load_table(table_path) -> dict:
    """
    attention: the table_path must be the .tsv path.
    Load the WikiTableQuestion from csv file. Result in a dict format like:
    {"header": [header1, header2,...], "rows": [[row11, row12, ...], [row21,...]... [...rownm]]}
    """

    def __extract_content(_line: str):
        _vals = [_.replace("\n", " ").strip() for _ in _line.strip("\n").split("\t")]
        return _vals

    with open(table_path, "r") as f:
        lines = f.readlines()

        rows = []
        for i, line in enumerate(lines):
            line = line.strip('\n')
            if i == 0:
                header = line.split("\t")
            else:
                rows.append(__extract_content(line))

    table_item = {"header": header, "rows": rows}

    # Defense assertion
    for i in range(len(rows) - 1):
        if not len(rows[i]) == len(rows[i - 1]):
            raise ValueError('some rows have diff cols.')

    return table_item


def majority_vote(
        nsqls: List,
        pred_answer_list: List,
        allow_none_and_empty_answer: bool = False,
        allow_error_answer: bool = False,
        answer_placeholder: Union[str, int] = '<error|empty>',
        vote_method: str = 'prob',
        answer_biased: Union[str, int] = None,
        answer_biased_weight: float = None,
):
    """
    Determine the final nsql execution answer by majority vote.
    """

    def _compare_answer_vote_simple(a, b):
        """
        First compare occur times. If equal, then compare max nsql logprob.
        """
        if a[1]['count'] > b[1]['count']:
            return 1
        elif a[1]['count'] < b[1]['count']:
            return -1
        else:
            if a[1]['nsqls'][0][1] > b[1]['nsqls'][0][1]:
                return 1
            elif a[1]['nsqls'][0][1] == b[1]['nsqls'][0][1]:
                return 0
            else:
                return -1

    def _compare_answer_vote_with_prob(a, b):
        """
        Compare prob sum.
        """
        return 1 if sum([math.exp(nsql[1]) for nsql in a[1]['nsqls']]) > sum(
            [math.exp(nsql[1]) for nsql in b[1]['nsqls']]) else -1

    # Vote answers
    candi_answer_dict = dict()
    for (nsql, logprob), pred_answer in zip(nsqls, pred_answer_list):
        if allow_none_and_empty_answer:
            if pred_answer == [None] or pred_answer == []:
                pred_answer = [answer_placeholder]
        if allow_error_answer:
            if pred_answer == '<error>':
                pred_answer = [answer_placeholder]

        # Invalid execution results
        if pred_answer == '<error>' or pred_answer == [None] or pred_answer == []:
            continue
        if candi_answer_dict.get(tuple(pred_answer), None) is None:
            candi_answer_dict[tuple(pred_answer)] = {
                'count': 0,
                'nsqls': []
            }
        answer_info = candi_answer_dict.get(tuple(pred_answer), None)
        answer_info['count'] += 1
        answer_info['nsqls'].append([nsql, logprob])

    # All candidates execution errors
    if len(candi_answer_dict) == 0:
        return answer_placeholder, [(nsqls[0][0], nsqls[0][-1])]

    # Sort
    if vote_method == 'simple':
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_simple), reverse=True)
    elif vote_method == 'prob':
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_with_prob), reverse=True)
    elif vote_method == 'answer_biased':
        # Specifically for Tabfact entailed answer, i.e., `1`.
        # If there exists nsql that produces `1`, we consider it more significant because `0` is very common.
        assert answer_biased_weight is not None and answer_biased_weight > 0
        for answer, answer_dict in candi_answer_dict.items():
            if answer == (answer_biased,):
                answer_dict['count'] *= answer_biased_weight
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_simple), reverse=True)
    elif vote_method == 'lf_biased':
        # Assign weights to different types of logic forms (lf) to control interpretability and coverage
        for answer, answer_dict in candi_answer_dict.items():
            count = 0
            for nsql, _ in answer_dict['nsqls']:
                if 'map@' in nsql:
                    count += 10
                elif 'ans@' in nsql:
                    count += 10
                else:
                    count += 1
            answer_dict['count'] = count
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_simple), reverse=True)
    else:
        raise ValueError(f"Vote method {vote_method} is not supported.")

    pred_answer_info = sorted_candi_answer_list[0]
    pred_answer, pred_answer_nsqls = list(pred_answer_info[0]), pred_answer_info[1]['nsqls']
    return pred_answer, pred_answer_nsqls


def load_data_split(dataset_to_load, split, data_dir=os.path.join(ROOT_DIR, 'datasets/')):
    # data_preprocess.py style: load from local preprocessed JSON, auto-generate if missing
    cache_dir = os.path.join(ROOT_DIR, 'output')
    os.makedirs(cache_dir, exist_ok=True)

    # Map dataset name to HF table-benchmark path
    hf_map = {
        'wikitq': 'table-benchmark/wikiqa',
        'tab_fact': 'table-benchmark/tabfact',
    }
    # All wikitq variants use the same data
    for key in ['has_squall', 'missing_squall', 'wikitq_sql_solvable', 'wikitq_sql_solvable_lower',
                'wikitq_sql_unsolvable', 'wikitq_sql_unsolvable_but_in_squall',
                'wikitq_scalability_ori', 'wikitq_scalability_100rows', 'wikitq_scalability_200rows',
                'wikitq_scalability_500rows', 'wikitq_robustness']:
        hf_map[key] = 'table-benchmark/wikiqa'

    if dataset_to_load in hf_map:
        cache_file = os.path.join(cache_dir, f'{dataset_to_load}_{split}.json')
        if not os.path.exists(cache_file):
            _preprocess_and_save(hf_map[dataset_to_load], dataset_to_load, split, cache_file)
        return _load_from_json(cache_file)

    # Original loading for other datasets (hybridqa, mmqa)
    dataset_split_loaded = load_dataset(
        path=os.path.join(data_dir, "{}.py".format(dataset_to_load)),
        cache_dir=os.path.join(data_dir, "data"))[split]

    # unify names of keys
    if dataset_to_load in ['hybridqa']:
        new_dataset_split_loaded = []
        for data_item in dataset_split_loaded:
            data_item['table']['page_title'] = data_item['context'].split(' | ')[0]
            new_dataset_split_loaded.append(data_item)
        dataset_split_loaded = new_dataset_split_loaded
    elif dataset_to_load == 'mmqa':
        new_dataset_split_loaded = []
        for data_item in dataset_split_loaded:
            data_item['table']['page_title'] = data_item['table']['title']
            new_dataset_split_loaded.append(data_item)
        dataset_split_loaded = new_dataset_split_loaded
    elif dataset_to_load not in ['wikitq', 'tab_fact']:
        pass  # Already handled above
    return dataset_split_loaded


def _preprocess_and_save(hf_path, dataset_name, split, cache_file):
    """Download from HF table-benchmark, preprocess, save as local JSON."""
    import ast
    from utils.table_parser import _parse_table_universal

    print(f"[Preprocessing] Downloading {hf_path} ({split}) to {cache_file}...")
    ds = load_dataset(hf_path, split=split)

    items = []
    for sample in ds:
        table = sample.get("table")
        headers, rows = _parse_table_universal(table, task_name=dataset_name)

        if dataset_name == 'tab_fact':
            table_title = sample.get("table_title", "") or ""
            table_id = str(sample.get("table_id", sample.get("id", "")))
            answer = sample.get("answer", sample.get("label", None))
            if isinstance(answer, (int, float)):
                answer_text = "Entailed" if int(answer) == 1 else "Refuted"
            else:
                ans_lower = str(answer).strip().lower()
                answer_text = "Entailed" if ans_lower in ["entailed", "entailment", "1", "true"] else "Refuted" if ans_lower in ["refuted", "contradiction", "0", "false"] else str(answer)
            items.append({
                "id": str(sample.get("id", "")),
                "question": str(sample.get("question", sample.get("statement", ""))),
                "table": {"id": table_id, "page_title": str(table_title), "header": headers, "rows": rows},
                "answer_text": [answer_text],
            })
        else:
            answer = sample.get("answer", sample.get("answers", []))
            if isinstance(answer, str):
                try:
                    answer = ast.literal_eval(answer)
                except (ValueError, SyntaxError):
                    answer = [answer]
            if not isinstance(answer, list):
                answer = [str(answer)]
            items.append({
                "id": str(sample.get("id", "")),
                "question": str(sample.get("question", "")),
                "table": {"page_title": "", "header": headers, "rows": rows},
                "answer_text": [str(a) for a in answer],
            })

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False)
    print(f"[Preprocessing] Saved {len(items)} items to {cache_file}")


def _load_from_json(cache_file):
    """Load preprocessed data from local JSON."""
    with open(cache_file, 'r', encoding='utf-8') as f:
        items = json.load(f)
    return items


def pprint_dict(dic):
    print(json.dumps(dic, indent=2))


def flatten(nested_list):
    for x in nested_list:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x
