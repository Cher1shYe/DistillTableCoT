"""Run Binder annotation + execution pipeline with configurable datasets and sample count.

Usage:
    python run.py                                    # wikitq + tab_fact, 3 items each
    python run.py --max_items 10                     # wikitq + tab_fact, 10 items each
    python run.py --datasets wikitq                  # only wikitq
    python run.py --datasets tab_fact --max_items 5  # only tab_fact, 5 items
"""
import os
import argparse
import platform

ROOT_DIR = os.path.dirname(__file__)

if platform.system() == "Windows":
    PREFIX = f"set TOKENIZERS_PARALLELISM=false&& set PYTHONPATH={ROOT_DIR}&& "
else:
    PREFIX = "export TOKENIZERS_PARALLELISM=false\n"


# ── Dataset configurations ──────────────────────────────────────
DATASET_CONFIGS = {
    "wikitq": {
        "prompt_file": "templates/prompts/wikitq_binder.txt",
        "max_generation_tokens": 512,
        "temperature": 0.4,
        "n_shots": 8,
        "exec_args": "--vote_method simple",
    },
    "tab_fact": {
        "prompt_file": "templates/prompts/tab_fact_binder.txt",
        "max_generation_tokens": 256,
        "temperature": 0.6,
        "n_shots": 18,
        "exec_args": "--allow_none_and_empty_answer --vote_method answer_biased --answer_biased 1 --answer_biased_weight 3",
    },
}


def run_phase(name, script, extra_args):
    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")
    cmd = f'{PREFIX}python {ROOT_DIR}/{script} {extra_args}'
    print(f"[CMD] {cmd}")
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run Binder pipeline")
    parser.add_argument("--max_items", type=int, default=3,
                        help="Number of items per dataset (default: 3)")
    parser.add_argument("--datasets", type=str, nargs="+",
                        choices=["wikitq", "tab_fact"],
                        default=["wikitq", "tab_fact"],
                        help="Datasets to run (default: wikitq tab_fact)")
    args = parser.parse_args()

    for ds in args.datasets:
        cfg = DATASET_CONFIGS[ds]
        ann_args = (
            f"--dataset {ds} --dataset_split test "
            f"--prompt_file {cfg['prompt_file']} "
            f"--max_generation_tokens {cfg['max_generation_tokens']} "
            f"--temperature {cfg['temperature']} "
            f"--n_shots {cfg['n_shots']} "
            f"--sampling_n 1 "
            f"--max_items {args.max_items} "
            f"--n_parallel_prompts 1 --n_processes 1 --max_api_total_tokens 28000 -v"
        )
        run_phase(f"Phase: {ds} Annotation",
                  "scripts/annotate_binder_program.py", ann_args)

        exe_args = (
            f"--dataset {ds} --dataset_split test "
            f"--qa_retrieve_pool_file templates/qa_retrieve_pool/qa_retrieve_pool.json "
            f"--input_program_file binder_program_{ds}_test.json "
            f"--output_program_execution_file binder_program_{ds}_test_exec.json "
            f"--n_processes 1 --max_items {args.max_items} --verbose "
            f"{cfg['exec_args']}"
        )
        run_phase(f"Phase: {ds} Execution",
                  "scripts/execute_binder_program.py", exe_args)

    print(f"\n{'=' * 60}")
    print("All done! Results saved in results/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
