"""Run TabFact annotation + execution only (3 items)."""
import os, platform

ROOT_DIR = os.path.dirname(__file__)

if platform.system() == "Windows":
    PREFIX = f"set TOKENIZERS_PARALLELISM=false&& set PYTHONPATH={ROOT_DIR}&& "
else:
    PREFIX = "export TOKENIZERS_PARALLELISM=false\n"

print("=" * 60)
print("TabFact Annotation")
print("=" * 60)
os.system(fr"""{PREFIX}python {ROOT_DIR}/scripts/annotate_binder_program.py --dataset tab_fact --dataset_split test --prompt_file templates/prompts/tab_fact_binder.txt --max_generation_tokens 256 --temperature 0.6 --sampling_n 1 --n_shots 18 --max_items 3 --n_parallel_prompts 1 --n_processes 1 --max_api_total_tokens 28000 -v""")

print("\n" + "=" * 60)
print("TabFact Execution")
print("=" * 60)
os.system(fr"""{PREFIX}python {ROOT_DIR}/scripts/execute_binder_program.py --dataset tab_fact --dataset_split test --qa_retrieve_pool_file templates/qa_retrieve_pool/qa_retrieve_pool.json --input_program_file binder_program_tab_fact_test.json --output_program_execution_file binder_program_tab_fact_test_exec.json --allow_none_and_empty_answer --vote_method answer_biased --answer_biased 1 --answer_biased_weight 3 --n_processes 1 --max_items 3 --verbose""")

print("\nDone! Results in results/")
