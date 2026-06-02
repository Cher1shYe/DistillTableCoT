import os

ROOT_DIR = os.path.join(os.path.dirname(__file__))

# Disable the TOKENIZERS_PARALLELISM
import platform
if platform.system() == "Windows":
    TOKENIZER_FALSE = f"set TOKENIZERS_PARALLELISM=false&& set PYTHONPATH={ROOT_DIR}&& "
else:
    TOKENIZER_FALSE = "export TOKENIZERS_PARALLELISM=false\n"

# Common args for annotation
ANN_ARGS = "--max_items 3 --n_parallel_prompts 1 --n_processes 1 --max_api_total_tokens 28000 -v"
# Common args for execution
EXE_ARGS = "--n_processes 1 --verbose"

print("=" * 60)
print("Phase 1: WikiTQ Annotation")
print("=" * 60)
os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/annotate_binder_program.py --dataset wikitq --dataset_split test --prompt_file templates/prompts/wikitq_binder.txt --max_generation_tokens 512 --temperature 0.4 --sampling_n 1 {ANN_ARGS}""")

print("\n" + "=" * 60)
print("Phase 2: WikiTQ Execution")
print("=" * 60)
os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/execute_binder_program.py --dataset wikitq --dataset_split test --qa_retrieve_pool_file templates/qa_retrieve_pool/qa_retrieve_pool.json --input_program_file binder_program_wikitq_test.json --output_program_execution_file binder_program_wikitq_test_exec.json --vote_method simple {EXE_ARGS}""")

print("\n" + "=" * 60)
print("Phase 3: TabFact Annotation")
print("=" * 60)
os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/annotate_binder_program.py --dataset tab_fact --dataset_split test --prompt_file templates/prompts/tab_fact_binder.txt --max_generation_tokens 256 --temperature 0.6 --sampling_n 1 --n_shots 18 {ANN_ARGS}""")

print("\n" + "=" * 60)
print("Phase 4: TabFact Execution")
print("=" * 60)
os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/execute_binder_program.py --dataset tab_fact --dataset_split test --qa_retrieve_pool_file templates/qa_retrieve_pool/qa_retrieve_pool.json --input_program_file binder_program_tab_fact_test.json --output_program_execution_file binder_program_tab_fact_test_exec.json --allow_none_and_empty_answer --vote_method answer_biased --answer_biased 1 --answer_biased_weight 3 {EXE_ARGS}""")

print("\n" + "=" * 60)
print("All done! Results saved in results/ directory")
print("=" * 60)
