# table_llm_eval/main.py

import argparse
from configs import TASK_CONFIGS
from run_inference import run_generation
from run_evaluate import run_evaluation

def main():
    parser = argparse.ArgumentParser(description="运行大语言模型在表格理解任务上的评估流程。")
    
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_CONFIGS.keys()),
        help="要执行的评估任务名称。"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["generate", "evaluate", "all"],
        help="要执行的模式: 'generate' 只生成结果, 'evaluate' 只评估现有结果, 'all' 两者都执行。"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="用于生成和评估的样本数量。"
    )

    args = parser.parse_args()

    if args.mode in ["generate", "all"]:
        run_generation(task_name=args.task, num_samples=args.num_samples)
    