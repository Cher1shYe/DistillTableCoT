"""
构建示例池的脚本
"""

import argparse
from example_pool import ExamplePool

def main():
    parser = argparse.ArgumentParser(description="构建 CoT 示例池")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["wikitableqa", "tabfact", "fetaqa"],
        help="要构建示例池的任务名称"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=20,
        help="要生成的示例数量"
    )
    parser.add_argument(
        "--pool_dir",
        type=str,
        default="example_pools",
        help="示例池保存目录"
    )
    
    args = parser.parse_args()
    
    # 构建示例池
    pool_manager = ExamplePool(pool_dir=args.pool_dir)
    pool_file = pool_manager.build_pool_for_task(
        task_name=args.task,
        num_examples=args.num_examples
    )
    
    if pool_file:
        # 显示统计信息
        stats = pool_manager.get_pool_stats(args.task)
        print(f"\n--- 示例池统计 ---")
        for key, value in stats.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()