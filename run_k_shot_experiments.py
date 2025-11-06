"""
运行 k-shot 对比实验
"""

import subprocess
import sys
import os

def run_experiment(task_name, num_samples, k_shots=[0, 1, 2, 3]):
    """运行不同 k-shot 的实验"""
    
    print(f"\n{'='*50}")
    print(f"开始 {task_name} 的 k-shot 实验")
    print(f"{'='*50}")
    
    for k in k_shots:
        print(f"\n--- 运行 {task_name} k={k} ---")
        
        # 构建命令
        cmd = [
            sys.executable, "run_inference.py",
            "--task_name", task_name,
            "--num_samples", str(num_samples),
            "--k_shot", str(k)
        ]
        
        # 执行命令
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"k={k} 完成")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"k={k} 失败: {e}")
            print(e.stderr)

def main():
    tasks = ["wikitableqa", "tabfact", "fetaqa"]
    num_samples = 10  # 每个实验的样本数
    k_shots = [0, 1, 2, 3]  # 要测试的 shot 数量
    
    for task in tasks:
        run_experiment(task, num_samples, k_shots)


if __name__ == "__main__":
    main()