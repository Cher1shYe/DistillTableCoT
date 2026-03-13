import os
import json
from datasets import load_dataset
from configs import TASK_CONFIGS  # 导入你刚才的配置字典

def download_and_inspect():
    # 创建本地主文件夹
    base_dir = "local_datasets"
    os.makedirs(base_dir, exist_ok=True)

    for task_name, config in TASK_CONFIGS.items():
        print(f"正在下载并处理数据集: {task_name} ...")
        
        # 1. 按照配置从 Hugging Face 加载数据
        dataset = load_dataset(config["dataset_name"], split=config["dataset_split"])
        
        # 2. 为每个任务创建一个专属子文件夹
        task_dir = os.path.join(base_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        
        # 3. 将完整的数据集保存到本地 (保存为 jsonl 格式方便大批量读取)
        full_data_path = os.path.join(task_dir, f"{config['dataset_split']}.jsonl")
        dataset.to_json(full_data_path, force_ascii=False)
        print(f"✅ {task_name} 完整数据已保存至: {full_data_path}")
        
        # 4. 核心：提取第 1 条数据，并以极其易读的格式保存下来，供你研究
        sample_data = dataset[0]
        sample_path = os.path.join(task_dir, "sample_1.json")
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=4)
        print(f"👀 {task_name} 的样本结构已保存至: {sample_path}\n")

if __name__ == "__main__":
    download_and_inspect()