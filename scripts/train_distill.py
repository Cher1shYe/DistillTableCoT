#!/usr/bin/env python3
import argparse
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_train.config_loader import load_config
from models.qwen_trainer import QwenDistillTrainer

def main():
    parser = argparse.ArgumentParser(description="Qwen小模型蒸馏训练脚本")
    parser.add_argument("--config", type=str, required=True,
                       help="配置文件路径，如 configs/qwen2.5-1.7b.yaml")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="自定义输出目录")
    parser.add_argument("--data_paths", nargs="+", default=None,
                       help="自定义数据路径")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置（如果提供了命令行参数）
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.data_paths:
        config['data']['data_paths'] = args.data_paths
    
    # 创建输出目录
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    
    # 保存使用的配置
    import yaml
    with open(os.path.join(config['training']['output_dir'], 'config_used.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # 创建训练器并开始训练
    trainer = QwenDistillTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()