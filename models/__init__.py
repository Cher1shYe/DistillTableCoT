"""
模型训练模块
提供Qwen模型蒸馏训练功能
"""

from .qwen_trainer import QwenDistillTrainer

__all__ = [
    "QwenDistillTrainer"
]

__version__ = "1.0.0"