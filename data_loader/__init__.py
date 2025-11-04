"""
数据加载模块
提供多数据集CoT数据加载功能
"""

from .cot_dataset import CoTDataset, DataCollatorForCoT

__all__ = [
    "CoTDataset",
    "DataCollatorForCoT"
]

__version__ = "1.0.0"