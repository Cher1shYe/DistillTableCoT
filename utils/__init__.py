"""
工具函数模块
提供配置加载等工具功能
"""

from .config_loader import load_config, deep_merge

__all__ = [
    "load_config",
    "deep_merge"
]

__version__ = "1.0.0"