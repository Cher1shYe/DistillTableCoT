import yaml
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置，支持继承"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 处理继承
    if '_base_' in config:
        base_path = os.path.join(os.path.dirname(config_path), config['_base_'])
        base_config = load_config(base_path)
        # 合并配置（当前配置覆盖基础配置）
        merged_config = deep_merge(base_config, config)
        return merged_config
    
    return config

def deep_merge(base: Dict, update: Dict) -> Dict:
    """深度合并两个字典"""
    result = base.copy()
    
    for key, value in update.items():
        if key == '_base_':
            continue
        if (key in result and isinstance(result[key], dict) 
            and isinstance(value, dict)):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result