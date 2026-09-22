from typing import Dict, Any


def load_observation_from_json(path: str) -> Dict[str, Any]:
    """从 JSON 文件加载观测数据 (仅加载, 不解析数组)"""
    import json
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


