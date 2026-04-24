"""
类别多样性计算
N@K: 推荐列表中的独特类别数量
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List
from math import log


def calculate_variety(categories: List[str]) -> int:
    """统计独特类别数"""
    return len(set(categories))


def gini_variety(categories: List[str]) -> float:
    """基于基尼系数的多样性指标"""
    if not categories:
        return 0.0
    counter = Counter(categories)
    values = np.array(sorted(counter.values(), reverse=True))
    n = len(values)
    gini = (2 * np.sum((n - np.arange(1, n + 1)) * values)) / (n * np.sum(values)) - (n + 1) / n
    return 1 - gini


def simpson_diversity(categories: List[str]) -> float:
    """辛普森多样性指数，随机选两个item来自不同类别的概率"""
    if not categories or len(categories) < 2:
        return 0.0
    counter = Counter(categories)
    n = len(categories)
    return 1 - sum((count * (count - 1)) / (n * (n - 1)) for count in counter.values())


def shannon_variety(categories: List[str]) -> float:
    """香农多样性的指数形式，有效类别数"""
    if not categories:
        return 0.0
    counter = Counter(categories)
    n = len(categories)
    probs = [count / n for count in counter.values()]
    h = -sum(p * log(p) if p > 0 else 0 for p in probs)
    return np.exp(h)


def variety_for_user(user_categories: List[str], top_k: int = None) -> Dict[str, float]:
    """计算用户的所有多样性指标"""
    if top_k:
        user_categories = user_categories[:top_k]
    return {
        'unique_categories': calculate_variety(user_categories),
        'gini_variety': gini_variety(user_categories),
        'simpson_diversity': simpson_diversity(user_categories),
        'shannon_variety': shannon_variety(user_categories),
        'total_items': len(user_categories),
        'variety_ratio': calculate_variety(user_categories) / len(user_categories) if user_categories else 0
    }


class VarietyTracker:
    """跟踪多样性随时间变化"""

    def __init__(self):
        self.history = []

    def add(self, categories: List[str], timestamp: str = None):
        self.history.append({
            'timestamp': timestamp,
            'unique_categories': calculate_variety(categories),
            'gini_variety': gini_variety(categories),
            'simpson': simpson_diversity(categories),
            'total': len(categories)
        })

    def get_series(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def calculate_trend(self) -> float:
        """负斜率表示多样性降低"""
        if len(self.history) < 2:
            return 0.0
        series = self.get_series()
        x = np.arange(len(series))
        y = series['unique_categories'].values
        n = len(x)
        return (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
               (n * np.sum(x * x) - np.sum(x) ** 2)
