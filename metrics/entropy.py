"""
信息熵计算
H@K: 推荐列表的类别信息熵
熵值越低表示类别越集中，茧房风险越高
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List


def calculate_entropy(categories: List[str]) -> float:
    """计算类别分布的香农熵"""
    if not categories:
        return 0.0
    counter = Counter(categories)
    total = len(categories)
    probs = [count / total for count in counter.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    return entropy


def calculate_max_entropy(n_categories: int) -> float:
    """计算给定类别数的最大熵"""
    if n_categories <= 1:
        return 0.0
    return np.log2(n_categories)


def normalized_entropy(categories: List[str]) -> float:
    """归一化熵，0-1范围，1表示完全多样"""
    if not categories:
        return 0.0
    entropy = calculate_entropy(categories)
    n_cats = len(set(categories))
    max_ent = calculate_max_entropy(n_cats)
    if max_ent == 0:
        return 0.0
    return entropy / max_ent


def entropy_for_user(user_categories: List[str], top_k: int = None) -> Dict[str, float]:
    """计算用户的所有熵相关指标"""
    if top_k:
        user_categories = user_categories[:top_k]
    return {
        'entropy': calculate_entropy(user_categories),
        'normalized_entropy': normalized_entropy(user_categories),
        'unique_categories': len(set(user_categories)),
        'total_items': len(user_categories)
    }


class EntropyTracker:
    """跟踪用户熵随时间的变化"""

    def __init__(self):
        self.history = []

    def add(self, categories: List[str], timestamp: str = None):
        """添加一个时间点的类别数据"""
        self.history.append({
            'timestamp': timestamp,
            'entropy': calculate_entropy(categories),
            'normalized_entropy': normalized_entropy(categories),
            'unique_categories': len(set(categories)),
            'total_items': len(categories)
        })

    def get_series(self) -> pd.DataFrame:
        """获取熵时间序列"""
        return pd.DataFrame(self.history)

    def calculate_trend(self) -> float:
        """计算熵的趋势斜率，负值表示多样性降低"""
        if len(self.history) < 2:
            return 0.0
        series = self.get_series()
        x = np.arange(len(series))
        y = series['normalized_entropy'].values
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
                (n * np.sum(x * x) - np.sum(x) ** 2)
        return slope


def main():
    """测试熵计算"""
    diverse = ['news', 'sports', 'health', 'lifestyle', 'tech'] * 3
    concentrated = ['news'] * 15 + ['sports'] * 10 + ['health'] * 5
    print("多样化分布:")
    print(f"  熵: {calculate_entropy(diverse):.3f}, 归一化熵: {normalized_entropy(diverse):.3f}")
    print("集中化分布:")
    print(f"  熵: {calculate_entropy(concentrated):.3f}, 归一化熵: {normalized_entropy(concentrated):.3f}")


if __name__ == "__main__":
    main()
