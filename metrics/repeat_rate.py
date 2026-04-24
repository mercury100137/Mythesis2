"""
兴趣重复率计算
R: 用户重复消费同类内容的程度
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List


def repeat_rate(categories: List[str]) -> float:
    """类别重复率，非独特类别占比"""
    if not categories:
        return 0.0
    counter = Counter(categories)
    n = len(categories)
    total_repeated = sum(count - 1 for count in counter.values() if count > 1)
    return total_repeated / n if n > 0 else 0.0


def category_entropy_ratio(categories: List[str]) -> float:
    """实际熵与最大熵的比值，低值表示重复倾向高"""
    if not categories:
        return 1.0
    counter = Counter(categories)
    n = len(categories)
    probs = [count / n for count in counter.values()]
    h = -sum(p * np.log(p) for p in probs if p > 0)
    h_max = np.log(len(counter))
    return h / h_max if h_max > 0 else 1.0


def sequential_repeat_rate(news_ids: List[str]) -> float:
    """连续重复率，相邻相同item的比例"""
    if not news_ids or len(news_ids) < 2:
        return 0.0
    repeats = sum(1 for i in range(1, len(news_ids)) if news_ids[i] == news_ids[i-1])
    return repeats / (len(news_ids) - 1)


def inter_arrival_concentration(categories: List[str]) -> float:
    """类别聚集程度，高值表示同类内容集中出现"""
    if not categories:
        return 0.0
    counter = Counter(categories)
    most_common_cat = counter.most_common(1)[0][0]
    transitions = 0
    for i in range(len(categories) - 1):
        if categories[i] == most_common_cat or categories[i+1] == most_common_cat:
            transitions += 1
    if len(categories) <= 1:
        return 0.0
    return transitions / (len(categories) - 1)


def interest_repeat_for_user(
    user_categories: List[str],
    user_news_ids: List[str] = None,
    top_k: int = None
) -> Dict[str, float]:
    """计算用户的所有重复率指标"""
    if top_k:
        user_categories = user_categories[:top_k]
        if user_news_ids:
            user_news_ids = user_news_ids[:top_k]
    metrics = {
        'category_repeat_rate': repeat_rate(user_categories),
        'entropy_ratio': category_entropy_ratio(user_categories),
        'total_items': len(user_categories)
    }
    if user_news_ids:
        metrics['sequential_repeat'] = sequential_repeat_rate(user_news_ids)
    metrics['inter_arrival'] = inter_arrival_concentration(user_categories)
    return metrics


class RepeatTracker:
    """跟踪重复率随时间变化"""

    def __init__(self):
        self.history = []

    def add(self, categories: List[str], timestamp: str = None):
        self.history.append({
            'timestamp': timestamp,
            'repeat_rate': repeat_rate(categories),
            'entropy_ratio': category_entropy_ratio(categories),
            'total': len(categories)
        })

    def get_series(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def calculate_trend(self) -> float:
        """正斜率表示重复增加"""
        if len(self.history) < 2:
            return 0.0
        series = self.get_series()
        x = np.arange(len(series))
        y = series['repeat_rate'].values
        n = len(x)
        return (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
               (n * np.sum(x * x) - np.sum(x) ** 2)
