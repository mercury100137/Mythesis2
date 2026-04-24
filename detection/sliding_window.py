"""
滑动时间窗口检测茧房趋势
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable
from datetime import datetime, timedelta


class SlidingWindow:
    """滑动时间窗口，跟踪用户行为随时间变化"""

    def __init__(self, window_size: int = 7, step_size: int = 1):
        self.window_size = window_size
        self.step_size = step_size
        self.windows = []

    def create_windows(self, timestamps: List[str], categories: List[str],
                       news_ids: List[str] = None) -> List[Dict]:
        """从带时间戳的行为数据创建滑动窗口"""
        if len(timestamps) != len(categories):
            raise ValueError("Timestamps and categories must have same length")
        dates = [pd.to_datetime(t) for t in timestamps]
        min_date = min(dates)
        max_date = max(dates)
        windows = []
        current = min_date
        while current <= max_date:
            window_end = current + timedelta(days=self.window_size)
            window_cats = []
            window_news = []
            window_ts = []
            for i, d in enumerate(dates):
                if current <= d < window_end:
                    window_cats.append(categories[i])
                    if news_ids:
                        window_news.append(news_ids[i])
                    window_ts.append(dates[i])
            if window_cats:
                windows.append({
                    'start': current,
                    'end': window_end,
                    'categories': window_cats,
                    'news_ids': window_news if news_ids else None,
                    'timestamps': window_ts,
                    'n_items': len(window_cats)
                })
            current += timedelta(days=self.step_size)
        self.windows = windows
        return windows

    def get_metrics_per_window(self, metrics_func: Callable) -> pd.DataFrame:
        """计算每个窗口的指标"""
        if not self.windows:
            raise ValueError("No windows created")
        results = []
        for i, w in enumerate(self.windows):
            metrics = metrics_func(w['categories'])
            metrics['window_id'] = i
            metrics['start'] = w['start']
            metrics['end'] = w['end']
            metrics['n_items'] = w['n_items']
            results.append(metrics)
        return pd.DataFrame(results)


class CocoonTrendDetector:
    """通过趋势分析检测茧房形成"""

    def __init__(self, min_windows: int = 3):
        self.min_windows = min_windows

    def detect_trend(self, values: List[float], method: str = 'linear') -> Dict:
        """检测指标值的趋势方向"""
        if len(values) < self.min_windows:
            return {'trend': 'insufficient_data', 'slope': 0.0}
        values = np.array(values)
        if method == 'linear':
            x = np.arange(len(values))
            n = len(x)
            slope = (n * np.sum(x * values) - np.sum(x) * np.sum(values)) / \
                    (n * np.sum(x * x) - np.sum(x) ** 2)
            intercept = np.mean(values) - slope * np.mean(x)
            y_pred = slope * x + intercept
            ss_res = np.sum((values - y_pred) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            return {
                'trend': 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable',
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_squared),
                'direction': 'cocoon' if slope < -0.01 else 'diverse'
            }
        elif method == 'moving_avg':
            window = min(3, len(values))
            ma = np.convolve(values, np.ones(window)/window, mode='valid')
            if len(ma) >= 2:
                return {'trend': 'increasing' if ma[-1] > ma[0] else 'decreasing'}
            return {'trend': 'unknown'}
        return {'trend': 'unknown'}

    def detect_cocoon(self, entropy_series: List[float], variety_series: List[float],
                      repeat_series: List[float]) -> Dict:
        """根据多指标检测茧房形成"""
        entropy_trend = self.detect_trend(entropy_series, 'linear')
        variety_trend = self.detect_trend(variety_series, 'linear')
        repeat_trend = self.detect_trend(repeat_series, 'linear')
        cocoon_score = 0
        cocoon_signals = []
        if entropy_trend['slope'] < -0.01:
            cocoon_score += 1
            cocoon_signals.append('entropy_declining')
        if variety_trend['slope'] < -0.1:
            cocoon_score += 1
            cocoon_signals.append('variety_declining')
        if repeat_trend['slope'] > 0.01:
            cocoon_score += 1
            cocoon_signals.append('repeat_increasing')
        severity = 'none'
        if cocoon_score >= 2:
            severity = 'moderate'
        if cocoon_score >= 3:
            severity = 'severe'
        return {
            'cocoon_detected': cocoon_score >= 2,
            'severity': severity,
            'cocoon_score': cocoon_score,
            'signals': cocoon_signals,
            'entropy_trend': entropy_trend,
            'variety_trend': variety_trend,
            'repeat_trend': repeat_trend
        }


class CocoonClassifier:
    """将用户按茧房严重程度分类"""

    SEVERITY_LEVELS = ['none', 'mild', 'moderate', 'severe']

    @staticmethod
    def classify(entropy: float, variety: int, repeat_rate: float,
                 entropy_threshold: float = 0.5, variety_threshold: int = 5,
                 repeat_threshold: float = 0.5) -> str:
        """基于阈值判断茧房严重程度"""
        signals = 0
        if entropy < entropy_threshold:
            signals += 1
        if variety < variety_threshold:
            signals += 1
        if repeat_rate > repeat_threshold:
            signals += 1
        if signals <= 0:
            return 'none'
        elif signals == 1:
            return 'mild'
        elif signals == 2:
            return 'moderate'
        else:
            return 'severe'

    @staticmethod
    def classify_from_index(cocoon_index: float) -> str:
        """基于融合茧房指数分类"""
        if cocoon_index < 0.3:
            return 'none'
        elif cocoon_index < 0.5:
            return 'mild'
        elif cocoon_index < 0.7:
            return 'moderate'
        else:
            return 'severe'
