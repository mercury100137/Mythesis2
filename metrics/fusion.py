"""
AHP层次分析法 + 熵权法融合
主观(AHP)和客观(熵权)权重结合
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class MetricWeights:
    """指标权重容器"""
    entropy_weight: float
    variety_weight: float
    repeat_weight: float
    cocoon_index: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'H@K_weight': self.entropy_weight,
            'N@K_weight': self.variety_weight,
            'R_weight': self.repeat_weight,
            'cocoon_index': self.cocoon_index
        }


class AHPCalculator:
    """层次分析法计算主观权重，对指标进行两两比较"""

    def __init__(self, metrics: List[str]):
        self.metrics = metrics
        self.n = len(metrics)
        self.matrix = self._create_default_matrix()

    def _create_default_matrix(self) -> np.ndarray:
        """创建等权重比较矩阵"""
        return np.ones((self.n, self.n))

    def set_preference(self, i: int, j: int, value: float):
        """设置指标i相对于指标j的偏好，1-9标度"""
        if i >= self.n or j >= self.n:
            raise ValueError("Index out of bounds")
        self.matrix[i, j] = value
        self.matrix[j, i] = 1.0 / value

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        """列归一化"""
        col_sums = matrix.sum(axis=0)
        return matrix / col_sums

    def _calculate_weights(self, normalized: np.ndarray) -> np.ndarray:
        """从归一化矩阵计算权重"""
        return normalized.mean(axis=1)

    def _check_consistency(self, matrix: np.ndarray, weights: np.ndarray) -> Tuple[float, bool]:
        """计算一致性比率CR，CR<0.1表示通过一致性检验"""
        n = matrix.shape[0]
        weighted_sum = matrix @ weights
        lambda_max = np.mean(weighted_sum / weights)
        ci = (lambda_max - n) / (n - 1) if n > 1 else 0
        ri_values = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12,
                     6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        ri = ri_values.get(n, 1.49)
        cr = ci / ri if ri > 0 else 0
        return cr, cr < 0.1

    def calculate(self) -> Tuple[np.ndarray, float, bool]:
        """计算AHP权重和一致性比率"""
        normalized = self._normalize(self.matrix.copy())
        weights = self._calculate_weights(normalized)
        cr, is_consistent = self._check_consistency(self.matrix, weights)
        return weights, cr, is_consistent


class EntropyWeightCalculator:
    """熵权法计算客观权重，熵值越高权重越低"""

    def __init__(self, metrics: List[str]):
        self.metrics = metrics

    def calculate(self, data: np.ndarray) -> np.ndarray:
        """根据数据矩阵计算熵权"""
        n_samples, n_metrics = data.shape
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        range_vals = data_max - data_min
        range_vals[range_vals == 0] = 1
        normalized = (data - data_min) / range_vals
        p = normalized / (normalized.sum(axis=0) + 1e-10)
        entropy = np.zeros(n_metrics)
        for j in range(n_metrics):
            pj = p[:, j]
            pj = pj[pj > 0]
            if len(pj) > 0:
                entropy[j] = -np.sum(pj * np.log(pj)) / np.log(n_samples)
        diversity = 1 - entropy
        weights = diversity / (diversity.sum() + 1e-10)
        return weights


class CocoonFusionIndex:
    """融合AHP和熵权，计算茧房综合指数"""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.ahp = AHPCalculator(['entropy', 'variety', 'repeat'])
        self.entropy_calc = EntropyWeightCalculator(['entropy', 'variety', 'repeat'])
        self.weights = None

    def set_ahp_preferences(self, preferences: List[Tuple[int, int, float]]):
        """设置AHP偏好偏好列表: [(i, j, value), ...]"""
        for i, j, val in preferences:
            self.ahp.set_preference(i, j, val)

    def fit(self, data: np.ndarray) -> MetricWeights:
        """计算融合权重"""
        ahp_weights, cr, is_consistent = self.ahp.calculate()
        entropy_weights = self.entropy_calc.calculate(data)
        fused_weights = self.alpha * ahp_weights + (1 - self.alpha) * entropy_weights
        fused_weights = fused_weights / fused_weights.sum()
        self.weights = fused_weights
        return MetricWeights(
            entropy_weight=fused_weights[0],
            variety_weight=fused_weights[1],
            repeat_weight=fused_weights[2],
            cocoon_index=0.0
        )

    def calculate_cocoon_index(
        self,
        entropy: float,
        variety: int,
        repeat_rate: float,
        normalize: bool = True
    ) -> float:
        """计算茧房指数，越高表示茧房越严重"""
        if self.weights is None:
            raise ValueError("Must call fit() first")
        if normalize:
            entropy_norm = min(entropy / 4.0, 1.0)
            variety_norm = min(variety / 20.0, 1.0)
            repeat_norm = repeat_rate
        else:
            entropy_norm = entropy
            variety_norm = variety
            repeat_norm = repeat_rate
        cocoon = (self.weights[2] * repeat_norm +
                  self.weights[0] * (1 - entropy_norm) +
                  self.weights[1] * (1 - variety_norm))
        return cocoon


class CocoonClassifier:
    """茧房严重程度分类器"""

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


class CocoonAnalyzer:
    """分析用户茧房状态"""

    def __init__(self, alpha: float = 0.5):
        self.fusion = CocoonFusionIndex(alpha=alpha)

    def analyze_window(self, categories: List[str], news_ids: List[str] = None) -> Dict:
        """分析单个时间窗口的茧房指标"""
        from .entropy import calculate_entropy, normalized_entropy
        from .variety import calculate_variety, simpson_diversity
        from .repeat_rate import repeat_rate
        entropy_val = calculate_entropy(categories)
        norm_entropy = normalized_entropy(categories)
        variety_val = calculate_variety(categories)
        simpson = simpson_diversity(categories)
        repeat_val = repeat_rate(categories)
        return {
            'entropy': entropy_val,
            'normalized_entropy': norm_entropy,
            'variety': variety_val,
            'simpson_diversity': simpson,
            'repeat_rate': repeat_val,
            'n_items': len(categories)
        }

    def analyze_trend(self, windows: List[Tuple[str, List[str]]]) -> pd.DataFrame:
        """分析多个窗口的趋势"""
        results = []
        for ts, cats in windows:
            r = self.analyze_window(cats)
            r['timestamp'] = ts
            results.append(r)
        df = pd.DataFrame(results)
        if len(df) >= 2:
            x = np.arange(len(df))
            n = len(x)
            y = df['normalized_entropy'].values
            df['entropy_trend'] = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
                                   (n * np.sum(x * x) - np.sum(x) ** 2)
            y = df['variety'].values
            df['variety_trend'] = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
                                   (n * np.sum(x * x) - np.sum(x) ** 2)
            y = df['repeat_rate'].values
            df['repeat_trend'] = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
                                 (n * np.sum(x * x) - np.sum(x) ** 2)
        return df
