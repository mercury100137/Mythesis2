"""
统计分析模块
ANOVA方差分析、Tukey多重比较、效应量Cohen's d
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, tukey_hsd
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class AnovaResult:
    """ANOVA结果"""
    f_statistic: float
    p_value: float
    significant: bool
    effect_size: float


@dataclass
class TukeyResult:
    """Tukey HSD结果"""
    group1: str
    group2: str
    meandiff: float
    p_value: float
    lower: float
    upper: float
    reject: bool


@dataclass
class CohensDResult:
    """Cohen's d效应量结果"""
    group1: str
    group2: str
    cohens_d: float
    interpretation: str


class AnovaAnalyzer:
    """单因素方差分析，比较不同算法或用户群体"""

    def __init__(self):
        self.results = None

    def test(self, groups: Dict[str, List[float]], alpha: float = 0.05) -> AnovaResult:
        """执行单因素方差分析"""
        group_values = [v for v in groups.values() if len(v) >= 2]
        if len(group_values) < 2:
            return AnovaResult(0, 1, False, 0)
        f_stat, p_value = f_oneway(*group_values)
        all_data = np.concatenate(group_values)
        grand_mean = np.mean(all_data)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_values)
        ss_total = sum((all_data - grand_mean) ** 2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        return AnovaResult(
            f_statistic=float(f_stat),
            p_value=float(p_value),
            significant=p_value < alpha,
            effect_size=float(eta_squared)
        )

    def test_multiple_metrics(self, df: pd.DataFrame, metric_col: str,
                               group_col: str, alpha: float = 0.05) -> Dict[str, AnovaResult]:
        """对多个指标分组检验"""
        groups = df.groupby(group_col)[metric_col].apply(list).to_dict()
        return {metric_col: self.test(groups, alpha)}


class TukeyAnalyzer:
    """Tukey HSD事后多重比较"""

    def test(self, groups: Dict[str, List[float]], alpha: float = 0.05) -> List[TukeyResult]:
        """执行Tukey HSD检验"""
        group_names = list(groups.keys())
        group_values = [np.array(groups[k]) for k in group_names]
        if len(group_values) < 2:
            return []
        result = tukey_hsd(*group_values)
        results = []
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                results.append(TukeyResult(
                    group1=group_names[i],
                    group2=group_names[j],
                    meandiff=float(result.reject[i, j]) if hasattr(result, 'reject') else 0,
                    p_value=float(result.pvalue[i, j]) if hasattr(result, 'pvalue') else 1,
                    lower=0, upper=0,
                    reject=bool(result.reject[i, j]) if hasattr(result, 'reject') else False
                ))
        return results


class EffectSizeAnalyzer:
    """Cohen's d效应量计算"""

    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """计算两组间Cohen's d"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        return float(d)

    @staticmethod
    def interpret(d: float) -> str:
        """解读Cohen's d: |d|<0.2微小, 0.2-0.5小, 0.5-0.8中, >=0.8大"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "微小"
        elif abs_d < 0.5:
            return "小"
        elif abs_d < 0.8:
            return "中"
        else:
            return "大"

    def compare_pairs(self, groups: Dict[str, List[float]], reference: str = None) -> List[CohensDResult]:
        """比较所有组对或与参照组比较"""
        results = []
        group_names = list(groups.keys())
        if reference:
            ref_values = np.array(groups[reference])
            for name in group_names:
                if name != reference:
                    d = self.cohens_d(ref_values, np.array(groups[name]))
                    results.append(CohensDResult(
                        group1=reference,
                        group2=name,
                        cohens_d=d,
                        interpretation=self.interpret(d)
                    ))
        else:
            for i in range(len(group_names)):
                for j in range(i + 1, len(group_names)):
                    d = self.cohens_d(
                        np.array(groups[group_names[i]]),
                        np.array(groups[group_names[j]])
                    )
                    results.append(CohensDResult(
                        group1=group_names[i],
                        group2=group_names[j],
                        cohens_d=d,
                        interpretation=self.interpret(d)
                    ))
        return results


class CorrelationAnalyzer:
    """指标间相关性分析"""

    @staticmethod
    def pearson_r(x: List[float], y: List[float]) -> Tuple[float, float]:
        """Pearson相关系数"""
        r, p = stats.pearsonr(x, y)
        return float(r), float(p)

    @staticmethod
    def spearman_r(x: List[float], y: List[float]) -> Tuple[float, float]:
        """Spearman等级相关系数"""
        r, p = stats.spearmanr(x, y)
        return float(r), float(p)

    def correlation_matrix(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """计算相关性矩阵"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        return df[columns].corr(method='pearson')

    def analyze_metrics_correlation(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """分析茧房指标间的相关性"""
        metrics = ['entropy', 'variety', 'repeat_rate', 'cocoon_index']
        available = [m for m in metrics if m in df.columns]
        results = {}
        for i, m1 in enumerate(available):
            for m2 in available[i + 1:]:
                r, p = self.pearson_r(df[m1].tolist(), df[m2].tolist())
                results[f"{m1}_vs_{m2}"] = {
                    'pearson_r': r,
                    'p_value': p,
                    'significant': p < 0.05
                }
        return results
