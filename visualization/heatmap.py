"""
可视化模块 - 茧房分布热力图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from typing import List, Dict
import pandas as pd
import io
import base64


def create_user_algo_heatmap(data: pd.DataFrame, value_col: str = 'cocoon_index',
                              title: str = "茧房指数热力图 (用户x算法)",
                              cmap: str = 'RdYlGn_r') -> str:
    """用户-算法茧房指数热力图"""
    pivot = data.pivot_table(
        values=value_col, index='user_id', columns='algorithm', aggfunc='mean'
    )
    if len(pivot) > 30:
        pivot = pivot.head(30)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap=cmap,
                center=pivot.values.mean(), ax=ax,
                cbar_kws={'label': value_col})
    ax.set_title(title, fontweight='bold', size=14)
    ax.set_xlabel('算法', fontsize=11)
    ax.set_ylabel('用户', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_str


def create_distribution_heatmap(cocoon_values: List[float], n_bins: int = 20,
                                 title: str = "茧房分布") -> str:
    """茧房严重程度分布热力图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    hist, xedges, yedges = np.histogram2d(
        cocoon_values, [v * 0.9 for v in cocoon_values], bins=n_bins
    )
    im = ax.imshow(hist.T, origin='lower', aspect='auto',
                   cmap='YlOrRd', extent=[0, n_bins, 0, n_bins])
    ax.set_xlabel('茧房指数区间', fontsize=11)
    ax.set_ylabel('频次', fontsize=11)
    ax.set_title(title, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_str


def create_correlation_heatmap(metrics_df: pd.DataFrame,
                                title: str = "指标相关性") -> str:
    """茧房指标间的相关性热力图"""
    numeric_cols = ['entropy', 'variety', 'repeat_rate', 'cocoon_index']
    available = [c for c in numeric_cols if c in metrics_df.columns]
    if len(available) < 2:
        return None
    corr = metrics_df[available].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, ax=ax, square=True,
                xticklabels=['熵', '多样性', '重复率', '茧房指数'],
                yticklabels=['熵', '多样性', '重复率', '茧房指数'])
    ax.set_title(title, fontweight='bold', size=13)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_str


def create_severity_heatmap(severity_counts: Dict[str, Dict[str, int]],
                             title: str = "各算法茧房严重程度分布") -> str:
    """茧房严重程度分布热力图"""
    df = pd.DataFrame(severity_counts).T
    df = df[['none', 'mild', 'moderate', 'severe']].fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt='.0f', cmap='Reds', ax=ax,
                cbar_kws={'label': '用户数'})
    ax.set_title(title, fontweight='bold', size=13)
    ax.set_xlabel('严重程度', fontsize=11)
    ax.set_ylabel('算法', fontsize=11)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_str
