"""
可视化模块 - 熵/指标时间折线图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from typing import List, Dict
import io
import base64


def create_entropy_line(entropy_series: List[float], timestamps: List[str] = None,
                        title: str = "熵随时间变化", color: str = '#4C78A8',
                        show_trend: bool = True) -> str:
    """创建熵时间序列折线图"""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(entropy_series))
    ax.plot(x, entropy_series, color=color, linewidth=2, marker='o',
            markersize=5, label='Entropy')
    ax.fill_between(x,
                    [v - 0.1 for v in entropy_series],
                    [v + 0.1 for v in entropy_series],
                    color=color, alpha=0.1)
    if show_trend and len(entropy_series) >= 2:
        z = np.polyfit(x, entropy_series, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), '--', color='red', linewidth=2, alpha=0.7,
                label=f'Trend (slope={z[0]:.4f})')
    ax.set_xlabel('时间窗口', fontsize=11)
    ax.set_ylabel('归一化熵', fontsize=11)
    ax.set_title(title, fontweight='bold', size=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_str


def create_metrics_comparison(metrics_dict: Dict[str, List[float]],
                               title: str = "多指标对比") -> str:
    """多指标对比折线图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#4C78A8', '#F58518', '#E45756', '#72B7B2', '#54A24B']
    x = np.arange(max(len(v) for v in metrics_dict.values()))
    for i, (name, values) in enumerate(metrics_dict.items()):
        ax.plot(x[:len(values)], values, color=colors[i % len(colors)],
                linewidth=2, marker='o', markersize=4, label=name)
    ax.set_xlabel('时间窗口', fontsize=11)
    ax.set_ylabel('指标值', fontsize=11)
    ax.set_title(title, fontweight='bold', size=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_str


def create_trend_analysis_chart(df: pd.DataFrame, title: str = "茧房趋势分析") -> str:
    """茧房趋势分析图"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    ax1 = axes[0]
    ax1.plot(df.index, df['normalized_entropy'], color='#4C78A8',
             linewidth=2, marker='o', label='Entropy')
    if 'entropy_trend' in df.columns:
        ax1.plot(df.index, df['entropy_trend'], '--', color='red',
                 linewidth=2, label='Trend')
    ax1.set_ylabel('归一化熵', fontsize=11)
    ax1.set_title('熵趋势 (茧房指标)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2 = axes[1]
    ax2.plot(df.index, df['repeat_rate'], color='#E45756',
             linewidth=2, marker='s', label='Repeat Rate')
    if 'repeat_trend' in df.columns:
        ax2.plot(df.index, df['repeat_trend'], '--', color='green',
                 linewidth=2, label='Trend')
    ax2.set_xlabel('时间窗口', fontsize=11)
    ax2.set_ylabel('重复率', fontsize=11)
    ax2.set_title('重复率趋势 (茧房指标)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.suptitle(title, fontweight='bold', size=14, y=1.01)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_str


def create_cocoon_index_chart(cocoon_indices: List[float], timestamps: List[str] = None,
                               threshold: float = 0.5, title: str = "茧房指数变化") -> str:
    """茧房指数折线图，带阈值线"""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cocoon_indices))
    ax.plot(x, cocoon_indices, color='#E45756', linewidth=2, marker='o',
            markersize=6, label='Cocoon Index')
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
               label=f'阈值 ({threshold})')
    ax.fill_between(x, cocoon_indices, threshold,
                    where=np.array(cocoon_indices) > threshold,
                    color='red', alpha=0.3)
    ax.set_xlabel('时间窗口', fontsize=11)
    ax.set_ylabel('茧房指数', fontsize=11)
    ax.set_title(title, fontweight='bold', size=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_str


if __name__ == "__main__":
    print("测试折线图模块...")
    entropy_series = [0.8, 0.7, 0.6, 0.5, 0.4]
    img = create_entropy_line(entropy_series, title="测试熵趋势")
    print(f"  熵趋势图生成成功，长度: {len(img)} 字符")
    
    cocoon_indices = [0.3, 0.4, 0.5, 0.6, 0.7]
    img2 = create_cocoon_index_chart(cocoon_indices, threshold=0.5, title="测试茧房指数")
    print(f"  茧房指数图生成成功，长度: {len(img2)} 字符")
    print("折线图模块测试完成")
