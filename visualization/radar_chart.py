"""
可视化模块 - 类别分布雷达图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from collections import Counter
from typing import List, Dict
import io
import base64


def create_radar_chart(categories: List[str], title: str = "类别分布",
                       max_categories: int = 10, color: str = '#4C78A8') -> str:
    """创建类别分布雷达图"""
    counter = Counter(categories)
    top_cats = counter.most_common(max_categories)
    labels = [c[0] for c in top_cats]
    values = [c[1] for c in top_cats]
    total = sum(values)
    values_norm = [v / total * 100 for v in values]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_plot = values_norm + [values_norm[0]]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values_plot, color=color, alpha=0.25)
    ax.plot(angles, values_plot, color=color, linewidth=2, marker='o', markersize=6)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, max(values_norm) * 1.2)
    ax.set_title(title, size=14, fontweight='bold', pad=20)
    for i, (angle, val) in enumerate(zip(angles[:-1], values_norm)):
        ax.annotate(f'{val:.1f}%', xy=(angle, val),
                    xytext=(angle, val + 5), ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_str


if __name__ == "__main__":
    # 测试雷达图
    test_categories = ['sports'] * 10 + ['news'] * 8 + ['tech'] * 5 + ['health'] * 2
    img = create_radar_chart(test_categories, title="测试雷达图")
    print(f"雷达图生成成功，数据长度: {len(img)} 字符")
    print("这是一个Base64编码的PNG图像")


def create_multi_radar(category_lists: Dict[str, List[str]],
                        title: str = "类别分布对比") -> str:
    """多个用户/算法的雷达图对比"""
    n_groups = len(category_lists)
    colors = ['#4C78A8', '#F58518', '#E45756', '#72B7B2', '#54A24B']
    all_cats = set()
    for cats in category_lists.values():
        all_cats.update(cats)
    labels = sorted(list(all_cats))[:10]
    n_cats = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 6),
                              subplot_kw=dict(polar=True))
    if n_groups == 1:
        axes = [axes]
    for ax, (label, cats), color in zip(axes, category_lists.items(), colors):
        counter = Counter(cats)
        values = [counter.get(c, 0) for c in labels]
        total = sum(values) if sum(values) > 0 else 1
        values_norm = [v / total * 100 for v in values] + [values[0]]
        ax.fill(angles, values_norm, color=color, alpha=0.25)
        ax.plot(angles, values_norm, color=color, linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=8)
        ax.set_title(label, size=12, fontweight='bold', pad=15)
    plt.suptitle(title, size=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_str


def create_category_bar(categories: List[str], title: str = "类别分布") -> str:
    """创建类别分布柱状图"""
    counter = Counter(categories)
    items = counter.most_common(15)
    labels, values = zip(*items)
    total = sum(values)
    percentages = [v / total * 100 for v in values]
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, percentages, color='#4C78A8', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('百分比 (%)')
    ax.set_title(title, fontweight='bold')
    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=9)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_str
