"""
可视化模块 - 用户茧房报告生成
"""

import numpy as np
from collections import Counter
from typing import Dict, List
import io
import base64


def generate_user_report(user_id: str, categories: List[str], cocoon_metrics: Dict,
                        algorithms: Dict[str, List[str]] = None,
                        trends: Dict = None) -> Dict[str, str]:
    """生成用户茧房完整报告"""
    from cocoon_detect.visualization.radar_chart import create_category_bar
    from cocoon_detect.visualization.line_chart import create_entropy_line, create_cocoon_index_chart

    reports = {}
    bar_img = create_category_bar(categories, f"用户{user_id} - 类别分布")
    reports['category_bar'] = bar_img

    if trends and 'entropy' in trends:
        entropy_img = create_entropy_line(trends['entropy'], title=f"用户{user_id} - 熵趋势")
        reports['entropy_trend'] = entropy_img

    if 'cocoon_index' in cocoon_metrics:
        index_img = create_cocoon_index_chart(
            cocoon_metrics['cocoon_index'], title=f"用户{user_id} - 茧房指数"
        )
        reports['cocoon_index'] = index_img

    return reports


def generate_summary_stats(user_categories: List[str], cocoon_metrics: Dict) -> str:
    """生成用户茧房状态文本摘要"""
    import sys
    from pathlib import Path
    if __name__ == "__main__":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from cocoon_detect.metrics.entropy import calculate_entropy
    from cocoon_detect.metrics.variety import calculate_variety
    from cocoon_detect.metrics.repeat_rate import repeat_rate

    entropy = calculate_entropy(user_categories)
    variety = calculate_variety(user_categories)
    repeat = repeat_rate(user_categories)
    counter = Counter(user_categories)
    top_cat = counter.most_common(1)[0]

    summary = []
    summary.append(f"用户茧房摘要")
    summary.append(f"总分析条数: {len(user_categories)}")
    summary.append(f"独特类别数: {variety}")
    summary.append(f"信息熵: {entropy:.3f}")
    summary.append(f"重复率: {repeat:.3f}")
    summary.append(f"主类别: {top_cat[0]} ({top_cat[1]}条)")

    if 'cocoon_index' in cocoon_metrics:
        idx = cocoon_metrics['cocoon_index']
        if isinstance(idx, list):
            idx = idx[-1] if idx else 0
        summary.append(f"茧房综合指数: {idx:.3f}")

    return "\n".join(summary)


if __name__ == "__main__":
    # 测试用户报告模块
    print("测试用户报告模块...")
    categories = ['news', 'sports', 'tech', 'news', 'sports', 'health']
    # cocoon_index 应该是时间序列列表
    cocoon_metrics = {'cocoon_index': [0.45, 0.52, 0.48, 0.55, 0.60]}

    summary = generate_summary_stats(categories, cocoon_metrics)
    print("  生成用户摘要:")
    for line in summary.split('\n'):
        print(f"    {line}")

    reports = generate_user_report('U123', categories, cocoon_metrics)
    print(f"  生成报告图表数: {len(reports)}")
    for key in reports:
        print(f"    - {key}: {len(reports[key])} 字符")
    print("用户报告模块测试完成")
