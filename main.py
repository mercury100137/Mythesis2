"""
信息茧房检测系统 - 主程序入口
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import cocoon_detect.config.settings as cfg
from cocoon_detect.data.loader import MINDLoader
from cocoon_detect.data.converter import simple_converter
from cocoon_detect.metrics.entropy import calculate_entropy, normalized_entropy
from cocoon_detect.metrics.variety import calculate_variety
from cocoon_detect.metrics.repeat_rate import repeat_rate
from cocoon_detect.metrics.fusion import CocoonFusionIndex, CocoonAnalyzer
from cocoon_detect.detection.sliding_window import SlidingWindow, CocoonTrendDetector, CocoonClassifier
from cocoon_detect.stat_analysis.stat_tests import AnovaAnalyzer, EffectSizeAnalyzer
from cocoon_detect.recommendation.runner import run as run_recommenders


def load_data():
    """加载MIND数据集"""
    print("=" * 60)
    print("加载MIND数据集")
    print("=" * 60)

    loader = MINDLoader(cfg.MIND_DEV_DIR)
    news_df, beh_df = loader.load_all()

    print(f"新闻记录: {len(news_df)}")
    print(f"行为记录: {len(beh_df)}")
    print(f"类别数: {news_df['category'].nunique()}")
    print(f"用户数: {beh_df['user_id'].nunique()}")

    return loader, news_df, beh_df


def compute_user_metrics(loader, beh_df, sample_users=100):
    """计算样本用户的茧房指标"""
    print("\n" + "=" * 60)
    print("计算用户茧房指标")
    print("=" * 60)

    users = beh_df['user_id'].unique()[:sample_users]
    results = []

    for user_id in users:
        history = loader.get_user_history(user_id)
        if len(history) < 5:
            continue

        categories = [loader.get_news_category(nid) for nid in history]

        ent = calculate_entropy(categories)
        norm_ent = normalized_entropy(categories)
        variety = calculate_variety(categories)
        repeat = repeat_rate(categories)

        severity = CocoonClassifier.classify(norm_ent, variety, repeat)

        results.append({
            'user_id': user_id,
            'n_history': len(history),
            'entropy': ent,
            'normalized_entropy': norm_ent,
            'variety': variety,
            'repeat_rate': repeat,
            'severity': severity
        })

    df = pd.DataFrame(results)
    print(f"计算了{len(df)}个用户的指标")
    print(f"\n茧房严重程度分布:")
    print(df['severity'].value_counts())

    return df


def analyze_trends(loader, beh_df):
    """分析茧房随时间的变化趋势"""
    print("\n" + "=" * 60)
    print("分析茧房趋势")
    print("=" * 60)

    sample_user = beh_df[beh_df['history'].str.len() > 20]['user_id'].iloc[0]
    print(f"分析用户: {sample_user}")

    history = loader.get_user_history(sample_user)
    categories = [loader.get_news_category(nid) for nid in history]

    window = SlidingWindow(window_size=cfg.SLIDING_WINDOW_SIZE, step_size=3)

    n = len(categories)
    timestamps = pd.date_range('2019-11-01', periods=n, freq='D').tolist()
    timestamps_str = [str(t) for t in timestamps]

    windows = window.create_windows(timestamps_str, categories)

    print(f"创建了{len(windows)}个时间窗口")

    detector = CocoonTrendDetector()
    entropy_series = []
    variety_series = []
    repeat_series = []

    for w in windows:
        cats = w['categories']
        entropy_series.append(normalized_entropy(cats))
        variety_series.append(calculate_variety(cats))
        repeat_series.append(repeat_rate(cats))

    result = detector.detect_cocoon(entropy_series, variety_series, repeat_series)
    print(f"\n茧房检测结果:")
    print(f"  检测到茧房: {result['cocoon_detected']}")
    print(f"  严重程度: {result['severity']}")
    print(f"  信号: {result['signals']}")
    print(f"  熵趋势斜率: {result['entropy_trend']['slope']:.4f}")

    return windows, result


def real_algorithm_comparison(topk: int = 10, sample_users: int = 500):
    """基于真实推荐模型计算各算法的茧房指数对比。

    1) 训练多种模型 (Popularity / ItemKNN / UserKNN / 可选 BPR) 并生成 Top-K 推荐
    2) 用 news.tsv 的 category 把推荐 item 列表映射为类别序列
    3) 对每个用户计算 H@K, N@K, R 与融合茧房指数
    4) 跨算法做 ANOVA + 与 ItemKNN 的 Cohen's d 对比
    """
    print("\n" + "=" * 60)
    print("真实推荐模型 - 茧房指数对比")
    print("=" * 60)

    # 训练模型并产出 Top-K (推荐结果文件已由 runner 落盘)
    all_recs = run_recommenders(
        topk=topk,
        sample_users=sample_users,
        include_bpr=True,
        include_recbole=False,
        prepare_recbole_files=False,
    )

    # 拿 item -> category 映射 (合并 train + dev 的 news.tsv)
    train_loader = MINDLoader(cfg.MIND_TRAIN_DIR)
    train_loader.load_news()
    dev_loader = MINDLoader(cfg.MIND_DEV_DIR)
    dev_loader.load_news()
    news_all = pd.concat([train_loader.news_df, dev_loader.news_df], ignore_index=True) \
        .drop_duplicates(subset=['news_id'])
    item_cat = dict(zip(news_all['news_id'].astype(str), news_all['category'].fillna('unknown')))

    # 逐用户计算指标
    fusion = CocoonFusionIndex(alpha=0.5)
    rows = []
    for algo, recs in all_recs.items():
        per_user = []
        for uid, items in recs.items():
            if not items:
                continue
            cats = [item_cat.get(str(it), 'unknown') for it in items]
            ent = normalized_entropy(cats)
            var = calculate_variety(cats)
            rep = repeat_rate(cats)
            per_user.append((uid, ent, var, rep))

        if not per_user:
            continue
        # 用本算法所有用户的指标拟合融合权重
        data = np.array([[e, v, r] for _, e, v, r in per_user], dtype=float)
        fusion.fit(data)
        for uid, ent, var, rep in per_user:
            cocoon = fusion.calculate_cocoon_index(ent, var, rep, normalize=True)
            rows.append({
                'algorithm': algo,
                'user_id': uid,
                'entropy': ent,
                'variety': var,
                'repeat_rate': rep,
                'cocoon_index': cocoon,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("没有可用推荐结果，跳过算法对比")
        return df

    print(f"\n各算法均值:")
    print(df.groupby('algorithm')[['entropy', 'variety', 'repeat_rate', 'cocoon_index']].mean())

    groups = df.groupby('algorithm')['cocoon_index'].apply(list).to_dict()
    if len(groups) >= 2:
        anova_result = AnovaAnalyzer().test(groups)
        print(f"\nANOVA: F={anova_result.f_statistic:.4f}, "
              f"p={anova_result.p_value:.4f}, η²={anova_result.effect_size:.4f}")

        ref = 'ItemKNN' if 'ItemKNN' in groups else next(iter(groups))
        sizes = EffectSizeAnalyzer().compare_pairs(groups, reference=ref)
        print(f"\n与 {ref} 的 Cohen's d (按效应量绝对值排序):")
        for s in sorted(sizes, key=lambda x: abs(x.cohens_d), reverse=True):
            print(f"  {s.group2}: d={s.cohens_d:.3f} ({s.interpretation})")

    return df


def main():
    parser = argparse.ArgumentParser(description='信息茧房检测系统')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['data', 'metrics', 'trends', 'analysis', 'full'],
                        help='运行模式')
    parser.add_argument('--sample', type=int, default=100, help='样本用户数')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("信息茧房检测系统")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if args.mode in ['data', 'full']:
        loader, news_df, beh_df = load_data()

    if args.mode in ['metrics', 'full']:
        if 'loader' not in dir():
            loader, news_df, beh_df = load_data()
        metrics_df = compute_user_metrics(loader, beh_df, sample_users=args.sample)
        metrics_df.to_csv(cfg.REPORT_DIR + '/user_metrics.csv', index=False)
        print(f"指标已保存到 {cfg.REPORT_DIR}/user_metrics.csv")

    if args.mode in ['trends', 'full']:
        if 'loader' not in dir():
            loader, news_df, beh_df = load_data()
        windows, trend_result = analyze_trends(loader, beh_df)

    if args.mode in ['analysis', 'full']:
        algo_df = real_algorithm_comparison(topk=10, sample_users=args.sample)
        if not algo_df.empty:
            algo_df.to_csv(cfg.REPORT_DIR + '/algorithm_comparison.csv', index=False)
            print(f"算法对比已保存到 {cfg.REPORT_DIR}/algorithm_comparison.csv")

    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
