"""训练多种推荐模型并输出每用户 Top-K 推荐结果文件。

用法:
    python -m cocoon_detect.recommendation.runner --topk 10 --sample-users 500

输出:
    config/reports/recommendations/<algo>_topk.csv   每行: user_id,item_ids (空格分隔)
    config/reports/recommendations/<algo>_long.csv   每行: user_id,rank,item_id
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import cocoon_detect.config.settings as cfg
from cocoon_detect.data.loader import MINDLoader
from cocoon_detect.data.converter import MINDToRecBoleConverter, build_item_feature_map

from cocoon_detect.recommendation.base import BaseRecommender
from cocoon_detect.recommendation.popularity import PopularityRecommender
from cocoon_detect.recommendation.item_knn import ItemKNNRecommender
from cocoon_detect.recommendation.user_knn import UserKNNRecommender


RECS_DIR = Path(cfg.REPORT_DIR) / "recommendations"
RECS_DIR.mkdir(parents=True, exist_ok=True)


def load_train_test() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """返回 train_inter / test_inter / item_id->category 映射。"""
    print("[runner] 加载训练集 (MINDsmall_train)...")
    train_loader = MINDLoader(cfg.MIND_TRAIN_DIR)
    train_loader.load_all()
    train_clicks = train_loader.get_click_interactions()
    train_hist = train_loader.get_history_interactions()
    train_inter = pd.concat([train_hist, train_clicks], ignore_index=True)
    train_inter = train_inter.drop_duplicates(subset=['user_id', 'item_id'])

    print("[runner] 加载测试集 (MINDsmall_dev)...")
    dev_loader = MINDLoader(cfg.MIND_DEV_DIR)
    dev_loader.load_all()
    test_inter = dev_loader.get_click_interactions()

    item_cat = build_item_feature_map(
        pd.concat([train_loader.news_df, dev_loader.news_df], ignore_index=True)
        .drop_duplicates(subset=['news_id']),
        feature='category',
    )
    print(
        f"[runner] train交互: {len(train_inter)} / test交互: {len(test_inter)} / "
        f"items: {len(item_cat)}"
    )
    return train_inter, test_inter, item_cat


def build_models(
    include_bpr: bool = True,
    include_recbole: bool = False,
    recbole_data_dir: str = None,
) -> List[BaseRecommender]:
    models: List[BaseRecommender] = [
        PopularityRecommender(),
        ItemKNNRecommender(k_neighbors=50),
        UserKNNRecommender(k_neighbors=50),
    ]
    if include_bpr:
        try:
            from cocoon_detect.recommendation.bpr import BPRRecommender
            models.append(BPRRecommender(factors=64, iterations=30))
        except Exception as e:
            print(f"[runner] 跳过 BPR: {e}")
    if include_recbole and recbole_data_dir:
        try:
            from cocoon_detect.recommendation.recbole_adapter import RecBoleRecommender
            for rb_model in ["BPR", "NeuMF"]:
                models.append(
                    RecBoleRecommender(
                        model_name=rb_model,
                        data_path=recbole_data_dir,
                        epochs=10,
                    )
                )
        except Exception as e:
            print(f"[runner] 跳过 RecBole: {e}")
    return models


def save_recommendations(algo: str, recs: Dict[str, List[str]]) -> Tuple[Path, Path]:
    """保存推荐结果：宽表（一行一用户）+ 长表（user,rank,item）。"""
    wide_rows = [
        {'user_id': u, 'item_ids': ' '.join(items)}
        for u, items in recs.items() if items
    ]
    wide_df = pd.DataFrame(wide_rows)
    wide_path = RECS_DIR / f"{algo}_topk.csv"
    wide_df.to_csv(wide_path, index=False)

    long_rows = [
        {'user_id': u, 'rank': r + 1, 'item_id': it}
        for u, items in recs.items()
        for r, it in enumerate(items)
    ]
    long_df = pd.DataFrame(long_rows)
    long_path = RECS_DIR / f"{algo}_long.csv"
    long_df.to_csv(long_path, index=False)
    return wide_path, long_path


def run(
    topk: int = 10,
    sample_users: int = 500,
    include_bpr: bool = True,
    include_recbole: bool = False,
    prepare_recbole_files: bool = False,
) -> Dict[str, Dict[str, List[str]]]:
    train_inter, test_inter, _ = load_train_test()

    if prepare_recbole_files:
        print("[runner] 生成 RecBole atomic 文件...")
        MINDToRecBoleConverter().convert()

    # 评估用户: 在 test 中有点击且在 train 中出现过的
    test_users = set(test_inter['user_id'].astype(str).unique())
    train_users = set(train_inter['user_id'].astype(str).unique())
    overlap_users = sorted(test_users & train_users)
    if sample_users and sample_users < len(overlap_users):
        overlap_users = overlap_users[:sample_users]
    print(f"[runner] 评估用户数: {len(overlap_users)}")

    models = build_models(
        include_bpr=include_bpr,
        include_recbole=include_recbole,
        recbole_data_dir=str(Path(cfg.BASE_DIR) / "recbole_data" / "mind"),
    )

    all_recs: Dict[str, Dict[str, List[str]]] = {}
    for model in models:
        t0 = time.time()
        print(f"\n[runner] 训练 {model.name} ...")
        try:
            model.fit(train_inter)
        except Exception as e:
            print(f"[runner] {model.name} 训练失败: {e}")
            continue
        print(f"[runner] {model.name} 训练耗时 {time.time() - t0:.1f}s，生成 Top-{topk}...")
        recs = model.recommend(overlap_users, k=topk, exclude_seen=True)
        wide_p, long_p = save_recommendations(model.name, recs)
        print(f"[runner] {model.name} 结果: {wide_p}")
        all_recs[model.name] = recs
    return all_recs


def main():
    parser = argparse.ArgumentParser(description="训练真实推荐模型并产出 Top-K")
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--sample-users', type=int, default=500,
                        help='评估/推荐的用户采样数，0 表示全部')
    parser.add_argument('--no-bpr', action='store_true')
    parser.add_argument('--recbole', action='store_true',
                        help='额外使用 RecBole 训练 BPR/NeuMF')
    parser.add_argument('--prepare-recbole', action='store_true',
                        help='生成 .inter atomic 文件')
    args = parser.parse_args()

    run(
        topk=args.topk,
        sample_users=args.sample_users if args.sample_users > 0 else None,
        include_bpr=not args.no_bpr,
        include_recbole=args.recbole,
        prepare_recbole_files=args.prepare_recbole or args.recbole,
    )


if __name__ == "__main__":
    main()
