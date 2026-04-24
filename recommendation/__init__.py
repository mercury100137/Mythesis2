"""推荐算法模块: 基于真实 MIND 数据训练并产生 Top-K 推荐。

各模型统一接口:
    model.fit(train_df)                                # train_df: user_id,item_id,(timestamp)
    model.recommend(users, k, exclude_seen=True)       # -> {user_id: [item_id, ...]}
base.py 👉 基类（规则定义）
bpr.py / item_knn.py / user_knn.py 👉 具体算法实现
recbole_adapter.py 👉 桥接器（调用RecBole的工具）
runner.py 👉 真正执行流程的入口（重点
"""

from .base import BaseRecommender, InteractionMatrix
from .popularity import PopularityRecommender
from .item_knn import ItemKNNRecommender
from .user_knn import UserKNNRecommender

__all__ = [
    'BaseRecommender',
    'InteractionMatrix',
    'PopularityRecommender',
    'ItemKNNRecommender',
    'UserKNNRecommender',
]
