"""推荐器基类与稀疏交互矩阵。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


@dataclass
class InteractionMatrix:
    """用户-物品稀疏交互矩阵 + ID 双向映射。"""

    matrix: csr_matrix
    user_to_idx: Dict[str, int]
    item_to_idx: Dict[str, int]
    idx_to_user: List[str] = field(init=False)
    idx_to_item: List[str] = field(init=False)

    def __post_init__(self):
        self.idx_to_user = [None] * len(self.user_to_idx)
        for u, i in self.user_to_idx.items():
            self.idx_to_user[i] = u
        self.idx_to_item = [None] * len(self.item_to_idx)
        for it, i in self.item_to_idx.items():
            self.idx_to_item[i] = it

    @property
    def n_users(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_items(self) -> int:
        return self.matrix.shape[1]

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'item_id',
        user_to_idx: Optional[Dict[str, int]] = None,
        item_to_idx: Optional[Dict[str, int]] = None,
    ) -> "InteractionMatrix":
        df = df[[user_col, item_col]].dropna().astype(str)
        if user_to_idx is None:
            users = df[user_col].unique().tolist()
            user_to_idx = {u: i for i, u in enumerate(users)}
        if item_to_idx is None:
            items = df[item_col].unique().tolist()
            item_to_idx = {it: i for i, it in enumerate(items)}
        # 过滤映射外的行
        df = df[df[user_col].isin(user_to_idx) & df[item_col].isin(item_to_idx)]
        rows = df[user_col].map(user_to_idx).to_numpy()
        cols = df[item_col].map(item_to_idx).to_numpy()
        data = np.ones(len(df), dtype=np.float32)
        mat = csr_matrix(
            (data, (rows, cols)),
            shape=(len(user_to_idx), len(item_to_idx)),
        )
        # 同一对去重 (取 1)
        mat.sum_duplicates()
        mat.data = np.ones_like(mat.data)
        return cls(matrix=mat, user_to_idx=user_to_idx, item_to_idx=item_to_idx)


class BaseRecommender(ABC):
    """所有推荐器的统一接口。"""

    name: str = "base"

    def __init__(self):
        self.inter: Optional[InteractionMatrix] = None

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "BaseRecommender":
        ...

    @abstractmethod
    def _score_user(self, user_idx: int) -> np.ndarray:
        """返回 shape=(n_items,) 的打分向量。"""

    def recommend(
        self,
        users: Iterable[str],
        k: int = 10,
        exclude_seen: bool = True,
    ) -> Dict[str, List[str]]:
        assert self.inter is not None, "请先 fit"
        out: Dict[str, List[str]] = {}
        n_items = self.inter.n_items
        k_eff = min(k, n_items)
        for u in users:
            uidx = self.inter.user_to_idx.get(str(u))
            if uidx is None:
                out[u] = []
                continue
            scores = self._score_user(uidx).astype(np.float64, copy=False)
            if exclude_seen:
                seen = self.inter.matrix[uidx].indices
                if len(seen):
                    scores[seen] = -np.inf
            if k_eff >= n_items:
                top_idx = np.argsort(-scores)[:k_eff]
            else:
                # 部分排序
                cand = np.argpartition(-scores, k_eff)[:k_eff]
                top_idx = cand[np.argsort(-scores[cand])]
            out[u] = [self.inter.idx_to_item[i] for i in top_idx if np.isfinite(scores[i])]
        return out
