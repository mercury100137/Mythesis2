"""ItemKNN: 物品相似度协同过滤 (cosine, 截断 top-k 邻居)。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from .base import BaseRecommender, InteractionMatrix


class ItemKNNRecommender(BaseRecommender):
    name = "ItemKNN"

    def __init__(self, k_neighbors: int = 50, shrink: float = 0.0):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.shrink = shrink
        self.sim_: csr_matrix | None = None

    def fit(self, train: pd.DataFrame) -> "ItemKNNRecommender":
        self.inter = InteractionMatrix.from_dataframe(train)
        ui = self.inter.matrix.tocsr().astype(np.float32)
        # cosine: 列归一化后 (item x user) @ (user x item)
        item_user = ui.T.tocsr()
        item_user_norm = normalize(item_user, norm='l2', axis=1)
        sim = item_user_norm @ item_user_norm.T  # (n_items, n_items)
        sim = sim.tolil()
        sim.setdiag(0)
        sim = sim.tocsr()
        sim.eliminate_zeros()
        self.sim_ = self._truncate_topk(sim, self.k_neighbors)
        return self

    @staticmethod
    def _truncate_topk(sim: csr_matrix, k: int) -> csr_matrix:
        sim = sim.tocsr()
        n = sim.shape[0]
        rows, cols, vals = [], [], []
        for i in range(n):
            start, end = sim.indptr[i], sim.indptr[i + 1]
            if end - start <= k:
                rows.extend([i] * (end - start))
                cols.extend(sim.indices[start:end].tolist())
                vals.extend(sim.data[start:end].tolist())
            else:
                row_data = sim.data[start:end]
                row_idx = sim.indices[start:end]
                top = np.argpartition(-row_data, k)[:k]
                rows.extend([i] * k)
                cols.extend(row_idx[top].tolist())
                vals.extend(row_data[top].tolist())
        return csr_matrix((vals, (rows, cols)), shape=sim.shape, dtype=np.float32)

    def _score_user(self, user_idx: int) -> np.ndarray:
        user_row = self.inter.matrix[user_idx]  # (1, n_items)
        scores = user_row @ self.sim_  # (1, n_items)
        return np.asarray(scores.todense()).ravel()
