"""UserKNN: 用户相似度协同过滤 (cosine, top-k 邻居)。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from .base import BaseRecommender, InteractionMatrix


class UserKNNRecommender(BaseRecommender):
    name = "UserKNN"

    def __init__(self, k_neighbors: int = 50):
        super().__init__()
        self.k_neighbors = k_neighbors
        self._user_norm: csr_matrix | None = None

    def fit(self, train: pd.DataFrame) -> "UserKNNRecommender":
        self.inter = InteractionMatrix.from_dataframe(train)
        self._user_norm = normalize(self.inter.matrix.astype(np.float32), norm='l2', axis=1).tocsr()
        return self

    def _score_user(self, user_idx: int) -> np.ndarray:
        u = self._user_norm[user_idx]
        sims = (u @ self._user_norm.T).toarray().ravel()  # (n_users,)
        sims[user_idx] = 0.0
        if self.k_neighbors < len(sims):
            cutoff_idx = np.argpartition(-sims, self.k_neighbors)[:self.k_neighbors]
            mask = np.zeros_like(sims)
            mask[cutoff_idx] = sims[cutoff_idx]
            sims = mask
        # scores = sum_v sim(u,v) * R[v]
        sims_sp = csr_matrix(sims.reshape(1, -1))
        scores = sims_sp @ self.inter.matrix
        return np.asarray(scores.todense()).ravel()
