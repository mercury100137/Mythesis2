"""BPR (Bayesian Personalized Ranking) 矩阵分解。

依赖可选库 ``implicit``。若未安装则在 fit 时抛出 ImportError，
runner 会自动跳过该模型，不影响整体流程。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseRecommender, InteractionMatrix


class BPRRecommender(BaseRecommender):
    name = "BPR"

    def __init__(self, factors: int = 64, iterations: int = 30, lr: float = 0.01):
        super().__init__()
        self.factors = factors
        self.iterations = iterations
        self.lr = lr
        self._model = None

    def fit(self, train: pd.DataFrame) -> "BPRRecommender":
        try:
            from implicit.bpr import BayesianPersonalizedRanking
        except ImportError as e:  # noqa: F401
            raise ImportError(
                "BPR 需要安装 implicit 库: pip install implicit"
            ) from e
        self.inter = InteractionMatrix.from_dataframe(train)
        self._model = BayesianPersonalizedRanking(
            factors=self.factors,
            iterations=self.iterations,
            learning_rate=self.lr,
            use_gpu=False,
        )
        # implicit 期望 user-item csr
        self._model.fit(self.inter.matrix.astype(np.float32), show_progress=False)
        return self

    def _score_user(self, user_idx: int) -> np.ndarray:
        u_factors = self._model.user_factors[user_idx]
        scores = self._model.item_factors @ u_factors
        return np.asarray(scores).ravel()
