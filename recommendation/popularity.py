"""全局热门基线: 按 item 训练频次排序。"""

import numpy as np
import pandas as pd

from .base import BaseRecommender, InteractionMatrix


class PopularityRecommender(BaseRecommender):
    name = "Popularity"

    def fit(self, train: pd.DataFrame) -> "PopularityRecommender":
        self.inter = InteractionMatrix.from_dataframe(train)
        # 物品被多少用户点过
        item_pop = np.asarray(self.inter.matrix.sum(axis=0)).ravel()
        self._scores = item_pop.astype(np.float32)
        return self

    def _score_user(self, user_idx: int) -> np.ndarray:
        return self._scores.copy()
