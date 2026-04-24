"""可选: 通过 RecBole 训练 BPR/NeuMF/ItemKNN 等模型并产出 Top-K。

需要先运行 data.converter 生成 recbole_data/mind/*.inter 等文件。
如果 recbole 未安装，整个模块在 import 时不会失败，只有 fit 时报错。

python -m recbole_adapter
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from base import BaseRecommender


class RecBoleRecommender(BaseRecommender):
    """用 RecBole 训练指定模型 (如 'BPR', 'NeuMF', 'ItemKNN')，再按 Top-K 输出。"""

    def __init__(
        self,
        model_name: str,
        data_path: str,
        dataset: str = "mind",
        config_dict: Optional[Dict] = None,
        epochs: int = 20,
    ):
        super().__init__()
        self.name = f"RecBole-{model_name}"
        self.model_name = model_name
        self.data_path = str(Path(data_path).parent)  # RecBole 要求父目录
        self.dataset = dataset
        self.config_dict = config_dict or {}
        self.epochs = epochs
        self._trainer = None
        self._model = None
        self._dataset = None
        self._dataloader = None

    def fit(self, train: pd.DataFrame) -> "RecBoleRecommender":
        try:
            from recbole.config import Config
            from recbole.data import create_dataset, data_preparation
            from recbole.utils import init_seed, get_model, get_trainer
        except ImportError as e:
            raise ImportError(
                "RecBoleRecommender 需要安装 recbole: pip install recbole"
            ) from e

        cfg = {
            'model': self.model_name,
            'dataset': self.dataset,
            'data_path': self.data_path,
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': 'item_id',
            'TIME_FIELD': 'timestamp',
            'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
            'epochs': self.epochs,
            'train_neg_sample_args': {'distribution': 'uniform', 'sample_num': 1},
        }
        cfg.update(self.config_dict)
        config = Config(model=self.model_name, dataset=self.dataset, config_dict=cfg)
        init_seed(config['seed'], config['reproducibility'])
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
        trainer.fit(train_data, valid_data, saved=False, show_progress=False)
        self._model = model
        self._dataset = dataset
        self._trainer = trainer
        # 构造 user/item 映射供 recommend 使用
        user_tokens = dataset.field2token_id[config['USER_ID_FIELD']]
        item_tokens = dataset.field2token_id[config['ITEM_ID_FIELD']]
        # field2token_id: token -> internal id
        self._user_token_to_id = user_tokens
        self._item_id_to_token = {v: k for k, v in item_tokens.items()}
        return self

    def recommend(
        self, users: Iterable[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        import torch
        out: Dict[str, List[str]] = {}
        self._model.eval()
        with torch.no_grad():
            for u in users:
                uid = self._user_token_to_id.get(str(u))
                if uid is None:
                    out[u] = []
                    continue
                user_tensor = torch.tensor([uid], device=self._model.device)
                scores = self._model.full_sort_predict({
                    'user_id': user_tensor,
                }).detach().cpu().numpy().ravel()
                top = np.argsort(-scores)[:k]
                out[u] = [self._item_id_to_token.get(i, str(i)) for i in top]
        return out

    def _score_user(self, user_idx):
        raise NotImplementedError
