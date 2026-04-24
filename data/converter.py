"""MIND -> RecBole 格式转换。

输出目录结构:
    <output_dir>/mind/
        mind.inter      # 训练集交互 (来自 MINDsmall_train: history + impressions clicks)
        mind.test.inter # 测试集交互 (来自 MINDsmall_dev: impressions 中 label=1)
        mind.item       # item_id, category, subcategory
        mind.user       # user_id

字段类型遵循 RecBole atomic file 规范:
    user_id:token  item_id:token  timestamp:float  category:token
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from cocoon_detect import config as cfg

import cocoon_detect.config.settings as cfg
from cocoon_detect.data.loader import MINDLoader


DATASET_NAME = "mind"


class MINDToRecBoleConverter:
    """将 MIND 训练集 + 验证集转换为 RecBole atomic 文件。"""

    def __init__(
        self,
        train_dir: str = cfg.MIND_TRAIN_DIR,
        dev_dir: str = cfg.MIND_DEV_DIR,
        output_dir: str = None,
        include_history_in_train: bool = True,
    ):
        self.train_dir = Path(train_dir)
        self.dev_dir = Path(dev_dir)
        self.output_dir = Path(output_dir or (Path(cfg.BASE_DIR) / "recbole_data" / DATASET_NAME))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_history_in_train = include_history_in_train

    def _build_split(self, mind_dir: Path, use_history: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
        loader = MINDLoader(str(mind_dir))
        news_df = loader.load_news()
        loader.load_behaviors()
        click_inter = loader.get_click_interactions()
        print("11111111111111")
        if use_history:
            hist_inter = loader.get_history_interactions()
            inter = pd.concat([hist_inter, click_inter], ignore_index=True)
            inter = inter.drop_duplicates(subset=['user_id', 'item_id'])
        else:
            inter = click_inter
        return inter, news_df

    def convert(self) -> Dict[str, Path]:
        print(f"[converter] 处理训练集: {self.train_dir}")
        train_inter, train_news = self._build_split(self.train_dir, self.include_history_in_train)
        print(f"[converter] 处理验证集(测试): {self.dev_dir}")
        test_inter, dev_news = self._build_split(self.dev_dir, use_history=False)

        # 合并 item 表
        items = pd.concat([train_news, dev_news], ignore_index=True)
        items = items.drop_duplicates(subset=['news_id'])
        items = items[['news_id', 'category', 'subcategory']].copy()
        items.columns = ['item_id', 'category', 'subcategory']
        items['category'] = items['category'].fillna('unknown')
        items['subcategory'] = items['subcategory'].fillna('unknown')

        users = pd.concat(
            [train_inter[['user_id']], test_inter[['user_id']]], ignore_index=True
        ).drop_duplicates()

        paths = self._write(train_inter, test_inter, items, users)
        print(
            f"[converter] 完成: train={len(train_inter)} test={len(test_inter)} "
            f"users={len(users)} items={len(items)}"
        )
        return paths

    def _write(self, train_inter, test_inter, items, users) -> Dict[str, Path]:
        train_path = self.output_dir / f"{DATASET_NAME}.inter"
        test_path = self.output_dir / f"{DATASET_NAME}.test.inter"
        item_path = self.output_dir / f"{DATASET_NAME}.item"
        user_path = self.output_dir / f"{DATASET_NAME}.user"

        train_out = train_inter.rename(columns={
            'user_id': 'user_id:token',
            'item_id': 'item_id:token',
            'timestamp': 'timestamp:float',
        })
        test_out = test_inter.rename(columns={
            'user_id': 'user_id:token',
            'item_id': 'item_id:token',
            'timestamp': 'timestamp:float',
        })
        item_out = items.rename(columns={
            'item_id': 'item_id:token',
            'category': 'category:token',
            'subcategory': 'subcategory:token',
        })
        user_out = users.rename(columns={'user_id': 'user_id:token'})
        print("11111111111112")
        train_out.to_csv(train_path, sep='\t', index=False)
        test_out.to_csv(test_path, sep='\t', index=False)
        item_out.to_csv(item_path, sep='\t', index=False)
        user_out.to_csv(user_path, sep='\t', index=False)
        print("11111111111113")
        return {
            'train_inter': train_path,
            'test_inter': test_path,
            'item': item_path,
            'user': user_path,
        }


def build_item_feature_map(items_df: pd.DataFrame, feature: str = 'category') -> Dict[str, str]:
    """item_id -> 内容特征 (category 或 subcategory) 的映射。"""
    col = 'item_id' if 'item_id' in items_df.columns else 'news_id'
    print("11111111111114")
    return dict(zip(items_df[col].astype(str), items_df[feature].fillna('unknown')))


def simple_converter(mind_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """简单转换，用于快速测试"""
    loader = MINDLoader(mind_dir)
    loader.load_all()
    interactions = []
    print("11111111111115")
    for _, row in loader.behaviors_df.iterrows():
        user_id = row['user_id']
        if pd.notna(row['history']):
            for news_id in str(row['history']).split():
                interactions.append({
                    'user_id': user_id,
                    'item_id': news_id,
                    'rating': 1.0
                })

    interactions_df = pd.DataFrame(interactions)
    items = loader.news_df[['news_id', 'category']].copy()
    items.columns = ['item_id', 'category']
    return interactions_df, items


if __name__ == "__main__":
    MINDToRecBoleConverter().convert()
