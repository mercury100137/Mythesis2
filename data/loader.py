"""
MIND数据集加载器
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import os

# 添加父目录到路径
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import cocoon_detect.config.settings as cfg


class MINDLoader:
    """加载MIND数据文件"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.news_df = None
        self.behaviors_df = None
        self.entity_emb = None
        self.relation_emb = None

    def load_news(self) -> pd.DataFrame:
        """加载新闻数据"""
        news_path = self.data_dir / cfg.NEWS_FILE
        cols = ['news_id', 'category', 'subcategory', 'title',
                'abstract', 'url', 'entities', 'sab_entities']
        self.news_df = pd.read_csv(news_path, sep='\t', header=None,
                                    names=cols, on_bad_lines='skip')
        return self.news_df

    def load_behaviors(self) -> pd.DataFrame:
        """加载用户行为数据"""
        beh_path = self.data_dir / cfg.BEHAVIORS_FILE
        cols = ['impression_id', 'user_id', 'time', 'history', 'impressions']
        self.behaviors_df = pd.read_csv(beh_path, sep='\t', header=None,
                                         names=cols, on_bad_lines='skip')
        self._parse_impressions()
        return self.behaviors_df

    def _parse_impressions(self):
        """解析impression字符串，分离点击(label=1)和展示"""
        def parse_imp(imp_str):
            if pd.isna(imp_str):
                return [], []
            clicks = []
            impressions = []
            for item in str(imp_str).split():
                if '-' not in item:
                    continue
                news_id, label = item.rsplit('-', 1)
                impressions.append(news_id)
                if label == '1':
                    clicks.append(news_id)
            return clicks, impressions

        parsed = self.behaviors_df['impressions'].apply(parse_imp)
        self.behaviors_df['clicks'] = parsed.apply(lambda x: x[0])
        self.behaviors_df['impressions_list'] = parsed.apply(lambda x: x[1])

    def get_click_interactions(self) -> pd.DataFrame:
        """从 impressions 中抽取 label=1 的点击，得到 (user_id, item_id, timestamp) 交互。
        若 behaviors 尚未加载则自动加载。"""
        if self.behaviors_df is None:
            self.load_behaviors()
        rows = []
        for _, r in self.behaviors_df.iterrows():
            uid = r['user_id']
            try:
                ts = pd.to_datetime(r['time']).timestamp()
            except Exception:
                ts = 0.0
            for nid in r['clicks']:
                rows.append((uid, nid, ts))
        return pd.DataFrame(rows, columns=['user_id', 'item_id', 'timestamp'])

    def get_history_interactions(self) -> pd.DataFrame:
        """从 history 字段抽取历史点击（每用户去重），用作训练正样本补充。"""
        if self.behaviors_df is None:
            self.load_behaviors()
        rows = []
        for _, r in self.behaviors_df.iterrows():
            uid = r['user_id']
            try:
                ts = pd.to_datetime(r['time']).timestamp()
            except Exception:
                ts = 0.0
            if pd.notna(r['history']):
                for nid in str(r['history']).split():
                    rows.append((uid, nid, ts))
        df = pd.DataFrame(rows, columns=['user_id', 'item_id', 'timestamp'])
        return df.drop_duplicates(subset=['user_id', 'item_id'])

    def load_entity_embeddings(self) -> Dict[str, np.ndarray]:
        """加载实体嵌入向量"""
        emb_path = self.data_dir / cfg.ENTITY_EMB_FILE
        embeddings = {}
        with open(emb_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 100:
                    entity_id = parts[0]
                    vec = np.array([float(x) for x in parts[1:]])
                    embeddings[entity_id] = vec
        self.entity_emb = embeddings
        return embeddings

    def load_relation_embeddings(self) -> Dict[str, np.ndarray]:
        """加载关系嵌入向量"""
        emb_path = self.data_dir / cfg.RELATION_EMB_FILE
        embeddings = {}
        with open(emb_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 100:
                    rel_id = parts[0]
                    vec = np.array([float(x) for x in parts[1:]])
                    embeddings[rel_id] = vec
        self.relation_emb = embeddings
        return embeddings

    def get_user_history(self, user_id: str) -> List[str]:
        """获取用户点击历史"""
        user_rows = self.behaviors_df[self.behaviors_df['user_id'] == user_id]
        history = []
        for _, row in user_rows.iterrows():
            if pd.notna(row['history']):
                history.extend(str(row['history']).split())
        return list(set(history))

    def get_news_category(self, news_id: str) -> str:
        """获取新闻的类别"""
        if self.news_df is None:
            self.load_news()
        row = self.news_df[self.news_df['news_id'] == news_id]
        if len(row) > 0:
            return row.iloc[0]['category']
        return 'unknown'

    def get_user_category_distribution(self, user_id: str) -> Dict[str, int]:
        """获取用户的历史类别分布"""
        history = self.get_user_history(user_id)
        dist = {}
        for news_id in history:
            cat = self.get_news_category(news_id)
            dist[cat] = dist.get(cat, 0) + 1
        return dist

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载所有主要数据"""
        self.load_news()
        self.load_behaviors()
        return self.news_df, self.behaviors_df


def main():
    """测试加载器"""
    print("加载MIND验证集...")
    loader = MINDLoader(cfg.MIND_DEV_DIR)
    news_df, beh_df = loader.load_all()
    print(f"新闻数: {len(news_df)}, 行为记录: {len(beh_df)}")
    print(f"类别数: {news_df['category'].nunique()}, 用户数: {beh_df['user_id'].nunique()}")


if __name__ == "__main__":
    main()
