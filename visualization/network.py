"""
可视化模块 - 用户兴趣网络图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import networkx as nx
from collections import Counter
from typing import List, Dict
import io
import base64


def create_user_interest_network(user_categories: Dict[str, List[str]],
                                 min_weight: float = 0.1,
                                 title: str = "用户兴趣网络") -> str:
    """创建用户与兴趣类别的关系网络图"""
    G = nx.Graph()
    cat_counter = Counter()
    for cats in user_categories.values():
        cat_counter.update(cats)
    for cat, count in cat_counter.most_common(20):
        G.add_node(cat, node_type='category', weight=count)
    for user_id, cats in list(user_categories.items())[:50]:
        G.add_node(user_id, node_type='user')
        for cat in set(cats):
            if cat in G.nodes:
                weight = cats.count(cat) / len(cats)
                if G.has_edge(user_id, cat):
                    G[user_id][cat]['weight'] += weight
                else:
                    G.add_edge(user_id, cat, weight=weight)
    for u, v in list(G.edges()):
        if G[u][v]['weight'] < min_weight:
            G.remove_edge(u, v)
    fig, ax = plt.subplots(figsize=(14, 12))
    cats = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'category']
    users = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'user']
    pos = {}
    if cats:
        radius = 5
        n_cats = len(cats)
        for i, cat in enumerate(cats):
            angle = 2 * np.pi * i / n_cats
            pos[cat] = (radius * np.cos(angle), radius * np.sin(angle))
    for i, user in enumerate(users):
        angle = 2 * np.pi * i / max(len(users), 1)
        pos[user] = (15 * np.cos(angle), 15 * np.sin(angle))
    if cats:
        cat_weights = [G.nodes[c].get('weight', 1) for c in cats]
        nx.draw_networkx_nodes(G, pos, nodelist=cats, node_size=[w * 50 for w in cat_weights],
                               node_color='#E45756', alpha=0.8, ax=ax)
    if users:
        nx.draw_networkx_nodes(G, pos, nodelist=users, node_size=100,
                               node_color='#4C78A8', alpha=0.6, ax=ax)
    edges = G.edges(data=True)
    if edges:
        widths = [d['weight'] * 3 for u, v, d in edges]
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.3, ax=ax)
    if cats:
        labels = {c: c for c in cats}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    ax.set_title(title, fontweight='bold', size=14)
    ax.axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_str


def create_category_network(categories: List[str], co_occurrence: bool = True,
                            title: str = "类别共现网络") -> str:
    """创建类别共现网络图"""
    G = nx.Graph()
    for i in range(len(categories) - 1):
        for j in range(i + 1, min(i + 5, len(categories))):
            c1, c2 = categories[i], categories[j]
            if c1 != c2:
                if G.has_edge(c1, c2):
                    G[c1][c2]['weight'] += 1
                else:
                    G.add_edge(c1, c2, weight=1)
    threshold = 2
    for u, v in list(G.edges()):
        if G[u][v]['weight'] < threshold:
            G.remove_edge(u, v)
    G.remove_nodes_from(list(nx.isolates(G)))
    if len(G.nodes) == 0:
        return None
    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, k=2, iterations=50)
    degrees = dict(G.degree())
    sizes = [degrees[n] * 100 for n in G.nodes()]
    centrality = nx.betweenness_centrality(G)
    colors = [centrality[n] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors,
                           cmap='YlOrRd', alpha=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
    widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5, ax=ax)
    ax.set_title(title, fontweight='bold', size=14)
    ax.axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_str


if __name__ == "__main__":
    print("网络图模块测试")
    # 测试用户兴趣网络
    user_cats = {
        "user1": ["sports", "sports", "tech", "news"],
        "user2": ["tech", "tech", "lifestyle"],
        "user3": ["sports", "news", "news"]
    }
    img = create_user_interest_network(user_cats, title="测试网络")
    print(f"用户兴趣网络图生成成功，长度: {len(img)} 字符")
    print("网络图模块测试通过")
