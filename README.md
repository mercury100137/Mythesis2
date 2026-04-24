# Information Cocoon Detection System

本项目实现基于MIND新闻推荐数据集的信息茧房检测与分析系统。

## 功能特性

- 多算法推荐：NRMS、NAML、LSTUR、DKN、TANR、NPA、Hi-Fi Ark
- 茧房指标：信息熵(H@K)、类别多样性(N@K)、兴趣重复率(R)
- 融合权重：AHP层次分析法 + 熵权法
- 滑动窗口趋势分析：检测茧房形成过程
- 可视化：雷达图、折线图、桑基图、网络图、热力图

## 快速开始

```bash
pip install -r requirements.txt
python main.py --mode full
```

## 项目结构

```
cocoon_detect/
├── config/settings.py      # 全局配置
├── data/
│   ├── loader.py           # MIND数据加载器
│   └── converter.py        # MIND → RecBole格式转换
├── metrics/
│   ├── entropy.py          # 信息熵计算
│   ├── variety.py          # 类别多样性计算
│   ├── repeat_rate.py      # 兴趣重复率计算
│   └── fusion.py           # AHP+熵权融合
├── detection/
│   └── sliding_window.py   # 滑动时间窗口 + 趋势分析
├── statistics/
│   └── stat_tests.py       # ANOVA/Tukey/Cohen's d
├── visualization/
│   ├── radar_chart.py      # 雷达图
│   ├── line_chart.py       # 折线图
│   ├── heatmap.py          # 热力图
│   ├── network.py          # 网络图
│   └── user_report.py      # 用户茧房报告
├── dashboard/app.py        # 交互式仪表板（streamlit run c:/Users/w2840/Desktop/你是谁/cocoon_detect/dashboard/app.py，换自己的路径）
└── main.py                 # 主程序入口
```
