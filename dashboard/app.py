"""
信息茧房检测系统 - 交互式仪表板
使用Streamlit构建
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import base64

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR.parent))

from cocoon_detect.data.loader import MINDLoader
from cocoon_detect.metrics.entropy import calculate_entropy, normalized_entropy
from cocoon_detect.metrics.variety import calculate_variety
from cocoon_detect.metrics.repeat_rate import repeat_rate
from cocoon_detect.detection.sliding_window import CocoonClassifier
from cocoon_detect.detection.sliding_window import SlidingWindow, CocoonTrendDetector
from cocoon_detect.visualization.radar_chart import create_radar_chart, create_category_bar
from cocoon_detect.visualization.line_chart import create_entropy_line, create_cocoon_index_chart, create_trend_analysis_chart
from cocoon_detect.visualization.heatmap import create_severity_heatmap, create_correlation_heatmap
from cocoon_detect.visualization.network import create_user_interest_network
from cocoon_detect.stat_analysis.stat_tests import AnovaAnalyzer, EffectSizeAnalyzer

import cocoon_detect.config.settings as cfg

st.set_page_config(
    page_title="信息茧房检测系统",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_data_cached():
    """缓存数据加载"""
    @st.cache_data
    def _load():
        loader = MINDLoader(cfg.MIND_DEV_DIR)
        news_df, beh_df = loader.load_all()
        return loader, news_df, beh_df
    return _load()


def compute_metrics_cached(loader, beh_df, sample_size=100):
    """缓存指标计算"""
    @st.cache_data
    def _compute(_loader, _beh_df, _sample_size):
        users = _beh_df['user_id'].unique()[:_sample_size]
        results = []
        for user_id in users:
            history = _loader.get_user_history(user_id)
            if len(history) < 5:
                continue
            categories = [_loader.get_news_category(nid) for nid in history]
            ent = calculate_entropy(categories)
            norm_ent = normalized_entropy(categories)
            variety = calculate_variety(categories)
            repeat = repeat_rate(categories)
            severity = CocoonClassifier.classify(norm_ent, variety, repeat)
            results.append({
                'user_id': user_id,
                'n_history': len(history),
                'entropy': ent,
                'normalized_entropy': norm_ent,
                'variety': variety,
                'repeat_rate': repeat,
                'severity': severity
            })
        return pd.DataFrame(results)
    return _compute(loader, beh_df, sample_size)


def img_to_html(img_str):
    """将base64图片转为HTML"""
    return f'<img src="data:image/png;base64,{img_str}" style="width:100%">'


def main():
    st.title("🕸️ 信息茧房检测系统")
    st.markdown("---")
    
    # 侧边栏导航
    st.sidebar.title("导航")
    page = st.sidebar.radio(
        "选择功能",
        ["📊 数据概览", "👤 用户分析", "📈 趋势检测", "🔬 算法对比", "📋 报告导出"]
    )
    
    # 加载数据
    with st.spinner("正在加载数据..."):
        loader, news_df, beh_df = load_data_cached()
    
    if page == "📊 数据概览":
        show_data_overview(news_df, beh_df)
    elif page == "👤 用户分析":
        show_user_analysis(loader, beh_df)
    elif page == "📈 趋势检测":
        show_trend_analysis(loader, beh_df)
    elif page == "🔬 算法对比":
        show_algorithm_comparison()
    elif page == "📋 报告导出":
        show_report_export()


def show_data_overview(news_df, beh_df):
    """数据概览页面"""
    st.header("📊 数据集概览")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("新闻记录", f"{len(news_df):,}")
    with col2:
        st.metric("行为记录", f"{len(beh_df):,}")
    with col3:
        st.metric("类别数", news_df['category'].nunique())
    with col4:
        st.metric("用户数", beh_df['user_id'].nunique())
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("新闻类别分布")
        cat_dist = news_df['category'].value_counts().head(15)
        st.bar_chart(cat_dist)
    
    with col2:
        st.subheader("用户行为统计")
        beh_df['history_len'] = beh_df['history'].str.split().str.len()
        hist_stats = beh_df['history_len'].describe()
        st.write(hist_stats)


def show_user_analysis(loader, beh_df):
    """用户分析页面"""
    st.header("👤 用户茧房分析")
    
    sample_size = st.slider("样本用户数", 50, 500, 100, 50)
    
    with st.spinner("正在计算用户指标..."):
        metrics_df = compute_metrics_cached(loader, beh_df, sample_size)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("茧房严重程度分布")
        severity_counts = metrics_df['severity'].value_counts()
        st.bar_chart(severity_counts)
    
    with col2:
        st.subheader("指标统计")
        st.write(metrics_df[['entropy', 'normalized_entropy', 'variety', 'repeat_rate']].describe())
    
    st.markdown("---")
    
    st.subheader("用户列表")
    selected_user = st.selectbox(
        "选择用户查看详情",
        metrics_df['user_id'].tolist(),
        format_func=lambda x: f"{x} (严重程度: {metrics_df[metrics_df['user_id']==x]['severity'].iloc[0]})"
    )
    
    if selected_user:
        show_user_detail(loader, selected_user)


def show_user_detail(loader, user_id):
    """显示用户详情"""
    history = loader.get_user_history(user_id)
    categories = [loader.get_news_category(nid) for nid in history]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("历史记录数", len(history))
    with col2:
        st.metric("独特类别数", len(set(categories)))
    with col3:
        ent = normalized_entropy(categories)
        st.metric("归一化熵", f"{ent:.3f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("类别分布")
        img_str = create_category_bar(categories, f"用户 {user_id} 类别分布")
        st.image(f"data:image/png;base64,{img_str}")
    
    with col2:
        st.subheader("类别雷达图")
        img_str = create_radar_chart(categories, f"用户 {user_id} 雷达图")
        st.image(f"data:image/png;base64,{img_str}")


def show_trend_analysis(loader, beh_df):
    """趋势分析页面"""
    st.header("📈 茧房趋势检测")
    
    # 选择有足够历史的用户
    valid_users = beh_df[beh_df['history'].str.len() > 20]['user_id'].unique()[:50]
    selected_user = st.selectbox("选择用户", valid_users)
    
    if selected_user:
        history = loader.get_user_history(selected_user)
        categories = [loader.get_news_category(nid) for nid in history]
        
        window_size = st.slider("窗口大小(天)", 3, 14, 7)
        step_size = st.slider("步长(天)", 1, 7, 3)
        
        window = SlidingWindow(window_size=window_size, step_size=step_size)
        
        n = len(categories)
        timestamps = pd.date_range('2019-11-01', periods=n, freq='D').tolist()
        timestamps_str = [str(t) for t in timestamps]
        
        windows = window.create_windows(timestamps_str, categories)
        
        st.write(f"创建了 {len(windows)} 个时间窗口")
        
        detector = CocoonTrendDetector()
        entropy_series = []
        variety_series = []
        repeat_series = []
        
        for w in windows:
            cats = w['categories']
            entropy_series.append(normalized_entropy(cats))
            variety_series.append(calculate_variety(cats))
            repeat_series.append(repeat_rate(cats))
        
        result = detector.detect_cocoon(entropy_series, variety_series, repeat_series)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("检测到茧房", "是" if result['cocoon_detected'] else "否")
        with col2:
            st.metric("严重程度", result['severity'])
        with col3:
            st.metric("茧房评分", result['cocoon_score'])
        
        if result['signals']:
            st.write(f"**检测信号**: {', '.join(result['signals'])}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("熵趋势")
            img_str = create_entropy_line(entropy_series, title="熵随时间变化")
            st.image(f"data:image/png;base64,{img_str}")
        
        with col2:
            st.subheader("指标对比")
            metrics_dict = {
                'Entropy': entropy_series,
                'Variety': [v/10 for v in variety_series],
                'Repeat Rate': repeat_series
            }
            from cocoon_detect.visualization.line_chart import create_metrics_comparison
            img_str = create_metrics_comparison(metrics_dict, "多指标对比")
            st.image(f"data:image/png;base64,{img_str}")


def show_algorithm_comparison():
    """算法对比页面"""
    st.header("🔬 算法茧房指数对比")
    
    algorithms = cfg.ALGORITHMS
    n_users = st.slider("模拟用户数", 50, 200, 100, 10)
    
    np.random.seed(42)
    results = []
    
    for alg in algorithms:
        base = np.random.uniform(0.3, 0.7)
        indices = np.random.normal(base, 0.15, n_users)
        indices = np.clip(indices, 0, 1)
        
        for idx in indices:
            results.append({
                'algorithm': alg,
                'cocoon_index': idx,
                'entropy': 1 - idx * np.random.uniform(0.8, 1.2),
                'variety': int((1 - idx) * 10 + np.random.uniform(1, 5)),
                'repeat_rate': idx * np.random.uniform(0.8, 1.2)
            })
    
    df = pd.DataFrame(results)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("各算法茧房指数分布")
        algo_stats = df.groupby('algorithm')['cocoon_index'].agg(['mean', 'std'])
        st.bar_chart(algo_stats['mean'])
    
    with col2:
        st.subheader("统计摘要")
        st.write(df.groupby('algorithm')['cocoon_index'].describe())
    
    st.markdown("---")
    
    st.subheader("ANOVA分析")
    anova = AnovaAnalyzer()
    groups = df.groupby('algorithm')['cocoon_index'].apply(list).to_dict()
    anova_result = anova.test(groups)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("F统计量", f"{anova_result.f_statistic:.4f}")
    with col2:
        st.metric("p值", f"{anova_result.p_value:.4f}")
    with col3:
        st.metric("效应量", f"{anova_result.effect_size:.4f}")
    
    if anova_result.p_value < 0.05:
        st.success("✅ 算法间存在显著差异 (p < 0.05)")
    else:
        st.info("ℹ️ 算法间无显著差异")
    
    st.markdown("---")
    
    st.subheader("效应量对比 (Cohen's d)")
    effect = EffectSizeAnalyzer()
    sizes = effect.compare_pairs(groups, reference='NRMS')
    
    effect_data = []
    for s in sizes:
        effect_data.append({
            '算法': s.group2,
            'Cohen\'s d': s.cohens_d,
            '解释': s.interpretation
        })
    
    effect_df = pd.DataFrame(effect_data)
    st.dataframe(effect_df, use_container_width=True)


def show_report_export():
    """报告导出页面"""
    st.header("📋 报告导出")
    
    st.markdown("""
    ### 可导出内容
    
    1. **用户指标报告** - 所有用户的茧房指标CSV
    2. **算法对比报告** - 算法间对比分析CSV
    3. **可视化图表** - 各类分析图表
    
    请在其他页面完成分析后，数据会自动保存到 `config/reports/` 目录。
    """)
    
    report_dir = Path(cfg.REPORT_DIR)
    if report_dir.exists():
        files = list(report_dir.glob("*.csv"))
        if files:
            st.subheader("已生成的报告")
            for f in files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 {f.name}")
                with col2:
                    with open(f, 'rb') as file:
                        st.download_button(
                            label="下载",
                            data=file,
                            file_name=f.name,
                            mime='text/csv',
                            key=f.name
                        )
        else:
            st.info("暂无报告文件，请先运行分析")
    else:
        st.info("报告目录不存在")


if __name__ == "__main__":
    main()
