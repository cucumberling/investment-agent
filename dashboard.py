"""期货投资 Agent - 可视化看板"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="期货投资 Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Apple 风格 CSS ──
st.markdown("""
<style>
/* ═══ 全局字体与背景 ═══ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'SF Pro Display', 'Helvetica Neue', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #f5f5f7 0%, #fbfbfd 100%);
}

/* ═══ 主内容区 ═══ */
.block-container {
    padding: 2rem 3rem !important;
    max-width: 1400px;
}

/* ═══ 侧边栏 ═══ */
section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.72) !important;
    backdrop-filter: saturate(180%) blur(20px);
    -webkit-backdrop-filter: saturate(180%) blur(20px);
    border-right: 1px solid rgba(0, 0, 0, 0.06);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    color: #1d1d1f;
    font-size: 0.9rem;
    letter-spacing: -0.01em;
}

/* ═══ Tab 标签栏 ═══ */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(12px);
    border-radius: 12px;
    padding: 4px;
    gap: 2px;
    border: 1px solid rgba(0, 0, 0, 0.04);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
    font-size: 0.85rem;
    color: #86868b;
    letter-spacing: -0.01em;
    transition: all 0.2s ease;
    border: none !important;
    background: transparent;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #1d1d1f;
    background: rgba(0, 0, 0, 0.03);
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #1d1d1f !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.04);
    font-weight: 600;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}
.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* ═══ Metric 指标卡片 ═══ */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0, 0, 0, 0.04);
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
}
[data-testid="stMetricLabel"] {
    color: #86868b !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: #1d1d1f !important;
    font-weight: 600 !important;
    font-size: 1.5rem !important;
    letter-spacing: -0.02em;
}
[data-testid="stMetricDelta"] {
    font-weight: 500 !important;
    font-size: 0.8rem !important;
}

/* ═══ 按钮 ═══ */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em;
    padding: 0.5rem 1.25rem !important;
    transition: all 0.2s ease !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
    background: rgba(255, 255, 255, 0.8) !important;
    color: #1d1d1f !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
    background: #ffffff !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(180deg, #007AFF, #0066DD) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 1px 3px rgba(0, 122, 255, 0.3) !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(180deg, #0088FF, #007AFF) !important;
    box-shadow: 0 4px 14px rgba(0, 122, 255, 0.35) !important;
}

/* ═══ 数据表格 ═══ */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(0, 0, 0, 0.06);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

/* ═══ Expander 折叠面板 ═══ */
.streamlit-expanderHeader {
    background: rgba(255, 255, 255, 0.6) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0, 0, 0, 0.04) !important;
    font-weight: 500;
    letter-spacing: -0.01em;
    padding: 12px 16px !important;
}
details {
    border-radius: 12px !important;
    border: 1px solid rgba(0, 0, 0, 0.06) !important;
    background: rgba(255, 255, 255, 0.5);
}

/* ═══ 提示框 (info/success/warning/error) ═══ */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
    backdrop-filter: blur(8px);
}
[data-testid="stAlert"] > div {
    border-radius: 12px !important;
}

/* ═══ SelectBox / Input ═══ */
[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border-color: rgba(0, 0, 0, 0.08) !important;
    background: rgba(255, 255, 255, 0.8) !important;
    transition: all 0.2s ease;
}
[data-baseweb="select"] > div:hover {
    border-color: rgba(0, 122, 255, 0.4) !important;
}
[data-baseweb="select"] > div:focus-within {
    border-color: #007AFF !important;
    box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.15) !important;
}

/* ═══ Slider ═══ */
[data-testid="stSlider"] [role="slider"] {
    background: #007AFF !important;
}

/* ═══ Checkbox / Radio ═══ */
.stCheckbox label, .stRadio label {
    font-weight: 400;
    letter-spacing: -0.01em;
}

/* ═══ Divider ═══ */
hr {
    border: none;
    border-top: 1px solid rgba(0, 0, 0, 0.06);
    margin: 1.5rem 0;
}

/* ═══ 标题层级 ═══ */
h1 {
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    color: #1d1d1f !important;
    font-size: 2rem !important;
}
h2, h3 {
    font-weight: 600 !important;
    letter-spacing: -0.025em !important;
    color: #1d1d1f !important;
}
h4, h5 {
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: #1d1d1f !important;
}
.stCaption, [data-testid="stCaptionContainer"] {
    color: #86868b !important;
    letter-spacing: -0.01em;
}

/* ═══ 进度条 ═══ */
.stProgress > div > div {
    border-radius: 6px;
    height: 6px !important;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, #007AFF, #34C759) !important;
    border-radius: 6px;
}

/* ═══ Plotly 图表容器 ═══ */
[data-testid="stPlotlyChart"] {
    background: rgba(255, 255, 255, 0.6);
    border-radius: 16px;
    padding: 12px;
    border: 1px solid rgba(0, 0, 0, 0.04);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

/* ═══ 内置图表 ═══ */
[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"] {
    background: rgba(255, 255, 255, 0.6);
    border-radius: 16px;
    padding: 12px;
    border: 1px solid rgba(0, 0, 0, 0.04);
}

/* ═══ Markdown 表格 ═══ */
table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid rgba(0, 0, 0, 0.06);
    width: 100%;
}
th {
    background: rgba(0, 0, 0, 0.02) !important;
    font-weight: 600 !important;
    font-size: 0.85rem;
    letter-spacing: 0.01em;
    padding: 10px 16px !important;
    border-bottom: 1px solid rgba(0, 0, 0, 0.06) !important;
}
td {
    padding: 10px 16px !important;
    border-bottom: 1px solid rgba(0, 0, 0, 0.03) !important;
    font-size: 0.9rem;
}

/* ═══ 滚动条 ═══ */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(0, 0, 0, 0.15);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 0, 0, 0.25);
}

/* ═══ Spinner ═══ */
.stSpinner > div {
    border-top-color: #007AFF !important;
}

/* ═══ 底部免责声明 ═══ */
.stApp > footer {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ── 侧边栏 ──
st.sidebar.title("⚙️ 控制面板")
auto_refresh = st.sidebar.checkbox("自动刷新（每60秒）", value=False)
if auto_refresh:
    try:
        from streamlit_autorefresh import st_autorefresh  # type: ignore
        st_autorefresh(interval=60000, key="auto_refresh")
    except ImportError:
        st.sidebar.caption("需安装 streamlit-autorefresh 才能自动刷新")

show_boll = st.sidebar.checkbox("显示布林带", value=True)
show_macd = st.sidebar.checkbox("显示 MACD", value=True)
show_rsi = st.sidebar.checkbox("显示 RSI", value=True)
history_days = st.sidebar.slider("历史数据天数", 30, 250, 120)

st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ **风险提示**")
st.sidebar.markdown("Agent 仅供参考，所有买卖决定由人来做。")

# ── 主标题 ──
st.markdown("""
<div style="
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 0 24px 0;
">
    <div>
        <h1 style="margin:0; font-size:2.2rem; font-weight:700; letter-spacing:-0.04em; color:#1d1d1f;">
            期货投资 Agent
        </h1>
        <p style="margin:4px 0 0 0; color:#86868b; font-size:0.9rem; letter-spacing:-0.01em;">
            实时行情 · 技术分析 · AI 驱动决策
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

col_time, col_btn = st.columns([4, 1])
with col_time:
    st.caption(f"最后更新  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col_btn:
    if st.button("刷新数据", use_container_width=True, key="refresh_btn"):
        st.cache_data.clear()
        st.query_params["_t"] = str(datetime.now().timestamp())
        st.rerun()

# ── 加载数据 ──
from data_fetcher import (
    get_futures_realtime,
    get_futures_history,
    calc_technical_indicators,
    get_futures_news,
    generate_signal,
    get_minute_data,
    detect_large_orders,
    get_all_futures_realtime,
    get_intraday_data,
    calc_turtle_signals,
    calc_turtle_position_size,
    get_jin10_all_flash,
    get_research_reports,
    get_options_shfe,
    get_options_gfex,
    get_futures_hold_position,
    generate_smart_analysis,
    generate_fundamental_analysis,
)
from analyzer import analyze_market


@st.cache_data(ttl=60)
def load_realtime_by_name(name):
    """按品种中文名加载实时全合约行情"""
    import akshare as ak
    try:
        df = ak.futures_zh_realtime(symbol=name)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df[['symbol', 'name', 'trade', 'open', 'high', 'low',
                 'volume', 'position', 'changepercent', 'prevsettlement']]
        df.columns = ['合约', '名称', '最新价', '开盘价', '最高价', '最低价',
                      '成交量', '持仓量', '涨跌幅%', '昨结算']
        df['最新价'] = pd.to_numeric(df['最新价'], errors='coerce')
        df['涨跌幅%'] = pd.to_numeric(df['涨跌幅%'], errors='coerce') * 100
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_history(symbol, days):
    """按连续合约代码加载历史 + 技术指标"""
    df = get_futures_history(symbol=symbol, days=days)
    if df is not None and not df.empty:
        df = calc_technical_indicators(df)
        df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
    return df


@st.cache_data(ttl=120)
def load_news():
    return get_futures_news()


@st.cache_data(ttl=120)
def load_all_futures():
    return get_all_futures_realtime()


@st.cache_data(ttl=300)
def load_options_shfe():
    return get_options_shfe()


@st.cache_data(ttl=300)
def load_options_gfex():
    return get_options_gfex()


# ── 品种名 → 连续合约代码 映射（从全品种数据中获取） ──
@st.cache_data(ttl=120)
def get_symbol_map():
    """返回 {品种中文名: 连续合约代码} 的映射"""
    all_df = load_all_futures()
    if all_df is not None and not all_df.empty:
        return dict(zip(all_df['品种'], all_df['合约']))
    return {}


# ── Tab 布局 ──
tab0, tab1, tab_turtle, tab_hold, tab_smart, tab_opt, tab5, tab3, tab4 = st.tabs([
    "🌍 全品种行情", "📈 品种详情", "🐢 海龟趋势", "🏦 主力持仓",
    "🧠 傻瓜分析", "📊 期权行情", "🔔 大单提醒", "📰 新闻资讯", "🤖 AI 分析"
])

# ══════════════════════════════════════
# Tab 0: 全品种行情总览
# ══════════════════════════════════════
with tab0:
    st.markdown("#### 🌍 期货全品种实时行情")
    st.caption("涵盖贵金属、有色、黑色系、能化、农产品、新材料六大板块主力合约")

    try:
        all_df = load_all_futures()
        if all_df is not None and not all_df.empty:
            # 涨跌统计
            up_count = len(all_df[all_df['涨跌幅%'] > 0])
            down_count = len(all_df[all_df['涨跌幅%'] < 0])
            flat_count = len(all_df[all_df['涨跌幅%'] == 0])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("品种总数", f"{len(all_df)}")
            c2.metric("上涨", f"{up_count}", delta=f"{up_count}", delta_color="normal")
            c3.metric("下跌", f"{down_count}", delta=f"-{down_count}", delta_color="normal")
            c4.metric("平盘", f"{flat_count}")

            # 涨幅 / 跌幅 TOP5
            col_top, col_bottom = st.columns(2)
            with col_top:
                st.markdown("##### 🔴 涨幅 TOP 5")
                top5 = all_df.nlargest(5, '涨跌幅%')[['品种', '最新价', '涨跌幅%']].reset_index(drop=True)
                st.dataframe(top5, use_container_width=True, hide_index=True)
            with col_bottom:
                st.markdown("##### 🟢 跌幅 TOP 5")
                bottom5 = all_df.nsmallest(5, '涨跌幅%')[['品种', '最新价', '涨跌幅%']].reset_index(drop=True)
                st.dataframe(bottom5, use_container_width=True, hide_index=True)

            # 按板块筛选
            st.markdown("---")
            categories = ['全部'] + sorted(all_df['板块'].unique().tolist())
            selected_cat = st.selectbox("按板块筛选", categories, index=0, key="cat_filter")
            display_df = all_df if selected_cat == '全部' else all_df[all_df['板块'] == selected_cat]

            def color_change(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        return 'color: #ff4b4b'
                    elif val < 0:
                        return 'color: #00cc66'
                return ''

            styled = display_df.style.applymap(color_change, subset=['涨跌幅%'])
            st.dataframe(styled, use_container_width=True, hide_index=True, height=600)

            with st.expander("查看板块涨跌分布"):
                sector_avg = all_df.groupby('板块')['涨跌幅%'].mean().round(2).sort_values()
                st.bar_chart(sector_avg, use_container_width=True, height=300)
        else:
            st.warning("⏳ 非交易时间或数据暂不可用，请在交易时段刷新")
    except Exception as e:
        st.error(f"获取全品种行情失败：{e}")

# ══════════════════════════════════════
# Tab 1: 品种详情（通用）
# ══════════════════════════════════════
ALL_NAMES = [
    '碳酸锂', '黄金', '白银', '沪铜', '沪铝', '沪锌', '沪铅', '沪镍', '沪锡',
    '螺纹钢', '热轧卷板', '铁矿石', '焦煤', '焦炭', '锰硅', '硅铁', '不锈钢',
    '原油', '燃油', '沥青', 'PTA', '甲醇', '乙二醇', '苯乙烯',
    '聚丙烯', '聚乙烯', 'PVC', '短纤', '液化石油气',
    '豆粕', '豆油', '棕榈油', '菜粕', '菜油', '玉米', '棉花',
    '白糖', '花生', '红枣', '苹果', '生猪',
    '工业硅', '纯碱', '玻璃', '尿素', '橡胶',
]

with tab1:
    # ── 品种选择器 ──
    selected_name = st.selectbox("选择品种", ALL_NAMES, index=0, key="detail_symbol")
    st.markdown(f"### 📈 {selected_name} 详情")

    try:
        # 1) 实时全合约行情
        rt = load_realtime_by_name(selected_name)
        if rt is not None and not rt.empty:
            main_row = rt.iloc[0]
            cont_symbol = main_row['合约']  # 连续合约代码，如 AU0, LC0

            # 找到成交量最大的合约作为主力合约
            rt_sorted = rt.sort_values('成交量', ascending=False)
            active_row = rt_sorted.iloc[0]
            active_contract = active_row['合约']

            # 指标卡片
            c1, c2, c3, c4 = st.columns(4)
            price = float(active_row['最新价'])
            chg = float(active_row['涨跌幅%'])
            c1.metric(f"{active_contract} 最新价", f"¥{price:,.2f}", f"{chg:+.2f}%")
            c2.metric("开盘价", f"¥{float(active_row['开盘价']):,.2f}")
            c3.metric("成交量", f"{int(active_row['成交量']):,}")
            c4.metric("持仓量", f"{int(active_row['持仓量']):,}")

            # 全合约列表
            with st.expander(f"查看 {selected_name} 全部合约", expanded=False):
                st.dataframe(rt, use_container_width=True, hide_index=True)
        else:
            cont_symbol = None
            st.warning("实时行情暂不可用")

        # 2) 日内K线走势
        st.markdown("---")
        st.markdown("#### 📊 当日走势（5分钟K线）")
        # 用连续合约代码，如果实时不可用则从 symbol_map 获取
        intraday_symbol = cont_symbol
        if not intraday_symbol:
            sym_map = get_symbol_map()
            intraday_symbol = sym_map.get(selected_name, '')
        if intraday_symbol:
            intraday = get_intraday_data(symbol=intraday_symbol, period="5")
            if intraday is not None and not intraday.empty:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                # 日内统计
                day_open = float(intraday.iloc[0]['open'])
                day_high = float(intraday['high'].max())
                day_low = float(intraday['low'].min())
                day_last = float(intraday.iloc[-1]['close'])
                day_vol = int(intraday['volume'].sum())
                day_range = (day_high - day_low) / day_open * 100

                # 指标卡片
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("开盘", f"¥{day_open:,.0f}")
                c2.metric("最高", f"¥{day_high:,.0f}")
                c3.metric("最低", f"¥{day_low:,.0f}")
                c4.metric("振幅", f"{day_range:.2f}%")
                c5.metric("总量", f"{day_vol:,}")

                # Plotly 上下两图：价格 + 成交量
                fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.75, 0.25],
                )

                # 价格走势（面积图，Y轴自适应）
                price_color = '#ff4b4b' if day_last >= day_open else '#00cc66'
                fill_color = 'rgba(255,75,75,0.1)' if price_color == '#ff4b4b' else 'rgba(0,204,102,0.1)'
                y_margin = (day_high - day_low) * 0.15 or day_high * 0.005

                # 底线（用于 tonexty 填充）
                fig.add_trace(go.Scatter(
                    x=intraday['时间'], y=[day_low - y_margin] * len(intraday),
                    mode='lines', line=dict(width=0), showlegend=False,
                    hoverinfo='skip',
                ), row=1, col=1)
                # 价格线 + 填充到底线
                fig.add_trace(go.Scatter(
                    x=intraday['时间'], y=intraday['close'],
                    mode='lines', name='价格',
                    line=dict(color=price_color, width=2),
                    fill='tonexty', fillcolor=fill_color,
                    hovertemplate='%{x}<br>价格: ¥%{y:,.0f}<extra></extra>',
                ), row=1, col=1)

                # 成交量柱状图（红涨绿跌）
                bar_colors = ['#ff4b4b' if c >= o else '#00cc66'
                              for c, o in zip(intraday['close'], intraday['open'])]
                fig.add_trace(go.Bar(
                    x=intraday['时间'], y=intraday['volume'],
                    name='成交量', marker_color=bar_colors,
                    hovertemplate='%{x}<br>成交量: %{y:,}<extra></extra>',
                ), row=2, col=1)

                # 布局优化
                fig.update_layout(
                    height=420, margin=dict(l=0, r=0, t=10, b=0),
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    hovermode='x unified',
                )
                fig.update_yaxes(title_text='价格', row=1, col=1,
                                 range=[day_low - y_margin, day_high + y_margin],
                                 gridcolor='rgba(128,128,128,0.1)')
                fig.update_yaxes(title_text='量', row=2, col=1, gridcolor='rgba(128,128,128,0.1)')
                fig.update_xaxes(gridcolor='rgba(128,128,128,0.1)')

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("⏳ 日内数据暂不可用，请在交易时段查看")

        # 3) 历史走势 + 技术分析
        st.markdown("---")

        # 用连续合约代码获取历史
        hist_symbol = cont_symbol
        if not hist_symbol:
            sym_map = get_symbol_map()
            hist_symbol = sym_map.get(selected_name, '')
        hist = load_history(hist_symbol, history_days) if hist_symbol else None

        if hist is not None and not hist.empty:
            # 价格走势
            st.markdown("#### 价格走势与均线")
            chart_data = hist[['日期', '收盘价', 'MA5', 'MA20']].dropna().set_index('日期')
            if show_boll:
                boll_data = hist[['日期', 'BOLL_UP', 'BOLL_MID', 'BOLL_DN']].dropna().set_index('日期')
                chart_data = chart_data.join(boll_data, how='left')
            st.line_chart(chart_data, use_container_width=True, height=400)

            # 成交量
            st.markdown("#### 成交量")
            vol_data = hist[['日期', '成交量']].dropna().set_index('日期')
            st.bar_chart(vol_data, use_container_width=True, height=200)

            # MACD
            if show_macd:
                st.markdown("#### MACD")
                macd_data = hist[['日期', 'MACD', 'MACD_Signal']].dropna().set_index('日期')
                st.line_chart(macd_data, use_container_width=True, height=200)
                macd_hist_data = hist[['日期', 'MACD_Hist']].dropna().set_index('日期')
                st.bar_chart(macd_hist_data, use_container_width=True, height=150)

            # RSI
            if show_rsi:
                st.markdown("#### RSI (14)")
                rsi_data = hist[['日期', 'RSI']].dropna().set_index('日期')
                st.line_chart(rsi_data, use_container_width=True, height=200)
                st.caption("RSI > 70 超买 | RSI < 30 超卖")

            # 信号
            st.markdown("#### 📡 当前信号")
            signals = generate_signal(hist)
            st.info(signals)

            # 数据表
            with st.expander("查看最近 10 日数据"):
                display_cols = ['日期', '开盘价', '最高价', '最低价', '收盘价', '成交量',
                                'MA5', 'MA20', 'RSI', 'MACD', 'ATR']
                display_cols = [c for c in display_cols if c in hist.columns]
                st.dataframe(
                    hist[display_cols].tail(10).round(2),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.warning("历史数据暂不可用")

    except Exception as e:
        st.error(f"加载品种详情失败：{e}")

# ══════════════════════════════════════
# Tab: 主力持仓排名
# ══════════════════════════════════════
with tab_hold:
    st.markdown("#### 🏦 主力持仓排名（前20名）")
    st.caption("查看各品种前20名期货公司的多空持仓及每日变化，了解主力资金动向")

    hold_name = st.selectbox("选择品种", ALL_NAMES, index=0, key="hold_symbol")

    # 获取品种代码
    import re as _re
    sym_map = get_symbol_map()
    hold_sym = sym_map.get(hold_name, '')
    if hold_sym:
        hold_code = _re.match(r'^([A-Za-z]+)', hold_sym)
        hold_code = hold_code.group(1).upper() if hold_code else hold_sym[:2].upper()
    else:
        hold_code = ''

    if st.button("🔍 查询主力持仓", type="primary", key="query_hold"):
        if hold_code:
            with st.spinner(f"正在获取 {hold_name}({hold_code}) 主力持仓数据..."):
                try:
                    pos_data = get_futures_hold_position(hold_code)
                    contract_name = pos_data.get('contract', '')

                    if contract_name:
                        st.info(f"📋 查询合约：**{contract_name}**（自动选择成交量最大的主力合约）")

                    # 显示汇总信息
                    if pos_data.get('summary'):
                        summary = pos_data['summary']
                        st.markdown("##### 📊 多空力量对比")
                        sc1, sc2, sc3, sc4 = st.columns(4)
                        if '多头总持仓' in summary:
                            sc1.metric("多头总持仓（前20）", f"{summary['多头总持仓']:,}")
                        if '空头总持仓' in summary:
                            sc2.metric("空头总持仓（前20）", f"{summary['空头总持仓']:,}")
                        if '多空净持仓' in summary:
                            net = summary['多空净持仓']
                            sc3.metric("多空净持仓", f"{net:+,}", delta="多头占优" if net > 0 else "空头占优")
                        if '多头占比%' in summary:
                            sc4.metric("多头占比", f"{summary['多头占比%']:.1f}%")

                        # 增减变化
                        if '多头增减' in summary or '空头增减' in summary:
                            st.markdown("##### 📈 今日增减变化")
                            dc1, dc2, dc3 = st.columns(3)
                            if '多头增减' in summary:
                                chg = summary['多头增减']
                                dc1.metric("多头增减", f"{chg:+,}", delta="增仓" if chg > 0 else "减仓")
                            if '空头增减' in summary:
                                chg = summary['空头增减']
                                dc2.metric("空头增减", f"{chg:+,}", delta="增仓" if chg > 0 else "减仓")
                            if '净增减' in summary:
                                net_chg = summary['净增减']
                                dc3.metric("净增减（多-空）", f"{net_chg:+,}",
                                          delta="多头加仓更多" if net_chg > 0 else "空头加仓更多")

                        st.markdown("---")

                    # 显示持买单量排名
                    if not pos_data.get('long', pd.DataFrame()).empty:
                        st.markdown("##### 🔴 持买单量排名（多头前20）")
                        st.caption("多头持仓越多的期货公司，代表越看好后市上涨")
                        st.dataframe(pos_data['long'], use_container_width=True, hide_index=True)

                    # 显示持卖单量排名
                    if not pos_data.get('short', pd.DataFrame()).empty:
                        st.markdown("##### 🟢 持卖单量排名（空头前20）")
                        st.caption("空头持仓越多的期货公司，代表越看好后市下跌")
                        st.dataframe(pos_data['short'], use_container_width=True, hide_index=True)

                    # 显示成交量排名
                    if not pos_data.get('volume', pd.DataFrame()).empty:
                        st.markdown("##### 📊 成交量排名（前20）")
                        st.caption("成交量大的期货公司，代表该公司客户交易活跃度高")
                        st.dataframe(pos_data['volume'], use_container_width=True, hide_index=True)

                    if (pos_data.get('long', pd.DataFrame()).empty and
                        pos_data.get('short', pd.DataFrame()).empty and
                        pos_data.get('volume', pd.DataFrame()).empty):
                        st.warning(f"暂未获取到 {hold_name}({hold_code}) 的持仓排名数据。"
                                  "可能是非交易时间或该品种暂不支持。")

                    # 解读说明
                    with st.expander("📖 如何解读主力持仓数据？"):
                        st.markdown("""
**什么是主力持仓排名？**
- 各期货交易所每天收盘后公布前20名期货公司的**成交量、持买单量（多头）、持卖单量（空头）**排名
- 这些期货公司背后代表了大量的机构和大户资金

**核心看点：**
1. **多空净持仓 = 多头总持仓 - 空头总持仓**
   - 正值 → 主力整体偏多（看涨）
   - 负值 → 主力整体偏空（看跌）

2. **今日增减变化更重要！**
   - 多头大幅加仓 + 空头减仓 → 强烈看多信号
   - 空头大幅加仓 + 多头减仓 → 强烈看空信号
   - 多空同时加仓 → 分歧加大，即将选择方向
   - 多空同时减仓 → 观望情绪浓厚

3. **关注哪些公司在加仓？**
   - 知名期货公司（如中信、国泰君安、永安等）的动向更有参考价值
   - 单个公司大幅加仓可能代表其背后的大客户在布局
                        """)

                except Exception as e:
                    st.error(f"获取持仓数据失败：{e}")
                    st.info("可能原因：非交易时间、品种代码不支持、网络问题。请稍后重试。")
        else:
            st.warning("未找到该品种的合约代码")

# ══════════════════════════════════════
# Tab: 傻瓜式行情分析
# ══════════════════════════════════════
with tab_smart:
    st.markdown("#### 🧠 傻瓜式行情分析")
    st.caption("不只是告诉你信号，更解释**为什么**这个信号有效、市场背后的逻辑是什么")

    smart_name = st.selectbox("选择品种", ALL_NAMES, index=0, key="smart_symbol")

    if st.button("🔍 开始分析", type="primary", key="smart_analyze", use_container_width=True):
        sym_map = get_symbol_map()
        smart_sym = sym_map.get(smart_name, '')
        if not smart_sym:
            rt = load_realtime_by_name(smart_name)
            if rt is not None and not rt.empty:
                smart_sym = rt.iloc[0]['合约']

        if smart_sym:
            with st.spinner(f"正在深度分析 {smart_name}..."):
                try:
                    hist = load_history(smart_sym, max(history_days, 120))
                    if hist is not None and not hist.empty and len(hist) >= 60:
                        analysis = generate_smart_analysis(hist)

                        # ── 综合判定卡片 ──
                        st.markdown("---")
                        verdict = analysis['verdict']
                        verdict_color = analysis['verdict_color']
                        total = analysis['total_score']
                        bull = analysis['bull_score']
                        bear = analysis['bear_score']

                        # 彩色判定显示
                        if verdict_color == 'bullish':
                            st.success(f"### 📈 综合判定：{verdict}（多头得分 {bull:.1f} vs 空头得分 {bear:.1f}）")
                        elif verdict_color == 'bearish':
                            st.error(f"### 📉 综合判定：{verdict}（多头得分 {bull:.1f} vs 空头得分 {bear:.1f}）")
                        else:
                            st.warning(f"### ⚖️ 综合判定：{verdict}（多头得分 {bull:.1f} vs 空头得分 {bear:.1f}）")

                        st.info(f"💡 **操作建议：** {analysis['verdict_detail']}")

                        # ── 多空力量柱 ──
                        st.markdown("##### ⚡ 多空力量对比")
                        score_col1, score_col2 = st.columns(2)
                        with score_col1:
                            st.metric("🔴 多头得分", f"{bull:.1f}")
                            st.progress(min(bull / 8, 1.0))
                        with score_col2:
                            st.metric("🟢 空头得分", f"{bear:.1f}")
                            st.progress(min(bear / 8, 1.0))

                        # ── 技术面信号 ──
                        st.markdown("---")
                        st.markdown("##### 📐 技术面信号（带原理解释）")

                        def render_signal_cards(sig_list, start_expanded=3):
                            _dir_color = {'bullish': '#FF3B30', 'bearish': '#34C759', 'neutral': '#FF9F0A'}
                            _dir_bg = {'bullish': 'rgba(255,59,48,0.06)', 'bearish': 'rgba(52,199,89,0.06)', 'neutral': 'rgba(255,159,10,0.06)'}
                            _dir_label = {'bullish': '多', 'bearish': '空', 'neutral': '中性'}
                            _strength_bar = {'强': '████', '中强': '███░', '中': '██░░', '弱': '█░░░'}

                            for idx, sig in enumerate(sig_list):
                                d = sig['direction']
                                c = _dir_color.get(d, '#8E8E93')
                                bg = _dir_bg.get(d, 'rgba(142,142,147,0.05)')
                                label = _dir_label.get(d, '中性')
                                bar = _strength_bar.get(sig['strength'], '█░░░')
                                preview = sig['signal'][:55] + ('…' if len(sig['signal']) > 55 else '')

                                st.markdown(f"""
<div style="background:{bg}; border-left:3px solid {c}; border-radius:0 10px 10px 0;
            padding:12px 16px; margin:6px 0;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
    <span style="font-weight:600; color:#1d1d1f; font-size:0.9rem;">{sig['name']}</span>
    <span style="font-size:0.75rem; color:{c}; font-weight:600; background:rgba(0,0,0,0.04);
                 padding:2px 8px; border-radius:6px;">{label} · {sig['strength']} {bar}</span>
  </div>
  <div style="color:#3a3a3c; font-size:0.82rem;">{preview}</div>
</div>
""", unsafe_allow_html=True)
                                with st.expander("查看原理解释与可靠性评估", expanded=(idx < start_expanded)):
                                    st.markdown(f"**信号详情：** {sig['signal']}")
                                    st.markdown(f"**原理（为什么有效）：** {sig['principle']}")
                                    st.markdown(f"**可靠性：** {sig['reliability']}")

                        render_signal_cards(analysis['signals'], start_expanded=2)

                        # ── 基本面信号 ──
                        st.markdown("---")
                        st.markdown("##### 🏭 基本面信号（供需·估值·季节性）")

                        try:
                            rt_for_fund = load_realtime_by_name(smart_name)
                            fund = generate_fundamental_analysis(
                                hist,
                                symbol_name=smart_name,
                                rt_df=rt_for_fund
                            )
                            fund_signals = fund.get('signals', [])
                            fund_bull = fund.get('bull_score', 0)
                            fund_bear = fund.get('bear_score', 0)

                            if fund_signals:
                                # 基本面多空分数
                                fb1, fb2 = st.columns(2)
                                with fb1:
                                    st.metric("基本面多头评分", f"{fund_bull:.1f}")
                                    st.progress(min(fund_bull / 6, 1.0))
                                with fb2:
                                    st.metric("基本面空头评分", f"{fund_bear:.1f}")
                                    st.progress(min(fund_bear / 6, 1.0))

                                render_signal_cards(fund_signals, start_expanded=2)

                                # 综合（技术面 + 基本面）
                                combined_bull = bull + fund_bull
                                combined_bear = bear + fund_bear
                                combined_net = combined_bull - combined_bear
                                st.markdown("---")
                                st.markdown("##### 🔮 综合判定（技术面 + 基本面）")
                                if combined_net >= 4:
                                    st.success(f"**强烈看多** — 综合多头 {combined_bull:.1f} vs 空头 {combined_bear:.1f}")
                                elif combined_net >= 1.5:
                                    st.success(f"**偏多** — 综合多头 {combined_bull:.1f} vs 空头 {combined_bear:.1f}")
                                elif combined_net <= -4:
                                    st.error(f"**强烈看空** — 综合多头 {combined_bull:.1f} vs 空头 {combined_bear:.1f}")
                                elif combined_net <= -1.5:
                                    st.error(f"**偏空** — 综合多头 {combined_bull:.1f} vs 空头 {combined_bear:.1f}")
                                else:
                                    st.warning(f"**多空均衡** — 综合多头 {combined_bull:.1f} vs 空头 {combined_bear:.1f}")
                            else:
                                st.info("基本面数据获取中，请稍后重试或切换至交易时段查看")
                        except Exception as fe:
                            st.caption(f"基本面分析加载异常：{fe}")

                        # ── 关键价位与技术指标 ──
                        st.markdown("---")
                        st.markdown("##### 🎯 关键价位与技术指标")

                        last_row = hist.iloc[-1]
                        kp1, kp2, kp3, kp4 = st.columns(4)
                        kp1.metric("当前价格", f"¥{analysis['price']:,.0f}")
                        if analysis['ma5']:
                            kp2.metric("MA5", f"¥{analysis['ma5']:,.0f}")
                        if analysis['ma20']:
                            kp3.metric("MA20", f"¥{analysis['ma20']:,.0f}")
                        if analysis['rsi']:
                            kp4.metric("RSI(14)", f"{analysis['rsi']:.1f}")

                        kp5, kp6, kp7, kp8 = st.columns(4)
                        kdj_k = last_row.get('KDJ_K', None)
                        kdj_j = last_row.get('KDJ_J', None)
                        cci_val = last_row.get('CCI', None)
                        wr_val = last_row.get('WR', None)
                        if kdj_k is not None and not pd.isna(kdj_k):
                            kp5.metric("KDJ-K", f"{kdj_k:.1f}")
                        if kdj_j is not None and not pd.isna(kdj_j):
                            kp6.metric("KDJ-J", f"{kdj_j:.1f}")
                        if cci_val is not None and not pd.isna(cci_val):
                            kp7.metric("CCI", f"{cci_val:.0f}")
                        if wr_val is not None and not pd.isna(wr_val):
                            kp8.metric("WR(14)", f"{wr_val:.1f}")

                    else:
                        st.warning("历史数据不足（需要至少60天），无法进行完整分析")
                except Exception as e:
                    st.error(f"分析失败：{e}")
        else:
            st.warning("未找到该品种的合约代码")

    else:
        st.markdown("""
        ---
        ##### 🤔 这个功能是做什么的？

        与传统技术分析不同，**傻瓜式分析**不只是画线和显示数字——它会告诉你：

        | 传统分析 | 傻瓜式分析 |
        |---------|-----------|
        | "RSI = 75，超买" | "RSI=75意味着过去14天几乎都在涨，追涨性价比低，**为什么？**因为弹簧已经被拉到极限..." |
        | "金叉买入" | "金叉意味着短期资金正在加速流入，**但胜率只有55%**，需要配合成交量放大才可靠" |
        | "突破布林上轨" | "统计上95%的时间价格在布林带内，突破上轨属于极端情况，**回归概率约70%**" |

        每个信号都会解释：
        - 🔬 **原理**：为什么这个指标有效？市场逻辑是什么？
        - 📏 **可靠性**：历史胜率大概多少？什么情况下更可靠？
        - ⭐ **强度**：这个信号有多重要？

        最终给出**多空力量评分**和**综合判定**，帮你做出更理性的决策。
        """)

# ══════════════════════════════════════
# Tab: 期权行情
# ══════════════════════════════════════
with tab_opt:
    st.markdown("#### 📊 期权行情")
    st.caption("数据来源：上海期货交易所 (SHFE) · 广州期货交易所 (GFEX) | 每5分钟刷新")

    opt_tab1, opt_tab2 = st.tabs(["🏛 上期所期权", "🆕 广期所期权（工业硅）"])

    # ── 上期所期权 ──
    with opt_tab1:
        try:
            contracts, series_iv = load_options_shfe()
            if contracts is not None and not contracts.empty:
                # 品种选择
                available_products = sorted(contracts['品种'].dropna().unique())
                available_products = [p for p in available_products if p]
                if available_products:
                    sel_product = st.selectbox("选择品种", available_products, key="opt_shfe_product")
                    prod_df = contracts[contracts['品种'] == sel_product].copy()
                    prod_iv = series_iv[series_iv['品种'] == sel_product].copy() if not series_iv.empty and '品种' in series_iv.columns else pd.DataFrame()

                    # ── 系列 IV 汇总 ──
                    if not prod_iv.empty:
                        st.markdown("##### 隐含波动率（各到期月）")
                        iv_cols = ['合约系列', 'IV%', '成交量', '持仓量']
                        iv_cols = [c for c in iv_cols if c in prod_iv.columns]
                        st.dataframe(prod_iv[iv_cols], use_container_width=True, hide_index=True)

                    # ── PCR 持仓比 ──
                    call_oi = prod_df[prod_df['类型'] == '认购']['持仓量'].sum()
                    put_oi = prod_df[prod_df['类型'] == '认沽']['持仓量'].sum()
                    pcr = put_oi / call_oi if call_oi > 0 else 0
                    c1, c2, c3 = st.columns(3)
                    c1.metric("认购总持仓", f"{int(call_oi):,}")
                    c2.metric("认沽总持仓", f"{int(put_oi):,}")
                    c3.metric("PCR（认沽/认购）", f"{pcr:.2f}",
                              delta="偏空" if pcr > 1.2 else ("偏多" if pcr < 0.8 else "中性"),
                              delta_color="inverse")
                    st.caption("PCR > 1.2 市场偏空情绪 | PCR < 0.8 市场偏多情绪")

                    # ── 期权链（主力到期月）──
                    expiries = sorted(prod_df['到期月'].dropna().unique())
                    if expiries:
                        sel_exp = st.selectbox("选择到期月", expiries, key="opt_expiry")
                        exp_df = prod_df[prod_df['到期月'] == sel_exp]
                        calls = exp_df[exp_df['类型'] == '认购'][['行权价', '收盘价', '德尔塔', '成交量', '持仓量']].sort_values('行权价').rename(columns={'收盘价': '认购价', '德尔塔': 'Delta', '成交量': '量(C)', '持仓量': '仓(C)'})
                        puts = exp_df[exp_df['类型'] == '认沽'][['行权价', '收盘价', '德尔塔', '成交量', '持仓量']].sort_values('行权价').rename(columns={'收盘价': '认沽价', '德尔塔': 'Delta_P', '成交量': '量(P)', '持仓量': '仓(P)'})
                        chain = pd.merge(calls, puts, on='行权价', how='outer').sort_values('行权价')
                        st.markdown(f"##### 期权链 · {sel_product} {sel_exp}")
                        st.dataframe(chain.round(4), use_container_width=True, hide_index=True)
            else:
                st.info("⏳ 上期所期权数据暂不可用（可能在非交易时段）")
        except Exception as e:
            st.error(f"加载上期所期权失败：{e}")

    # ── 广期所期权（工业硅）──
    with opt_tab2:
        st.info("💡 碳酸锂期权暂未上市，广期所期权目前仅有工业硅（SI）")
        try:
            gfex_df = load_options_gfex()
            if gfex_df is not None and not gfex_df.empty:
                # PCR
                call_oi = gfex_df[gfex_df['类型'] == '认购']['持仓量'].sum()
                put_oi = gfex_df[gfex_df['类型'] == '认沽']['持仓量'].sum()
                pcr = put_oi / call_oi if call_oi > 0 else 0
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("认购持仓", f"{int(call_oi):,}")
                c2.metric("认沽持仓", f"{int(put_oi):,}")
                c3.metric("PCR", f"{pcr:.2f}")
                avg_iv = gfex_df['IV%'].mean()
                c4.metric("平均 IV", f"{avg_iv:.1f}%")

                # 认购 / 认沽 分栏
                col_c, col_p = st.columns(2)
                disp_cols = ['合约名称', '收盘价', 'Delta', 'IV%', '成交量', '持仓量']
                disp_cols = [c for c in disp_cols if c in gfex_df.columns]
                with col_c:
                    st.markdown("##### 认购（Call）")
                    calls = gfex_df[gfex_df['类型'] == '认购'][disp_cols].sort_values('行权价') if '行权价' in gfex_df.columns else gfex_df[gfex_df['类型'] == '认购'][disp_cols]
                    st.dataframe(calls.round(4), use_container_width=True, hide_index=True, height=400)
                with col_p:
                    st.markdown("##### 认沽（Put）")
                    puts = gfex_df[gfex_df['类型'] == '认沽'][disp_cols].sort_values('行权价') if '行权价' in gfex_df.columns else gfex_df[gfex_df['类型'] == '认沽'][disp_cols]
                    st.dataframe(puts.round(4), use_container_width=True, hide_index=True, height=400)
            else:
                st.info("⏳ 广期所期权数据暂不可用")
        except Exception as e:
            st.error(f"加载广期所期权失败：{e}")


# Tab: 大单提醒
# ══════════════════════════════════════
with tab5:
    st.markdown("#### 🔔 大单成交监控")
    st.caption("检测分钟K线中成交量显著放大的异常交易，可能代表主力资金动向")

    col_contract, col_threshold = st.columns(2)
    with col_contract:
        monitor_contract = st.selectbox("监控合约", ["LC2605", "LC2609", "LC2607"], index=0)
    with col_threshold:
        vol_threshold = st.slider("量比阈值（越大越严格）", 2.0, 8.0, 3.0, 0.5,
                                  help="成交量超过近20根K线均量的N倍时触发提醒")

    try:
        minute_df = get_minute_data(symbol=monitor_contract, period="1")
        if minute_df is not None and not minute_df.empty:
            large_orders = detect_large_orders(minute_df, threshold_multiplier=vol_threshold)

            if not large_orders.empty:
                buy_count = len(large_orders[large_orders['方向'].str.contains('买入')])
                sell_count = len(large_orders[large_orders['方向'].str.contains('卖出')])
                c1, c2, c3 = st.columns(3)
                c1.metric("大单总数", f"{len(large_orders)} 笔")
                c2.metric("大单买入", f"{buy_count} 笔", delta=f"{buy_count}" if buy_count > sell_count else None)
                c3.metric("大单卖出", f"{sell_count} 笔", delta=f"-{sell_count}" if sell_count > buy_count else None)

                if buy_count > sell_count * 1.5:
                    st.success(f"📊 多头占优：大单买入 {buy_count} 笔 vs 卖出 {sell_count} 笔，主力可能在吸筹")
                elif sell_count > buy_count * 1.5:
                    st.warning(f"📊 空头占优：大单卖出 {sell_count} 笔 vs 买入 {buy_count} 笔，主力可能在出货")
                else:
                    st.info(f"📊 多空均衡：买入 {buy_count} 笔 vs 卖出 {sell_count} 笔")

                st.markdown("##### 最近大单明细")
                st.dataframe(large_orders.sort_values('时间', ascending=False),
                             use_container_width=True, hide_index=True)

                st.markdown("##### 大单成交量分布")
                chart_df = large_orders[['时间', '成交量']].copy()
                chart_df['时间'] = chart_df['时间'].dt.strftime('%m-%d %H:%M')
                st.bar_chart(chart_df.set_index('时间'), use_container_width=True, height=250)
            else:
                st.info(f"✅ 当前量比阈值 {vol_threshold}x 下未检测到大单异动，市场交投平稳")

            with st.expander("查看分钟成交量走势（最近60根）"):
                recent = minute_df.tail(60).copy()
                recent['时间'] = recent['datetime'].dt.strftime('%H:%M')
                st.bar_chart(recent[['时间', 'volume']].set_index('时间'),
                             use_container_width=True, height=200)
        else:
            st.warning("⏳ 分钟数据暂不可用，请在交易时段查看")
    except Exception as e:
        st.error(f"大单监控加载失败：{e}")

# ══════════════════════════════════════
# Tab: 新闻资讯（重构版）
# ══════════════════════════════════════
with tab3:
    news_tab1, news_tab2, news_tab3 = st.tabs(["⚡ 实时快讯", "📊 机构研报", "🌐 宏观资讯"])

    # ── 实时快讯（碳酸锂/锂电产业精准过滤）──
    with news_tab1:
        st.markdown("#### ⚡ 碳酸锂 · 锂电产业资讯")
        st.caption("金十数据实时快讯 · 精准过滤碳酸锂/锂盐/锂矿/动力电池相关内容")

        # 从快讯中提取现货价格
        try:
            news = load_news()
            if news is not None and not news.empty and news.iloc[0]['内容'] != '':
                # 提取含价格的条目优先展示
                import re as _re
                price_items = []
                info_items = []
                for _, row in news.iterrows():
                    content = str(row.get('内容', ''))
                    if any(kw in content for kw in ['价格', '报价', '元/吨', '万元/吨', '涨', '跌', '均价']):
                        price_items.append(row)
                    else:
                        info_items.append(row)

                # 价格动态卡片
                if price_items:
                    st.markdown("##### 💰 价格动态")
                    for row in price_items[:5]:
                        time_str = str(row.get('发布时间', ''))
                        content = str(row.get('内容', ''))
                        time_short = time_str[11:16] if len(time_str) > 16 else time_str
                        # 高亮价格数字
                        display = _re.sub(r'(\d{4,6}(?:\.\d+)?(?:元|万元))', r'**\1**', content)
                        for kw in ['碳酸锂', '氢氧化锂', '锂矿', '锂价']:
                            display = display.replace(kw, f"**{kw}**")
                        st.info(f"🕐 `{time_short}`  {display}")

                # 产业资讯
                all_remaining = info_items + (price_items[5:] if len(price_items) > 5 else [])
                if all_remaining:
                    st.markdown("##### 📋 产业动态")
                    for row in all_remaining:
                        time_str = str(row.get('发布时间', ''))
                        content = str(row.get('内容', ''))
                        time_short = time_str[11:16] if len(time_str) > 16 else time_str
                        for kw in ['碳酸锂', '锂电', '锂矿', '锂盐', '赣锋', '天齐', '动力电池', '储能']:
                            content = content.replace(kw, f"**{kw}**")
                        st.markdown(f"`{time_short}` {content}")
                        st.divider()

                if not price_items and not info_items:
                    st.info("📭 今日暂无碳酸锂/锂电相关快讯")
            else:
                st.info("📭 今日暂无碳酸锂/锂电相关快讯")
        except Exception as e:
            st.error(f"获取快讯失败：{e}")

    # ── 机构研报（东方财富）──
    with news_tab2:
        st.markdown("#### 📊 机构研报")
        st.caption("来源：东方财富研报中心 | 覆盖赣锋锂业、天齐锂业、宁德时代")

        report_stocks = st.multiselect(
            "选择关注的股票",
            ['002460 赣锋锂业', '002466 天齐锂业', '300750 宁德时代',
             '002812 恩捷股份', '300014 亿纬锂能'],
            default=['002460 赣锋锂业', '002466 天齐锂业', '300750 宁德时代'],
            key="report_stocks"
        )
        stock_codes = [s.split(' ')[0] for s in report_stocks]

        try:
            @st.cache_data(ttl=1800)
            def load_reports(codes_tuple):
                return get_research_reports(symbols=list(codes_tuple))

            reports = load_reports(tuple(stock_codes))
            if reports is not None and not reports.empty and reports.iloc[0]['报告名称'] != '':
                # 评级统计
                rating_counts = reports['评级'].value_counts()
                if not rating_counts.empty:
                    cols = st.columns(min(len(rating_counts), 4))
                    for i, (rating, count) in enumerate(rating_counts.items()):
                        if i < 4 and rating:
                            cols[i].metric(f"评级：{rating}", f"{count} 篇")

                st.markdown("---")

                # 研报列表
                for _, row in reports.iterrows():
                    rating_icon = {'买入': '🔴', '增持': '🟠', '推荐': '🟡', '中性': '⚪', '减持': '🟢'}.get(row['评级'], '⚫')
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"**{row['报告名称']}**")
                        st.caption(f"{row['日期']} | {row['机构']} | {row['股票']} | {rating_icon} {row['评级']}")
                    with col2:
                        if row.get('PDF链接') and row['PDF链接'] != '':
                            st.markdown(f"[📄 PDF]({row['PDF链接']})")
                    st.markdown("---")
            else:
                st.info("暂无研报数据")
        except Exception as e:
            st.error(f"获取研报失败：{e}")

    # ── 宏观资讯（金十全部快讯）──
    with news_tab3:
        st.markdown("#### 🌐 全球宏观快讯")
        st.caption("金十数据实时快讯 | 包含全球财经、大宗商品、央行政策等")
        try:
            @st.cache_data(ttl=120)
            def load_macro_news():
                return get_jin10_all_flash(count=30)

            macro = load_macro_news()
            if macro is not None and not macro.empty and macro.iloc[0]['标题'] != '暂无快讯':
                for _, row in macro.iterrows():
                    time_str = str(row.get('发布时间', ''))
                    content = str(row.get('内容', ''))
                    time_short = time_str[11:16] if len(time_str) > 16 else time_str
                    st.markdown(f"`{time_short}` {content[:120]}")
                    st.markdown("---")
            else:
                st.info("暂无宏观快讯")
        except Exception as e:
            st.error(f"获取宏观资讯失败：{e}")

# ══════════════════════════════════════
# Tab: AI 分析
# ══════════════════════════════════════
with tab4:
    st.markdown("#### 🤖 AI 综合分析")
    st.caption("基于实时行情、技术指标和新闻资讯，由 Claude 生成分析报告")

    if st.button("🔍 生成 AI 分析报告", type="primary", use_container_width=True):
        with st.spinner("正在分析中，请稍候..."):
            try:
                rt = load_realtime_by_name('碳酸锂')
                hist = load_history('LC0', history_days)
                news = load_news()

                price_str = rt.to_string(index=False) if rt is not None else "数据不可用"
                signal_str = generate_signal(hist) if hist is not None and not hist.empty else "无信号"
                news_str = "\n".join(
                    news.apply(lambda r: r.get('标题', str(r.iloc[0])), axis=1).tolist()
                ) if news is not None and not news.empty else "无新闻"

                report = analyze_market(price_str, news_str, signal_str)
                st.markdown(report)
            except Exception as e:
                st.error(f"AI 分析失败：{e}")
    else:
        st.info("👆 点击按钮生成分析报告（需要配置 ANTHROPIC_API_KEY）")

# ══════════════════════════════════════
# Tab: 海龟趋势分析
# ══════════════════════════════════════
with tab_turtle:
    st.markdown("#### 🐢 海龟交易法则 · 趋势分析")
    st.caption("基于《海龟交易法则》，系统1: 20日突破入市/10日退出，系统2: 55日突破入市/20日退出，止损2N")

    # 品种选择（可多选扫描）
    turtle_mode = st.radio("模式", ["单品种详情", "全品种扫描"], horizontal=True, key="turtle_mode")

    if turtle_mode == "单品种详情":
        turtle_name = st.selectbox("选择品种", ALL_NAMES, index=0, key="turtle_symbol")

        # 获取数据（需要至少55天）
        sym_map = get_symbol_map()
        t_symbol = sym_map.get(turtle_name, '')
        if not t_symbol:
            # fallback: 尝试实时数据获取
            rt = load_realtime_by_name(turtle_name)
            if rt is not None and not rt.empty:
                t_symbol = rt.iloc[0]['合约']

        if t_symbol:
            t_hist = load_history(t_symbol, max(history_days, 120))
            if t_hist is not None and not t_hist.empty and len(t_hist) >= 56:
                # 用实时价格更新最新一行，确保信号基于最新数据
                rt_now = load_realtime_by_name(turtle_name)
                if rt_now is not None and not rt_now.empty:
                    rt_sorted = rt_now.sort_values('成交量', ascending=False)
                    live_price = float(rt_sorted.iloc[0]['最新价'])
                    if live_price > 0:
                        t_hist.loc[t_hist.index[-1], '收盘价'] = live_price
                        t_hist.loc[t_hist.index[-1], '最高价'] = max(float(t_hist.iloc[-1]['最高价']), live_price)
                        t_hist.loc[t_hist.index[-1], '最低价'] = min(float(t_hist.iloc[-1]['最低价']), live_price)
                        # 重新计算指标
                        from data_fetcher import calc_technical_indicators as recalc
                        t_hist = recalc(t_hist)

                turtle = calc_turtle_signals(t_hist)
                if turtle and turtle['N']:
                    # 显示数据来源
                    if rt_now is not None and not rt_now.empty:
                        st.caption(f"实时价格已更新 | 数据截至 {datetime.now().strftime('%H:%M:%S')}")
                    else:
                        st.caption("使用历史收盘价（实时数据不可用）")

                    # ── 核心指标 ──
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("当前价格", f"¥{turtle['close']:,.2f}")
                    c2.metric("N值 (ATR20)", f"{turtle['N']:,.2f}")
                    trend_n = turtle['stats'].get('trend_strength')
                    if trend_n is not None:
                        trend_label = "强多" if trend_n > 2 else "偏多" if trend_n > 0.5 else "中性" if trend_n > -0.5 else "偏空" if trend_n > -2 else "强空"
                        c3.metric("趋势强度", f"{trend_n:+.1f}N", delta=trend_label)
                    vol20 = turtle['stats'].get('volatility_20d')
                    if vol20 is not None:
                        c4.metric("年化波动率", f"{vol20:.1f}%")

                    # ── 系统1 & 2 信号 ──
                    st.markdown("---")
                    col_s1, col_s2 = st.columns(2)

                    with col_s1:
                        st.markdown("##### 系统1 (20日突破)")
                        sig1 = turtle['sys1']['signal']
                        if '做多' in sig1:
                            st.success(f"📈 {sig1}")
                        elif '做空' in sig1:
                            st.error(f"📉 {sig1}")
                        elif '退出' in sig1:
                            st.warning(f"⚠️ {sig1}")
                        else:
                            st.info(f"😐 {sig1}")

                        st.markdown(f"""
| 项目 | 价格 |
|------|------|
| 做多入市 | ¥{turtle['sys1']['entry_long']:,.2f} |
| 做空入市 | ¥{turtle['sys1']['entry_short']:,.2f} |
| 多头退出 | ¥{turtle['sys1']['exit_long']:,.2f} |
| 空头退出 | ¥{turtle['sys1']['exit_short']:,.2f} |
| 止损距离 | {turtle['N'] * 2:,.2f} (2N) |
""")

                    with col_s2:
                        st.markdown("##### 系统2 (55日突破)")
                        sig2 = turtle['sys2']['signal']
                        if '做多' in sig2:
                            st.success(f"📈 {sig2}")
                        elif '做空' in sig2:
                            st.error(f"📉 {sig2}")
                        elif '退出' in sig2:
                            st.warning(f"⚠️ {sig2}")
                        else:
                            st.info(f"😐 {sig2}")

                        st.markdown(f"""
| 项目 | 价格 |
|------|------|
| 做多入市 | ¥{turtle['sys2']['entry_long']:,.2f} |
| 做空入市 | ¥{turtle['sys2']['entry_short']:,.2f} |
| 多头退出 | ¥{turtle['sys2']['exit_long']:,.2f} |
| 空头退出 | ¥{turtle['sys2']['exit_short']:,.2f} |
| 止损距离 | {turtle['N'] * 2:,.2f} (2N) |
""")

                    # ── 头寸计算器 ──
                    st.markdown("---")
                    st.markdown("##### 💰 头寸规模计算器")
                    st.caption("根据海龟法则：1个头寸单位 = 账户净值1% ÷ (N × 合约乘数)")
                    col_acc, col_mult = st.columns(2)
                    with col_acc:
                        account = st.number_input("账户资金 (元)", value=500000, step=50000, key="turtle_account")
                    with col_mult:
                        multiplier = st.number_input("合约乘数", value=1, step=1, key="turtle_mult",
                                                      help="碳酸锂=1吨/手, 黄金=1000克/手, 螺纹钢=10吨/手")

                    units = calc_turtle_position_size(account, turtle['N'], multiplier)
                    max_units = units * 4  # 海龟最多4个头寸单位
                    risk_per_unit = turtle['N'] * 2 * multiplier
                    st.markdown(f"""
- **每个头寸单位**: {units} 手
- **最大头寸 (4单位)**: {max_units} 手
- **单位止损金额**: ¥{risk_per_unit:,.0f} (2N × 乘数)
- **单位风险占比**: {risk_per_unit * units / account * 100:.2f}% 的账户资金
""")

                    # ── 唐奇安通道图 ──
                    st.markdown("---")
                    st.markdown("##### 📊 唐奇安通道 + 价格走势")

                    try:
                        import plotly.graph_objects as go

                        plot_df = t_hist.tail(120).copy()
                        plot_df['日期'] = pd.to_datetime(plot_df['日期'])

                        fig = go.Figure()

                        # 55日通道（浅色背景）
                        fig.add_trace(go.Scatter(
                            x=plot_df['日期'], y=plot_df['DC55_HIGH'],
                            mode='lines', name='55日高点', line=dict(color='rgba(255,165,0,0.3)', dash='dot'),
                        ))
                        fig.add_trace(go.Scatter(
                            x=plot_df['日期'], y=plot_df['DC55_LOW'],
                            mode='lines', name='55日低点', line=dict(color='rgba(255,165,0,0.3)', dash='dot'),
                            fill='tonexty', fillcolor='rgba(255,165,0,0.05)',
                        ))

                        # 20日通道
                        fig.add_trace(go.Scatter(
                            x=plot_df['日期'], y=plot_df['DC20_HIGH'],
                            mode='lines', name='20日高点', line=dict(color='rgba(0,150,255,0.5)', width=1),
                        ))
                        fig.add_trace(go.Scatter(
                            x=plot_df['日期'], y=plot_df['DC20_LOW'],
                            mode='lines', name='20日低点', line=dict(color='rgba(0,150,255,0.5)', width=1),
                            fill='tonexty', fillcolor='rgba(0,150,255,0.05)',
                        ))

                        # 价格线
                        fig.add_trace(go.Scatter(
                            x=plot_df['日期'], y=plot_df['收盘价'],
                            mode='lines', name='收盘价', line=dict(color='#ff4b4b', width=2),
                        ))

                        # 10日退出线
                        fig.add_trace(go.Scatter(
                            x=plot_df['日期'], y=plot_df['DC10_LOW'],
                            mode='lines', name='10日低点(退出)', line=dict(color='rgba(0,200,0,0.4)', dash='dash', width=1),
                        ))

                        price_min = plot_df[['DC55_LOW', '收盘价']].min().min()
                        price_max = plot_df[['DC55_HIGH', '收盘价']].max().max()
                        margin = (price_max - price_min) * 0.05

                        fig.update_layout(
                            height=500, margin=dict(l=0, r=0, t=30, b=0),
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            yaxis=dict(range=[price_min - margin, price_max + margin],
                                       gridcolor='rgba(128,128,128,0.1)'),
                            xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                            hovermode='x unified',
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        # fallback: 用 st.line_chart
                        dc_cols = ['日期', '收盘价', 'DC20_HIGH', 'DC20_LOW', 'DC55_HIGH', 'DC55_LOW']
                        dc_cols = [c for c in dc_cols if c in t_hist.columns]
                        st.line_chart(t_hist[dc_cols].tail(120).set_index('日期'), use_container_width=True, height=400)

                    # ── 历史统计 ──
                    st.markdown("---")
                    st.markdown("##### 📋 历史统计")
                    stats = turtle['stats']
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("区间最高", f"¥{stats.get('high_all', 0):,.2f}")
                    col_b.metric("区间最低", f"¥{stats.get('low_all', 0):,.2f}")
                    ret_total = stats.get('return_total', 0)
                    col_c.metric(f"区间涨跌 ({stats.get('days', 0)}日)",
                                 f"{ret_total:+.2f}%",
                                 delta=f"{'上涨' if ret_total > 0 else '下跌'}")

                    col_d, col_e, col_f = st.columns(3)
                    r30 = stats.get('return_30d')
                    r60 = stats.get('return_60d')
                    col_d.metric("近30日涨跌", f"{r30:+.2f}%" if r30 is not None else "N/A")
                    col_e.metric("近60日涨跌", f"{r60:+.2f}%" if r60 is not None else "N/A")
                    col_f.metric("20日波动率(年化)", f"{stats.get('volatility_20d', 0):.1f}%")

                else:
                    st.warning("数据不足（需至少56个交易日），无法计算海龟指标")
            else:
                st.warning("历史数据不足或不可用")
        else:
            st.warning("无法获取该品种的合约代码")

    else:
        # ── 全品种扫描 ──
        st.caption("扫描所有品种，找出正在突破或接近突破的品种")

        with st.spinner("正在扫描所有品种的海龟信号..."):
            scan_results = []
            sym_map = get_symbol_map()

            # 预加载全品种实时数据
            all_rt = load_all_futures()
            rt_price_map = {}
            if all_rt is not None and not all_rt.empty:
                for _, r in all_rt.iterrows():
                    rt_price_map[r['品种']] = float(r['最新价'])

            for name in ALL_NAMES:
                t_sym = sym_map.get(name, '')
                if not t_sym:
                    continue
                try:
                    t_h = load_history(t_sym, 120)
                    if t_h is not None and not t_h.empty and len(t_h) >= 56:
                        # 用实时价格更新
                        live_p = rt_price_map.get(name)
                        if live_p and live_p > 0:
                            t_h.loc[t_h.index[-1], '收盘价'] = live_p
                            t_h.loc[t_h.index[-1], '最高价'] = max(float(t_h.iloc[-1]['最高价']), live_p)
                            t_h.loc[t_h.index[-1], '最低价'] = min(float(t_h.iloc[-1]['最低价']), live_p)
                            from data_fetcher import calc_technical_indicators as recalc2
                            t_h = recalc2(t_h)
                        t_sig = calc_turtle_signals(t_h)
                        if t_sig and t_sig['N']:
                            close = t_sig['close']
                            n_val = t_sig['N']
                            # 距离突破的百分比
                            dist_long_20 = (t_sig['sys1']['entry_long'] - close) / close * 100
                            dist_short_20 = (close - t_sig['sys1']['entry_short']) / close * 100
                            dist_long_55 = (t_sig['sys2']['entry_long'] - close) / close * 100

                            scan_results.append({
                                '品种': name,
                                '价格': close,
                                'N值': round(n_val, 2),
                                '系统1': t_sig['sys1']['signal'],
                                '系统2': t_sig['sys2']['signal'],
                                '趋势强度(N)': t_sig['stats'].get('trend_strength', 0),
                                '距20日高点%': round(dist_long_20, 2),
                                '距20日低点%': round(dist_short_20, 2),
                                '距55日高点%': round(dist_long_55, 2),
                                '30日涨跌%': t_sig['stats'].get('return_30d', 0),
                            })
                except Exception:
                    continue

            if scan_results:
                scan_df = pd.DataFrame(scan_results)

                # 突破品种
                breakers = scan_df[(scan_df['系统1'] != '无信号') | (scan_df['系统2'] != '无信号')]
                if not breakers.empty:
                    st.markdown("##### 🚨 有信号的品种")
                    st.dataframe(breakers, use_container_width=True, hide_index=True)
                else:
                    st.info("当前没有品种触发海龟突破信号")

                # 接近突破的品种（距20日高点 < 2%）
                near_long = scan_df[(scan_df['距20日高点%'] > 0) & (scan_df['距20日高点%'] < 2)].sort_values('距20日高点%')
                if not near_long.empty:
                    st.markdown("##### 📈 接近做多突破（距20日高点 < 2%）")
                    st.dataframe(near_long[['品种', '价格', '距20日高点%', '趋势强度(N)', '30日涨跌%']],
                                 use_container_width=True, hide_index=True)

                near_short = scan_df[(scan_df['距20日低点%'] > 0) & (scan_df['距20日低点%'] < 2)].sort_values('距20日低点%')
                if not near_short.empty:
                    st.markdown("##### 📉 接近做空突破（距20日低点 < 2%）")
                    st.dataframe(near_short[['品种', '价格', '距20日低点%', '趋势强度(N)', '30日涨跌%']],
                                 use_container_width=True, hide_index=True)

                # 趋势强度排行
                st.markdown("---")
                st.markdown("##### 🏆 趋势强度排行（偏离55日均线的N倍数）")
                st.caption("正值=多头趋势，负值=空头趋势，绝对值越大趋势越强")
                ranked = scan_df[['品种', '价格', '趋势强度(N)', '30日涨跌%', 'N值']].sort_values('趋势强度(N)', ascending=False)
                st.dataframe(ranked, use_container_width=True, hide_index=True, height=500)
            else:
                st.warning("扫描数据不足")

# ── 底部 ──
st.markdown("---")
st.caption("⚠️ 免责声明：本工具仅供学习研究使用，不构成任何投资建议。投资有风险，入市需谨慎。")
