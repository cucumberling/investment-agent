"""数据获取模块 - 从 akshare 获取期货行情和新闻"""
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def get_futures_realtime():
    """获取碳酸锂期货全部合约实时行情"""
    df = ak.futures_zh_realtime(symbol="碳酸锂")
    df = df[['symbol', 'name', 'trade', 'open', 'high', 'low',
             'volume', 'position', 'changepercent', 'prevsettlement']]
    df.columns = ['合约', '名称', '最新价', '开盘价', '最高价', '最低价',
                  '成交量', '持仓量', '涨跌幅%', '昨结算']
    df['最新价'] = pd.to_numeric(df['最新价'], errors='coerce')
    df['涨跌幅%'] = pd.to_numeric(df['涨跌幅%'], errors='coerce') * 100
    return df.reset_index(drop=True)


def get_futures_history(symbol="LC0", days=120):
    """获取碳酸锂期货历史日K线数据"""
    try:
        df = ak.futures_main_sina(symbol=symbol, start_date=(datetime.now() - timedelta(days=days*2)).strftime("%Y%m%d"))
        if df is not None and len(df) > 0:
            df.columns = ['日期', '开盘价', '最高价', '最低价', '收盘价', '成交量', '持仓量', '结算价']
            for col in ['开盘价', '最高价', '最低价', '收盘价', '成交量', '持仓量']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.tail(days)
            return df.reset_index(drop=True)
    except Exception:
        pass
    # 备用方案：用连续合约
    try:
        df = ak.futures_zh_daily_sina(symbol=symbol)
        if df is not None and len(df) > 0:
            df.columns = ['日期', '开盘价', '最高价', '最低价', '收盘价', '成交量', '持仓量', '结算价']
            for col in ['开盘价', '最高价', '最低价', '收盘价', '成交量', '持仓量']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.tail(days)
            return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def calc_technical_indicators(df):
    """计算技术指标：MA、RSI、ATR、MACD、布林带"""
    if df is None or df.empty:
        return df

    close = df['收盘价']
    high = df['最高价']
    low = df['最低价']

    # 均线
    df['MA5'] = close.rolling(5).mean()
    df['MA10'] = close.rolling(10).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA60'] = close.rolling(60).mean()

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # 布林带
    df['BOLL_MID'] = close.rolling(20).mean()
    std = close.rolling(20).std()
    df['BOLL_UP'] = df['BOLL_MID'] + 2 * std
    df['BOLL_DN'] = df['BOLL_MID'] - 2 * std

    # ── KDJ 随机指标 ──
    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9).replace(0, np.nan) * 100
    df['KDJ_K'] = rsv.ewm(com=2, adjust=False).mean()
    df['KDJ_D'] = df['KDJ_K'].ewm(com=2, adjust=False).mean()
    df['KDJ_J'] = 3 * df['KDJ_K'] - 2 * df['KDJ_D']

    # ── OBV 能量潮 ──
    obv_sign = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
    df['OBV'] = (df['成交量'] * obv_sign).cumsum()
    df['OBV_MA20'] = df['OBV'].rolling(20).mean()

    # ── 威廉指标 WR(14) ──
    high_14 = high.rolling(14).max()
    low_14 = low.rolling(14).min()
    df['WR'] = (high_14 - close) / (high_14 - low_14).replace(0, np.nan) * -100

    # ── 动量指标 ROC(12) ──
    df['ROC'] = (close - close.shift(12)) / close.shift(12).replace(0, np.nan) * 100

    # ── CCI 顺势指标 ──
    tp = (high + low + close) / 3
    tp_ma = tp.rolling(20).mean()
    tp_md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df['CCI'] = (tp - tp_ma) / (0.015 * tp_md.replace(0, np.nan))

    # ── 海龟交易法则指标 ──
    # N值 = ATR(20)，即20日真实波动幅度均值
    tr20 = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df['N'] = tr20.rolling(20).mean()

    # 唐奇安通道
    df['DC20_HIGH'] = high.rolling(20).max()   # 系统1入市：20日最高
    df['DC20_LOW'] = low.rolling(20).min()     # 系统1入市：20日最低
    df['DC10_HIGH'] = high.rolling(10).max()   # 系统1退出：10日最高
    df['DC10_LOW'] = low.rolling(10).min()     # 系统1退出：10日最低
    df['DC55_HIGH'] = high.rolling(55).max()   # 系统2入市：55日最高
    df['DC55_LOW'] = low.rolling(55).min()     # 系统2入市：55日最低
    df['DC20_HIGH_EXIT'] = high.rolling(20).max()  # 系统2退出：20日最高
    df['DC20_LOW_EXIT'] = low.rolling(20).min()    # 系统2退出：20日最低

    return df


def calc_turtle_signals(df):
    """
    海龟交易法则信号生成。
    系统1: 20日突破入市，10日突破退出，止损2N
    系统2: 55日突破入市，20日突破退出，止损2N
    返回当前状态的字典。
    """
    if df is None or df.empty or len(df) < 56:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = last['收盘价']
    n_val = last['N']

    result = {
        'close': close,
        'N': round(n_val, 2) if not pd.isna(n_val) else None,
        'sys1': {},
        'sys2': {},
        'stats': {},
    }

    # ── 系统1: 20日突破 ──
    dc20_high = prev['DC20_HIGH']  # 用前一日的通道（今天突破昨天的通道）
    dc20_low = prev['DC20_LOW']
    dc10_high = prev['DC10_HIGH']
    dc10_low = prev['DC10_LOW']

    result['sys1']['entry_long'] = round(dc20_high, 2)
    result['sys1']['entry_short'] = round(dc20_low, 2)
    result['sys1']['exit_long'] = round(dc10_low, 2)
    result['sys1']['exit_short'] = round(dc10_high, 2)

    if close > dc20_high:
        result['sys1']['signal'] = '做多突破'
        result['sys1']['stop'] = round(close - 2 * n_val, 2)
    elif close < dc20_low:
        result['sys1']['signal'] = '做空突破'
        result['sys1']['stop'] = round(close + 2 * n_val, 2)
    elif close < dc10_low:
        result['sys1']['signal'] = '多头退出'
    elif close > dc10_high:
        result['sys1']['signal'] = '空头退出'
    else:
        result['sys1']['signal'] = '无信号'

    # ── 系统2: 55日突破 ──
    dc55_high = prev['DC55_HIGH']
    dc55_low = prev['DC55_LOW']
    dc20_exit_high = prev['DC20_HIGH_EXIT']
    dc20_exit_low = prev['DC20_LOW_EXIT']

    result['sys2']['entry_long'] = round(dc55_high, 2)
    result['sys2']['entry_short'] = round(dc55_low, 2)
    result['sys2']['exit_long'] = round(dc20_exit_low, 2)
    result['sys2']['exit_short'] = round(dc20_exit_high, 2)

    if close > dc55_high:
        result['sys2']['signal'] = '做多突破'
        result['sys2']['stop'] = round(close - 2 * n_val, 2)
    elif close < dc55_low:
        result['sys2']['signal'] = '做空突破'
        result['sys2']['stop'] = round(close + 2 * n_val, 2)
    elif close < dc20_exit_low:
        result['sys2']['signal'] = '多头退出'
    elif close > dc20_exit_high:
        result['sys2']['signal'] = '空头退出'
    else:
        result['sys2']['signal'] = '无信号'

    # ── 历史统计 ──
    if len(df) >= 60:
        closes = df['收盘价']
        result['stats']['days'] = len(df)
        result['stats']['high_all'] = round(closes.max(), 2)
        result['stats']['low_all'] = round(closes.min(), 2)
        result['stats']['return_total'] = round(
            (closes.iloc[-1] / closes.iloc[0] - 1) * 100, 2)
        result['stats']['return_30d'] = round(
            (closes.iloc[-1] / closes.iloc[-30] - 1) * 100, 2) if len(df) >= 30 else None
        result['stats']['return_60d'] = round(
            (closes.iloc[-1] / closes.iloc[-60] - 1) * 100, 2) if len(df) >= 60 else None
        result['stats']['volatility_20d'] = round(
            closes.pct_change().tail(20).std() * np.sqrt(252) * 100, 2)
        # 趋势强度: 价格偏离55日均线的N倍数
        ma55 = closes.rolling(55).mean().iloc[-1]
        if not pd.isna(ma55) and not pd.isna(n_val) and n_val > 0:
            result['stats']['trend_strength'] = round((close - ma55) / n_val, 2)
        else:
            result['stats']['trend_strength'] = None

    return result


def calc_turtle_position_size(account_size, n_val, point_value=1):
    """
    海龟头寸规模计算。
    1个头寸单位 = 账户净值的1% / (N × 每点价值)
    account_size: 账户总资金
    n_val: N值（ATR20）
    point_value: 每点代表的金额（期货合约乘数）
    """
    if n_val <= 0:
        return 0
    unit = int(account_size * 0.01 / (n_val * point_value))
    return max(unit, 0)


def get_futures_news():
    """获取碳酸锂/锂电直接相关快讯（精准过滤）"""
    # 精准词：必须与锂电产业直接相关
    li_keywords = [
        '碳酸锂', '氢氧化锂', '锂矿', '锂盐', '锂电', '锂离子',
        '赣锋', '天齐', '雅化', '盛新锂能', '永兴材料',
        '电池级', 'MMLC', '工业级锂', '锂价', '锂资源',
        '磷酸铁锂', '三元锂', '正极材料', '负极材料',
        '新能源汽车销量', '动力电池', '储能电池',
    ]
    return get_jin10_flash(keywords=li_keywords, count=30)


def get_jin10_flash(keywords=None, count=30):
    """
    获取金十数据实时快讯。
    keywords: 过滤关键词列表，None 则返回全部
    """
    import requests
    import re
    try:
        url = "https://flash-api.jin10.com/get_flash_list"
        params = {'channel': '-8200', 'vip': 0, 'max_time': ''}
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
            'x-app-id': 'bVBF4FyRTn5NJF5n',
            'x-version': '1.0.0',
        }
        r = requests.get(url, params=params, headers=headers, timeout=10)
        data = r.json()
        items = data.get('data', [])

        results = []
        for item in items:
            content = item.get('data', {}).get('content', '') or item.get('data', {}).get('title', '')
            content = re.sub(r'<[^>]+>', '', content).strip()
            if not content or len(content) < 10:
                continue
            time_str = item.get('time', '')

            # 关键词过滤
            if keywords:
                if not any(kw in content for kw in keywords):
                    continue

            results.append({
                '标题': content[:80],
                '内容': content,
                '发布时间': time_str,
                '类型': '快讯',
            })
            if len(results) >= count:
                break

        if results:
            return pd.DataFrame(results)
    except Exception:
        pass
    return pd.DataFrame({'标题': ['暂无快讯'], '内容': [''], '发布时间': [''], '类型': ['']})


def get_jin10_all_flash(count=50):
    """获取金十全部快讯（不过滤关键词），用于宏观资讯"""
    return get_jin10_flash(keywords=None, count=count)


def get_options_shfe():
    """
    上期所期权日行情（铝/铜/黄金/白银/锌/螺纹钢等）。
    返回 (合约明细df, 系列IV摘要df)，失败则返回 (空df, 空df)。
    """
    import re
    prefix_map = {
        'al': '沪铝', 'cu': '沪铜', 'au': '黄金', 'ag': '白银',
        'zn': '沪锌', 'rb': '螺纹钢', 'ni': '沪镍', 'sn': '沪锡',
    }
    empty = (pd.DataFrame(), pd.DataFrame())
    try:
        result = ak.option_shfe_daily()
        contracts, series_iv = result[0].copy(), result[1].copy()
        if contracts.empty:
            return empty
        # 标注品种名 + 认购/认沽
        def parse_code(code):
            m = re.match(r'^([a-z]+)(\d{4})([CP])(\d+)$', str(code))
            if m:
                prefix, exp, cp_type, strike = m.groups()
                return prefix_map.get(prefix, prefix.upper()), exp, '认购' if cp_type == 'C' else '认沽', int(strike)
            return '', '', '', 0
        parsed = contracts['合约代码'].apply(parse_code)
        contracts['品种'] = parsed.apply(lambda x: x[0])
        contracts['到期月'] = parsed.apply(lambda x: x[1])
        contracts['类型'] = parsed.apply(lambda x: x[2])
        contracts['行权价'] = parsed.apply(lambda x: x[3])
        # 转数值
        for col in ['收盘价', '结算价', '德尔塔', '成交量', '持仓量']:
            if col in contracts.columns:
                contracts[col] = pd.to_numeric(contracts[col], errors='coerce')
        # 系列IV去掉汇总行
        if '合约系列' in series_iv.columns:
            series_iv = series_iv[series_iv['合约系列'] != '小计'].copy()
            series_iv['品种'] = series_iv['合约系列'].apply(
                lambda s: prefix_map.get(re.match(r'^([a-z]+)', str(s)).group(1) if re.match(r'^([a-z]+)', str(s)) else '', str(s))
            )
            series_iv['隐含波动率%'] = (pd.to_numeric(series_iv['隐含波动率'], errors='coerce') * 100).round(2)
        return contracts, series_iv
    except Exception:
        return empty


def get_options_gfex():
    """
    广期所期权日行情（工业硅 SI，碳酸锂期权暂未上市）。
    返回处理后的 DataFrame，失败返回空 DataFrame。
    """
    import re
    try:
        df = ak.option_gfex_daily()
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        # 从合约名称识别类型和行权价，如 si2309-C-11000
        def parse_gfex(name):
            m = re.match(r'^(\w+)-([CP])-(\d+)$', str(name))
            if m:
                _, cp_type, strike = m.groups()
                return '认购' if cp_type == 'C' else '认沽', int(strike)
            return '', 0
        parsed = df['合约名称'].apply(parse_gfex)
        df['类型'] = parsed.apply(lambda x: x[0])
        df['行权价'] = parsed.apply(lambda x: x[1])
        for col in ['收盘价', '结算价', 'Delta', '成交量', '持仓量', '隐含波动率']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['IV%'] = (df['隐含波动率'] * 100).round(2)
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_research_reports(symbols=None):
    """
    获取机构研报（东方财富）。
    symbols: 股票代码列表，如 ['002460', '002466', '300750']
    """
    if symbols is None:
        symbols = ['002460', '002466', '300750']  # 赣锋锂业, 天齐锂业, 宁德时代

    all_reports = []
    for code in symbols:
        try:
            df = ak.stock_research_report_em(symbol=code)
            if df is not None and len(df) > 0:
                for _, row in df.head(10).iterrows():
                    all_reports.append({
                        '股票': f"{row.get('股票简称', '')}({code})",
                        '报告名称': str(row.get('报告名称', '')),
                        '机构': str(row.get('机构', '')),
                        '评级': str(row.get('东财评级', '')),
                        '日期': str(row.get('日期', '')),
                        'PDF链接': str(row.get('报告PDF链接', '')),
                    })
        except Exception:
            continue

    if all_reports:
        df = pd.DataFrame(all_reports)
        df = df.sort_values('日期', ascending=False).head(30)
        return df.reset_index(drop=True)
    return pd.DataFrame({'股票': ['暂无研报'], '报告名称': [''], '机构': [''], '评级': [''], '日期': [''], 'PDF链接': ['']})


def get_related_stocks_realtime():
    """获取锂电相关股票实时行情"""
    stocks = {"002460": "赣锋锂业", "002466": "天齐锂业", "600549": "厦门钨业"}
    results = []
    for code, name in stocks.items():
        try:
            prefix = "sz" if code.startswith("0") or code.startswith("3") else "sh"
            df = ak.stock_zh_a_spot_em()
            row = df[df['代码'] == code]
            if not row.empty:
                results.append({
                    "代码": code,
                    "名称": name,
                    "最新价": row.iloc[0].get('最新价', 0),
                    "涨跌幅%": row.iloc[0].get('涨跌幅', 0),
                    "成交额(亿)": round(row.iloc[0].get('成交额', 0) / 1e8, 2),
                })
        except Exception:
            results.append({"代码": code, "名称": name, "最新价": "-", "涨跌幅%": "-", "成交额(亿)": "-"})
    return pd.DataFrame(results) if results else pd.DataFrame()


def get_intraday_data(symbol="AU0", period="5"):
    """获取日内分钟K线（默认5分钟），只保留最新交易日数据"""
    try:
        df = ak.futures_zh_minute_sina(symbol=symbol, period=period)
        if df is not None and len(df) > 0:
            df['datetime'] = pd.to_datetime(df['datetime'])
            for col in ['open', 'high', 'low', 'close', 'volume', 'hold']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # 只取最新交易日
            latest_date = df['datetime'].dt.date.max()
            df = df[df['datetime'].dt.date == latest_date].copy()
            df['时间'] = df['datetime'].dt.strftime('%H:%M')
            return df.reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()


def get_minute_data(symbol="LC2605", period="1"):
    """获取分钟K线数据，用于大单检测"""
    try:
        df = ak.futures_zh_minute_sina(symbol=symbol, period=period)
        if df is not None and len(df) > 0:
            df['datetime'] = pd.to_datetime(df['datetime'])
            for col in ['open', 'high', 'low', 'close', 'volume', 'hold']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
    except Exception:
        return pd.DataFrame()


def detect_large_orders(df, threshold_multiplier=3.0):
    """
    大单检测：找出成交量显著高于近期均值的分钟K线。
    逻辑：某分钟成交量 > 近20根K线均量 × 倍数阈值，判定为大单异动。
    同时结合价格变动方向判断多空方向。
    """
    if df is None or df.empty or len(df) < 25:
        return pd.DataFrame()

    df = df.copy()
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma20']
    df['price_change'] = df['close'] - df['open']

    # 筛选成交量超过均值 N 倍的K线
    large = df[df['vol_ratio'] >= threshold_multiplier].copy()
    if large.empty:
        return pd.DataFrame()

    large['方向'] = large['price_change'].apply(
        lambda x: '🔴 大单买入' if x > 0 else ('🟢 大单卖出' if x < 0 else '⚪ 方向不明')
    )
    large['波动幅度'] = ((large['high'] - large['low']) / large['open'] * 100).round(3)

    result = large[['datetime', 'close', 'volume', 'vol_ratio', '方向', '波动幅度', 'hold']].copy()
    result.columns = ['时间', '成交价', '成交量', '量比', '方向', '波动%', '持仓量']
    result['量比'] = result['量比'].round(1)
    return result.tail(20).reset_index(drop=True)


def _fetch_one_future(args):
    """获取单个品种行情（供并发调用）"""
    category, sym = args
    try:
        df = ak.futures_zh_realtime(symbol=sym)
        if df is not None and not df.empty:
            row = df.iloc[0]
            price = pd.to_numeric(row.get('trade', 0), errors='coerce')
            prev = pd.to_numeric(row.get('prevsettlement', 0), errors='coerce')
            chg_pct = pd.to_numeric(row.get('changepercent', 0), errors='coerce') * 100
            vol = pd.to_numeric(row.get('volume', 0), errors='coerce')
            pos = pd.to_numeric(row.get('position', 0), errors='coerce')
            high = pd.to_numeric(row.get('high', 0), errors='coerce')
            low = pd.to_numeric(row.get('low', 0), errors='coerce')
            return {
                '板块': category,
                '品种': sym,
                '合约': row.get('symbol', ''),
                '最新价': price,
                '涨跌幅%': round(chg_pct, 2),
                '最高': high,
                '最低': low,
                '成交量': int(vol) if not pd.isna(vol) else 0,
                '持仓量': int(pos) if not pd.isna(pos) else 0,
                '昨结算': prev,
            }
    except Exception:
        pass
    return None


def get_all_futures_realtime():
    """获取所有主要期货品种的实时行情（主力合约），并发请求加速"""
    from concurrent.futures import ThreadPoolExecutor
    categories = {
        '贵金属': ['黄金', '白银'],
        '有色金属': ['沪铜', '沪铝', '沪锌', '沪铅', '沪镍', '沪锡'],
        '黑色系': ['螺纹钢', '热轧卷板', '铁矿石', '焦煤', '焦炭', '锰硅', '硅铁', '不锈钢'],
        '能源化工': ['原油', '燃油', '沥青', 'PTA', '甲醇', '乙二醇', '苯乙烯',
                  '聚丙烯', '聚乙烯', 'PVC', '短纤', '液化石油气'],
        '农产品': ['豆粕', '豆油', '棕榈油', '菜粕', '菜油', '玉米', '棉花',
                '白糖', '花生', '红枣', '苹果', '生猪'],
        '新材料': ['碳酸锂', '工业硅', '纯碱', '玻璃', '尿素', '橡胶'],
    }
    tasks = [(cat, sym) for cat, syms in categories.items() for sym in syms]
    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(_fetch_one_future, tasks))
    all_rows = [r for r in results if r is not None]
    if all_rows:
        return pd.DataFrame(all_rows)
    return pd.DataFrame()


def generate_signal(df):
    """基于技术指标生成交易信号"""
    if df is None or df.empty or len(df) < 20:
        return "数据不足，无法生成信号"

    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []

    # 均线金叉/死叉
    if last['MA5'] > last['MA20'] and prev['MA5'] <= prev['MA20']:
        signals.append("🔴 MA5上穿MA20 金叉买入信号")
    elif last['MA5'] < last['MA20'] and prev['MA5'] >= prev['MA20']:
        signals.append("🟢 MA5下穿MA20 死叉卖出信号")

    # RSI 超买超卖
    if last['RSI'] > 70:
        signals.append(f"⚠️ RSI={last['RSI']:.1f} 超买区域，注意回调风险")
    elif last['RSI'] < 30:
        signals.append(f"📈 RSI={last['RSI']:.1f} 超卖区域，关注反弹机会")

    # MACD
    if last['MACD_Hist'] > 0 and prev['MACD_Hist'] <= 0:
        signals.append("📊 MACD柱由负转正，多头动能增强")
    elif last['MACD_Hist'] < 0 and prev['MACD_Hist'] >= 0:
        signals.append("📊 MACD柱由正转负，空头动能增强")

    # 布林带
    if last['收盘价'] > last['BOLL_UP']:
        signals.append("⚡ 价格突破布林上轨，可能超涨")
    elif last['收盘价'] < last['BOLL_DN']:
        signals.append("⚡ 价格跌破布林下轨，可能超跌")

    if not signals:
        signals.append("😐 当前无明显交易信号，建议观望")

    return "\n".join(signals)


# ═══════════════════════════════════════════════════
# 主力持仓排名（前20名多空持仓及变化）
# ═══════════════════════════════════════════════════

def get_futures_hold_position(symbol="LC", date=None, contract=None):
    """
    获取期货品种前20名主力持仓排名数据。
    使用 futures_hold_pos_sina 接口，参数：
      symbol: 指标类型 '成交量'/'持买单量'/'持卖单量'
      contract: 合约代码如 'LC2609'
      date: 日期 'YYYYMMDD'
    本函数的 symbol 参数是品种代码如 'LC'，会自动查找主力合约。
    返回 dict: {'volume': df, 'long': df, 'short': df, 'summary': dict}
    """
    result = {'volume': pd.DataFrame(), 'long': pd.DataFrame(), 'short': pd.DataFrame(), 'summary': {}, 'contract': ''}

    # 如果没指定合约，先通过实时行情找到主力合约（成交量最大的）
    if contract is None:
        contract = _find_main_contract(symbol)

    if not contract:
        return result

    result['contract'] = contract

    # 获取三种排名数据
    # akshare futures_hold_pos_sina 的 symbol 参数: "成交量" / "多单持仓" / "空单持仓"
    indicator_map = {
        '成交量': 'volume',
        '多单持仓': 'long',
        '空单持仓': 'short',
    }

    # 自动查找最近交易日的日期
    if date is None:
        from datetime import date as dt_date
        dates_to_try = [(dt_date.today() - timedelta(days=i)).strftime("%Y%m%d") for i in range(10)]
    else:
        dates_to_try = [date]

    for try_date in dates_to_try:
        got_data = False
        for indicator, key in indicator_map.items():
            try:
                df = ak.futures_hold_pos_sina(symbol=indicator, contract=contract, date=try_date)
                if df is not None and not df.empty:
                    df = df.copy()
                    df.columns = [c.strip() for c in df.columns]
                    for col in ['成交量', '多单持仓', '空单持仓', '比上交易增减']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    result[key] = df.head(20).reset_index(drop=True)
                    got_data = True
            except Exception:
                continue
        if got_data:
            result['date'] = try_date
            break

    # 计算多空汇总
    result['summary'] = _calc_position_summary(result)
    return result


def _find_main_contract(symbol):
    """根据品种代码找到主力合约（成交量最大的）"""
    # 品种中文名映射
    symbol_name_map = {
        'LC': '碳酸锂', 'AU': '黄金', 'AG': '白银',
        'CU': '沪铜', 'AL': '沪铝', 'ZN': '沪锌', 'PB': '沪铅',
        'NI': '沪镍', 'SN': '沪锡', 'RB': '螺纹钢', 'HC': '热轧卷板',
        'I': '铁矿石', 'J': '焦炭', 'JM': '焦煤',
        'SC': '原油', 'FU': '燃油', 'BU': '沥青',
        'TA': 'PTA', 'MA': '甲醇', 'EG': '乙二醇',
        'PP': '聚丙烯', 'L': '聚乙烯', 'V': 'PVC',
        'M': '豆粕', 'Y': '豆油', 'P': '棕榈油',
        'CF': '棉花', 'SR': '白糖', 'C': '玉米',
        'SA': '纯碱', 'FG': '玻璃', 'UR': '尿素',
        'SI': '工业硅', 'RU': '橡胶', 'LH': '生猪',
        'AP': '苹果', 'CJ': '红枣', 'PK': '花生',
        'RM': '菜粕', 'OI': '菜油', 'EB': '苯乙烯',
        'SS': '不锈钢', 'SF': '硅铁', 'SM': '锰硅',
    }

    name = symbol_name_map.get(symbol.upper(), '')
    if not name:
        return None

    try:
        df = ak.futures_zh_realtime(symbol=name)
        if df is not None and not df.empty:
            df['volume'] = pd.to_numeric(df.get('volume', 0), errors='coerce')
            # 排除连续合约（如 LC0, AU0），只保留具体月份合约
            import re as _re
            df = df[~df['symbol'].apply(lambda s: bool(_re.match(r'^[A-Za-z]+0$', str(s))))]
            if df.empty:
                return None
            # 找成交量最大的合约
            main = df.sort_values('volume', ascending=False).iloc[0]
            return str(main.get('symbol', ''))
    except Exception:
        pass

    return None


def _calc_position_summary(result):
    """计算持仓汇总信息（多空对比）"""
    summary = {}

    # 多头（多单持仓）
    if not result['long'].empty:
        if '多单持仓' in result['long'].columns:
            summary['多头总持仓'] = int(pd.to_numeric(result['long']['多单持仓'], errors='coerce').sum())
        if '比上交易增减' in result['long'].columns:
            summary['多头增减'] = int(pd.to_numeric(result['long']['比上交易增减'], errors='coerce').sum())

    # 空头（空单持仓）
    if not result['short'].empty:
        if '空单持仓' in result['short'].columns:
            summary['空头总持仓'] = int(pd.to_numeric(result['short']['空单持仓'], errors='coerce').sum())
        if '比上交易增减' in result['short'].columns:
            summary['空头增减'] = int(pd.to_numeric(result['short']['比上交易增减'], errors='coerce').sum())

    # 汇总计算
    if '多头总持仓' in summary and '空头总持仓' in summary:
        summary['多空净持仓'] = summary['多头总持仓'] - summary['空头总持仓']
        total = summary['多头总持仓'] + summary['空头总持仓']
        if total > 0:
            summary['多头占比%'] = round(summary['多头总持仓'] / total * 100, 1)
    if '多头增减' in summary and '空头增减' in summary:
        summary['净增减'] = summary['多头增减'] - summary['空头增减']

    return summary


# ═══════════════════════════════════════════════════
# 傻瓜式行情分析（带原理解释）
# ═══════════════════════════════════════════════════

def generate_smart_analysis(df):
    """
    生成带原理解释的傻瓜式行情分析。
    不只是告诉你信号，还解释：
    1. 为什么这个信号好用
    2. 信号背后的市场逻辑
    3. 当前可信度评估
    4. 综合判断和操作建议
    """
    if df is None or df.empty or len(df) < 60:
        return {"overall": "数据不足（需要至少60天历史数据）", "signals": [], "verdict": ""}

    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = last['收盘价']
    signals = []
    bull_score = 0  # 多头得分
    bear_score = 0  # 空头得分

    # ═══ 1. 趋势分析：均线系统 ═══
    ma5 = last['MA5']
    ma10 = last['MA10']
    ma20 = last['MA20']
    ma60 = last.get('MA60', None)

    # 多头排列 / 空头排列
    if not pd.isna(ma5) and not pd.isna(ma20):
        if ma5 > ma10 > ma20:
            bull_score += 2
            signals.append({
                'name': '均线多头排列',
                'direction': 'bullish',
                'strength': '强',
                'signal': f'MA5({ma5:.0f}) > MA10({ma10:.0f}) > MA20({ma20:.0f})，短中期均线呈多头排列',
                'principle': '均线多头排列说明短期买入者的平均成本高于中期，中期又高于长期。'
                            '这意味着无论短线还是中线的持仓者都处于盈利状态，市场信心强。'
                            '当持仓者普遍盈利时，抛压小、买入意愿强，价格倾向继续上涨。',
                'reliability': '均线排列是趋势跟踪最基础的指标，在趋势行情中胜率约60-65%，'
                              '但在震荡行情中容易反复交叉产生假信号。'
            })
        elif ma5 < ma10 < ma20:
            bear_score += 2
            signals.append({
                'name': '均线空头排列',
                'direction': 'bearish',
                'strength': '强',
                'signal': f'MA5({ma5:.0f}) < MA10({ma10:.0f}) < MA20({ma20:.0f})，短中期均线呈空头排列',
                'principle': '均线空头排列说明近期买入者普遍被套，越晚买的亏得越多。'
                            '被套的持仓者有止损需求，形成持续的卖出压力。'
                            '同时价格低于各均线也会让潜在买家观望，减少买入动力。',
                'reliability': '空头排列在下跌趋势中胜率约60-65%，但注意跌到极端位置可能出现反转。'
            })

    # 金叉 / 死叉
    if not pd.isna(ma5) and not pd.isna(ma20):
        if last['MA5'] > last['MA20'] and prev['MA5'] <= prev['MA20']:
            bull_score += 1.5
            signals.append({
                'name': 'MA5/MA20 金叉',
                'direction': 'bullish',
                'strength': '中',
                'signal': 'MA5 上穿 MA20，形成金叉买入信号',
                'principle': '金叉意味着近5天平均成本开始超过近20天平均成本。'
                            '本质上说明短期资金正在加速流入，短线买方力量超过了中线卖方。'
                            '这是趋势反转或加速的早期信号，类似于"风向开始变了"。',
                'reliability': '单独金叉的胜率约50-55%。需要配合成交量放大（说明真的有资金进来，'
                              '而非缩量反弹），以及RSI未超买来确认。如果是在低位金叉，可靠性更高。'
            })
        elif last['MA5'] < last['MA20'] and prev['MA5'] >= prev['MA20']:
            bear_score += 1.5
            signals.append({
                'name': 'MA5/MA20 死叉',
                'direction': 'bearish',
                'strength': '中',
                'signal': 'MA5 下穿 MA20，形成死叉卖出信号',
                'principle': '死叉意味着近5天的平均买入价开始低于近20天，'
                            '说明短线资金正在撤出，多方力量在减弱。'
                            '高位死叉尤其危险——说明短线获利盘正在离场。',
                'reliability': '死叉信号胜率约50-55%。如果同时伴随成交量放大、RSI从高位回落，'
                              '可靠性显著提升。低位死叉可能只是震荡，高位死叉更值得重视。'
            })

    # 价格与MA60的关系（长期趋势）
    if ma60 is not None and not pd.isna(ma60):
        dist_pct = (close - ma60) / ma60 * 100
        if close > ma60:
            bull_score += 1
            signals.append({
                'name': '长期趋势偏多',
                'direction': 'bullish',
                'strength': '中' if dist_pct < 5 else '强',
                'signal': f'价格({close:.0f})在60日均线({ma60:.0f})之上，偏离{dist_pct:.1f}%',
                'principle': '60日均线代表近3个月的市场平均成本。价格在其上方说明'
                            '大部分持仓者是盈利的，这是牛市的基本特征。'
                            '60日线也是很多机构的持仓成本参考线。',
                'reliability': '60日均线是判断中长期趋势最稳定的参考之一。偏离过大（>10%）'
                              '时要注意回调风险。'
            })
        else:
            bear_score += 1
            signals.append({
                'name': '长期趋势偏空',
                'direction': 'bearish',
                'strength': '中' if dist_pct > -5 else '强',
                'signal': f'价格({close:.0f})在60日均线({ma60:.0f})之下，偏离{dist_pct:.1f}%',
                'principle': '价格低于60日均线意味着近3个月买入的人大多亏损，'
                            '市场整体情绪偏弱。每次反弹到60日均线附近都会遇到解套盘的抛压。',
                'reliability': '60日线是强支撑/阻力。如果连续多日无法收回60日线，'
                              '说明空方力量确实较强。'
            })

    # ═══ 2. 动量分析：RSI ═══
    rsi = last.get('RSI', None)
    if rsi is not None and not pd.isna(rsi):
        if rsi > 80:
            bear_score += 2
            signals.append({
                'name': 'RSI 极度超买',
                'direction': 'bearish',
                'strength': '强',
                'signal': f'RSI = {rsi:.1f}，进入极度超买区域',
                'principle': 'RSI衡量的是近14天上涨幅度占总波动的比例。RSI>80意味着'
                            '过去14天几乎都在涨，上涨动能已经"用力过猛"。'
                            '就像弹簧被拉到极限，继续上涨的空间小，回调概率大。'
                            '这不代表一定会跌，但代表"追涨的性价比很低"。',
                'reliability': 'RSI超买在震荡市中胜率高达70%+，但在强趋势中可能持续超买。'
                              '配合价格出现滞涨或放量长上影线时，回调信号更可靠。'
            })
        elif rsi > 70:
            bear_score += 1
            signals.append({
                'name': 'RSI 超买',
                'direction': 'bearish',
                'strength': '中',
                'signal': f'RSI = {rsi:.1f}，进入超买区域',
                'principle': 'RSI>70说明近期上涨动能较强，但也意味着短期涨幅已经较大。'
                            '此时追涨风险增加——即使趋势继续，短期回调概率也在升高。'
                            '比较聪明的做法是等RSI回落到60附近再考虑入场。',
                'reliability': 'RSI超买在单独使用时胜率约55-60%，配合均线死叉或MACD顶背离时更可靠。'
            })
        elif rsi < 20:
            bull_score += 2
            signals.append({
                'name': 'RSI 极度超卖',
                'direction': 'bullish',
                'strength': '强',
                'signal': f'RSI = {rsi:.1f}，进入极度超卖区域',
                'principle': 'RSI<20意味着过去14天几乎都在跌，卖方力量已经衰竭。'
                            '所有想卖的人基本都卖完了，继续杀跌的动力不足。'
                            '物极必反的道理——跌到极致往往意味着反弹随时可能出现。',
                'reliability': 'RSI极度超卖的反弹概率高达75%+，是最可靠的抄底信号之一。'
                              '但注意：极端行情中RSI可能持续在超卖区，设好止损很重要。'
            })
        elif rsi < 30:
            bull_score += 1
            signals.append({
                'name': 'RSI 超卖',
                'direction': 'bullish',
                'strength': '中',
                'signal': f'RSI = {rsi:.1f}，进入超卖区域',
                'principle': 'RSI<30说明近期跌幅已经较大，空方动能在减弱。'
                            '虽然趋势可能还没完全反转，但继续大幅下跌的概率在降低。'
                            '可以开始关注反弹机会，但不宜重仓。',
                'reliability': 'RSI超卖信号在震荡市中胜率约60%，需要配合其他信号确认底部。'
            })
        else:
            direction = 'bullish' if rsi > 50 else 'bearish'
            if rsi > 50:
                bull_score += 0.5
            else:
                bear_score += 0.5
            signals.append({
                'name': 'RSI 中性',
                'direction': direction,
                'strength': '弱',
                'signal': f'RSI = {rsi:.1f}，处于中性区域（{"偏强" if rsi > 55 else "偏弱" if rsi < 45 else "中性"}）',
                'principle': 'RSI在30-70之间属于正常波动范围，不构成强信号。'
                            f'RSI={rsi:.0f}{"偏向多方" if rsi > 55 else "偏向空方" if rsi < 45 else "，多空均衡"}，'
                            '目前没有极端情绪，需要依赖其他指标来判断方向。',
                'reliability': '中性RSI不具有独立的交易信号价值，需要结合趋势和其他指标综合判断。'
            })

    # ═══ 3. 趋势动能：MACD ═══
    macd = last.get('MACD', None)
    macd_signal = last.get('MACD_Signal', None)
    macd_hist = last.get('MACD_Hist', None)
    prev_hist = prev.get('MACD_Hist', None)

    if macd_hist is not None and not pd.isna(macd_hist) and prev_hist is not None and not pd.isna(prev_hist):
        if macd_hist > 0 and prev_hist <= 0:
            bull_score += 1.5
            signals.append({
                'name': 'MACD 金叉（柱由负转正）',
                'direction': 'bullish',
                'strength': '中强',
                'signal': f'MACD柱状图由负转正（{prev_hist:.2f} → {macd_hist:.2f}），DIF上穿DEA',
                'principle': 'MACD金叉意味着短期趋势动能（12日EMA）开始超过中期（26日EMA）。'
                            '柱由负转正表示下跌动能已经耗尽，上涨动能开始积累。'
                            '这就像汽车从减速到开始加速——虽然速度还不快，但方向已经改变了。',
                'reliability': 'MACD金叉在零轴下方出现（弱势转强）胜率约55%；'
                              '在零轴上方出现（强势延续）胜率更高约60-65%。'
                              '配合成交量放大确认更可靠。'
            })
        elif macd_hist < 0 and prev_hist >= 0:
            bear_score += 1.5
            signals.append({
                'name': 'MACD 死叉（柱由正转负）',
                'direction': 'bearish',
                'strength': '中强',
                'signal': f'MACD柱状图由正转负（{prev_hist:.2f} → {macd_hist:.2f}），DIF下穿DEA',
                'principle': 'MACD死叉意味着上涨动能开始减弱，短期趋势开始弱于中期。'
                            '柱由正转负表示多方力量耗尽，空方开始占上风。'
                            '如果在高位出现死叉，往往是阶段性顶部的信号。',
                'reliability': 'MACD死叉在零轴上方出现时，可能只是回调而非反转，胜率约55%；'
                              '在零轴下方出现（弱势继续）胜率约60%。'
            })
        elif macd_hist > 0:
            bull_score += 0.5
            # 柱子在变长还是变短？
            growing = macd_hist > prev_hist
            signals.append({
                'name': f'MACD 多头动能{"增强" if growing else "减弱"}',
                'direction': 'bullish' if growing else 'neutral',
                'strength': '中' if growing else '弱',
                'signal': f'MACD柱为正值({macd_hist:.2f})，且柱体{"在变长↑" if growing else "在缩短↓"}',
                'principle': f'MACD柱为正说明多头动能仍在，{"柱体变长表示动能在加速，趋势正在增强" if growing else "但柱体缩短表示动能在减弱，上涨可能即将放缓或结束"}。'
                            '关注柱子的变化趋势比柱子本身的大小更重要——变化趋势是"加速度"。',
                'reliability': '柱体连续缩短3根以上时，反转概率显著增加。'
            })
        elif macd_hist < 0:
            bear_score += 0.5
            growing = abs(macd_hist) > abs(prev_hist)
            signals.append({
                'name': f'MACD 空头动能{"增强" if growing else "减弱"}',
                'direction': 'bearish' if growing else 'neutral',
                'strength': '中' if growing else '弱',
                'signal': f'MACD柱为负值({macd_hist:.2f})，且柱体{"在变长↓" if growing else "在缩短↑"}',
                'principle': f'MACD柱为负说明空头动能仍在，{"柱体变长表示下跌在加速" if growing else "但柱体缩短表示空方力量在减弱，下跌可能即将放缓"}。'
                            '柱体缩短是底部将近的早期信号，但不等于马上反转。',
                'reliability': '空头柱体连续缩短是底部信号，配合成交量萎缩更可靠。'
            })

    # ═══ 4. 波动分析：布林带 ═══
    boll_up = last.get('BOLL_UP', None)
    boll_dn = last.get('BOLL_DN', None)
    boll_mid = last.get('BOLL_MID', None)

    if boll_up is not None and not pd.isna(boll_up):
        boll_width = (boll_up - boll_dn) / boll_mid * 100 if boll_mid and boll_mid > 0 else 0

        if close > boll_up:
            bear_score += 1
            signals.append({
                'name': '突破布林上轨',
                'direction': 'bearish',
                'strength': '中',
                'signal': f'价格({close:.0f})突破布林上轨({boll_up:.0f})，偏离均值过大',
                'principle': '布林带上轨 = 20日均线 + 2倍标准差。价格突破上轨意味着'
                            '当前价格偏离了"正常波动范围"——统计上说，95%的时间价格'
                            '应该在布林带内运行。突破上轨说明短期超涨，回归均值的概率大。',
                'reliability': '布林带突破在震荡市中回归概率约70%+，非常可靠。'
                              '但在强趋势突破时（如利好消息驱动），可能沿着上轨运行，'
                              '此时需要观察是否伴随成交量持续放大。'
            })
        elif close < boll_dn:
            bull_score += 1
            signals.append({
                'name': '跌破布林下轨',
                'direction': 'bullish',
                'strength': '中',
                'signal': f'价格({close:.0f})跌破布林下轨({boll_dn:.0f})，偏离均值过大',
                'principle': '价格跌破布林下轨说明已经超出了正常波动范围的下限。'
                            '统计上只有约2.5%的概率出现，属于极端情况。'
                            '大多数情况下价格会回归到布林中轨附近。',
                'reliability': '跌破布林下轨的反弹概率约65-70%，是不错的抄底参考。'
                              '但如果带宽很窄（波动率很低），突破可能是新趋势的起点而非超跌。'
            })
        else:
            # 价格在布林带内的位置
            pos = (close - boll_dn) / (boll_up - boll_dn) * 100 if (boll_up - boll_dn) > 0 else 50
            signals.append({
                'name': '布林带位置分析',
                'direction': 'bullish' if pos > 60 else 'bearish' if pos < 40 else 'neutral',
                'strength': '弱',
                'signal': f'价格在布林带中的位置：{pos:.0f}%（0%=下轨，100%=上轨），带宽{boll_width:.1f}%',
                'principle': f'当前价格偏向{"上轨" if pos > 60 else "下轨" if pos < 40 else "中轨"}。'
                            f'带宽{boll_width:.1f}%{"较窄，说明波动率收缩，即将迎来方向选择（突破行情）" if boll_width < 5 else "正常" if boll_width < 10 else "较宽，说明波动剧烈，趋势可能正在进行中"}。',
                'reliability': '布林带宽度收缩到极致后往往有大行情出现，这是最有价值的布林带信号。'
            })

    # ═══ 5. 成交量分析 ═══
    vol = last.get('成交量', 0)
    if vol and not pd.isna(vol):
        vol_ma5 = df['成交量'].tail(5).mean()
        vol_ma20 = df['成交量'].tail(20).mean()
        if vol_ma20 > 0:
            vol_ratio = vol / vol_ma20
            if vol_ratio > 2:
                signals.append({
                    'name': '成交量异常放大',
                    'direction': 'neutral',
                    'strength': '强',
                    'signal': f'今日成交量是20日均量的{vol_ratio:.1f}倍，成交量显著放大',
                    'principle': '成交量大幅放大说明市场分歧加剧，多空双方都在积极交易。'
                               '放量上涨 = 多方强势进攻，趋势可能加速；'
                               '放量下跌 = 恐慌性抛售，可能见底也可能破位。'
                               '需要结合价格方向来判断。',
                    'reliability': '量价配合是技术分析中最核心的原则。放量突破的可靠性远高于缩量突破。'
                })
            elif vol_ratio < 0.5:
                signals.append({
                    'name': '成交量明显萎缩',
                    'direction': 'neutral',
                    'strength': '中',
                    'signal': f'今日成交量仅为20日均量的{vol_ratio:.1f}倍，成交量明显萎缩',
                    'principle': '缩量说明市场参与热情低迷，观望情绪浓厚。'
                               '缩量上涨 = 上涨缺乏支撑，持续性存疑；'
                               '缩量下跌 = 抛压减轻，可能即将企稳。'
                               '极度缩量后往往会有方向选择。',
                    'reliability': '缩量本身不是交易信号，但缩量到极致后的放量突破是高胜率信号。'
                })

    # ═══ 6. KDJ 随机指标策略 ═══
    k_val = last.get('KDJ_K', None)
    d_val = last.get('KDJ_D', None)
    j_val = last.get('KDJ_J', None)
    prev_k = prev.get('KDJ_K', None)
    prev_d = prev.get('KDJ_D', None)

    if k_val is not None and not pd.isna(k_val) and d_val is not None and not pd.isna(d_val):
        # KDJ 金叉/死叉
        if prev_k is not None and not pd.isna(prev_k) and prev_d is not None and not pd.isna(prev_d):
            if k_val > d_val and prev_k <= prev_d and k_val < 30:
                bull_score += 2
                signals.append({
                    'name': 'KDJ 低位金叉',
                    'direction': 'bullish',
                    'strength': '强',
                    'signal': f'K({k_val:.1f})上穿D({d_val:.1f})，在超卖区形成金叉，J={j_val:.1f}',
                    'principle': 'KDJ在超卖区（K<30）形成金叉，是经典的买入信号。'
                                '此时说明价格已经触底反弹，短线动量从向下转为向上。'
                                '低位金叉意味着卖方力量耗尽，买方开始主动进场。',
                    'reliability': '低位KDJ金叉胜率约65-70%，是短线最可靠的买入信号之一。'
                                  '如果同时RSI超卖+MACD柱缩短，形成"三重共振"，胜率可达75%+。'
                })
            elif k_val > d_val and prev_k <= prev_d and k_val >= 30:
                bull_score += 1
                signals.append({
                    'name': 'KDJ 金叉',
                    'direction': 'bullish',
                    'strength': '中',
                    'signal': f'K({k_val:.1f})上穿D({d_val:.1f})，形成金叉，J={j_val:.1f}',
                    'principle': 'KDJ金叉反映短期动量转向上行，K线上穿D线说明近期'
                                '的价格波动趋势开始偏向上涨。但中位区金叉不如低位金叉可靠。',
                    'reliability': '中位区KDJ金叉胜率约55%，需要配合趋势方向确认。'
                })
            elif k_val < d_val and prev_k >= prev_d and k_val > 70:
                bear_score += 2
                signals.append({
                    'name': 'KDJ 高位死叉',
                    'direction': 'bearish',
                    'strength': '强',
                    'signal': f'K({k_val:.1f})下穿D({d_val:.1f})，在超买区形成死叉，J={j_val:.1f}',
                    'principle': 'KDJ在超买区（K>70）形成死叉，是经典的卖出信号。'
                                '说明短线涨势已经到达极致，获利盘开始离场。'
                                '高位死叉意味着买方力量枯竭，回调即将开始。',
                    'reliability': '高位KDJ死叉胜率约65-70%，短线卖出信号很可靠。'
                                  '如果J值曾超过100后回落，信号更强。'
                })
            elif k_val < d_val and prev_k >= prev_d and k_val <= 70:
                bear_score += 1
                signals.append({
                    'name': 'KDJ 死叉',
                    'direction': 'bearish',
                    'strength': '中',
                    'signal': f'K({k_val:.1f})下穿D({d_val:.1f})，形成死叉，J={j_val:.1f}',
                    'principle': 'KDJ死叉说明短期动量转向下行。但中低位死叉'
                                '可能只是短暂回调而非趋势反转。',
                    'reliability': '中位区死叉胜率约55%，需要结合趋势分析。'
                })

        # J值极端信号
        if j_val is not None and not pd.isna(j_val):
            if j_val > 100:
                bear_score += 1
                signals.append({
                    'name': 'KDJ-J值超买钝化',
                    'direction': 'bearish',
                    'strength': '中强',
                    'signal': f'J值 = {j_val:.1f}，超过100进入钝化区域',
                    'principle': 'J值超过100是极度超买的信号。J = 3K - 2D，当J>100时，'
                                '说明短期上涨速度远超均值，这种"过热"状态不可持续。'
                                '就像发动机转速过高，必须降温（回调）才能继续运行。',
                    'reliability': 'J值>100后回落的概率约70%+。但强趋势中J值可能持续钝化在100以上，'
                                  '因此更关注J值从100以上开始拐头向下的时刻。'
                })
            elif j_val < 0:
                bull_score += 1
                signals.append({
                    'name': 'KDJ-J值超卖钝化',
                    'direction': 'bullish',
                    'strength': '中强',
                    'signal': f'J值 = {j_val:.1f}，跌破0进入钝化区域',
                    'principle': 'J值跌破0是极度超卖的信号。J<0意味着下跌动量已经'
                                '远超正常范围，恐慌已经到了极致。'
                                '物极必反，极度超卖往往是短线反弹的最佳时机。',
                    'reliability': 'J值<0后反弹的概率约70%+。关注J值从负数开始拐头向上的信号。'
                })

    # ═══ 7. 威廉指标 WR 策略 ═══
    wr = last.get('WR', None)
    if wr is not None and not pd.isna(wr):
        if wr < -80:
            bull_score += 1
            signals.append({
                'name': 'WR 超卖信号',
                'direction': 'bullish',
                'strength': '中',
                'signal': f'威廉指标 WR(14) = {wr:.1f}，进入超卖区（< -80）',
                'principle': 'WR衡量当前价格相对于过去14天最高价和最低价的位置。'
                            'WR < -80 说明价格已经接近14天内的最低水平，'
                            '卖方力量接近衰竭，反弹概率增大。'
                            'WR是RSI的"镜像版"，但对短线价格变化更敏感。',
                'reliability': 'WR超卖区信号胜率约60%。WR比RSI更灵敏，所以更适合短线操作。'
                              '与KDJ超卖信号共振时可靠性大幅提升。'
            })
        elif wr > -20:
            bear_score += 1
            signals.append({
                'name': 'WR 超买信号',
                'direction': 'bearish',
                'strength': '中',
                'signal': f'威廉指标 WR(14) = {wr:.1f}，进入超买区（> -20）',
                'principle': 'WR > -20 说明价格接近14天内的最高水平，'
                            '短期涨幅已经较大。买方力量可能即将衰竭，'
                            '获利了结的压力在增大。',
                'reliability': 'WR超买区信号胜率约60%。在强趋势中WR可能持续在超买区运行，'
                              '需要等WR从超买区回落确认。'
            })

    # ═══ 8. CCI 顺势指标策略 ═══
    cci = last.get('CCI', None)
    if cci is not None and not pd.isna(cci):
        if cci > 200:
            bear_score += 1.5
            signals.append({
                'name': 'CCI 极度超买',
                'direction': 'bearish',
                'strength': '中强',
                'signal': f'CCI = {cci:.0f}，远超+100标准线，处于极度超买',
                'principle': 'CCI（顺势指标）衡量价格偏离统计均值的程度。'
                            'CCI > 200意味着价格已经严重偏离正常范围。'
                            '这不是"顺势做多"的时候，而是应该警惕回归。'
                            'CCI的独特之处在于它没有上下限，能更好地捕捉极端走势。',
                'reliability': 'CCI > 200后回调概率约65%。CCI从极端值回落时是较好的做空时点。'
            })
        elif cci > 100:
            bull_score += 0.5
            signals.append({
                'name': 'CCI 强势区间',
                'direction': 'bullish',
                'strength': '中',
                'signal': f'CCI = {cci:.0f}，在+100以上运行，处于强势区间',
                'principle': 'CCI > 100说明价格在统计上处于"异常强势"状态。'
                            '这通常意味着有新的上涨趋势正在形成。'
                            '在CCI从+100附近回落到0再次上穿+100时，是最佳买入时机。',
                'reliability': 'CCI在100以上运行是顺势信号，配合均线多头排列胜率更高。'
            })
        elif cci < -200:
            bull_score += 1.5
            signals.append({
                'name': 'CCI 极度超卖',
                'direction': 'bullish',
                'strength': '中强',
                'signal': f'CCI = {cci:.0f}，远低于-100标准线，处于极度超卖',
                'principle': 'CCI < -200意味着价格严重偏离正常范围的下方，'
                            '恐慌性抛售可能已经过度。统计上这种极端偏离往往会回归。'
                            '是逆向投资者关注的抄底信号。',
                'reliability': 'CCI < -200后反弹概率约65%。但注意趋势性下跌中CCI可能持续在低位。'
            })
        elif cci < -100:
            bear_score += 0.5
            signals.append({
                'name': 'CCI 弱势区间',
                'direction': 'bearish',
                'strength': '中',
                'signal': f'CCI = {cci:.0f}，在-100以下运行，处于弱势区间',
                'principle': 'CCI < -100说明价格处于"异常弱势"状态，下跌趋势可能正在进行中。'
                            '当CCI持续在-100以下运行时，不宜轻易抄底。',
                'reliability': 'CCI在-100以下是弱势信号。等CCI回升到-100以上再考虑买入。'
            })

    # ═══ 9. ROC 动量策略 ═══
    roc = last.get('ROC', None)
    prev_roc = prev.get('ROC', None)
    if roc is not None and not pd.isna(roc):
        if roc > 15:
            bear_score += 1
            signals.append({
                'name': 'ROC 动量过热',
                'direction': 'bearish',
                'strength': '中',
                'signal': f'12日动量 ROC = {roc:.1f}%，涨幅过大',
                'principle': 'ROC衡量12天的价格变化百分比。ROC > 15%意味着12天内涨了超过15%，'
                            '这种速度的上涨很难持续。就像短跑——速度太快了就需要喘气。'
                            '历史上大部分急涨之后都会有回调整理的过程。',
                'reliability': '短期急涨后回调概率约60-65%，但回调幅度不一定大。'
                              '强趋势中可能只是横盘整理而非大幅下跌。'
            })
        elif roc < -15:
            bull_score += 1
            signals.append({
                'name': 'ROC 动量冰点',
                'direction': 'bullish',
                'strength': '中',
                'signal': f'12日动量 ROC = {roc:.1f}%，跌幅过大',
                'principle': 'ROC < -15%意味着12天内跌了超过15%，下跌速度已经很极端。'
                            '急跌往往意味着恐慌性抛售接近尾声，技术性反弹的概率增大。',
                'reliability': '急跌后反弹概率约60-65%。但要区分"急跌反弹"和"趋势反转"'
                              '——反弹不等于见底。'
            })
        # ROC 零轴穿越
        if prev_roc is not None and not pd.isna(prev_roc):
            if roc > 0 and prev_roc <= 0:
                bull_score += 0.5
                signals.append({
                    'name': 'ROC 突破零轴',
                    'direction': 'bullish',
                    'strength': '弱',
                    'signal': f'ROC由负转正（{prev_roc:.1f}% → {roc:.1f}%），12日前买入者开始盈利',
                    'principle': 'ROC穿越零轴说明当前价格超过了12天前的价格，'
                                '意味着12天前的持仓者从亏损转为盈利，市场情绪改善。'
                                '这是中短期趋势转换的信号。',
                    'reliability': 'ROC零轴穿越胜率约55%，是辅助确认信号，需要配合趋势指标使用。'
                })
            elif roc < 0 and prev_roc >= 0:
                bear_score += 0.5
                signals.append({
                    'name': 'ROC 跌破零轴',
                    'direction': 'bearish',
                    'strength': '弱',
                    'signal': f'ROC由正转负（{prev_roc:.1f}% → {roc:.1f}%），12日前买入者开始亏损',
                    'principle': 'ROC跌破零轴说明当前价格低于12天前，持仓者开始亏损，'
                                '可能产生止损卖出压力。',
                    'reliability': 'ROC零轴跌破胜率约55%，是辅助信号。'
                })

    # ═══ 10. OBV 能量潮量价背离 ═══
    obv = last.get('OBV', None)
    obv_ma = last.get('OBV_MA20', None)
    if obv is not None and not pd.isna(obv) and len(df) >= 20:
        # 检查量价背离：价格创新高但OBV未创新高（顶背离），或价格创新低但OBV未创新低（底背离）
        recent_20 = df.tail(20)
        price_max_idx = recent_20['收盘价'].idxmax()
        obv_max_idx = recent_20['OBV'].idxmax()
        price_min_idx = recent_20['收盘价'].idxmin()
        obv_min_idx = recent_20['OBV'].idxmin()

        # 价格在近5日创新高，但OBV的最高点在更早之前（顶背离）
        recent_5_prices = df.tail(5)['收盘价']
        recent_5_obv = df.tail(5)['OBV']
        high_20 = recent_20['收盘价'].max()
        high_5 = recent_5_prices.max()

        if high_5 >= high_20 * 0.99:  # 价格接近20日高点
            obv_high_20 = recent_20['OBV'].max()
            obv_high_5 = recent_5_obv.max()
            if obv_high_5 < obv_high_20 * 0.95:  # OBV没有同步创新高
                bear_score += 1.5
                signals.append({
                    'name': 'OBV 顶背离（量价背离）',
                    'direction': 'bearish',
                    'strength': '中强',
                    'signal': '价格接近20日高点，但OBV（成交量累积）未能同步创新高',
                    'principle': '价格创新高但成交量没有跟上（OBV不创新高），说明上涨动力在衰竭。'
                                '真正的上涨需要"量增价涨"——买方资金持续流入。'
                                '顶背离意味着虽然价格还在涨，但参与的资金在减少，'
                                '是"最后一波"上涨的典型特征，顶部可能即将出现。',
                    'reliability': 'OBV顶背离是最可靠的顶部信号之一，胜率约65-70%。'
                                  '如果同时出现RSI超买或MACD死叉，基本可以确认见顶。'
                })

        low_20 = recent_20['收盘价'].min()
        low_5 = recent_5_prices.min()
        if low_5 <= low_20 * 1.01:  # 价格接近20日低点
            obv_low_20 = recent_20['OBV'].min()
            obv_low_5 = recent_5_obv.min()
            if obv_low_5 > obv_low_20 * 1.05:  # OBV没有同步创新低
                bull_score += 1.5
                signals.append({
                    'name': 'OBV 底背离（量价背离）',
                    'direction': 'bullish',
                    'strength': '中强',
                    'signal': '价格接近20日低点，但OBV（成交量累积）未能同步创新低',
                    'principle': '价格创新低但OBV未创新低，说明下跌过程中卖出量在减少。'
                                '这意味着"该卖的都已经卖了"，下方的卖出压力在枯竭。'
                                '底背离是"暗流涌动"——虽然价格还在跌，但聪明钱已经开始低吸。',
                    'reliability': 'OBV底背离是最可靠的底部信号之一，胜率约65-70%。'
                                  '配合RSI超卖和KDJ金叉更可靠。'
                })

        # OBV趋势 vs 价格趋势
        if obv_ma is not None and not pd.isna(obv_ma):
            if obv > obv_ma and close < ma20 and not pd.isna(ma20):
                bull_score += 0.5
                signals.append({
                    'name': 'OBV 资金暗中流入',
                    'direction': 'bullish',
                    'strength': '弱',
                    'signal': 'OBV在20日均线之上（资金流入），但价格在MA20之下',
                    'principle': '价格偏弱但OBV偏强，说明有资金在低位持续买入。'
                                '这种"暗中吸筹"的行为往往出现在主力建仓阶段，'
                                '价格一旦止跌，可能出现较强的反弹。',
                    'reliability': '量价分歧的信号胜率约55%，是前瞻性信号，可能需要几天才会兑现。'
                })

    # ═══ 11. 连涨连跌分析 ═══
    consecutive = 0
    for i in range(len(df) - 1, max(len(df) - 11, 0), -1):
        if df.iloc[i]['收盘价'] > df.iloc[i - 1]['收盘价']:
            if consecutive >= 0:
                consecutive += 1
            else:
                break
        elif df.iloc[i]['收盘价'] < df.iloc[i - 1]['收盘价']:
            if consecutive <= 0:
                consecutive -= 1
            else:
                break
        else:
            break

    if consecutive >= 5:
        bear_score += 1.5
        signals.append({
            'name': f'连续上涨{consecutive}天',
            'direction': 'bearish',
            'strength': '中强',
            'signal': f'价格已经连续上涨{consecutive}个交易日',
            'principle': f'连续{consecutive}天上涨在统计上属于较为罕见的事件（概率约{2**(- consecutive)*100:.1f}%）。'
                        '连涨天数越多，回调的概率越大。这不是说"一定会跌"，而是说'
                        '"继续涨的赔率在降低"。理性的做法是等回调后再入场，而非追涨。',
            'reliability': f'连涨{consecutive}天后次日回调的概率约{min(50 + consecutive * 5, 80)}%。'
                          '但需注意在大牛市中连涨可能持续更久。'
        })
    elif consecutive >= 3:
        bear_score += 0.5
        signals.append({
            'name': f'连续上涨{consecutive}天',
            'direction': 'neutral',
            'strength': '弱',
            'signal': f'价格已经连续上涨{consecutive}个交易日，短期有获利回吐压力',
            'principle': '短期连涨说明多方力量较强，但也在累积获利盘的回吐压力。'
                        '如果量能无法持续放大，回调风险增大。',
            'reliability': '连涨3-4天的回调概率约55%，不算强信号。'
        })
    elif consecutive <= -5:
        bull_score += 1.5
        days = abs(consecutive)
        signals.append({
            'name': f'连续下跌{days}天',
            'direction': 'bullish',
            'strength': '中强',
            'signal': f'价格已经连续下跌{days}个交易日',
            'principle': f'连续{days}天下跌意味着恐慌情绪已经充分释放。'
                        '就像橡皮筋被拉到极限——跌得越久，反弹的力度可能越大。'
                        '连跌中后期往往是恐慌盘的集中释放，一旦卖盘枯竭就会反弹。',
            'reliability': f'连跌{days}天后反弹的概率约{min(50 + days * 5, 80)}%。'
                          '但要区分技术反弹和趋势反转。'
        })
    elif consecutive <= -3:
        bull_score += 0.5
        days = abs(consecutive)
        signals.append({
            'name': f'连续下跌{days}天',
            'direction': 'neutral',
            'strength': '弱',
            'signal': f'价格已经连续下跌{days}个交易日，短线可能出现技术反弹',
            'principle': '短期连跌后市场可能出现超跌反弹，但不一定代表趋势反转。',
            'reliability': f'连跌3-4天后反弹概率约55%。'
        })

    # ═══ 12. ATR 波动率策略 ═══
    atr = last.get('ATR', None)
    if atr is not None and not pd.isna(atr) and close > 0:
        atr_pct = atr / close * 100  # ATR占比
        # 与历史ATR比较
        atr_series = df['ATR'].dropna().tail(60)
        if len(atr_series) >= 20:
            atr_percentile = (atr_series < atr).sum() / len(atr_series) * 100

            if atr_percentile > 90:
                signals.append({
                    'name': 'ATR 波动率极高',
                    'direction': 'neutral',
                    'strength': '中强',
                    'signal': f'ATR = {atr:.0f}（占价格{atr_pct:.1f}%），处于近60日{atr_percentile:.0f}%分位，波动极剧烈',
                    'principle': '波动率处于历史极高水平，说明市场情绪极度不稳定。'
                                '极高波动率通常出现在趋势的末端——不管是疯狂上涨的顶部还是恐慌下跌的底部。'
                                '此时持仓风险极大，建议降低仓位或设置更宽的止损。'
                                f'当前单日波动约{atr_pct:.1f}%，止损建议至少设{atr_pct*2:.1f}%。',
                    'reliability': '极高波动率后行情往往会收敛。但不能单独判断方向，要配合趋势指标。'
                })
            elif atr_percentile < 10:
                signals.append({
                    'name': 'ATR 波动率极低（收缩）',
                    'direction': 'neutral',
                    'strength': '中强',
                    'signal': f'ATR = {atr:.0f}（占价格{atr_pct:.1f}%），处于近60日{atr_percentile:.0f}%分位，波动极低',
                    'principle': '波动率收缩到极低水平，这是"暴风雨前的平静"。'
                                '历史上波动率极低之后往往会迎来一波大行情（突破行情）。'
                                '方向不确定，但幅度可能会很大。'
                                '策略建议：可以用唐奇安通道或布林带突破来捕捉方向。',
                    'reliability': '波动率收缩后突破行情的概率约70%+，是最可靠的预警信号之一。'
                                  '关键是不知道方向——需要等突破信号出现后再跟进。'
                })

    # ═══ 13. 支撑阻力位分析 ═══
    if len(df) >= 30:
        recent_30 = df.tail(30)
        resistance = recent_30['最高价'].max()
        support = recent_30['最低价'].min()
        range_pct = (resistance - support) / support * 100

        dist_to_resistance = (resistance - close) / close * 100
        dist_to_support = (close - support) / close * 100

        if dist_to_resistance < 1:
            signals.append({
                'name': '接近30日阻力位',
                'direction': 'bearish',
                'strength': '中',
                'signal': f'价格({close:.0f})距30日最高点({resistance:.0f})仅{dist_to_resistance:.1f}%，面临阻力',
                'principle': '30日最高点是重要的心理阻力位——前高买入的人到达解套位时可能抛售。'
                            '同时，前高附近也是很多交易者设定的止盈位。'
                            '突破前高需要放量确认，否则可能再次回落。',
                'reliability': '前高阻力位的有效率约60%。如果放量突破前高，则阻力转为支撑，'
                              '可能开启新一轮上涨。'
            })
        elif dist_to_support < 1:
            signals.append({
                'name': '接近30日支撑位',
                'direction': 'bullish',
                'strength': '中',
                'signal': f'价格({close:.0f})距30日最低点({support:.0f})仅{dist_to_support:.1f}%，接近支撑',
                'principle': '30日最低点是重要的心理支撑位——前低附近往往有买盘支撑。'
                            '如果能在前低附近获得支撑并反弹，形成"双底"结构，是可靠的买入信号。'
                            '但如果跌破前低，可能引发新一轮下跌。',
                'reliability': '前低支撑位的有效率约60%。配合缩量企稳更可靠。'
                              '跌破前低则止损。'
            })

    # ═══ 综合评判 ═══
    total_score = bull_score - bear_score
    if total_score >= 4:
        verdict = '强烈看多'
        verdict_detail = '多个技术指标共振看多，市场处于明显的上升趋势中。建议顺势做多，在回调到均线附近时加仓。'
        verdict_color = 'bullish'
    elif total_score >= 2:
        verdict = '偏多'
        verdict_detail = '技术面偏向多方，但信号不够强烈。可以轻仓做多，但要设好止损。关注能否放量突破确认。'
        verdict_color = 'bullish'
    elif total_score >= 0.5:
        verdict = '略偏多'
        verdict_detail = '技术面略偏多方，但不够明确。建议以观望为主，等待更多确认信号。如果已经持有多单，可以继续持有。'
        verdict_color = 'neutral'
    elif total_score <= -4:
        verdict = '强烈看空'
        verdict_detail = '多个技术指标共振看空，市场处于明显的下降趋势中。建议回避做多，已有多单考虑止损。'
        verdict_color = 'bearish'
    elif total_score <= -2:
        verdict = '偏空'
        verdict_detail = '技术面偏向空方。建议谨慎操作，不宜追多。如果做空有条件的话可以考虑轻仓试空。'
        verdict_color = 'bearish'
    elif total_score <= -0.5:
        verdict = '略偏空'
        verdict_detail = '技术面略偏空，但不够明确。建议观望为主，等待方向确认。持有多单注意设好止损。'
        verdict_color = 'neutral'
    else:
        verdict = '多空均衡'
        verdict_detail = '技术面多空力量均衡，方向不明。建议等待，不要急于操作。等出现明确突破信号后再入场。'
        verdict_color = 'neutral'

    return {
        'signals': signals,
        'bull_score': bull_score,
        'bear_score': bear_score,
        'total_score': total_score,
        'verdict': verdict,
        'verdict_detail': verdict_detail,
        'verdict_color': verdict_color,
        'price': close,
        'ma5': ma5 if not pd.isna(ma5) else None,
        'ma20': ma20 if not pd.isna(ma20) else None,
        'rsi': rsi if rsi is not None and not pd.isna(rsi) else None,
    }
