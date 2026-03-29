"""AI 分析模块 - 调用 Claude API 生成投资分析"""
import os

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def analyze_market(price_data: str, news_data: str, signals: str) -> str:
    """调用 Claude API 综合分析市场数据"""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")

    if not HAS_ANTHROPIC or not api_key:
        return _local_analysis(price_data, signals)

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""你是一位专业的碳酸锂期货分析师，请根据以下数据给出简洁的市场分析和操作建议。

## 实时行情
{price_data}

## 技术信号
{signals}

## 最新资讯
{news_data}

请按以下格式输出：
### 行情概述
（2-3句话概括当前走势）

### 技术面判断
（基于技术指标的多空判断）

### 消息面影响
（新闻对价格的可能影响）

### 操作建议
（具体的操作方向和关键价位，注明仅供参考）

⚠️ 以上分析仅供参考，不构成投资建议。"""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text
    except Exception as e:
        return f"AI 分析暂不可用（{e}），使用本地分析：\n\n" + _local_analysis(price_data, signals)


def _local_analysis(price_data: str, signals: str) -> str:
    """本地简单分析（不依赖 API）"""
    return f"""### 行情数据
{price_data}

### 技术信号
{signals}

### 提示
未配置 ANTHROPIC_API_KEY，显示原始数据。
设置方法：export ANTHROPIC_API_KEY="your-key"

⚠️ 以上数据仅供参考，不构成投资建议。"""
