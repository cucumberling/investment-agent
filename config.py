"""投资 Agent 配置文件"""
import os

# Claude API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# 关注的期货合约
FUTURES_CONTRACTS = {
    "LC0": "碳酸锂连续",
    "LC2605": "碳酸锂2605",
    "LC2609": "碳酸锂2609",
}

# 主力合约
MAIN_CONTRACT = "LC2605"

# 关注的相关股票（后续扩展用）
RELATED_STOCKS = {
    "002460": "赣锋锂业",
    "002466": "天齐锂业",
    "600549": "厦门钨业",
}

# 企业微信 Webhook（填入你的 key）
WECHAT_WEBHOOK_KEY = os.getenv("WECHAT_WEBHOOK_KEY", "")
WECHAT_WEBHOOK_URL = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={WECHAT_WEBHOOK_KEY}"

# 技术指标参数
MA_FAST = 5
MA_SLOW = 20
RSI_PERIOD = 14
ATR_PERIOD = 14

# 风控参数
MAX_SINGLE_RISK = 0.02      # 单笔风险不超过 2%
MAX_MARGIN_RATIO = 0.50     # 保证金占用不超过 50%
MAX_CONSECUTIVE_LOSS = 3    # 连续亏损 3 笔暂停
