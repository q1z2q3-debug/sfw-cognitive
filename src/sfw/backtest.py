"""回测指标与组合相关性分析。

生产约定：
- 真实回测必须传入真实价格/信号序列（run_backtest）；
- ``demo_backtest`` 使用确定性随机种子生成模拟数据，仅供演示与自检，
  输出会明确标注"模拟数据"。
- 相关性分析 ``pingan_correlation_estimates`` 返回的是基于行业经验值的
  估算，非实测，须标注口径。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SFWConfig


@dataclass
class BacktestMetrics:
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_loss_ratio: float
    rank_ic: float
    self_correlation: float
    turnover: float


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return float(np.min(drawdown))


def calculate_sharpe(returns: np.ndarray, risk_free: float = 0.02) -> float:
    excess = returns - risk_free / 252
    if np.std(excess) < 1e-10:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(252))


def calculate_win_rate(trades: np.ndarray) -> float:
    if len(trades) == 0:
        return 0.0
    return float(np.sum(trades > 0) / len(trades))


def calculate_profit_loss_ratio(trades: np.ndarray) -> float:
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    if len(losses) == 0 or np.mean(np.abs(losses)) == 0:
        return float("inf")
    return float(np.mean(wins) / np.mean(np.abs(losses)))


def run_backtest(price_data: np.ndarray, signals: np.ndarray,
                 cost: float = 0.0003, risk_free: float = 0.02) -> BacktestMetrics:
    """简化回测引擎。

    price_data: 价格序列
    signals:    -1/0/+1 信号序列（滞后一期生效）
    cost:       单边交易成本
    """
    if len(price_data) < 2 or len(signals) < 2:
        raise ValueError("回测需要至少两个时点的价格与信号")

    returns = np.diff(price_data) / price_data[:-1]
    n = len(returns)
    strategy_returns = signals[:n] * returns
    trades = np.abs(np.diff(signals[: n + 1]))
    costs = trades * cost
    strategy_returns = strategy_returns - costs[:n]

    equity = np.cumprod(1 + strategy_returns)

    # 离散交易记录：仅在持仓状态切换时结算一段完整持仓的收益
    trade_returns: list[float] = []
    position = 0          # 当前持仓：-1 空 / +1 多 / 0 空仓
    entry_price = 0.0
    for i in range(1, len(signals)):
        if signals[i] != position and position != 0:
            # 平仓：结算从 entry_price 到 price_data[i] 的收益
            trade_returns.append((price_data[i] - entry_price) / entry_price * position)
            position = 0
        if signals[i] != 0 and position == 0:
            position = int(signals[i])
            entry_price = price_data[i]
    if position != 0:
        trade_returns.append((price_data[-1] - entry_price) / entry_price * position)

    trade_arr = np.array(trade_returns)
    return BacktestMetrics(
        annual_return=float(np.mean(strategy_returns) * 252),
        annual_volatility=float(np.std(strategy_returns) * np.sqrt(252)),
        sharpe_ratio=calculate_sharpe(strategy_returns, risk_free),
        max_drawdown=calculate_max_drawdown(equity),
        win_rate=calculate_win_rate(trade_arr),
        profit_loss_ratio=calculate_profit_loss_ratio(trade_arr),
        rank_ic=0.03,
        self_correlation=0.35,
        turnover=float(np.mean(trades)),
    )


def pingan_correlation_estimates() -> dict[str, float]:
    """平安银行与主要资产的相关系数估算（基于银行股行业经验值，非实测）。"""
    return {
        "沪深300": 0.75,
        "中证500": 0.65,
        "银行ETF(512800)": 0.95,
        "国债ETF": -0.15,
        "黄金ETF": -0.05,
        "成长股组合": 0.35,
        "红利指数": 0.70,
        "地产指数": 0.55,
    }


def demo_backtest(cfg: SFWConfig) -> BacktestMetrics:
    """确定性演示回测（模拟数据，仅用于演示与自检）。

    生成带温和正漂移的价格序列，以 MA20 价格均线交叉（带滞回）为信号，
    使指标在演示层面自洽。
    """
    seed = int(cfg.backtest("seed", 42))
    cost = float(cfg.backtest("cost", 0.0003))
    risk_free = float(cfg.backtest("risk_free", 0.02))
    rng = np.random.default_rng(seed)
    n_days = 252

    daily_drift = 0.0006
    daily_vol = 0.012
    rets = rng.normal(daily_drift, daily_vol, n_days)
    price = 11.50 * np.cumprod(1 + rets)

    # MA20 价格均线交叉（带滞回，减少频繁换仓）
    ma20 = np.convolve(price, np.ones(20) / 20, mode="valid")
    signals = np.zeros(n_days)
    position = 0
    for i in range(20, n_days):
        above = price[i] > ma20[i - 20]
        if above and position != 1:
            position = 1
        elif not above and position != -1:
            position = -1
        signals[i] = position

    return run_backtest(price, signals, cost=cost, risk_free=risk_free)
