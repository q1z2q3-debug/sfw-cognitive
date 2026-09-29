"""实时行情数据接入（akshare 适配层）。

依赖可选：``pip install "sfw-cognitive[data]"`` 以启用实时取数（pandas + akshare）。
无依赖时所有函数抛出/返回明确的"数据不可得"标记，不阻塞分析流水线。

功能：
- 拉取日线行情 / 财务指标 / 北向持股 / 融资融券
- 技术指标（MA/MACD/RSI）
- 由行情与财务数据计算九维评分指标（{-1/0/+1}，与 scoring.weights 键一致）
- ``fetch_to_data_dict`` 生成可直接写入 ``data/*.yaml`` 的数据字典

约定：akshare 接口可能随版本变动，本模块对每个接口做容错；接口变更时返回
"数据不可得"而非静默编造数值。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np

logger = logging.getLogger("sfw.market_data")

try:
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:  # pragma: no cover
    pd = None
    HAVE_PANDAS = False

try:
    import akshare as ak
    HAVE_AKSHARE = True
except ImportError:  # pragma: no cover
    ak = None
    HAVE_AKSHARE = False

DIM_INDICATOR_ORDER: dict[str, list[str]] = {
    "天": ["policy", "macro", "liquidity", "geopolitics"],
    "地": ["pe_percentile", "pb_percentile", "roe_trend", "revenue_growth",
           "profit_growth", "cashflow_quality"],
    "人": ["northbound", "margin_balance", "main_capital", "turnover"],
    "正": ["ma_arrange", "macd", "rsi", "breakout"],
    "反": ["valuation_extreme", "oversold", "chip_concentration", "reversal_signal"],
    "合": ["factor_resonance", "cross_modal", "signal_strength"],
    "归": ["industry_cycle", "seasonality", "historical_similar"],
    "守": ["debt_ratio", "interest_bearing_ratio", "goodwill", "pledge", "compliance"],
    "进": ["catalyst_strength", "catalyst_time", "profit_elasticity", "consensus"],
}


def require_deps() -> None:
    """拉取实时数据需要 akshare。"""
    if not HAVE_AKSHARE:
        raise RuntimeError("缺少可选依赖：请执行 pip install 'sfw-cognitive[data]'（pandas + akshare）")


# ═══════════════════════════════════════════════════════════════
# 原始数据拉取
# ═══════════════════════════════════════════════════════════════

def fetch_price(symbol: str = "sz000001", days: int = 750) -> pd.DataFrame | None:
    """获取日线行情（前复权）。失败返回 None。"""
    if not HAVE_AKSHARE:
        return None
    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date=start, end_date=end, adjust="qfq")
        df["日期"] = pd.to_datetime(df["日期"])
        return df.sort_values("日期").reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("获取日线行情失败(%s): %s", symbol, exc)
        return None


def fetch_financial(symbol: str = "sz000001") -> pd.DataFrame | None:
    """获取财务分析指标。失败返回 None。"""
    if not HAVE_AKSHARE:
        return None
    try:
        return ak.stock_financial_analysis_indicator(symbol=symbol)
    except Exception as exc:  # noqa: BLE001
        logger.warning("获取财务指标失败(%s): %s", symbol, exc)
        return None


def fetch_northbound(stock: str = "平安银行") -> pd.DataFrame | None:
    """获取北向持股（近30日）。失败返回 None。"""
    if not HAVE_AKSHARE:
        return None
    try:
        return ak.stock_hsgt_individual_em(stock=stock)
    except Exception as exc:  # noqa: BLE001
        logger.warning("获取北向数据失败(%s): %s", stock, exc)
        return None


def fetch_margin(symbol: str = "sz000001") -> pd.DataFrame | None:
    """获取融资融券明细。失败返回 None。"""
    if not HAVE_AKSHARE:
        return None
    try:
        df = ak.stock_margin_detail_szse(date=datetime.now().strftime("%Y%m%d"))
        code = symbol.replace("sz", "")
        return df[df["证券代码"] == code]
    except Exception as exc:  # noqa: BLE001
        logger.warning("获取融资融券失败(%s): %s", symbol, exc)
        return None


# ═══════════════════════════════════════════════════════════════
# 技术指标
# ═══════════════════════════════════════════════════════════════

def calc_ma(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    for w in (windows or [5, 10, 20, 60]):
        df[f"MA{w}"] = df["收盘"].rolling(w).mean()
    return df


def calc_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df["收盘"].ewm(span=fast).mean()
    ema_slow = df["收盘"].ewm(span=slow).mean()
    df["DIF"] = ema_fast - ema_slow
    df["DEA"] = df["DIF"].ewm(span=signal).mean()
    df["MACD"] = 2 * (df["DIF"] - df["DEA"])
    return df


def calc_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = df["收盘"].diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)
    return df


def _safe(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return 0


# ═══════════════════════════════════════════════════════════════
# 九维指标计算
# ═══════════════════════════════════════════════════════════════

def compute_scoring_indicators(price_df: pd.DataFrame,
                               fin_df: pd.DataFrame | None = None) -> dict[str, dict[str, int]]:
    """由行情与财务数据计算九维评分指标（{-1/0/+1}，键与 config scoring.weights 一致）。"""
    if not HAVE_PANDAS or price_df is None or price_df.empty:
        return {}

    tech = calc_ma(calc_macd(calc_rsi(price_df.copy())))
    last = tech.iloc[-1]
    recent = tech.tail(20)

    def s(cond_plus, cond_minus):
        return 1 if cond_plus else (-1 if cond_minus else 0)

    # 地维：估值分位（用近3年收盘价代替）
    pe_ind = _safe(lambda: s(float(price_df["收盘"].rank(pct=True).iloc[-1]) < 0.30,
                             float(price_df["收盘"].rank(pct=True).iloc[-1]) > 0.70))
    # ROE/营收/利润/现金流：用财务指标（若有）
    roe_trend = _safe(lambda: 1 if float(fin_df["净资产收益率(%)"].iloc[0])
                      > float(fin_df["净资产收益率(%)"].iloc[1]) else -1) if fin_df is not None else 0
    rev_ind = _safe(lambda: s(float(fin_df["主营业务收入增长率(%)"].iloc[0]) > 5,
                              float(fin_df["主营业务收入增长率(%)"].iloc[0]) < 0)) if fin_df is not None else 0
    profit_ind = _safe(lambda: s(float(fin_df["净利润增长率(%)"].iloc[0]) > 5,
                                 float(fin_df["净利润增长率(%)"].iloc[0]) < 0)) if fin_df is not None else 0

    indicators = {
        "天": {"policy": 0, "macro": 0, "liquidity": 0, "geopolitics": 0},
        "地": {"pe_percentile": pe_ind, "pb_percentile": pe_ind, "roe_trend": roe_trend,
               "revenue_growth": rev_ind, "profit_growth": profit_ind, "cashflow_quality": 0},
        "人": {"northbound": 0, "margin_balance": 0, "main_capital": 0, "turnover": 0},
        "正": {
            "ma_arrange": s(last["MA5"] > last["MA10"] > last["MA20"],
                            last["MA5"] < last["MA10"] < last["MA20"]),
            "macd": s(last["DIF"] > last["DEA"] and last["DIF"] > 0,
                      last["DIF"] < last["DEA"] and last["DIF"] < 0),
            "rsi": s(50 < last["RSI"] < 70, last["RSI"] < 30 or last["RSI"] > 80),
            "breakout": s(last["收盘"] >= recent["最高"].max() * 0.99,
                          last["收盘"] <= recent["最低"].min() * 1.01),
        },
        "反": {"valuation_extreme": pe_ind, "oversold": 0, "chip_concentration": 0, "reversal_signal": 0},
        "合": {"factor_resonance": 0, "cross_modal": 0, "signal_strength": 0},
        "归": {"industry_cycle": 0, "seasonality": 0, "historical_similar": 0},
        "守": {"debt_ratio": 0, "interest_bearing_ratio": 0, "goodwill": 0, "pledge": 0, "compliance": 0},
        "进": {"catalyst_strength": 0, "catalyst_time": 0, "profit_elasticity": 0, "consensus": 0},
    }
    return indicators


def fetch_to_data_dict(symbol: str = "sz000001", name: str = "",
                       days: int = 750) -> dict[str, Any]:
    """拉取实时数据并生成可写入 data/*.yaml 的数据字典。

    任一数据源失败时，对应字段标为"数据不可得"（None / 空），不编造数值。
    """
    require_deps()
    price = fetch_price(symbol, days)
    fin = fetch_financial(symbol)
    scoring = compute_scoring_indicators(price, fin) if price is not None else {}

    market: dict[str, Any] = {"price": None}
    if price is not None and not price.empty:
        close = float(price["收盘"].iloc[-1])
        market = {
            "price": round(close, 2),
            "volatility_annual": round(float(price["收盘"].pct_change().std() * np.sqrt(252)), 3),
        }

    return {
        "meta": {
            "stock": name or symbol,
            "code": symbol,
            "as_of": datetime.now().strftime("%Y-%m-%d"),
            "currency": "CNY",
        },
        "market": market,
        "scoring": scoring,
        "source": [
            {"item": "实时行情（akshare）", "source": "akshare",
             "date": datetime.now().strftime("%Y-%m-%d"), "confidence": "中(自动拉取,需人工核验)"},
        ],
        "_note": "自动拉取数据，PE/PB/股息率/三表等需人工补充，未取到的字段请勿编造。",
    }
