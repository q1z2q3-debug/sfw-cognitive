"""输入数据模式、加载与校验。

数据文件为 YAML（见 data/pingan_000001.yaml）。生产约定：
- ``meta`` 与 ``market`` 为必填；缺失即校验失败。
- ``scoring`` 与 ``bank_special`` 为选填；缺失指标在评分时按 0（中性）
  处理并在报告中标注"数据不可得"，不静默失败。
- ``valuation`` 为选填；缺失时估值方法返回不可用标记。

校验失败会抛出带有明确原因的异常；缺失数据只降级不中断。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StockData:
    meta: dict[str, Any] = field(default_factory=dict)
    market: dict[str, Any] = field(default_factory=dict)
    scoring: dict[str, dict[str, int]] = field(default_factory=dict)
    bank_special: dict[str, float] = field(default_factory=dict)
    valuation: dict[str, Any] = field(default_factory=dict)
    source: list[dict[str, str]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return str(self.meta.get("stock", "未知标的"))

    @property
    def code(self) -> str:
        return str(self.meta.get("code", ""))


def _validate_meta(meta: dict[str, Any]) -> None:
    if not isinstance(meta, dict) or not meta.get("stock"):
        raise ValueError("数据文件缺少 meta.stock（标的名称）")


def _validate_market(market: dict[str, Any]) -> None:
    if not isinstance(market, dict):
        raise ValueError("数据文件缺少 market 段")
    price = market.get("price")
    if price is None or not isinstance(price, (int, float)) or price <= 0:
        raise ValueError("market.price 必须为正数")


def _normalize_scoring(scoring: Any) -> dict[str, dict[str, int]]:
    """评分指标归一化：值域必须为 {-1, 0, 1}。"""
    out: dict[str, dict[str, int]] = {}
    if not scoring:
        return out
    for dim, indicators in scoring.items():
        if not isinstance(indicators, dict):
            continue
        norm = {}
        for name, val in indicators.items():
            if val is None:
                continue
            v = int(val)
            if v not in (-1, 0, 1):
                raise ValueError(f"评分指标 {dim}.{name} 取值必须为 -1/0/+1，当前 {val}")
            norm[name] = v
        out[dim] = norm
    return out


def load_stock_data(path: str | Path) -> StockData:
    """加载并校验数据文件。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据文件不存在: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError("数据文件顶层必须为映射（YAML 字典）")

    meta = raw.get("meta", {})
    _validate_meta(meta)
    market = raw.get("market", {})
    _validate_market(market)

    return StockData(
        meta=meta,
        market=market,
        scoring=_normalize_scoring(raw.get("scoring", {})),
        bank_special={k: float(v) for k, v in (raw.get("bank_special") or {}).items()},
        valuation=raw.get("valuation", {}) or {},
        source=raw.get("source", []) or [],
    )
