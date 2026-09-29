"""配置加载与校验。

配置来源（优先级从高到低）：
1. 用户通过 ``--config`` 显式指定的 YAML 文件
2. 包内 ``config/default.yaml``
3. 内置默认值（见本模块 DEFAULT_CONFIG）

采用"深度合并 + 白名单校验"：未知键忽略并告警，缺失键回退默认值，
关键数值做类型与范围校验。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

logger = logging.getLogger("sfw.config")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.yaml"

# 内置兜底配置（当 default.yaml 缺失或不可读时使用）
DEFAULT_CONFIG: dict[str, Any] = {
    "engine": {
        "temperature_default": 0.15,
        "temperature_high_vol": 0.25,
        "high_volatility_threshold": 0.30,
        "fractal_entropy_threshold": 0.7,
        "fractal_conflict_threshold": 3,
        "fractal_max_depth": 3,
        "tension_micro_threshold": 0.55,
        "tension_macro_threshold": 0.80,
        "fission_cooldown": 3,
        "fission_entropy_threshold": 0.95,
        "fission_conflict_threshold": 3,
        "self_correlation_warning": 0.65,
        "topology_focused_volume": 3.0,
        "topology_diffuse_volume": 7.0,
        "coupling_strong": 0.8,
        "coupling_anti": -0.8,
        "emergence_high_entropy": 0.8,
        "emergence_high_entropy_count": 4,
        "emergence_min_mutual_info": 0.1,
        "emergence_track_cycles": 3,
    },
    "scoring": {
        "weights": {},
        "bank_special": {
            "nim": {"plus": 1.75, "zero": 1.70},
            "npl": {"plus": 1.10, "zero": 1.30},
            "provision_coverage": {"plus": 200.0, "zero": 150.0},
            "capital_adequacy": {"plus": 13.0, "zero": 11.0},
            "dividend_yield": {"plus": 4.0, "zero": 2.5},
        },
    },
    "valuation": {
        "weights": {"DCF": 0.0, "PE": 0.10, "PB": 0.50, "EVEBITDA": 0.0, "SOTP": 0.30},
        "defaults": {
            "target_pe": 6.0,
            "target_pb": 0.56,
            "wacc": 0.10,
            "terminal_growth": 0.03,
            "ev_ebitda_multiple": 8.0,
        },
    },
    "backtest": {"risk_free": 0.02, "cost": 0.0003, "seed": 42},
    "report": {"disclaimer": ""},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并：override 覆盖 base。"""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _validate_engine(cfg: dict[str, Any]) -> None:
    e = cfg.get("engine", {})
    # 温度与阈值必须为正且合理
    for key in ("temperature_default", "temperature_high_vol", "high_volatility_threshold"):
        val = e.get(key)
        if not isinstance(val, (int, float)) or val <= 0:
            raise ValueError(f"engine.{key} 必须为正数，当前: {val!r}")
    if e.get("temperature_high_vol") <= e.get("temperature_default"):
        raise ValueError("engine.temperature_high_vol 必须大于 temperature_default")


@dataclass
class SFWConfig:
    """运行时配置对象。"""

    raw: dict[str, Any] = field(default_factory=dict)

    def engine(self, key: str, default: Any = None) -> Any:
        return self.raw.get("engine", {}).get(key, default)

    def scoring_weights(self, dim: str) -> dict[str, float]:
        return self.raw.get("scoring", {}).get("weights", {}).get(dim, {})

    def bank_special(self, key: str, default: Any = None) -> Any:
        return self.raw.get("scoring", {}).get("bank_special", {}).get(key, default)

    def valuation(self, key: str, default: Any = None) -> Any:
        return self.raw.get("valuation", {}).get(key, default)

    def backtest(self, key: str, default: Any = None) -> Any:
        return self.raw.get("backtest", {}).get(key, default)

    def disclaimer(self) -> str:
        return self.raw.get("report", {}).get("disclaimer", "")


def _load_bundled_default() -> dict[str, Any]:
    """读取打包进包的默认配置（src/sfw/conf/default.yaml）。"""
    try:
        from importlib import resources
        text = resources.files("sfw").joinpath("conf", "default.yaml").read_text(encoding="utf-8")
        return yaml.safe_load(text) or {}
    except Exception:  # noqa: BLE001
        return {}


def load_config(path: str | os.PathLike | None = None, *, extra: dict[str, Any] | None = None) -> SFWConfig:
    """加载配置。

    合并顺序（后覆盖前）：内置默认 → 包内 conf/default.yaml → 项目
    config/default.yaml → 用户显式文件(path) → 代码覆盖(extra)。

    参数:
        path: 用户配置 YAML 路径（可选）。
        extra: 以代码方式覆盖的额外配置（优先级最高）。

    返回:
        校验通过后的 SFWConfig。
    """
    if yaml is None:  # pragma: no cover
        raise RuntimeError("缺少依赖 PyYAML，请先执行 pip install PyYAML")

    merged = _deep_merge(DEFAULT_CONFIG, {})

    # 1) 包内 conf/default.yaml（wheel 安装时唯一兜底）
    merged = _deep_merge(merged, _load_bundled_default())

    # 2) 项目级 config/default.yaml（若存在，便于用户直接编辑）
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
            file_cfg = yaml.safe_load(fh) or {}
        merged = _deep_merge(merged, file_cfg)

    # 3) 用户显式配置文件
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        merged = _deep_merge(merged, user_cfg)

    # 4) 代码层覆盖
    if extra:
        merged = _deep_merge(merged, extra)

    _validate_engine(merged)
    return SFWConfig(raw=merged)
