"""九维评分卡（概率云态版）+ 银行专项评分。

设计：
- 每个维度由若干指标组成，每项指标根据其原始值映射为 -1/0/+1；
- 维度加权分 s ∈ [-1, +1]，再由 ``core.softmax_cloud`` 转为概率云。
- 指标权重来自配置 ``scoring.weights.<dim>``；银行专项阈值来自
  ``scoring.bank_special``。

通用评分卡不绑定具体标的；标的的原始指标值由数据文件（data/*.yaml）
提供，缺失指标按 0（中性）处理并在报告中标注"数据不可得"。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SFWConfig
from .core import DIM_NAMES, ProbabilityCloud, softmax_cloud

DIM_CN = {
    "tian": "天·宏观叙事",
    "di": "地·基本面估值",
    "ren": "人·情绪资金",
    "zheng": "正·趋势动量",
    "fan": "反·反转信号",
    "he": "合·多因子整合",
    "gui": "归·历史周期",
    "shou": "守·风险边界",
    "jin": "进·进攻确信度",
}


@dataclass
class DimensionScore:
    dimension: str          # 九维键名
    raw_score: float        # 加权分 ∈ [-1, +1]
    cloud: ProbabilityCloud
    expectation: float
    entropy: float
    judgment: str
    temperature: float
    missing_indicators: list[str]   # 数据不可得的指标名
    indicator_map: dict[str, float] # 指标名 -> 映射值


def _clamp01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


# ── 指标值 → -1/0/+1 的三档映射器（通用）──

def _three_bucket(value: float, plus: float, zero: float, lower_better: bool = False) -> int:
    """通用三档映射。lower_better=True 时数值越低越偏多（如不良率）。"""
    if lower_better:
        if value < zero:
            return 1
        if value < plus:
            return 0
        return -1
    if value > plus:
        return 1
    if value > zero:
        return 0
    return -1


class ScoringCard:
    """九维量化评分卡（概率云态版）。"""

    def __init__(self, cfg: SFWConfig, market_volatility: float = 0.15):
        self.cfg = cfg
        self.temperature = self.cfg.engine("temperature_default", 0.15)
        if market_volatility > self.cfg.engine("high_volatility_threshold", 0.30):
            self.temperature = self.cfg.engine("temperature_high_vol", 0.25)
        self.results: dict[str, DimensionScore] = {}

    # ── 各维度：由指标映射字典加权合成 ──
    def score(self, indicators: dict[str, dict[str, int]]) -> dict[str, DimensionScore]:
        """indicators: {维度键: {指标键: -1/0/+1}}。"""
        for dim in DIM_NAMES:
            weights = self.cfg.scoring_weights(dim)
            ind_map = indicators.get(dim, {})
            weighted = 0.0
            w_sum = 0.0
            missing: list[str] = []
            per_indicator: dict[str, float] = {}
            for ind_name, w in weights.items():
                if ind_name in ind_map and ind_map[ind_name] is not None:
                    val = int(ind_map[ind_name])
                    per_indicator[ind_name] = float(val)
                    weighted += w * val
                    w_sum += w
                else:
                    missing.append(ind_name)  # 数据不可得 → 不参与
            raw = weighted / w_sum if w_sum > 0 else 0.0
            cloud = ProbabilityCloud(probs=softmax_cloud(raw, self.temperature))
            self.results[dim] = DimensionScore(
                dimension=dim,
                raw_score=raw,
                cloud=cloud,
                expectation=cloud.expectation,
                entropy=cloud.entropy,
                judgment=cloud.judgment(),
                temperature=self.temperature,
                missing_indicators=missing,
                indicator_map=per_indicator,
            )
        return self.results

    def get_state_vector(self) -> np.ndarray:
        return np.array([self.results[d].expectation for d in DIM_NAMES])

    def get_scores_dict(self) -> dict[str, float]:
        return {d: self.results[d].raw_score for d in DIM_NAMES}

    def get_conflict_count(self) -> int:
        vals = {d: self.results[d].expectation for d in DIM_NAMES}
        names = list(vals.keys())
        conflicts = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if vals[names[i]] * vals[names[j]] < -0.09:
                    conflicts += 1
        return conflicts

    def to_summary(self) -> str:
        lines = ["维度     原始分  p(-1)    p(0)     p(+1)    期望μ    熵H    判定"]
        lines.append("-" * 64)
        for d in DIM_NAMES:
            r = self.results[d]
            lines.append(
                f"{d:<6} {r.raw_score:>+6.2f} {r.cloud.probs[0]:>7.3f} {r.cloud.probs[1]:>7.3f} "
                f"{r.cloud.probs[2]:>7.3f} {r.expectation:>+7.3f} {r.entropy:>6.3f} {r.judgment}"
            )
        return "\n".join(lines)


# ── 银行专项评分（阈值来自配置）──

def bank_special_scoring(bank_data: dict[str, float], cfg: SFWConfig) -> dict[str, float]:
    """银行股专项评分：净息差/不良率/拨备/资本/股息。

    bank_data 字段: net_interest_margin, npl_ratio, provision_coverage,
                    capital_adequacy, dividend_yield
    返回: {指标中文名: -1/0/+1}
    """
    spec = cfg.raw.get("scoring", {}).get("bank_special", {})
    out: dict[str, float] = {}

    nim = spec.get("nim", {"plus": 1.75, "zero": 1.70})
    npl = spec.get("npl", {"plus": 1.10, "zero": 1.30})
    pcr = spec.get("provision_coverage", {"plus": 200.0, "zero": 150.0})
    car = spec.get("capital_adequacy", {"plus": 13.0, "zero": 11.0})
    dy = spec.get("dividend_yield", {"plus": 4.0, "zero": 2.5})

    out["净息差"] = float(_three_bucket(bank_data.get("net_interest_margin", 1.80), nim["plus"], nim["zero"]))
    out["资产质量"] = float(_three_bucket(bank_data.get("npl_ratio", 1.05), npl["plus"], npl["zero"], lower_better=True))
    out["拨备安全垫"] = float(_three_bucket(bank_data.get("provision_coverage", 219.58), pcr["plus"], pcr["zero"]))
    out["资本充足"] = float(_three_bucket(bank_data.get("capital_adequacy", 13.46), car["plus"], car["zero"]))
    out["股息价值"] = float(_three_bucket(bank_data.get("dividend_yield", 5.36), dy["plus"], dy["zero"]))
    return out
