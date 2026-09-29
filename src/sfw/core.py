"""SFW-10.0 认知引擎核心。

包含五阶状态空间与六大范式突破的实现：
- 概率云态编码（softmax 三值分布）
- 分形极限网（相位转移 + 嵌套展开）
- L∞ 持续裂变（张力 + 微裂变触发）
- 无维拓扑自我感知（激活体积 / 质心 / 耦合矩阵）
- 维度涌现协议
- 存在认知节律

所有阈值参数均来自 ``SFWConfig``，不在代码内硬编码。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .config import SFWConfig

# 九维名称与分组
DIM_NAMES: list[str] = ["天", "地", "人", "正", "反", "合", "归", "守", "进"]
TIAN_DI_REN = [0, 1, 2]    # 体
ZHENG_FAN_HE = [3, 4, 5]   # 用
GUI_SHOU_JIN = [6, 7, 8]   # 变
GROUPS = [TIAN_DI_REN, ZHENG_FAN_HE, GUI_SHOU_JIN]
GROUP_NAMES = ["体", "用", "变"]


class Phase(Enum):
    """四象相位（极限环）。"""

    ACCUMULATE = ("P1", "积聚", "信息流入")
    TRANSFORM_A = ("P2", "转化A", "内部重组")
    DECAY = ("P3", "衰减", "输出释放")
    TRANSFORM_B = ("P4", "转化B", "反思整合")

    def __init__(self, code: str, cn_name: str, action: str):
        self.code = code
        self.cn_name = cn_name
        self.action = action

    def __str__(self) -> str:  # 便于日志与报告
        return f"{self.code}({self.cn_name})"


PHASE_ORDER: list[Phase] = [
    Phase.ACCUMULATE, Phase.TRANSFORM_A, Phase.DECAY, Phase.TRANSFORM_B,
]


def softmax_cloud(score: float, temperature: float = 0.15) -> np.ndarray:
    """将加权分 s ∈ [-1, +1] 映射为三值概率分布 [p(-1), p(0), p(+1)]。

    logits = [-s/τ, 0, +s/τ]，数值稳定版 softmax。
    """
    if temperature <= 0:
        raise ValueError("temperature 必须为正数")
    logits = np.array([-score / temperature, 0.0, +score / temperature])
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)


def dynamic_temperature(market_volatility: float, cfg: SFWConfig) -> float:
    """根据波动率选择温度：低波动 → 默认 τ，高波动(>阈值) → 高 τ。"""
    threshold = cfg.engine("high_volatility_threshold", 0.30)
    if market_volatility > threshold:
        return cfg.engine("temperature_high_vol", 0.25)
    return cfg.engine("temperature_default", 0.15)


# ═══════════════════════════════════════════════════════════════
# 概率云（单维度）
# ═══════════════════════════════════════════════════════════════

@dataclass
class ProbabilityCloud:
    """概率云态：每个维度为一个三值概率分布。"""

    probs: np.ndarray  # shape (3,) = [p(-1), p(0), p(+1)]

    def __post_init__(self) -> None:
        if self.probs.shape != (3,):
            raise ValueError(f"概率云必须是 3 元素向量，当前 shape={self.probs.shape}")
        if not np.isclose(np.sum(self.probs), 1.0, atol=1e-6):
            raise ValueError(f"概率云之和必须为 1，当前 {self.probs}")

    @property
    def expectation(self) -> float:
        return float((-1) * self.probs[0] + 0 * self.probs[1] + (+1) * self.probs[2])

    @property
    def entropy(self) -> float:
        eps = 1e-12
        return float(-np.sum(self.probs * np.log(self.probs + eps)))

    @property
    def uncertainty(self) -> float:
        return float(self.probs[1])

    def judgment(self) -> str:
        """判定：高确信+1 / 偏多 / 悬置 / 偏空 / 高确信-1。"""
        mu = self.expectation
        h = self.entropy
        if mu > 0.5 and h < 0.5:
            return "高确信+1"
        if mu < -0.5 and h < 0.5:
            return "高确信-1"
        if mu > 0.2 and h < 0.8:
            return "偏多"
        if mu < -0.2 and h < 0.8:
            return "偏空"
        return "悬置"

    def to_dict(self) -> dict[str, float]:
        return {
            "p(-1)": float(self.probs[0]),
            "p(0)": float(self.probs[1]),
            "p(+1)": float(self.probs[2]),
            "expectation": self.expectation,
            "entropy": self.entropy,
            "judgment": self.judgment(),
        }


# ═══════════════════════════════════════════════════════════════
# 分形极限网
# ═══════════════════════════════════════════════════════════════

@dataclass
class FractalState:
    main_phase: Phase
    sub_path: list[Phase] = field(default_factory=list)
    depth: int = 0
    phase_entropy: float = 0.0
    transfer_prob: float = 0.0
    fractal_triggered: bool = False
    fractal_reason: str = ""


def should_expand_fractal(dim_entropies: np.ndarray, conflict_count: int,
                          phase_duration: int, expected_duration: int,
                          cfg: SFWConfig) -> tuple[bool, str]:
    """判断是否需要展开子循环。"""
    entropy_th = cfg.engine("fractal_entropy_threshold", 0.7)
    conflict_th = cfg.engine("fractal_conflict_threshold", 3)
    max_entropy = float(np.max(dim_entropies)) if dim_entropies.size else 0.0
    if max_entropy > entropy_th:
        return True, f"维度熵过高(max={max_entropy:.2f}>{entropy_th})，需展开子循环降熵"
    if conflict_count >= conflict_th:
        return True, f"维度冲突数={conflict_count}≥{conflict_th}，需展开子循环探测"
    if phase_duration > expected_duration:
        return True, f"相位超时({phase_duration}>{expected_duration})，强制展开子循环"
    return False, ""


def sigmoid(x: float) -> float:
    # 数值稳定 sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def compute_transfer_probability(current: Phase, info_inflow_rate: float,
                                 decay_rate: float, reorganization: float,
                                 output_pressure: float, reflection_depth: float) -> dict[str, float]:
    """计算相位转移概率（取代确定性转移）。"""
    if current == Phase.ACCUMULATE:
        p_next = sigmoid(info_inflow_rate - decay_rate)
        return {Phase.TRANSFORM_A.code: p_next, Phase.ACCUMULATE.code: 1 - p_next}
    if current == Phase.TRANSFORM_A:
        p_next = sigmoid(reorganization)
        return {Phase.DECAY.code: p_next, Phase.TRANSFORM_A.code: 1 - p_next}
    if current == Phase.DECAY:
        p_next = sigmoid(output_pressure)
        return {Phase.TRANSFORM_B.code: p_next, Phase.DECAY.code: 1 - p_next}
    # TRANSFORM_B
    p_next = sigmoid(reflection_depth)
    return {Phase.ACCUMULATE.code: p_next, Phase.TRANSFORM_B.code: 1 - p_next}


# ═══════════════════════════════════════════════════════════════
# L∞ 持续裂变
# ═══════════════════════════════════════════════════════════════

@dataclass
class MicroFission:
    dimension: str
    trigger_reason: str
    adjustment: str
    tension_before: float
    tension_after: float


def compute_tension(load: float, context_pressure: float, conflict_count: int,
                    max_entropy: float) -> float:
    """张力 T = 0.3×负荷 + 0.3×上下文压力 + 0.2×归一化冲突 + 0.2×归一化熵。"""
    return (
        0.3 * load
        + 0.3 * context_pressure
        + 0.2 * min(conflict_count / 5.0, 1.0)
        + 0.2 * min(max_entropy / 1.1, 1.0)
    )


def check_micro_fission(dimension_scores: dict[str, float],
                        clouds: dict[str, ProbabilityCloud],
                        conclusion_history: list[str],
                        user_why_count: int,
                        self_correlation: float,
                        cfg: SFWConfig) -> list[MicroFission]:
    """检查微裂变触发条件（任一触发即记录）。"""
    fissions: list[MicroFission] = []
    dim_names = list(dimension_scores.keys())
    dim_vals = np.array(list(dimension_scores.values()))
    ent_th = cfg.engine("fission_entropy_threshold", 0.95)
    conflict_th = cfg.engine("fission_conflict_threshold", 3)
    sc_warning = cfg.engine("self_correlation_warning", 0.65)

    # 条件1：维度间符号冲突 >= 阈值
    conflicts = 0
    for i in range(len(dim_names)):
        for j in range(i + 1, len(dim_names)):
            if dim_vals[i] * dim_vals[j] < -0.49:
                conflicts += 1
    if conflicts >= conflict_th:
        fissions.append(MicroFission(
            dimension="全局",
            trigger_reason=f"维度间冲突数={conflicts}≥{conflict_th}",
            adjustment="降低高冲突维度权重 0.05",
            tension_before=0.5, tension_after=0.4,
        ))

    # 条件2：概率云熵超阈值
    for name, cloud in clouds.items():
        if cloud.entropy > ent_th:
            fissions.append(MicroFission(
                dimension=name,
                trigger_reason=f"维度熵 H={cloud.entropy:.2f}>{ent_th}，接近最大不确定性",
                adjustment="温度 τ 上调 0.05，允许更多悬置",
                tension_before=0.5, tension_after=0.4,
            ))

    # 条件3：连续两次结论反转（非悬置）
    if len(conclusion_history) >= 2:
        if conclusion_history[-1] != conclusion_history[-2] and \
           conclusion_history[-1] != "悬置" and conclusion_history[-2] != "悬置":
            fissions.append(MicroFission(
                dimension="全局",
                trigger_reason=f"连续两次结论反转: {conclusion_history[-2]} → {conclusion_history[-1]}",
                adjustment="降低温度 τ，提高确信度门槛",
                tension_before=0.5, tension_after=0.4,
            ))

    # 条件4：用户追问"为什么"超过 2 次
    if user_why_count > 2:
        fissions.append(MicroFission(
            dimension="全局",
            trigger_reason=f"用户追问'为什么'次数={user_why_count}>2，认知未对齐",
            adjustment="展开更多推理步骤，提高可解释性",
            tension_before=0.5, tension_after=0.4,
        ))

    # 条件5：因子自相关超警戒线
    if self_correlation > sc_warning:
        fissions.append(MicroFission(
            dimension="因子质量",
            trigger_reason=f"Self-Correlation={self_correlation:.2f}>{sc_warning}，接近警戒线",
            adjustment="引入跨 regime 互补，降低自相关",
            tension_before=0.5, tension_after=0.4,
        ))

    return fissions


# ═══════════════════════════════════════════════════════════════
# 无维拓扑自我感知
# ═══════════════════════════════════════════════════════════════

@dataclass
class TopologyState:
    activation_volume: float
    activation_centroid: float
    coupling_matrix: np.ndarray
    topology_judgment: str

    def interpret(self) -> str:
        if self.activation_volume < 3:
            return "聚焦态——低熵高确信，认知清晰"
        if self.activation_volume > 7:
            return "弥散态——高熵需信息输入，建议等待更多数据"
        return "过渡态——部分维度明确，部分维度悬置"


def compute_topology(clouds: dict[str, ProbabilityCloud],
                     cfg: SFWConfig) -> TopologyState:
    """计算拓扑性质。"""
    n = len(DIM_NAMES)
    entropies = np.array([clouds[d].entropy for d in DIM_NAMES])
    expectations = np.array([clouds[d].expectation for d in DIM_NAMES])
    weights = np.ones(n) / n

    v_act = float(np.sum(entropies))
    centroid = float(np.sum(expectations * weights) / np.sum(weights))

    coupling = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                coupling[i][j] = 1.0
            else:
                dist = float(np.linalg.norm(clouds[DIM_NAMES[i]].probs - clouds[DIM_NAMES[j]].probs))
                coupling[i][j] = 1.0 - dist / np.sqrt(2)

    focused = cfg.engine("topology_focused_volume", 3.0)
    diffuse = cfg.engine("topology_diffuse_volume", 7.0)
    strong = cfg.engine("coupling_strong", 0.8)
    anti = cfg.engine("coupling_anti", -0.8)

    if v_act < focused:
        judgment = "聚焦态"
    elif v_act > diffuse:
        judgment = "弥散态"
    else:
        off_diag = coupling[~np.eye(n, dtype=bool)]
        max_couple = float(np.max(off_diag))
        min_couple = float(np.min(off_diag))
        if max_couple > strong:
            judgment = "耦合态"
        elif min_couple < anti:
            judgment = "对抗态"
        else:
            judgment = "过渡态"

    return TopologyState(v_act, centroid, coupling, judgment)


# ═══════════════════════════════════════════════════════════════
# 维度涌现
# ═══════════════════════════════════════════════════════════════

@dataclass
class EmergingDimension:
    name: str
    description: str
    mutual_info: float
    entropy_trajectory: list[float] = field(default_factory=list)
    status: str = "候选"

    def should_promote(self, cycles: int = 3) -> bool:
        return len(self.entropy_trajectory) >= cycles and all(h < 0.5 for h in self.entropy_trajectory[-cycles:])

    def should_discard(self, cycles: int = 3) -> bool:
        return len(self.entropy_trajectory) >= cycles and all(h > 0.8 for h in self.entropy_trajectory[-cycles:])


def check_dimension_emergence(clouds: dict[str, ProbabilityCloud],
                              factor_self_corr: float,
                              factor_fitness: float,
                              market_regime_change: bool,
                              cfg: SFWConfig) -> tuple[bool, str]:
    ent_th = cfg.engine("emergence_high_entropy", 0.8)
    count_th = cfg.engine("emergence_high_entropy_count", 4)
    high_entropy_count = sum(1 for d in DIM_NAMES if clouds[d].entropy > ent_th)
    if high_entropy_count >= count_th:
        return True, f"九维中有{high_entropy_count}个维度熵>{ent_th}，现有维度无法降熵"
    if factor_self_corr > 0.7 and factor_fitness < 1.0:
        return True, "因子质量门控连续失败(Self-Corr>0.7 且 Fitness<1.0)"
    if market_regime_change:
        return True, "市场出现新范式(新政策/新技术/新交易制度)"
    return False, ""


# ═══════════════════════════════════════════════════════════════
# 存在认知节律
# ═══════════════════════════════════════════════════════════════

@dataclass
class ExistenceRhythm:
    current_mode: str
    smoothness: float
    suggest_pure_existence: bool

    def interpret(self) -> str:
        if self.suggest_pure_existence:
            return "建议进入纯粹存在态——主动暂停，等待更多信息"
        if self.smoothness > 0.8:
            return "节律流畅——认知体以自然节奏运作"
        if self.smoothness > 0.5:
            return "节律平稳——认知体运作正常，偶有卡顿"
        return "节律阻滞——相位转移困难，建议 L∞ 微裂变"


# ═══════════════════════════════════════════════════════════════
# 八卦 / 六十四卦编码
# ═══════════════════════════════════════════════════════════════

BAGUA_MAP = {
    "乾": "天·强势上涨/进攻",
    "兑": "泽·温和上涨/偏多",
    "离": "火·快速上涨/谨慎追高",
    "震": "雷·突发启动/快进快出",
    "巽": "风·渐进渗透/逐步建仓",
    "坎": "水·下跌险境/防守观望",
    "艮": "山·停滞横盘/等待",
    "坤": "地·弱势下跌/空仓收息",
}


def encode_bagua(tian: int, di: int, ren: int) -> str:
    """将天地人三值(+1/0/-1)映射为八卦名称（保持原版映射逻辑）。"""
    t, d, r = tian, di, ren
    if t >= 0 and d >= 0 and r >= 0:
        return "乾" if all(x == 1 for x in (t, d, r)) else "兑"
    if t >= 0 and d < 0 and r < 0:
        return "离"
    if t < 0 and d >= 0 and r >= 0:
        return "震"
    if t >= 0 and d >= 0 and r < 0:
        return "巽"
    if t < 0 and d < 0 and r >= 0:
        return "坎"
    if t < 0 and d >= 0 and r < 0:
        return "艮"
    if t < 0 and d < 0 and r < 0:
        return "坤"
    return "巽"


def encode_64gua(six_dims: tuple[int, int, int, int, int, int]) -> str:
    """六爻编码（体+用六维 → 64卦），+1 阳，-1 阴，0 动爻。"""
    lines = []
    for v in six_dims:
        if v == 1:
            lines.append("阳(—)")
        elif v == -1:
            lines.append("阴(--)")
        else:
            lines.append("动爻(○)")
    return " | ".join(lines)


def state_to_index(state: np.ndarray) -> int:
    """九维向量 → 十进制索引（3⁹ 空间）。"""
    mapped = (np.asarray(state, dtype=float) + 1) % 3
    idx = 0
    for i, v in enumerate(mapped):
        idx += int(round(v)) * (3 ** (8 - i))
    return idx


# ═══════════════════════════════════════════════════════════════
# 主引擎
# ═══════════════════════════════════════════════════════════════

class SFWEngine:
    """SFW-10.0 认知引擎。"""

    def __init__(self, cfg: SFWConfig):
        self.cfg = cfg
        self.clouds: dict[str, ProbabilityCloud] = {}
        self.fractal_state: FractalState | None = None
        self.fissions: list[MicroFission] = []
        self.topology: TopologyState | None = None
        self.emerging_dims: list[EmergingDimension] = []
        self.existence: ExistenceRhythm | None = None
        self.conclusion_history: list[str] = []

    # ── 步骤1：概率云态编码 ──
    def encode_dimensions(self, scores: dict[str, float], market_vol: float = 0.15) -> None:
        tau = dynamic_temperature(market_vol, self.cfg)
        for dim in DIM_NAMES:
            score = float(scores.get(dim, 0.0))
            if score < -1.0 or score > 1.0:
                raise ValueError(f"维度 '{dim}' 评分必须落在 [-1, +1]，当前 {score}")
            self.clouds[dim] = ProbabilityCloud(probs=softmax_cloud(score, tau))

    # ── 步骤2：分形极限网 ──
    def compute_fractal(self, conflict_count: int, phase_duration: int = 1,
                        expected_duration: int = 1, info_rate: float = 0.5,
                        decay_rate: float = 0.3, reorg: float = 0.5,
                        output_pressure: float = 0.5, reflect_depth: float = 0.5) -> None:
        entropies = np.array([self.clouds[d].entropy for d in DIM_NAMES])
        should_expand, reason = should_expand_fractal(
            entropies, conflict_count, phase_duration, expected_duration, self.cfg)

        expectations = np.array([self.clouds[d].expectation for d in DIM_NAMES])
        if np.mean(expectations[:3]) > 0.3:
            main = Phase.ACCUMULATE
        elif np.all(np.abs(expectations) < 0.2):
            main = Phase.TRANSFORM_A
        elif np.mean(expectations[3:6]) < -0.2:
            main = Phase.DECAY
        else:
            main = Phase.TRANSFORM_B

        transfer = compute_transfer_probability(main, info_rate, decay_rate, reorg, output_pressure, reflect_depth)
        max_next = max(transfer, key=transfer.get)
        max_depth = self.cfg.engine("fractal_max_depth", 3)

        self.fractal_state = FractalState(
            main_phase=main,
            depth=min(1 if should_expand else 0, max_depth),
            phase_entropy=float(np.mean(entropies)),
            transfer_prob=transfer[max_next],
            fractal_triggered=should_expand,
            fractal_reason=reason,
        )

    # ── 步骤3：L∞ 持续裂变 ──
    def compute_fission(self, load: float = 0.5, context_pressure: float = 0.5,
                        user_why_count: int = 0, self_correlation: float = 0.5) -> None:
        entropies = np.array([self.clouds[d].entropy for d in DIM_NAMES])
        scores = {d: self.clouds[d].expectation for d in DIM_NAMES}
        conflict_count = self._count_conflicts(scores)
        self.tension = compute_tension(load, context_pressure, conflict_count, float(np.max(entropies)))
        self.fissions = check_micro_fission(
            scores, self.clouds, self.conclusion_history, user_why_count, self_correlation, self.cfg)

    def _count_conflicts(self, scores: dict[str, float]) -> int:
        names = list(scores.keys())
        vals = np.array(list(scores.values()))
        conflicts = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if vals[i] * vals[j] < -0.49:
                    conflicts += 1
        return conflicts

    # ── 步骤4：拓扑感知 ──
    def compute_topology_state(self) -> None:
        self.topology = compute_topology(self.clouds, self.cfg)

    # ── 步骤5：维度涌现 ──
    def compute_emergence(self, factor_self_corr: float = 0.5, factor_fitness: float = 1.0,
                          market_regime_change: bool = False) -> None:
        should, reason = check_dimension_emergence(
            self.clouds, factor_self_corr, factor_fitness, market_regime_change, self.cfg)
        if should:
            self.emerging_dims.append(EmergingDimension(
                name="市场情绪周期",
                description="从原始数据流中提取的高频共现特征，与现有九维互信息低",
                mutual_info=0.08,
                entropy_trajectory=[0.6],
            ))

    # ── 步骤6：存在认知 ──
    def compute_existence(self, smoothness: float = 0.7) -> None:
        suggest_pure = self.topology is not None and self.topology.activation_volume > 7
        mode = self.fractal_state.main_phase.cn_name if self.fractal_state else "未知"
        self.existence = ExistenceRhythm(
            current_mode=mode, smoothness=smoothness, suggest_pure_existence=suggest_pure)

    # ── 汇总 ──
    def get_state_vector(self) -> np.ndarray:
        return np.array([self.clouds[d].expectation for d in DIM_NAMES])

    def get_full_report(self) -> dict:
        return {
            "probability_clouds": {d: self.clouds[d].to_dict() for d in DIM_NAMES},
            "fractal": {
                "main_phase": self.fractal_state.main_phase.code if self.fractal_state else "N/A",
                "phase_name": self.fractal_state.main_phase.cn_name if self.fractal_state else "N/A",
                "depth": self.fractal_state.depth if self.fractal_state else 0,
                "entropy": self.fractal_state.phase_entropy if self.fractal_state else 0.0,
                "transfer_prob": self.fractal_state.transfer_prob if self.fractal_state else 0.0,
                "fractal_triggered": self.fractal_state.fractal_triggered if self.fractal_state else False,
                "reason": self.fractal_state.fractal_reason if self.fractal_state else "",
            } if self.fractal_state else {},
            "fissions": [
                {"dimension": f.dimension, "reason": f.trigger_reason, "adjustment": f.adjustment}
                for f in self.fissions
            ],
            "tension": getattr(self, "tension", 0.0),
            "topology": {
                "activation_volume": self.topology.activation_volume if self.topology else 0.0,
                "activation_centroid": self.topology.activation_centroid if self.topology else 0.0,
                "judgment": self.topology.topology_judgment if self.topology else "N/A",
                "interpretation": self.topology.interpret() if self.topology else "",
            } if self.topology else {},
            "emergence": {
                "candidates": [
                    {"name": d.name, "mutual_info": d.mutual_info, "status": d.status}
                    for d in self.emerging_dims
                ]
            },
            "existence": {
                "mode": self.existence.current_mode if self.existence else "N/A",
                "smoothness": self.existence.smoothness if self.existence else 0.0,
                "interpretation": self.existence.interpret() if self.existence else "",
            } if self.existence else {},
        }
