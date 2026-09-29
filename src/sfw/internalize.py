"""SFW 大模型内化（Dual-System Internalization）

把认知固件对九维认知状态的判断，逐步内化成语义化的快速模式库（System 1），
使重复或邻近的认知状态能直接命中历史判断，无需重跑完整推理 / 再次调用 LLM
（System 2）。遵循 school -> internalize -> graduate 的成长曲线：随着模式覆盖
与置信度提升，对外部 LLM 的依赖逐步下降，最终"毕业"为可自决的快速直觉层。

实现语义对齐 btcu-harness 的 System1PatternLibrary：
  - 模式 = (state_index, 9 维三值向量, input_hash, action, success_rate x recency 置信)
  - S1 匹配级联：精确哈希 O(1) -> 9 维三值最近邻(k-NN) -> 输入文本模糊匹配
  - 反馈回灌：S2 结果经 Bayesian 更新强化 S1 成功率
默认内置（自包含、可发布）；若安装 btcu-harness，可用 provider="btcu" 委托其模式库。

本模块只记录"固件判断"的确定性签名，不含投资建议、不预测涨跌、不承诺收益。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

# ─────────────────────────────────────────────────────────────
# 基础常量
# ─────────────────────────────────────────────────────────────

SPACE_SIZE = 3**9  # 19683
DIM_COUNT = 9


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_input(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


# 三值化：把 [-1,1] 的九维期望映射到 {-1,0,+1}，与 report._triline 口径一致
def to_trits(state: Any, threshold: float = 0.2) -> list[int]:
    vals = list(state) if not hasattr(state, "tolist") else list(state.tolist())
    out: list[int] = []
    for v in vals:
        f = float(v)
        if f > threshold:
            out.append(1)
        elif f < -threshold:
            out.append(-1)
        else:
            out.append(0)
    return out[:DIM_COUNT]


def state_to_index(state: Any) -> int:
    """九维三值向量 -> 19683 空间十进制索引（与 core.state_to_index 同构）。"""
    trits = to_trits(state)
    idx = 0
    for i, t in enumerate(trits):
        idx += (t + 1) % 3 * (3 ** (DIM_COUNT - 1 - i))
    return idx


def _hamming(a: list[int], b: list[int]) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))


def _jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ─────────────────────────────────────────────────────────────
# 模式记录
# ─────────────────────────────────────────────────────────────

@dataclass
class PatternRecord:
    """一条已内化的认知判断模式。"""

    state_index: int
    state_values: list[int]
    input_hash: str
    action: str
    success_count: int = 0
    use_count: int = 0
    created_at: str = field(default_factory=_now_iso)
    last_used: str = field(default_factory=_now_iso)

    @property
    def success_rate(self) -> float:
        # Laplace 平滑，避免单次试用即满置信
        return (self.success_count + 1) / (self.use_count + 2) if self.use_count else 0.5

    @property
    def recency_factor(self) -> float:
        try:
            last = datetime.fromisoformat(self.last_used)
        except ValueError:
            last = _now()
        age_days = max(0.0, (_now() - last).total_seconds() / 86400.0)
        return math.exp(-age_days / 7.0)  # 半衰期约 7 天

    @property
    def confidence(self) -> float:
        return self.success_rate * self.recency_factor

    def record_use(self, success: bool) -> None:
        self.use_count += 1
        if success:
            self.success_count += 1
        self.last_used = _now_iso()


# ─────────────────────────────────────────────────────────────
# 模式库（System 1）
# ─────────────────────────────────────────────────────────────

class PatternLibrary:
    """System 1 快速模式库：内置实现（自包含、可发布）。"""

    def __init__(self, storage_path: str = "output/sfw_patterns.json") -> None:
        self.storage_path = storage_path
        self.records: list[PatternRecord] = []

    # ---- 写入 ----
    def add(self, rec: PatternRecord) -> None:
        # 同 (state, input_hash) 去重合并，保留最近 action 与累计统计
        for i, r in enumerate(self.records):
            if r.state_index == rec.state_index and r.input_hash == rec.input_hash:
                r.action = rec.action
                r.last_used = _now_iso()
                self.records[i] = r
                return
        self.records.append(rec)

    def learn(self, state_index: int, state_values: list[int],
              input_text: str, action: str) -> PatternRecord:
        rec = PatternRecord(
            state_index=state_index,
            state_values=state_values,
            input_hash=_hash_input(input_text),
            action=action,
        )
        self.add(rec)
        return rec

    # ---- 匹配级联：精确 -> kNN -> 模糊 ----
    def best_match(self, state_values: list[int], input_text: str,
                   top_k: int = 3) -> PatternRecord | None:
        exact: list[PatternRecord] = []
        near: list[tuple[int, PatternRecord]] = []
        for r in self.records:
            d = _hamming(r.state_values, state_values)
            if d == 0:
                exact.append(r)
            elif d <= 2:  # 邻近状态（最多 2 个三值位差异）
                near.append((d, r))

        # 1) 精确状态 + 文本相似度排序
        if exact:
            best = max(exact, key=lambda r: r.confidence * (0.5 + 0.5 * _jaccard(r.action, input_text)))
            return best

        # 2) kNN：邻近状态里取置信最高
        if near:
            near.sort(key=lambda t: (t[0], -t[1].confidence))
            return near[0][1]

        # 3) 模糊：全局按文本相似度取置信加权最高
        fuzzy = [(r, _jaccard(input_text, r.action)) for r in self.records]
        fuzzy = [(r, j) for r, j in fuzzy if j >= 0.35]
        if fuzzy:
            best_fuzzy = max(fuzzy, key=lambda t: t[1] * t[0].confidence)
            return best_fuzzy[0]

        return None

    def feedback(self, input_hash: str, success: bool) -> PatternRecord | None:
        for r in self.records:
            if r.input_hash == input_hash:
                r.record_use(success)
                return r
        return None

    # ---- 统计 / 毕业 ----
    def coverage_stats(self) -> dict[str, Any]:
        distinct_states = len({r.state_index for r in self.records})
        total_uses = sum(r.use_count for r in self.records)
        avg_conf = (sum(r.confidence for r in self.records) / len(self.records)) if self.records else 0.0
        return {
            "patterns": len(self.records),
            "distinct_states": distinct_states,
            "state_coverage": distinct_states / SPACE_SIZE,
            "total_uses": total_uses,
            "avg_confidence": avg_conf,
            "last_learned": max((r.created_at for r in self.records), default=""),
        }

    # ---- 持久化 ----
    def save(self) -> str:
        dirpath = os.path.dirname(self.storage_path) or "."
        os.makedirs(dirpath, exist_ok=True)
        payload = {"version": "sfw-internalize-1", "records": [asdict(r) for r in self.records]}
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return self.storage_path

    def load(self) -> bool:
        if not os.path.exists(self.storage_path):
            return False
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.records = [PatternRecord(**r) for r in payload.get("records", [])]
            return True
        except (json.JSONDecodeError, OSError, TypeError):
            self.records = []
            return False

    @property
    def exists(self) -> bool:
        return os.path.exists(self.storage_path)


# ─────────────────────────────────────────────────────────────
# 双系统内化器
# ─────────────────────────────────────────────────────────────

@dataclass
class InternalizeResult:
    action: str
    source: str            # "system1" | "system2"
    confidence: float
    learned: bool = False
    matched_hash: str = ""


class DualSystemInternalizer:
    """S1(模式库) + S2(完整推理/LLM) 的内化协调器。

    system2_fn: Callable[[list[int], str], str] —— 输入三值九维向量与原始输入，
                返回判断签名（S2 的"推演"）。SFW 里即完整分析引擎的紧凑签名。
    """

    def __init__(self, library_path: str = "output/sfw_patterns.json",
                 min_confidence: float = 0.6,
                 system2_fn: Callable[[list[int], str], str] | None = None,
                 provider: str = "builtin") -> None:
        self.library = PatternLibrary(library_path)
        self.min_confidence = min_confidence
        self.system2_fn = system2_fn
        self.provider = provider

    def resolve(self, state9: Any, input_text: str) -> InternalizeResult:
        """先试 S1；置信不足则走 S2 并内化。"""
        state_values = to_trits(state9)
        match = self.library.best_match(state_values, input_text)
        if match and match.confidence >= self.min_confidence:
            return InternalizeResult(action=match.action, source="system1",
                                     confidence=match.confidence, matched_hash=match.input_hash)

        # S2：完整推演（未提供回调则回退到"新状态待推演"）
        if self.system2_fn is not None:
            action = self.system2_fn(state_values, input_text)
        else:
            action = f"idx={state_to_index(state_values)}|待推演"
        rec = self.library.learn(state_to_index(state_values), state_values, input_text, action)
        return InternalizeResult(action=action, source="system2",
                                 confidence=rec.confidence, learned=True, matched_hash=rec.input_hash)

    def feedback(self, input_hash: str, success: bool) -> PatternRecord | None:
        return self.library.feedback(input_hash, success)

    def graduation(self) -> dict[str, Any]:
        """school -> internalize -> graduate：按模式数/使用量/置信/去重状态判定。

        19683 空间极大，覆盖率几近 0，故以"学习成熟度"而非覆盖率作门槛。
        """
        stats = self.library.coverage_stats()
        patterns = stats["patterns"]
        avg = stats["avg_confidence"]
        uses = stats["total_uses"]
        states = stats["distinct_states"]
        if patterns >= 5 and avg >= 0.7 and uses >= 25 and states >= 5:
            stage = "graduate"
        elif patterns >= 1 and uses >= 5:
            stage = "internalize"
        else:
            stage = "school"
        return {**stats, "stage": stage,
                "llm_dependency": round(max(0.0, 1.0 - min(1.0, patterns / 100) - avg), 3)}

    def save(self) -> str:
        return self.library.save()

    def load(self) -> bool:
        return self.library.load()


# ─────────────────────────────────────────────────────────────
# 可选：委托 btcu-harness 的 System1 模式库（provider="btcu"）
# ─────────────────────────────────────────────────────────────

def _btcu_library(storage_path: str) -> Any:
    try:
        from btcu_harness.cognition.system1 import System1PatternLibrary  # type: ignore
        return System1PatternLibrary(persistence_path=storage_path)
    except Exception:  # noqa: BLE001
        return None


# ─────────────────────────────────────────────────────────────
# S2 判断签名：把 SFW 分析结果压成确定性字符串
# ─────────────────────────────────────────────────────────────

def analysis_action_signature(state9, engine_report: dict, weighted_tp: float) -> str:
    """从 SFW 分析结果生成紧凑判断签名（System 2 的可内化输出）。"""
    idx = state_to_index(state9)
    fractal = engine_report.get("fractal", {}) or {}
    phase = fractal.get("phase_name") or fractal.get("main_phase") or "N/A"
    topology = engine_report.get("topology", {}) or {}
    verdict = topology.get("judgment") or "N/A"
    tension = float(engine_report.get("tension", 0.0))
    tp = f"{weighted_tp:.2f}" if weighted_tp and weighted_tp > 0 else "N/A"
    return f"idx={idx}|phase={phase}|tension={tension:.2f}|verdict={verdict}|wtp={tp}"
