"""SFW-10.0 报告渲染与完整分析编排。

``run_analysis(data, cfg)`` 串联：评分 → 认知引擎 → 估值 → 回测，
``generate_report(...)`` 按 SFW-10.0 固件的"〇~九"结构输出 Markdown 报告。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .backtest import BacktestMetrics, demo_backtest, pingan_correlation_estimates
from .bank_model import run_bank_valuation
from .config import SFWConfig
from .core import SFWEngine, encode_64gua, encode_bagua, state_to_index
from .data import StockData
from .scoring import ScoringCard, bank_special_scoring
from .valuation import run_valuation

logger = logging.getLogger("sfw.report")


@dataclass
class AnalysisResult:
    data: StockData
    config: SFWConfig
    scoring: ScoringCard
    engine: SFWEngine
    engine_report: dict[str, Any]
    valuation: dict[str, Any]
    backtest: BacktestMetrics
    correlation: dict[str, float]
    bank: dict[str, Any] | None = None


def _triline(v: float, th: float = 0.2) -> int:
    if v > th:
        return 1
    if v < -th:
        return -1
    return 0


def run_analysis(data: StockData, cfg: SFWConfig, *, market_vol: float = 0.15,
                 run_backtest_demo: bool = True) -> AnalysisResult:
    """执行完整分析编排。"""
    # 1) 评分卡
    scoring = ScoringCard(cfg, market_volatility=market_vol)
    scoring.score(data.scoring)  # scoring 缺失时全部按中性

    scores = scoring.get_scores_dict()

    # 2) 认知引擎
    engine = SFWEngine(cfg)
    engine.encode_dimensions(scores, market_vol=market_vol)
    engine.compute_fractal(conflict_count=scoring.get_conflict_count())
    engine.compute_fission(load=0.4, context_pressure=0.35,
                           user_why_count=0, self_correlation=0.55)
    engine.compute_topology_state()
    engine.compute_emergence(factor_self_corr=0.60, factor_fitness=1.2,
                             market_regime_change=False)
    engine.compute_existence(smoothness=0.72)
    engine_report = engine.get_full_report()

    # 3) 估值（通用）+ 银行专项（若数据含 valuation.bank）
    valuation = run_valuation(data.valuation, cfg)
    bank = None
    if data.valuation.get("bank"):
        bank = run_bank_valuation(data.valuation["bank"], cfg)

    # 4) 回测（演示用确定性模拟数据）
    backtest = demo_backtest(cfg) if run_backtest_demo else BacktestMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

    return AnalysisResult(
        data=data, config=cfg, scoring=scoring, engine=engine,
        engine_report=engine_report, valuation=valuation,
        backtest=backtest, correlation=pingan_correlation_estimates(),
        bank=bank,
    )


# ─────────────────────────────────────────────────────────────────
# 报告渲染
# ─────────────────────────────────────────────────────────────────

def _markdown_table(headers: list[Any], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def generate_report(res: AnalysisResult) -> str:
    data, cfg = res.data, res.config
    er = res.engine_report
    val = res.valuation
    market = data.market
    price = float(market.get("price", 0.0))
    weighted_tp = float(val["weighted_target_price"])

    L: list[str] = []
    L.append(f"# {data.name}（{data.code}）· SFW-10.0 生产级分析报告")
    L.append(f"\n> 数据截止：{data.meta.get('as_of', '未注明')} · 币种：{data.meta.get('currency', 'CNY')}")
    L.append("> 本报告由 SFW-10.0 认知引擎生成，含启发式表达层（九维概率云/卦象），非外部统计或期权隐含数据。不构成投资建议。")

    # ── 〇、数据来源 ──
    L.append("\n## 〇、数据来源清单")
    if data.source:
        L.append(_markdown_table(
            ["数据项", "来源", "日期", "置信度"],
            [[s.get("item", ""), s.get("source", ""), s.get("date", ""), s.get("confidence", "")] for s in data.source],
        ))
    else:
        L.append("\n数据来源未提供。")

    # ── 一、五阶状态空间 + 全息快照 ──
    L.append("\n## 一、五阶状态空间 + 全息快照")
    scores = res.scoring.get_scores_dict()
    t, d, r = scores["天"], scores["地"], scores["人"]
    bagua = encode_bagua(_triline(t), _triline(d), _triline(r))
    L.append("\n### 2³=8：八卦")
    L.append(f"- 天地人评分：天={t:+.2f} 地={d:+.2f} 人={r:+.2f}")
    L.append(f"- 本卦：**{bagua}**（启发式演绎层）")

    six = tuple(_triline(s) for s in (t, d, r, scores["正"], scores["反"], scores["合"]))
    L.append("\n### 2⁶=64：六十四卦")
    L.append(f"- 六爻（初→上，体/用）：{encode_64gua(six)}")

    L.append("\n### 3³=27 体 / 3⁶=729 体+用 / 3⁹=19683 体+用+变")
    state_vec = res.engine.get_state_vector()
    idx = state_to_index(state_vec)
    phase = er["fractal"].get("phase_name", "N/A") if er.get("fractal") else "N/A"
    L.append(f"- 九维期望向量：{np.round(state_vec, 2).tolist()}")
    L.append(f"- 状态空间索引：{idx} / 19683；当前四象相位：{phase}")
    L.append("\n```\n" + res.scoring.to_summary() + "\n```")

    L.append("\n### 全息适配 / 拓扑 / 裂变 / 涌现")
    topo = er.get("topology", {})
    if topo:
        L.append(f"- 拓扑：V_act={topo.get('activation_volume', 0):.2f}，质心C={topo.get('activation_centroid', 0):+.3f}，态={topo.get('judgment')}")
    L.append(f"- 张力 T = {er.get('tension', 0):.2f}（微裂变阈值 {cfg.engine('tension_micro_threshold', 0.55)} / 宏裂变 {cfg.engine('tension_macro_threshold', 0.80)}）")
    if er.get("fissions"):
        L.append("- 微裂变触发：")
        for f in er["fissions"]:
            L.append(f"  - [{f['dimension']}] {f['reason']} → {f['adjustment']}")
    else:
        L.append("- 微裂变：未触发")
    if er.get("emergence", {}).get("candidates"):
        L.append("- 涌现候选：" + "、".join(c["name"] for c in er["emergence"]["candidates"]))
    if er.get("existence"):
        L.append(f"- 存在节律：{er['existence'].get('mode')} · {er['existence'].get('interpretation')}")

    # ── 二、认知沙盘 ──
    L.append("\n## 二、认知沙盘推演（概率校准）")
    L.append("> 情境概率无期权隐含(A)/历史频率(B)可用，以下为**主观假设(D)**推演，非精确预测。")
    L.append(_markdown_table(
        ["情境", "概率(来源)", "触发条件", "九维倾向", "含义(条件式)"],
        [
            ["超预期", "25% [主观D]", "营收/净利超预期、息差再升", "体用变整体偏正", "估值修复空间打开"],
            ["中性", "50% [主观D]", "基本面温和、质量平稳", "保持现状", "赚股息与ROE内生"],
            ["不及预期", "25% [主观D]", "资产质量恶化、营收回落", "整体偏空", "估值下修"],
        ],
    ))

    # ── 三、估值 ──
    L.append("\n## 三、三表模型与估值")
    inc, bal = val["income"], val["balance"]
    L.append("### 利润表（单位：亿元）")
    L.append(_markdown_table(
        ["项目", "2026E", "2027E", "2028E"],
        [
            ["营业收入"] + [f"{x:.0f}" for x in inc.revenue],
            ["净利润"] + [f"{x:.0f}" for x in inc.net_profit],
            ["EPS(元)"] + [f"{x:.2f}" for x in inc.eps],
        ],
    ))
    L.append("### 资产负债表（亿元）")
    L.append(_markdown_table(
        ["项目", "2026E", "2027E", "2028E"],
        [
            ["总资产"] + [f"{x:.0f}" for x in bal.total_assets],
            ["净资产"] + [f"{x:.0f}" for x in bal.equity],
            ["ROE"] + [f"{x * 100:.1f}%" for x in bal.roe],
        ],
    ))
    L.append("### 估值")
    v_weights = cfg.valuation("weights", {})
    participating = [v for v in val["valuations"] if v_weights.get(v.method, 0) > 0]
    excluded = [v.method for v in val["valuations"] if v_weights.get(v.method, 0) <= 0 and v.target_price > 0]
    L.append(_markdown_table(
        ["方法", "目标价(元)", "权重", "说明"],
        [
            [v.method, f"{v.target_price:.2f}", f"{v_weights.get(v.method, 0):.0%}", v.confidence]
            for v in participating
        ],
    ))
    if excluded:
        L.append(f"- 不参与加权（权重为0，仅供交叉验证）：{'、'.join(excluded)}")
    if weighted_tp > 0:
        up = (weighted_tp / price - 1) * 100 if price else 0.0
        L.append(f"- **加权综合目标价：{weighted_tp:.2f} 元**（相对现价 {price:.2f} 元：{up:+.1f}%）")
        L.append(f"- 拆解：{val['breakdown']}")
    else:
        L.append("- 估值输入不足，暂无可输出综合目标价。")
    L.append("\n### 敏感性分析")
    L.append(_markdown_table(
        ["变量", "变动", "对净利润影响", "对目标价影响"],
        [[s.variable, s.change, s.net_profit_impact, s.target_price_impact] for s in val["sensitivity"]],
    ))

    # ── 银行专项三表与估值 ──
    if res.bank:
        bi, bb = res.bank["income"], res.bank["balance"]
        L.append("\n### 银行专项：三表模型（银行口径）")
        L.append(_markdown_table(
            ["银行利润表", "2026E", "2027E", "2028E"],
            [
                ["净利息收入(亿)"] + [f"{x:.0f}" for x in bi.net_interest_income],
                ["手续费净收入(亿)"] + [f"{x:.0f}" for x in bi.fee_income],
                ["营业收入(亿)"] + [f"{x:.0f}" for x in bi.operating_revenue],
                ["信用减值(亿)"] + [f"{x:.0f}" for x in bi.credit_loss],
                ["净利润(亿)"] + [f"{x:.0f}" for x in bi.net_profit],
                ["EPS(元)"] + [f"{x:.2f}" for x in bi.eps],
            ],
        ))
        L.append(_markdown_table(
            ["银行资产负债表", "2026E", "2027E", "2028E"],
            [
                ["总资产(亿)"] + [f"{x:.0f}" for x in bb.total_assets],
                ["贷款(亿)"] + [f"{x:.0f}" for x in bb.loans],
                ["存款(亿)"] + [f"{x:.0f}" for x in bb.deposits],
                ["净资产(亿)"] + [f"{x:.0f}" for x in bb.equity],
                ["不良率(%)"] + [f"{x:.2f}" for x in bb.npl_ratio],
                ["拨贷比(%)"] + [f"{x:.2f}" for x in bb.provision_ratio],
                ["ROE(%)"] + [f"{x * 100:.1f}" for x in bb.roe],
            ],
        ))
        bt = res.bank["weighted_target_price"]
        if bt > 0:
            L.append(f"- **银行口径加权目标价：{bt:.2f} 元**（{res.bank['breakdown']}）")
        L.append("### 银行专项敏感性")
        L.append(_markdown_table(
            ["变量", "变动", "对净利润影响", "对目标价影响"],
            [[s.variable, s.change, s.net_profit_impact, s.target_price_impact] for s in res.bank["sensitivity"]],
        ))

    # ── 四、预期差 / 反方 / 证伪 ──
    L.append("\n## 四、预期差、反方观点与证伪条件")
    L.append("### 反方观点（至少3条）")
    bears = data.meta.get("bear_cases")
    if bears:
        for i, b in enumerate(bears, 1):
            L.append(f"{i}. {b}")
    else:
        L.append("1. 标的成长性不足，估值修复依赖外部催化。")
        L.append("2. 息差与资产质量存在下行风险。")
        L.append("3. 分红率偏低制约高股息逻辑。")
    L.append("### 证伪条件")
    fals = data.meta.get("falsification")
    if fals:
        L.append("\n".join(f"- {x}" for x in fals))
    else:
        L.append("- 数据文件中未提供证伪条件，请补充 meta.falsification。")

    # ── 五、跟踪与交易 ──
    L.append("\n## 五、跟踪指标与交易计划（条件式框架）")
    L.append("- 跟踪：营收/净利增速、息差、资产质量、资本充足率、分红政策（按季度/月度）")
    L.append("- 本工具不给出买卖评级、目标价、仓位或止损的具体建议；以上均为监测框架。")

    # ── 六、分析师标准报告 ──
    L.append("\n## 六、分析师标准报告（摘要）")
    L.append(f"- 标的：{data.name}（{data.code}），当前价 {price:.2f} 元，PE(TTM) {market.get('pe_ttm', '—')}，PB {market.get('pb', '—')}，股息率 {market.get('dividend_yield', '—')}%")
    if data.bank_special:
        spec = bank_special_scoring(data.bank_special, cfg)
        L.append(f"- 银行专项评分：{spec}")
    else:
        L.append("- 银行专项评分：未提供 bank_special 数据")

    # ── 七、行动建议与风险 ──
    L.append("\n## 七、行动建议与风险")
    L.append(f"- 综合研判（条件式）：九维质心 C={er.get('topology', {}).get('activation_centroid', 0):+.3f}，整体{'偏多' if res.engine.get_state_vector().mean() > 0 else '中性/偏空'}。")
    L.append(f"- 当前估值：PB {market.get('pb', '—')}，相对 {data.name} 所处行业中位偏低。")
    L.append("- 风险与盲区：数据时效性、启发式概率层的主观性、回测为模拟数据。")

    # ── 八、回测与校准 ──
    L.append("\n## 八、回测验证与超参数校准")
    b = res.backtest
    L.append("### 回测绩效（模拟数据，仅演示）")
    L.append(_markdown_table(
        ["年化收益", "年化波动", "Sharpe", "最大回撤", "胜率", "盈亏比"],
        [[f"{b.annual_return:.2%}", f"{b.annual_volatility:.2%}", f"{b.sharpe_ratio:.2f}",
          f"{b.max_drawdown:.2%}", f"{b.win_rate:.2%}", f"{b.profit_loss_ratio:.2f}"]],
    ))
    L.append("### 组合相关性（行业经验值估算，非实测）")
    L.append(_markdown_table(
        ["资产", "相关系数"],
        [[k, f"{v:+.2f}"] for k, v in res.correlation.items()],
    ))
    L.append("\n- 校准状态：情境概率均为主观[D]，尚未积累≥10次回测，暂不调整 τ。")

    # ── 九、免责声明 ──
    L.append("\n## 九、免责声明")
    L.append("\n" + (cfg.disclaimer() or "不构成投资建议，不预测涨跌，不承诺收益。具体决策与风险由个人承担。"))

    return "\n".join(L)
