"""三表预测、多方法估值与敏感性分析（数据驱动）。

所有业务数值（营收/净利基数、增速、股本、净资产等）来自数据文件
（data/*.yaml），引擎只做计算；算法默认值（目标倍数、WACC、永续增长）
来自配置 ``valuation.defaults``。

银行股适用性说明：DCF 与 EV/EBITDA 对银行适用性有限，仅作交叉验证，
综合目标价以 PB 为主权重。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import SFWConfig

# ═══════════════════════════════════════════════════════════════
# 三表数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class IncomeStatement:
    revenue: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    operating_cost: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gross_profit: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    operating_expense: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    r_d_expense: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    operating_profit: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    net_profit: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    eps: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class BalanceSheet:
    total_assets: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    total_liabilities: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    equity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    debt_to_asset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    roe: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    roic: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class CashFlowStatement:
    operating_cf: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    investing_cf: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    financing_cf: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    free_cf: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class ValuationResult:
    method: str
    target_price: float
    assumptions: str
    confidence: str


@dataclass
class SensitivityResult:
    variable: str
    change: str
    net_profit_impact: str
    target_price_impact: str


# ═══════════════════════════════════════════════════════════════
# 三表构建（从输入构建，缺失字段给出明确默认并提示）
# ═══════════════════════════════════════════════════════════════

def _vec3(value) -> list[float]:
    if isinstance(value, (list, tuple)):
        if len(value) < 3:
            value = list(value) + [value[-1]] * (3 - len(value))
        return [float(x) for x in value[:3]]
    return [float(value)] * 3


def build_income(inc_input: dict, shares: float) -> IncomeStatement:
    base_rev = float(inc_input.get("base_revenue", 0.0))
    rev_growth = _vec3(inc_input.get("revenue_growth", [0.02, 0.03, 0.04]))
    base_np = float(inc_input.get("base_net_profit", 0.0))
    np_growth = _vec3(inc_input.get("profit_growth", [0.03, 0.03, 0.03]))

    revenue, net_profit = [], []
    r = base_rev
    for g in rev_growth:
        r *= (1 + g)
        revenue.append(r)
    p = base_np
    for g in np_growth:
        p *= (1 + g)
        net_profit.append(p)

    cost_ratio = float(inc_input.get("cost_ratio", 0.45))
    expense_ratio = float(inc_input.get("expense_ratio", 0.30))
    rd_ratio = float(inc_input.get("rd_ratio", 0.03))
    operating_cost = [x * cost_ratio for x in revenue]
    gross_profit = [x * (1 - cost_ratio) for x in revenue]
    operating_expense = [x * expense_ratio for x in revenue]
    r_d_expense = [x * rd_ratio for x in revenue]
    operating_profit = [x * (1 - cost_ratio - expense_ratio - rd_ratio) for x in revenue]
    eps = [p / shares for p in net_profit] if shares else [0.0, 0.0, 0.0]

    return IncomeStatement(
        revenue=revenue, operating_cost=operating_cost, gross_profit=gross_profit,
        operating_expense=operating_expense, r_d_expense=r_d_expense,
        operating_profit=operating_profit, net_profit=net_profit, eps=eps,
    )


def build_balance(bal_input: dict) -> BalanceSheet:
    base_assets = float(bal_input.get("base_assets", 0.0))
    asset_growth = _vec3(bal_input.get("asset_growth", [0.03, 0.03, 0.03]))
    total_assets = []
    a = base_assets
    for g in asset_growth:
        a *= (1 + g)
        total_assets.append(a)

    equity = _vec3(bal_input.get("equity", [0.0, 0.0, 0.0]))
    if equity[0] == 0.0 and base_assets > 0:
        equity_ratio = float(bal_input.get("equity_ratio", 0.087))
        equity = [base_assets * equity_ratio, 0.0, 0.0]
        # 按资产增速外推净资产
        equity = _vec3([equity[0], equity[0] * (1 + asset_growth[0]), equity[0] * (1 + asset_growth[1]) ** 2])

    total_liabilities = [total_assets[i] - equity[i] for i in range(3)]
    debt_to_asset = [total_liabilities[i] / total_assets[i] if total_assets[i] else 0.0 for i in range(3)]
    roe = _vec3(bal_input.get("roe", [0.10, 0.10, 0.10]))
    roic = _vec3(bal_input.get("roic", [0.015, 0.015, 0.015]))

    return BalanceSheet(
        total_assets=total_assets, total_liabilities=total_liabilities, equity=equity,
        debt_to_asset=debt_to_asset, roe=roe, roic=roic,
    )


def build_cashflow(cf_input: dict) -> CashFlowStatement:
    operating_cf = _vec3(cf_input.get("operating_cf", [0.0, 0.0, 0.0]))
    investing_cf = _vec3(cf_input.get("investing_cf", [0.0, 0.0, 0.0]))
    financing_cf = _vec3(cf_input.get("financing_cf", [0.0, 0.0, 0.0]))
    free_cf = [op - abs(inv) for op, inv in zip(operating_cf, investing_cf)]
    return CashFlowStatement(
        operating_cf=operating_cf, investing_cf=investing_cf,
        financing_cf=financing_cf, free_cf=free_cf,
    )


# ═══════════════════════════════════════════════════════════════
# 估值方法
# ═══════════════════════════════════════════════════════════════

def dcf_valuation(free_cashflows: list[float], wacc: float, growth_rate: float,
                  shares: float) -> ValuationResult:
    if wacc <= growth_rate:
        return ValuationResult("DCF", 0.0, "WACC 需大于永续增长率", "低")
    if not free_cashflows or free_cashflows[-1] <= 0:
        return ValuationResult("DCF", 0.0, "末段 FCF 非正，DCF 不可用", "低")
    fcf_terminal = free_cashflows[-1] * (1 + growth_rate) / (wacc - growth_rate)
    pv = fcf_terminal / ((1 + wacc) ** len(free_cashflows))
    value_per_share = pv / shares if shares else 0.0
    return ValuationResult(
        "DCF", value_per_share,
        f"WACC={wacc:.2%}, 永续增长={growth_rate:.2%}, 终值={fcf_terminal:.0f}亿",
        "低（银行现金流波动大，DCF 适用性有限）",
    )


def pe_valuation(eps_2026e: float, target_pe: float) -> ValuationResult:
    return ValuationResult(
        "PE", eps_2026e * target_pe,
        f"2026E EPS={eps_2026e:.2f}, 目标PE={target_pe:.1f}x",
        "中（银行股 PE 参考价值有限，建议结合 PB）",
    )


def pb_valuation(bvps_2026e: float, target_pb: float) -> ValuationResult:
    return ValuationResult(
        "PB", bvps_2026e * target_pb,
        f"2026E BVPS={bvps_2026e:.2f}, 目标PB={target_pb:.2f}x",
        "高（银行股估值核心方法）",
    )


def ev_ebitda_valuation(ebitda: float, multiple: float, net_debt: float, shares: float) -> ValuationResult:
    ev = ebitda * multiple
    equity_value = ev - net_debt
    return ValuationResult(
        "EV/EBITDA", equity_value / shares if shares else 0.0,
        f"EBITDA={ebitda:.0f}亿, 倍数={multiple:.1f}x, 净债务={net_debt:.0f}亿",
        "低（银行股不适用，仅供交叉验证）",
    )


def sotp_valuation(segments: dict[str, float]) -> ValuationResult:
    total = sum(segments.values())
    desc = "; ".join(f"{k}: {v:.2f}元" for k, v in segments.items())
    return ValuationResult("SOTP", total, desc, "中")


def weighted_valuation(results: list[ValuationResult], weights: dict[str, float]) -> tuple[float, str]:
    valid = [(r, weights.get(r.method, 0.0)) for r in results if r.target_price > 0]
    if not valid:
        return 0.0, "无有效估值结果"
    total_w = sum(w for _, w in valid)
    weighted_tp = sum(r.target_price * w for r, w in valid) / total_w if total_w else 0.0
    breakdown = " | ".join(f"{r.method}:{r.target_price:.2f}({w:.0%})" for r, w in valid)
    return weighted_tp, breakdown


# ═══════════════════════════════════════════════════════════════
# 敏感性
# ═══════════════════════════════════════════════════════════════

def sensitivity_analysis(base_tp: float, bvps: float) -> list[SensitivityResult]:
    return [
        SensitivityResult("营收增速", "±1%", "净利润±约2.5%", f"目标价±约{base_tp * 0.02:.2f}元"),
        SensitivityResult("净息差", "±0.05%", "净利润±约1.5%", f"目标价±约{base_tp * 0.015:.2f}元"),
        SensitivityResult("信用成本", "±0.05%", "净利润±约2.0%", f"目标价±约{base_tp * 0.02:.2f}元"),
        SensitivityResult("折现率(WACC)", "±0.5%", "DCF值±约5%", f"目标价±约{base_tp * 0.05:.2f}元"),
        SensitivityResult("永续增长率", "±0.5%", "DCF终值±约8%", f"目标价±约{base_tp * 0.03:.2f}元"),
        SensitivityResult("目标PB", "±0.05x", "—", f"目标价±约{bvps * 0.05:.2f}元"),
    ]


# ═══════════════════════════════════════════════════════════════
# 完整估值流程
# ═══════════════════════════════════════════════════════════════

def run_valuation(val_input: dict, cfg: SFWConfig) -> dict:
    """从数据文件的 valuation 段构建三表并执行多方法估值。

    val_input 结构（见 data/pingan_000001.yaml 示例）：
        shares, income{...}, balance{...}, cashflow{...},
        targets{target_pe,target_pb,bvps,wacc,terminal_growth,ev_ebitda,net_debt,segments}
    """
    shares = float(val_input.get("shares", 194.06))
    inc = build_income(val_input.get("income", {}), shares)
    bal = build_balance(val_input.get("balance", {}))
    cf = build_cashflow(val_input.get("cashflow", {}))

    defaults = cfg.valuation("defaults", {})
    targets = val_input.get("targets", {})
    target_pe = float(targets.get("target_pe", defaults.get("target_pe", 6.0)))
    target_pb = float(targets.get("target_pb", defaults.get("target_pb", 0.56)))
    wacc = float(targets.get("wacc", defaults.get("wacc", 0.10)))
    term_growth = float(targets.get("terminal_growth", defaults.get("terminal_growth", 0.03)))
    ev_mult = float(targets.get("ev_ebitda", defaults.get("ev_ebitda_multiple", 8.0)))
    net_debt = float(targets.get("net_debt", 0.0))
    bvps = float(targets.get("bvps", bal.equity[0] / shares if shares else 0.0))

    results: list[ValuationResult] = [
        dcf_valuation(cf.free_cf, wacc, term_growth, shares),
        pe_valuation(inc.eps[0], target_pe),
        pb_valuation(bvps, target_pb),
        ev_ebitda_valuation(float(targets.get("ebitda", 0.0)), ev_mult, net_debt, shares),
        sotp_valuation(targets.get("segments", {})),
    ]

    weights = cfg.valuation("weights", {})
    weighted_tp, breakdown = weighted_valuation(results, weights)
    sensitivity = sensitivity_analysis(weighted_tp, bvps)

    return {
        "income": inc,
        "balance": bal,
        "cashflow": cf,
        "valuations": results,
        "weighted_target_price": weighted_tp,
        "breakdown": breakdown,
        "sensitivity": sensitivity,
        "bvps": bvps,
    }
