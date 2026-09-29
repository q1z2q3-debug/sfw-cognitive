"""银行专用三表模板、敏感性分析与专项估值（数据驱动）。

与通用估值模型不同，银行股以"生息资产 × 净息差 → 利息净收入"为利润表
主干驱动，资产负债表以"贷款/存款/拨备/资本"为核心，估值以 PB-ROE、
历史 PB 分位与 DDM 为主。该模块把附件（deepseek 专家指导）中的银行专项
模型落成可运行代码。

输入来自数据文件的 ``valuation.bank`` 段，所有业务数值不硬编码在代码中。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import SFWConfig

# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class BankIncome:
    """银行利润表（按银行口径）。"""
    years: list[int] = field(default_factory=list)
    net_interest_income: list[float] = field(default_factory=list)   # 利息净收入
    fee_income: list[float] = field(default_factory=list)            # 手续费及佣金净收入
    other_nonint_income: list[float] = field(default_factory=list)   # 其他非息收入
    operating_revenue: list[float] = field(default_factory=list)     # 营业收入
    operating_cost: list[float] = field(default_factory=list)        # 业务及管理费+税金
    credit_loss: list[float] = field(default_factory=list)           # 信用减值损失(拨备)
    operating_profit: list[float] = field(default_factory=list)      # 营业利润
    pre_tax_profit: list[float] = field(default_factory=list)        # 税前利润
    net_profit: list[float] = field(default_factory=list)            # 净利润
    eps: list[float] = field(default_factory=list)

    @property
    def net_interest_ratio(self) -> float:  # 利息净收入占营收比
        if not self.operating_revenue or not self.operating_revenue[0]:
            return 0.0
        return self.net_interest_income[0] / self.operating_revenue[0]


@dataclass
class BankBalance:
    """银行资产负债表（核心科目）。"""
    years: list[int] = field(default_factory=list)
    total_assets: list[float] = field(default_factory=list)
    loans: list[float] = field(default_factory=list)       # 发放贷款
    deposits: list[float] = field(default_factory=list)    # 吸收存款
    equity: list[float] = field(default_factory=list)      # 净资产
    provision: list[float] = field(default_factory=list)   # 贷款减值准备
    provision_ratio: list[float] = field(default_factory=list)   # 拨贷比(%)
    npl_ratio: list[float] = field(default_factory=list)   # 不良率(%)
    core_tier1: list[float] = field(default_factory=list)  # 核心一级资本充足率(%)
    roe: list[float] = field(default_factory=list)         # 净资产收益率


@dataclass
class BankCashFlow:
    """银行现金流量表（简化）。"""
    operating_cf: list[float] = field(default_factory=list)
    investing_cf: list[float] = field(default_factory=list)
    financing_cf: list[float] = field(default_factory=list)
    deposit_increase: list[float] = field(default_factory=list)
    loan_increase: list[float] = field(default_factory=list)


@dataclass
class BankSensitivity:
    variable: str
    change: str
    net_profit_impact: str
    target_price_impact: str


@dataclass
class BankValuation:
    method: str
    target_price: float
    assumptions: str


# ═══════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════

def _vec3(value, default=0.0) -> list[float]:
    if isinstance(value, (list, tuple)):
        v = [float(x) for x in value]
        if len(v) < 3:
            v = v + [v[-1]] * (3 - len(v))
        return v[:3]
    if value is None:
        return [float(default)] * 3
    return [float(value)] * 3


def _scale(start: float, growth: list[float]) -> list[float]:
    """按增速序列外推。"""
    out = []
    cur = start
    for g in growth:
        cur *= (1 + g)
        out.append(cur)
    return out


# ═══════════════════════════════════════════════════════════════
# 三表构建
# ═══════════════════════════════════════════════════════════════

def build_bank_income(inc: dict, shares: float, base_year: int = 2026) -> BankIncome:
    """银行利润表模型。

    inc 字段：
        base_earning_assets: 基期生息资产（亿元）
        earning_asset_growth: 生息资产增速 [g1,g2,g3]
        nim: 净息差(%) 序列或常量
        fee_ratio: 手续费净收入占营收比
        other_nonint_ratio: 其他非息收入占营收比
        cost_income_ratio: 成本收入比（业务及管理费/营收）
        credit_cost_ratio: 信用成本率（信用减值/贷款）
        loan_base: 贷款基数（用于信用减值测算，亿元）
        tax_rate: 有效税率
    """
    ea_growth = _vec3(inc.get("earning_asset_growth", [0.03, 0.03, 0.03]))
    nim = _vec3(inc.get("nim", [1.80, 1.80, 1.80]))
    fee_ratio = _vec3(inc.get("fee_ratio", [0.19, 0.19, 0.19]))
    other_ratio = _vec3(inc.get("other_nonint_ratio", [0.15, 0.15, 0.15]))
    cost_ratio = _vec3(inc.get("cost_income_ratio", [0.28, 0.27, 0.27]))
    credit_cost_ratio = _vec3(inc.get("credit_cost_ratio", [1.15, 1.10, 1.05]))  # %
    tax_rate = float(inc.get("tax_rate", 0.25))
    loan_base = float(inc.get("loan_base", 0.0))
    loan_growth = _vec3(inc.get("loan_growth", ea_growth))

    ea = _scale(float(inc.get("base_earning_assets", 0.0)), ea_growth)
    loans = _scale(loan_base if loan_base else ea[0] * 0.5, loan_growth)

    nii, fee, other, rev, cost, credit = [], [], [], [], [], []
    for i in range(3):
        nii.append(ea[i] * nim[i] / 100.0)
        fee.append(nii[i] * fee_ratio[i] / (1 - fee_ratio[i] - other_ratio[i]))
        other.append(nii[i] * other_ratio[i] / (1 - fee_ratio[i] - other_ratio[i]))
        rev.append(nii[i] + fee[i] + other[i])
        cost.append(rev[i] * cost_ratio[i])
        credit.append(loans[i] * credit_cost_ratio[i] / 100.0)

    # 简化：营业利润 = 营业收入 - 业务及管理费 - 信用减值 - 税金及附加(含在成本)
    op_profit = [rev[i] - cost[i] - credit[i] for i in range(3)]
    pre_tax = op_profit[:]
    net = [p * (1 - tax_rate) for p in pre_tax]
    eps = [p / shares for p in net] if shares else [0.0, 0.0, 0.0]

    return BankIncome(
        years=[base_year + i for i in range(3)],
        net_interest_income=nii, fee_income=fee, other_nonint_income=other,
        operating_revenue=rev, operating_cost=cost, credit_loss=credit,
        operating_profit=op_profit, pre_tax_profit=pre_tax,
        net_profit=net, eps=eps,
    )


def build_bank_balance(bal: dict, income: BankIncome, base_year: int = 2026) -> BankBalance:
    """银行资产负债表模型。

    bal 字段：
        base_loans / base_deposits / base_assets: 基期
        loan_growth / deposit_growth / asset_growth
        npl_ratio: 不良率(%) 序列
        provision_ratio: 拨贷比(%) 序列
        core_tier1: 核心一级资本充足率(%)
        roe: 净资产收益率
        equity: 净资产（可直接给，或按 ROE 由净利润推算）
    """
    loan_growth = _vec3(bal.get("loan_growth", [0.03, 0.03, 0.03]))
    deposit_growth = _vec3(bal.get("deposit_growth", [0.03, 0.03, 0.03]))
    asset_growth = _vec3(bal.get("asset_growth", [0.03, 0.03, 0.03]))
    npl = _vec3(bal.get("npl_ratio", [1.05, 1.05, 1.05]))
    prov_ratio = _vec3(bal.get("provision_ratio", [2.30, 2.30, 2.30]))
    core_tier1 = _vec3(bal.get("core_tier1", [9.30, 9.30, 9.30]))
    roe = _vec3(bal.get("roe", [0.105, 0.105, 0.105]))

    loans = _scale(float(bal.get("base_loans", 0.0)), loan_growth)
    deposits = _scale(float(bal.get("base_deposits", 0.0)), deposit_growth)
    total_assets = _scale(float(bal.get("base_assets", 0.0)), asset_growth)
    provision = [loans[i] * prov_ratio[i] / 100.0 for i in range(3)]

    equity = _vec3(bal.get("equity"))
    if equity[0] == 0.0:
        # 由净利润与 ROE 推算（ROE 为净资产收益率）
        equity = [income.net_profit[i] / roe[i] for i in range(3)]

    return BankBalance(
        years=[base_year + i for i in range(3)],
        total_assets=total_assets, loans=loans, deposits=deposits, equity=equity,
        provision=provision, provision_ratio=prov_ratio, npl_ratio=npl,
        core_tier1=core_tier1, roe=roe,
    )


def build_bank_cashflow(cf: dict, base_year: int = 2026) -> BankCashFlow:
    deposit_inc = _vec3(cf.get("deposit_increase", [0.0, 0.0, 0.0]))
    loan_inc = _vec3(cf.get("loan_increase", [0.0, 0.0, 0.0]))
    return BankCashFlow(
        operating_cf=_vec3(cf.get("operating_cf", [0.0, 0.0, 0.0])),
        investing_cf=_vec3(cf.get("investing_cf", [0.0, 0.0, 0.0])),
        financing_cf=_vec3(cf.get("financing_cf", [0.0, 0.0, 0.0])),
        deposit_increase=deposit_inc, loan_increase=loan_inc,
    )


# ═══════════════════════════════════════════════════════════════
# 银行敏感性
# ═══════════════════════════════════════════════════════════════

def bank_sensitivity(base_np: float, base_tp: float, loan_base: float,
                     eps: float) -> list[BankSensitivity]:
    """银行敏感性：LPR / 不良率 / 存款成本 / 信贷增速。"""
    # 净息差 -10bp → 利息净收入 ≈ -10bp×生息资产；按净利润/营收杠杆粗略估算
    return [
        BankSensitivity("LPR", "-10bp", "净利润约-3%", f"目标价约-{base_tp * 0.03:.2f}元"),
        BankSensitivity("不良率", "+10bp", f"信用成本+约{loan_base * 0.001:.0f}亿",
                        f"目标价约-{base_tp * 0.03:.2f}元"),
        BankSensitivity("存款成本", "-5bp", "净息差改善约+3bp，净利润约+2%",
                        f"目标价约+{base_tp * 0.02:.2f}元"),
        BankSensitivity("信贷增速", "-1pct", "净利润约-2%", f"目标价约-{base_tp * 0.02:.2f}元"),
        BankSensitivity("分红率", "+5pct", "股息率+0.3%", f"目标价约+{base_tp * 0.02:.2f}元"),
    ]


# ═══════════════════════════════════════════════════════════════
# 银行专项估值
# ═══════════════════════════════════════════════════════════════

def pb_roe_valuation(roe: float, coe: float, g: float, bvps: float) -> BankValuation:
    """PB-ROE：目标PB = (ROE - g)/(COE - g)，目标价 = 目标PB × BVPS。"""
    target_pb = (roe - g) / (coe - g) if coe > g else 0.0
    return BankValuation(
        "PB-ROE", target_pb * bvps,
        f"ROE={roe:.1%}, COE={coe:.1%}, g={g:.1%} → 目标PB={target_pb:.2f}x",
    )


def historical_pb_valuation(bvps: float, target_pb: float, note: str = "") -> BankValuation:
    return BankValuation("历史PB分位", bvps * target_pb, f"目标PB={target_pb:.2f}x{('；' + note) if note else ''}")


def ddm_valuation(dps: float, g: float, r: float) -> BankValuation:
    """DDM：目标价 = DPS/(r - g)。"""
    if r <= g:
        return BankValuation("DDM", 0.0, "要求回报率需大于股息增长率")
    return BankValuation("DDM", dps / (r - g), f"DPS={dps:.2f}, r={r:.1%}, g={g:.1%}")


def run_bank_valuation(bank_input: dict, cfg: SFWConfig) -> dict:
    """执行银行专项三表 + 估值。

    bank_input 为数据文件 ``valuation.bank`` 段，字段见各 build 函数 docstring。
    """
    shares = float(bank_input.get("shares", 194.06))
    income = build_bank_income(bank_input.get("income", {}), shares)
    balance = build_bank_balance(bank_input.get("balance", {}), income)
    cashflow = build_bank_cashflow(bank_input.get("cashflow", {}))
    val_input = bank_input.get("valuation", {})

    bvps = float(val_input.get("bvps", balance.equity[0] / shares if shares else 0.0))
    roe = float(val_input.get("roe", income.net_profit[0] / (balance.equity[0] if balance.equity[0] else 1e-9)))
    coe = float(val_input.get("coe", 0.10))
    g = float(val_input.get("g", 0.03))
    hist_pb = float(val_input.get("hist_target_pb", 0.56))
    dps = float(val_input.get("dps", income.net_profit[0] * 0.20 / shares))
    r = float(val_input.get("r", 0.10))

    valuations = [
        pb_roe_valuation(roe, coe, g, bvps),
        historical_pb_valuation(bvps, hist_pb, "国泰海通参考"),
        ddm_valuation(dps, g, r),
    ]
    weights = {"PB-ROE": 0.35, "历史PB分位": 0.45, "DDM": 0.20}

    # PB-ROE 前置条件：ROE 须显著高于 COE（隐含目标PB<1），否则该模型不适用
    # （如 ROE 处于下行通道且约等于 COE 时，公式给出 >1x 的偏乐观 PB，不应主导加权）。
    notes: dict[str, str] = {}
    pbroe = valuations[0]
    if pbroe.target_price > 0 and bvps > 0 and (pbroe.target_price / bvps) >= 1.0:
        weights["PB-ROE"] = 0.0
        notes["PB-ROE"] = "不适用（ROE 未显著高于 COE，隐含目标PB≥1x，退出加权）"

    valid = [(v, weights[v.method]) for v in valuations if weights[v.method] > 0 and v.target_price > 0]
    total_w = sum(w for _, w in valid)
    weighted_tp = sum(v.target_price * w for v, w in valid) / total_w if total_w else 0.0
    breakdown_parts = [
        f"{v.method}:{v.target_price:.2f}({weights[v.method]:.0%}{('·' + notes.get(v.method, '')) if notes.get(v.method) else ''})"
        for v in valuations
    ]

    sensitivity = bank_sensitivity(income.net_profit[0], weighted_tp,
                                   balance.loans[0] if balance.loans else 0.0, income.eps[0])

    return {
        "income": income,
        "balance": balance,
        "cashflow": cashflow,
        "valuations": valuations,
        "weights": weights,
        "notes": notes,
        "weighted_target_price": weighted_tp,
        "breakdown": " | ".join(breakdown_parts),
        "sensitivity": sensitivity,
        "bvps": bvps,
    }
