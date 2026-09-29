"""银行专用三表模板与专项估值单元测试。"""

import unittest

from sfw.bank_model import (
    BankIncome,
    bank_sensitivity,
    build_bank_balance,
    build_bank_income,
    ddm_valuation,
    historical_pb_valuation,
    pb_roe_valuation,
    run_bank_valuation,
)
from sfw.config import load_config

BANK_INPUT = {
    "shares": 194.06,
    "income": {
        "base_earning_assets": 54000,
        "earning_asset_growth": [0.015, 0.02, 0.02],
        "nim": [1.80, 1.82, 1.85],
        "fee_ratio": 0.19,
        "other_nonint_ratio": 0.15,
        "cost_income_ratio": [0.28, 0.27, 0.27],
        "credit_cost_ratio": [1.15, 1.10, 1.05],
        "loan_base": 34522.74,
        "loan_growth": [0.018, 0.02, 0.022],
        "tax_rate": 0.25,
    },
    "balance": {
        "base_loans": 34522.74, "loan_growth": [0.018, 0.02, 0.022],
        "base_deposits": 36605.87, "deposit_growth": [0.02, 0.022, 0.024],
        "base_assets": 60287.85, "asset_growth": [0.017, 0.02, 0.02],
        "npl_ratio": [1.05, 1.04, 1.03], "provision_ratio": [2.30, 2.28, 2.26],
        "core_tier1": [9.32, 9.35, 9.38], "roe": [0.105, 0.105, 0.106],
        "equity": [4700, 4850, 5000],
    },
    "valuation": {"bvps": 24.15, "roe": 0.105, "coe": 0.10, "g": 0.03,
                  "hist_target_pb": 0.56, "dps": 0.50, "r": 0.10},
}


class TestBankIncome(unittest.TestCase):
    def test_income_decomposition(self):
        inc = build_bank_income(BANK_INPUT["income"], 194.06)
        self.assertIsInstance(inc, BankIncome)
        self.assertEqual(len(inc.net_profit), 3)
        # 营业收入 = 净利息 + 手续费 + 其他非息
        for i in range(3):
            self.assertAlmostEqual(
                inc.operating_revenue[i],
                inc.net_interest_income[i] + inc.fee_income[i] + inc.other_nonint_income[i],
                places=6,
            )
        # 净利润为正
        self.assertGreater(inc.net_profit[0], 0)
        # 利息净收入占比应在 55%~70%（银行口径）
        self.assertGreater(inc.net_interest_ratio, 0.5)
        self.assertLess(inc.net_interest_ratio, 0.8)

    def test_income_growth(self):
        inc = build_bank_income(BANK_INPUT["income"], 194.06)
        self.assertGreater(inc.net_profit[2], inc.net_profit[0])


class TestBankBalance(unittest.TestCase):
    def test_balance_consistency(self):
        inc = build_bank_income(BANK_INPUT["income"], 194.06)
        bal = build_bank_balance(BANK_INPUT["balance"], inc)
        self.assertEqual(len(bal.loans), 3)
        self.assertGreater(bal.equity[0], 0)
        # 拨备 = 贷款 × 拨贷比
        self.assertAlmostEqual(bal.provision[0], bal.loans[0] * 0.0230, delta=0.5)


class TestBankValuation(unittest.TestCase):
    def test_pb_roe_formula(self):
        v = pb_roe_valuation(0.105, 0.10, 0.03, 24.15)
        target_pb = (0.105 - 0.03) / (0.10 - 0.03)
        self.assertAlmostEqual(v.target_price, target_pb * 24.15, places=4)

    def test_historical_pb(self):
        v = historical_pb_valuation(24.15, 0.56)
        self.assertAlmostEqual(v.target_price, 13.524, places=4)

    def test_ddm(self):
        v = ddm_valuation(0.50, 0.03, 0.10)
        self.assertAlmostEqual(v.target_price, 0.50 / 0.07, places=4)

    def test_ddm_invalid(self):
        v = ddm_valuation(0.50, 0.10, 0.10)
        self.assertEqual(v.target_price, 0.0)

    def test_sensitivity(self):
        s = bank_sensitivity(440, 13.0, 34000, 2.2)
        self.assertEqual(len(s), 5)


class TestRunBankValuation(unittest.TestCase):
    def test_full(self):
        cfg = load_config()
        res = run_bank_valuation(BANK_INPUT, cfg)
        for key in ("income", "balance", "cashflow", "valuations",
                    "weighted_target_price", "breakdown", "sensitivity", "bvps"):
            self.assertIn(key, res)
        self.assertGreater(res["weighted_target_price"], 0)

    def test_weighted_uses_only_valid(self):
        cfg = load_config()
        res = run_bank_valuation(BANK_INPUT, cfg)
        self.assertEqual(len(res["valuations"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
