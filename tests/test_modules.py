"""评分卡、估值、数据层与流水线单元测试。"""

import os
import unittest
from pathlib import Path

import numpy as np

from sfw.config import load_config
from sfw.data import load_stock_data
from sfw.report import generate_report, run_analysis
from sfw.scoring import ScoringCard, bank_special_scoring
from sfw.valuation import (
    build_balance,
    build_cashflow,
    build_income,
    pb_valuation,
    pe_valuation,
    run_valuation,
)

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "pingan_000001.yaml"


def _cfg():
    return load_config()


class TestScoring(unittest.TestCase):
    def test_score_bounds(self):
        card = ScoringCard(_cfg())
        indicators = {d: {k: 1 for k in _cfg().scoring_weights(d)} for d in ("天", "地", "正")}
        card.score(indicators)
        for dim in ("天", "地", "正"):
            r = card.results[dim]
            self.assertGreaterEqual(r.raw_score, -1)
            self.assertLessEqual(r.raw_score, 1)
            self.assertAlmostEqual(np.sum(r.cloud.probs), 1.0, places=6)

    def test_missing_indicator_marked(self):
        card = ScoringCard(_cfg())
        card.score({"天": {"policy": 1}})  # 其余指标缺失
        r = card.results["天"]
        self.assertTrue(r.missing_indicators)

    def test_bank_special(self):
        cfg = _cfg()
        spec = bank_special_scoring(
            {"net_interest_margin": 1.80, "npl_ratio": 1.05,
             "provision_coverage": 219.58, "capital_adequacy": 13.14,
             "dividend_yield": 5.27}, cfg)
        for val in spec.values():
            self.assertIn(val, (-1.0, 0.0, 1.0))


class TestValuation(unittest.TestCase):
    def test_income_growth(self):
        inc = build_income({"base_revenue": 1000, "revenue_growth": [0.02, 0.03, 0.04],
                            "base_net_profit": 100, "profit_growth": [0.03, 0.03, 0.03]}, 10.0)
        self.assertEqual(len(inc.revenue), 3)
        self.assertGreater(inc.revenue[2], inc.revenue[0])

    def test_balance(self):
        bal = build_balance({"base_assets": 1000, "asset_growth": [0.1, 0.1, 0.1],
                             "equity": [100, 110, 120]})
        self.assertGreater(bal.total_assets[2], bal.total_assets[0])
        self.assertAlmostEqual(bal.total_assets[0], bal.total_liabilities[0] + bal.equity[0], places=6)

    def test_cashflow(self):
        cf = build_cashflow({"operating_cf": [100, 120, 140], "investing_cf": [-10, -20, -30]})
        self.assertEqual(len(cf.operating_cf), 3)
        self.assertEqual(cf.free_cf[0], 90.0)

    def test_pb_pe(self):
        self.assertAlmostEqual(pb_valuation(24.84, 0.56).target_price, 13.9104, places=4)
        self.assertAlmostEqual(pe_valuation(2.24, 6.0).target_price, 13.44, places=6)

    def test_run_valuation(self):
        cfg = _cfg()
        val = run_valuation({"shares": 194.06,
                             "income": {"base_revenue": 1314.42, "base_net_profit": 426.33},
                             "balance": {"base_assets": 60287.85, "equity": [4700, 4850, 5000]},
                             "cashflow": {"operating_cf": [3500, 3800, 4100], "investing_cf": [-800, -900, -1000]},
                             "targets": {"bvps": 24.15, "target_pb": 0.56}}, cfg)
        self.assertGreater(val["weighted_target_price"], 0)


class TestData(unittest.TestCase):
    def test_load_valid(self):
        data = load_stock_data(DATA_FILE)
        self.assertEqual(data.code, "000001.SZ")

    def test_missing_price_rejected(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("meta:\n  stock: X\nmarket: {}\n")
            name = f.name
        try:
            with self.assertRaises(ValueError):
                load_stock_data(name)
        finally:
            os.unlink(name)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_stock_data("/no/such/file.yaml")


class TestPipeline(unittest.TestCase):
    def test_full_analysis(self):
        cfg = _cfg()
        data = load_stock_data(DATA_FILE)
        res = run_analysis(data, cfg)
        md = generate_report(res)
        self.assertIn("平安银行", md)
        self.assertIn("## 九、免责声明", md)
        self.assertGreater(res.valuation["weighted_target_price"], 0)

    def test_report_has_disclaimer(self):
        cfg = _cfg()
        data = load_stock_data(DATA_FILE)
        md = generate_report(run_analysis(data, cfg))
        self.assertIn("不构成投资建议", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
