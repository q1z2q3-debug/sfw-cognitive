"""实时行情接入模块测试（重点：无 akshare 时的优雅降级 + 纯 pandas 指标计算）。"""

import unittest

import numpy as np

from sfw.market_data import (
    HAVE_AKSHARE,
    HAVE_PANDAS,
    compute_scoring_indicators,
    fetch_financial,
    fetch_price,
    require_deps,
)


# 依赖风险：akshare 未必安装；此处验证"未安装时明确报错/降级"，避免假通过。
@unittest.skipUnless(not HAVE_AKSHARE, "akshare 已安装，跳过降级路径测试")
class TestGracefulDegradation(unittest.TestCase):
    def test_require_deps_raises(self):
        with self.assertRaises(RuntimeError):
            require_deps()

    def test_fetch_returns_none(self):
        self.assertIsNone(fetch_price("sz000001"))
        self.assertIsNone(fetch_financial("sz000001"))


@unittest.skipUnless(HAVE_PANDAS, "需要 pandas")
class TestIndicatorCalc(unittest.TestCase):
    def test_compute_scoring_indicators(self):
        import pandas as pd
        rng = np.random.default_rng(7)
        n = 120
        price = 10.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
        df = pd.DataFrame({
            "日期": pd.date_range("2024-01-01", periods=n, freq="D"),
            "开盘": price, "收盘": price, "最高": price * 1.01,
            "最低": price * 0.99, "成交额": [1e9] * n, "换手率": [2.0] * n,
        })
        out = compute_scoring_indicators(df, None)
        # 九维全部存在
        self.assertEqual(set(out.keys()), {"天", "地", "人", "正", "反", "合", "归", "守", "进"})
        # 值域 {-1,0,+1}
        for indicators in out.values():
            for val in indicators.values():
                self.assertIn(val, (-1, 0, 1))

    def test_empty_price(self):
        out = compute_scoring_indicators(None, None)
        self.assertEqual(out, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
