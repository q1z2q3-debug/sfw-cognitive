"""大模型内化（双系统）模块测试：S1 命中 / S2 学习 / 反馈回灌 / 持久化 / 毕业。"""

import os
import tempfile
import unittest

from sfw.internalize import (
    DualSystemInternalizer,
    PatternLibrary,
    analysis_action_signature,
    state_to_index,
    to_trits,
)

S1 = [1, 1, 0, 1, -1, 0, 0, 1, -1]     # 一个"看多企稳"态
S2 = [1, 1, 0, 1, -1, 0, 0, 1, 1]      # 邻近态（末位差异）


def _btcu_importable() -> bool:
    try:
        import btcu_harness  # noqa: F401
        return True
    except ImportError:
        return False


def _tmp_lib() -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(path)
    return path


def _s2_fn(state9, input_text):
    return f"idx={state_to_index(state9)}|S2推演:{input_text}"


class TestTritsAndIndex(unittest.TestCase):
    def test_to_trits_and_index(self):
        t = to_trits(S1)
        self.assertEqual(len(t), 9)
        self.assertEqual(t, S1)
        idx = state_to_index(S1)
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, 3**9)

    def test_index_distinct(self):
        self.assertNotEqual(state_to_index(S1), state_to_index(S2))


class TestPatternLibrary(unittest.TestCase):
    def test_learn_and_exact_hit(self):
        lib = PatternLibrary()
        lib.learn(state_to_index(S1), S1, "平安银行", "idx=X|verdict=偏多")
        match = lib.best_match(S1, "平安银行")
        self.assertIsNotNone(match)
        self.assertEqual(match.action, "idx=X|verdict=偏多")

    def test_knn_neighbor_hit(self):
        lib = PatternLibrary()
        lib.learn(state_to_index(S1), S1, "平安银行", "idx=X|verdict=偏多")
        # 邻近态（1 位差异）应命中
        match = lib.best_match([1, 1, 0, 1, -1, 0, 0, 1, 1], "平安银行")
        self.assertIsNotNone(match)

    def test_no_match_returns_none(self):
        lib = PatternLibrary()
        self.assertIsNone(lib.best_match(S1, "无记录"))

    def test_feedback_bayesian(self):
        lib = PatternLibrary()
        rec = lib.learn(state_to_index(S1), S1, "平安银行", "偏多")
        self.assertIsNotNone(lib.feedback(rec.input_hash, success=True))
        rec2 = lib.feedback(rec.input_hash, success=True)
        self.assertEqual(rec2.use_count, 2)
        self.assertEqual(rec2.success_count, 2)

    def test_persistence_roundtrip(self):
        path = _tmp_lib()
        lib = PatternLibrary(path)
        lib.learn(state_to_index(S1), S1, "平安银行", "偏多")
        lib.save()
        lib2 = PatternLibrary(path)
        self.assertTrue(lib2.load())
        self.assertEqual(len(lib2.records), 1)
        self.assertEqual(lib2.records[0].action, "偏多")
        os.remove(path)


class TestDualSystemInternalizer(unittest.TestCase):
    def test_s2_miss_then_learn(self):
        path = _tmp_lib()
        inter = DualSystemInternalizer(library_path=path, system2_fn=_s2_fn, min_confidence=0.6)
        r1 = inter.resolve(S1, "平安银行")
        self.assertEqual(r1.source, "system2")
        self.assertTrue(r1.learned)
        # 反馈强化后，S1 应能直接命中
        inter.feedback(r1.matched_hash, success=True)
        r2 = inter.resolve(S1, "平安银行")
        self.assertEqual(r2.source, "system1")
        self.assertEqual(r2.action, r1.action)
        inter.save()
        os.remove(path)

    def test_graduation_stages(self):
        path = _tmp_lib()
        inter = DualSystemInternalizer(library_path=path, system2_fn=_s2_fn)
        # school：空库
        self.assertEqual(inter.graduation()["stage"], "school")
        # 学习 + 多次成功反馈 → 至少 internalize
        for _ in range(10):
            r = inter.resolve(S1, "平安银行")
            inter.feedback(r.matched_hash, success=True)
        stage = inter.graduation()["stage"]
        self.assertIn(stage, ("internalize", "graduate"))
        inter.save()
        os.remove(path)

    def test_signature_builds(self):
        sig = analysis_action_signature(S1, {"fractal": {"phase_name": "积聚"},
                                             "topology": {"judgment": "偏多"},
                                             "tension": 0.42}, 11.56)
        self.assertIn("idx=", sig)
        self.assertIn("wtp=11.56", sig)


class TestBtcUProvider(unittest.TestCase):
    @unittest.skipUnless(_btcu_importable(), "未安装 btcu-harness")
    def test_btcu_bridge_available(self):
        from sfw.internalize import _btcu_library
        lib = _btcu_library(_tmp_lib())
        self.assertIsNotNone(lib)


def _btcu_importable() -> bool:
    try:
        import btcu_harness  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    unittest.main(verbosity=2)
