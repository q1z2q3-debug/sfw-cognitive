"""核心引擎单元测试。"""

import unittest

import numpy as np

from sfw.config import SFWConfig, load_config
from sfw.core import (
    DIM_NAMES,
    Phase,
    ProbabilityCloud,
    SFWEngine,
    check_dimension_emergence,
    compute_tension,
    compute_topology,
    compute_transfer_probability,
    dynamic_temperature,
    encode_bagua,
    should_expand_fractal,
    softmax_cloud,
    state_to_index,
)


def _cfg() -> SFWConfig:
    return load_config()


class TestProbabilityCloud(unittest.TestCase):
    def test_softmax_extreme_positive(self):
        probs = softmax_cloud(1.0, 0.15)
        self.assertAlmostEqual(np.sum(probs), 1.0, places=6)
        cloud = ProbabilityCloud(probs=probs)
        self.assertGreater(probs[2], 0.99)
        self.assertEqual(cloud.judgment(), "高确信+1")

    def test_softmax_extreme_negative(self):
        probs = softmax_cloud(-1.0, 0.15)
        cloud = ProbabilityCloud(probs=probs)
        self.assertAlmostEqual(np.sum(probs), 1.0, places=6)
        self.assertGreater(probs[0], 0.99)
        self.assertEqual(cloud.judgment(), "高确信-1")

    def test_softmax_zero(self):
        probs = softmax_cloud(0.0, 0.15)
        self.assertAlmostEqual(probs[0], probs[1], places=6)
        self.assertAlmostEqual(probs[1], probs[2], places=6)

    def test_entropy_range(self):
        for s in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            cloud = ProbabilityCloud(probs=softmax_cloud(s, 0.15))
            self.assertGreaterEqual(cloud.entropy, 0)
            self.assertLessEqual(cloud.entropy, np.log(3) + 0.01)

    def test_cloud_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            ProbabilityCloud(probs=np.array([0.3, 0.7]))

    def test_dynamic_temperature(self):
        cfg = _cfg()
        self.assertEqual(dynamic_temperature(0.10, cfg), 0.15)
        self.assertEqual(dynamic_temperature(0.35, cfg), 0.25)


class TestFractal(unittest.TestCase):
    def test_should_expand_high_entropy(self):
        cfg = _cfg()
        entropies = np.array([0.5, 0.95, 0.3, 0.4, 0.6, 0.2, 0.5, 0.4, 0.3])
        triggered, reason = should_expand_fractal(entropies, 1, 1, 2, cfg)
        self.assertTrue(triggered)
        self.assertIn("熵过高", reason)

    def test_should_expand_conflict(self):
        cfg = _cfg()
        triggered, reason = should_expand_fractal(np.array([0.3] * 9), 3, 1, 2, cfg)
        self.assertTrue(triggered)
        self.assertIn("冲突", reason)

    def test_should_not_expand(self):
        cfg = _cfg()
        triggered, _ = should_expand_fractal(np.array([0.2] * 9), 0, 1, 2, cfg)
        self.assertFalse(triggered)

    def test_transfer_probability_sum(self):
        probs = compute_transfer_probability(Phase.ACCUMULATE, 0.5, 0.3, 0.5, 0.5, 0.5)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)


class TestTensionFission(unittest.TestCase):
    def test_tension_range(self):
        t = compute_tension(0.5, 0.5, 2, 0.8)
        self.assertGreaterEqual(t, 0)
        self.assertLessEqual(t, 1)


class TestTopology(unittest.TestCase):
    def test_focused_state(self):
        cfg = _cfg()
        clouds = {d: ProbabilityCloud(probs=softmax_cloud(0.8, 0.15)) for d in DIM_NAMES}
        topo = compute_topology(clouds, cfg)
        self.assertEqual(topo.topology_judgment, "聚焦态")
        self.assertLess(topo.activation_volume, 3)

    def test_diffuse_state(self):
        cfg = _cfg()
        clouds = {d: ProbabilityCloud(probs=softmax_cloud(0.0, 0.15)) for d in DIM_NAMES}
        topo = compute_topology(clouds, cfg)
        self.assertEqual(topo.topology_judgment, "弥散态")
        self.assertGreater(topo.activation_volume, 7)

    def test_coupling_symmetric(self):
        cfg = _cfg()
        clouds = {d: ProbabilityCloud(probs=softmax_cloud(0.5, 0.15)) for d in DIM_NAMES}
        topo = compute_topology(clouds, cfg)
        self.assertTrue(np.allclose(topo.coupling_matrix, topo.coupling_matrix.T))


class TestEmergence(unittest.TestCase):
    def test_trigger(self):
        cfg = _cfg()
        clouds = {d: ProbabilityCloud(probs=softmax_cloud(0.0, 0.15)) for d in DIM_NAMES}
        should, reason = check_dimension_emergence(clouds, 0.6, 1.2, False, cfg)
        self.assertTrue(should)
        self.assertIn("无法降熵", reason)

    def test_no_trigger(self):
        cfg = _cfg()
        clouds = {d: ProbabilityCloud(probs=softmax_cloud(0.7, 0.15)) for d in DIM_NAMES}
        should, _ = check_dimension_emergence(clouds, 0.3, 1.5, False, cfg)
        self.assertFalse(should)


class TestBagua(unittest.TestCase):
    def test_all_positive(self):
        self.assertEqual(encode_bagua(1, 1, 1), "乾")

    def test_all_negative(self):
        self.assertEqual(encode_bagua(-1, -1, -1), "坤")

    def test_mixed(self):
        self.assertIn(encode_bagua(1, -1, 0), ("离", "巽", "艮"))


class TestStateIndex(unittest.TestCase):
    def test_range(self):
        idx = state_to_index(np.array([0.3] * 9))
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, 19683)


class TestEnginePipeline(unittest.TestCase):
    def test_full_pipeline(self):
        cfg = _cfg()
        engine = SFWEngine(cfg)
        scores = {d: 0.4 for d in DIM_NAMES}
        engine.encode_dimensions(scores)
        engine.compute_fractal(conflict_count=1)
        engine.compute_fission()
        engine.compute_topology_state()
        engine.compute_emergence()
        engine.compute_existence()
        report = engine.get_full_report()
        for key in ("probability_clouds", "fractal", "fissions", "topology", "emergence", "existence"):
            self.assertIn(key, report)

    def test_out_of_range_score_rejected(self):
        cfg = _cfg()
        engine = SFWEngine(cfg)
        with self.assertRaises(ValueError):
            engine.encode_dimensions({"天": 1.5, "地": 0, "人": 0, "正": 0, "反": 0, "合": 0, "归": 0, "守": 0, "进": 0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
