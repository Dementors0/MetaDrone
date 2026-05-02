import unittest

import torch

from worker_context_features import (
    WORKER_CONTEXT_FEATURE_DIM,
    WORKER_CONTEXT_LAGS,
    extract_worker_context_features,
)


class WorkerContextFeaturesTest(unittest.TestCase):
    def test_shape_and_rotation_transform(self):
        p_history = [
            torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([[-0.5, 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        ]
        p_target = torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float32)
        # body x -> world +y, body y -> world -x, body z -> world +z
        R_current = torch.tensor(
            [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        )

        feat = extract_worker_context_features(
            p_history_list=p_history,
            p_target=p_target,
            R_current=R_current,
        )
        self.assertEqual(feat.shape, torch.Size([1, WORKER_CONTEXT_FEATURE_DIM]))

        goal_dir_body = feat[0, 0:3]
        self.assertTrue(torch.allclose(goal_dir_body, torch.tensor([1.0, 0.0, 0.0]), atol=1e-6))

        goal_dist_log = feat[0, 3]
        self.assertAlmostEqual(goal_dist_log.item(), torch.log1p(torch.tensor(2.0)).item(), places=6)

        # lag=1 first delta_p_body block lives at indices [4:7]
        delta_p_body_lag1 = feat[0, 4:7]
        self.assertTrue(torch.allclose(delta_p_body_lag1, torch.tensor([0.0, 0.5, 0.0]), atol=1e-6))

        # delta_goal_dist for lag=1 is the first element in tail block.
        delta_goal_dist_lag1 = feat[0, 19]
        self.assertGreater(delta_goal_dist_lag1.item(), 0.0)

    def test_goal_distance_semantics(self):
        p_history = [
            torch.tensor([[0.0, -1.0, 0.0], [0.0, 3.5, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=torch.float32),
        ]
        p_target = torch.tensor([[0.0, 4.0, 0.0], [0.0, 4.0, 0.0]], dtype=torch.float32)
        R_current = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)

        feat = extract_worker_context_features(
            p_history_list=p_history,
            p_target=p_target,
            R_current=R_current,
        )

        goal_dist_log = feat[:, 3]
        self.assertGreater(goal_dist_log[0].item(), goal_dist_log[1].item())

        delta_goal_dist_lag1 = feat[:, 19]
        self.assertGreater(delta_goal_dist_lag1[0].item(), 0.0)
        self.assertLess(delta_goal_dist_lag1[1].item(), 0.0)

    def test_short_history_uses_earliest_frame(self):
        p_now = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        p_history = [p_now]
        p_target = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32)
        R_current = torch.eye(3, dtype=torch.float32).unsqueeze(0)

        feat = extract_worker_context_features(
            p_history_list=p_history,
            p_target=p_target,
            R_current=R_current,
        )

        self.assertEqual(feat.shape[-1], WORKER_CONTEXT_FEATURE_DIM)
        self.assertTrue(torch.isfinite(feat).all())
        self.assertTrue(torch.allclose(feat[0, 4:19], torch.zeros(15), atol=1e-6))
        self.assertTrue(torch.allclose(feat[0, 19:24], torch.zeros(5), atol=1e-6))

    def test_zero_distance_is_stable(self):
        p_now = torch.tensor([[3.0, -2.0, 0.5]], dtype=torch.float32)
        p_history = [p_now, p_now]
        p_target = p_now.clone()
        R_current = torch.eye(3, dtype=torch.float32).unsqueeze(0)

        feat = extract_worker_context_features(
            p_history_list=p_history,
            p_target=p_target,
            R_current=R_current,
        )

        self.assertTrue(torch.isfinite(feat).all())
        self.assertTrue(torch.allclose(feat[0, 0:3], torch.zeros(3), atol=1e-6))
        self.assertGreaterEqual(feat[0, 3].item(), 0.0)

    def test_default_lag_configuration_is_expected(self):
        self.assertEqual(WORKER_CONTEXT_LAGS, (1, 2, 4, 8, 16))


if __name__ == "__main__":
    unittest.main()
