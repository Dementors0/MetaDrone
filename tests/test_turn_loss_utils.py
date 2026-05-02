import unittest

import torch

from turn_loss_utils import compute_turn_preference_loss_xy_windowed


class TurnPreferenceLossXYWindowedTest(unittest.TestCase):
    def test_z_variation_does_not_change_turn_loss(self):
        v_up = torch.tensor(
            [
                [[1.0, 0.0, 0.0]],
                [[1.0, 0.0, 8.0]],
            ],
            dtype=torch.float32,
        )
        v_down = torch.tensor(
            [
                [[1.0, 0.0, 0.0]],
                [[1.0, 0.0, -8.0]],
            ],
            dtype=torch.float32,
        )

        loss_up = compute_turn_preference_loss_xy_windowed(v_up, speed_threshold=0.2, window=8)
        loss_down = compute_turn_preference_loss_xy_windowed(v_down, speed_threshold=0.2, window=8)

        self.assertTrue(torch.allclose(loss_up, loss_down, atol=1e-6))
        self.assertAlmostEqual(loss_up[1, 0].item(), 1.0, places=6)

    def test_sustained_arc_beats_small_wiggle(self):
        arc = torch.tensor(
            [
                [[1.0, 0.0, 0.0]],
                [[0.70710677, 0.70710677, 0.0]],
                [[0.0, 1.0, 0.0]],
                [[-0.70710677, 0.70710677, 0.0]],
            ],
            dtype=torch.float32,
        )
        wiggle = torch.tensor(
            [
                [[1.0, 0.0, 0.0]],
                [[1.0, 0.1, 0.0]],
                [[1.0, -0.1, 0.0]],
                [[1.0, 0.1, 0.0]],
            ],
            dtype=torch.float32,
        )

        loss_arc = compute_turn_preference_loss_xy_windowed(arc, speed_threshold=0.2, window=8)
        loss_wiggle = compute_turn_preference_loss_xy_windowed(wiggle, speed_threshold=0.2, window=8)
        self.assertLess(loss_arc[3, 0].item(), loss_wiggle[3, 0].item())

    def test_wrap_at_pi_boundary_is_stable(self):
        eps = 0.01
        v_history = torch.tensor(
            [
                [[-1.0, eps, 0.0]],
                [[-1.0, -eps, 0.0]],
            ],
            dtype=torch.float32,
        )
        loss = compute_turn_preference_loss_xy_windowed(v_history, speed_threshold=0.2, window=8)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertGreater(loss[1, 0].item(), 0.95)

    def test_opposite_turns_cancel_in_vector_sum(self):
        # 0 -> +45 -> 0 => (+45, -45) cancels in |sum(dtheta)|.
        v_history = torch.tensor(
            [
                [[1.0, 0.0, 0.0]],
                [[0.70710677, 0.70710677, 0.0]],
                [[1.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        loss = compute_turn_preference_loss_xy_windowed(v_history, speed_threshold=0.2, window=8)
        self.assertGreater(loss[2, 0].item(), 0.9)

    def test_low_horizontal_speed_is_masked_out(self):
        v_history = torch.tensor(
            [
                [[0.05, 0.0, 6.0]],
                [[0.05, 0.0, -6.0]],
            ],
            dtype=torch.float32,
        )

        loss = compute_turn_preference_loss_xy_windowed(v_history, speed_threshold=0.2, window=8)
        self.assertAlmostEqual(loss[1, 0].item(), 0.0, places=6)

    def test_short_prefix_is_stable(self):
        v_history = torch.tensor(
            [
                [[1.0, 1.0, 0.0]],
                [[2.0, 2.0, 0.0]],
            ],
            dtype=torch.float32,
        )

        loss = compute_turn_preference_loss_xy_windowed(v_history, speed_threshold=0.2, window=8)
        self.assertEqual(loss.shape, torch.Size([2, 1]))
        self.assertTrue(torch.isfinite(loss).all())


if __name__ == "__main__":
    unittest.main()
