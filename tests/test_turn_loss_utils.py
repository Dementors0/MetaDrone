import unittest

import torch

from turn_loss_utils import compute_turn_preference_loss_xy_unit


class TurnPreferenceLossXYUnitTest(unittest.TestCase):
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

        loss_up = compute_turn_preference_loss_xy_unit(v_up, speed_threshold=0.2)
        loss_down = compute_turn_preference_loss_xy_unit(v_down, speed_threshold=0.2)

        self.assertTrue(torch.allclose(loss_up, loss_down, atol=1e-6))
        self.assertAlmostEqual(loss_up[1, 0].item(), 1.0, places=6)

    def test_turn_angle_changes_reduce_consistency(self):
        # Batch 0: 90 deg turn -> loss 0.5
        # Batch 1: 180 deg turn -> loss 0.0
        v_history = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        )

        loss = compute_turn_preference_loss_xy_unit(v_history, speed_threshold=0.2)
        self.assertAlmostEqual(loss[1, 0].item(), 0.5, places=6)
        self.assertAlmostEqual(loss[1, 1].item(), 0.0, places=6)

    def test_low_horizontal_speed_is_masked_out(self):
        v_history = torch.tensor(
            [
                [[0.05, 0.0, 6.0]],
                [[0.05, 0.0, -6.0]],
            ],
            dtype=torch.float32,
        )

        loss = compute_turn_preference_loss_xy_unit(v_history, speed_threshold=0.2)
        self.assertAlmostEqual(loss[1, 0].item(), 0.0, places=6)

    def test_same_horizontal_direction_is_scale_invariant(self):
        v_history = torch.tensor(
            [
                [[1.0, 1.0, 0.0]],
                [[2.0, 2.0, 0.0]],
                [[10.0, 10.0, 0.0]],
            ],
            dtype=torch.float32,
        )

        loss = compute_turn_preference_loss_xy_unit(v_history, speed_threshold=0.2)
        self.assertAlmostEqual(loss[1, 0].item(), 1.0, places=6)
        self.assertAlmostEqual(loss[2, 0].item(), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
