import math
import unittest

import torch

from turn_loss_utils import compute_direction_stability_loss_3d


class DirectionStabilityLoss3DTest(unittest.TestCase):
    def test_loss_is_monotonic_with_three_dimensional_angle(self):
        angles_deg = [0.0, 5.0, 10.0, 30.0, 90.0, 180.0]
        losses = []
        for angle_deg in angles_deg:
            angle = math.radians(angle_deg)
            velocity = torch.tensor(
                [
                    [[1.0, 0.0, 0.0]],
                    [[math.cos(angle), 0.0, math.sin(angle)]],
                ],
                dtype=torch.float64,
            )
            loss = compute_direction_stability_loss_3d(velocity)
            losses.append(loss[1, 0].item())

        self.assertAlmostEqual(losses[0], 0.0, places=12)
        for previous, current in zip(losses, losses[1:]):
            self.assertLess(previous, current)
        self.assertAlmostEqual(losses[-1], 1.0, places=6)

    def test_vertical_direction_change_is_penalized(self):
        velocity = torch.tensor(
            [
                [[1.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        loss = compute_direction_stability_loss_3d(velocity)
        self.assertGreater(loss[1, 0].item(), 0.45)

    def test_same_direction_with_different_speed_has_zero_loss(self):
        velocity = torch.tensor(
            [
                [[0.5, 1.0, -0.5]],
                [[1.0, 2.0, -1.0]],
            ],
            dtype=torch.float32,
        )
        loss = compute_direction_stability_loss_3d(velocity)
        self.assertAlmostEqual(loss[1, 0].item(), 0.0, places=6)

    def test_soft_speed_gate_matches_threshold_weight(self):
        velocity = torch.tensor(
            [
                [[0.2, 0.0, 0.0]],
                [[0.0, 0.2, 0.0]],
            ],
            dtype=torch.float64,
        )
        loss = compute_direction_stability_loss_3d(
            velocity,
            speed_threshold=0.2,
            speed_softness=0.01,
            soft_angle_deg=10.0,
        )
        beta = math.radians(10.0)
        angle_loss = (0.5 * math.pi - 0.5 * beta) / (
            math.pi - 0.5 * beta
        )
        self.assertAlmostEqual(loss[1, 0].item(), 0.25 * angle_loss, places=6)

    def test_soft_speed_gate_keeps_velocity_gradient(self):
        velocity = torch.tensor(
            [
                [[0.2, 0.0, 0.0]],
                [[0.0, 0.2, 0.0]],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        loss = compute_direction_stability_loss_3d(velocity).sum()
        loss.backward()

        self.assertIsNotNone(velocity.grad)
        self.assertTrue(torch.isfinite(velocity.grad).all())
        self.assertGreater(velocity.grad.abs().sum().item(), 0.0)

    def test_low_speed_suppresses_but_does_not_hard_cut_loss(self):
        low_speed = torch.tensor(
            [
                [[0.15, 0.0, 0.0]],
                [[0.0, 0.15, 0.0]],
            ],
            dtype=torch.float64,
        )
        high_speed = torch.tensor(
            [
                [[0.25, 0.0, 0.0]],
                [[0.0, 0.25, 0.0]],
            ],
            dtype=torch.float64,
        )
        low_loss = compute_direction_stability_loss_3d(low_speed)[1, 0]
        high_loss = compute_direction_stability_loss_3d(high_speed)[1, 0]

        self.assertGreater(low_loss.item(), 0.0)
        self.assertLess(low_loss.item(), high_loss.item() * 1e-3)

    def test_single_step_is_zero_and_finite(self):
        velocity = torch.tensor([[[1.0, 2.0, 3.0]]])
        loss = compute_direction_stability_loss_3d(velocity)
        self.assertEqual(loss.shape, torch.Size([1, 1]))
        self.assertTrue(torch.isfinite(loss).all())
        self.assertEqual(loss[0, 0].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
