import math
import unittest

import torch

from utils.tensor_utils import decode_worker_action


class WorkerDirectAccelerationTest(unittest.TestCase):
    def test_v2_decodes_three_acceleration_channels_and_yaw(self):
        act = torch.tensor([[1.0, 2.0, 3.0, 0.5]], dtype=torch.float32)
        rotation = torch.eye(3, dtype=torch.float32).unsqueeze(0)

        acceleration, yaw_rate = decode_worker_action(act, rotation, math.pi)

        self.assertTrue(torch.allclose(acceleration, act[:, :3]))
        self.assertTrue(torch.allclose(yaw_rate, torch.tanh(act[:, 3:4]) * math.pi))

    def test_legacy_decodes_acceleration_without_velocity_or_yaw(self):
        act = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        yaw_90 = torch.tensor(
            [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        )

        acceleration, yaw_rate = decode_worker_action(act, yaw_90, math.pi)

        self.assertTrue(torch.allclose(acceleration, torch.tensor([[0.0, 1.0, 0.0]])))
        self.assertIsNone(yaw_rate)


if __name__ == '__main__':
    unittest.main()
