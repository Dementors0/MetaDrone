import unittest

import torch

from LossGenNet_transformer import LossGenNet


def _safe_normalize(x, eps=1e-6):
    return torch.nn.functional.normalize(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), dim=-1, eps=eps)


class LGNVRefIntegrationTest(unittest.TestCase):
    def test_lgn_forward_outputs_weight_and_vref_shapes(self):
        model = LossGenNet(
            state_dim=13,
            geom_dim=19,
            progress_dim=32,
            hidden_dim=64,
            nhead=4,
            num_layers=1,
            max_seq_len=8,
        )
        model.eval()

        B = 4
        depth = torch.randn(B, 1, 12, 16, dtype=torch.float32)
        state = torch.randn(B, 13, dtype=torch.float32)
        geom = torch.randn(B, 19, dtype=torch.float32)
        progress = torch.randn(B, 32, dtype=torch.float32)

        weights, vref, hx = model(depth, state, geom, progress, hx=None)
        self.assertEqual(weights.shape, torch.Size([B, 7]))
        self.assertEqual(vref.shape, torch.Size([B, 3]))
        self.assertEqual(hx.shape, torch.Size([B, 1, 64]))
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.isfinite(vref).all())
        self.assertTrue(torch.isfinite(hx).all())

    def test_vref_normalization_stable_for_zero_head_output(self):
        model = LossGenNet(
            state_dim=13,
            geom_dim=19,
            progress_dim=32,
            hidden_dim=64,
            nhead=4,
            num_layers=1,
            max_seq_len=8,
        )
        model.eval()
        with torch.no_grad():
            model.vref_head.weight.zero_()
            model.vref_head.bias.zero_()

        B = 3
        depth = torch.randn(B, 1, 12, 16, dtype=torch.float32)
        state = torch.randn(B, 13, dtype=torch.float32)
        geom = torch.randn(B, 19, dtype=torch.float32)
        progress = torch.randn(B, 32, dtype=torch.float32)

        _, vref, _ = model(depth, state, geom, progress, hx=None)
        self.assertTrue(torch.isfinite(vref).all())
        self.assertTrue(torch.allclose(vref, torch.zeros_like(vref), atol=1e-6))

    def test_real_velocity_world_to_body_and_lgn_vref_loss_range(self):
        T, B = 3, 2

        v_real_world = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        )

        r_identity = torch.eye(3, dtype=torch.float32)
        r_yaw90 = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        r_proxy_seq = torch.stack(
            [torch.stack([r_identity, r_yaw90], dim=0) for _ in range(T)],
            dim=0,
        )  # [T, B, 3, 3]

        v_real_body_seq = torch.squeeze(v_real_world[:, :, None, :] @ r_proxy_seq, 2)
        v_real_body_dir = _safe_normalize(v_real_body_seq)

        vref_same = v_real_body_dir.clone()
        loss_same = (1.0 - (v_real_body_dir * vref_same).sum(dim=-1)).clamp(0.0, 2.0)
        self.assertTrue(torch.allclose(loss_same, torch.zeros_like(loss_same), atol=1e-6))

        vref_opposite = -v_real_body_dir
        loss_opposite = (1.0 - (v_real_body_dir * vref_opposite).sum(dim=-1)).clamp(0.0, 2.0)
        self.assertTrue(torch.allclose(loss_opposite, torch.full_like(loss_opposite, 2.0), atol=1e-6))

        random_vref = _safe_normalize(torch.randn_like(v_real_body_seq))
        loss_random = (1.0 - (v_real_body_dir * random_vref).sum(dim=-1)).clamp(0.0, 2.0)
        self.assertTrue(torch.isfinite(loss_random).all())
        self.assertGreaterEqual(loss_random.min().item(), 0.0)
        self.assertLessEqual(loss_random.max().item(), 2.0)


if __name__ == "__main__":
    unittest.main()
