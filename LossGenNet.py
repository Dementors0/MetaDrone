import torch
from torch import nn

class LossGenNet(nn.Module):
    """损失生成网络: Output 4 dynamic weights based on visual & state context"""

    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        # 深度图处理: [B, 1, 12, 16] -> Flatten
        self.feature_embed = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # 输入维度: 768 (Image) + state_dim
        input_dim = 16 * 6 * 8 + state_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),  # [Vel, Dir, Obs, Smooth]
            nn.Softmax(dim=-1)
        )

    def forward(self, depth_feat, state):
        d_emb = self.feature_embed(depth_feat)
        x = torch.cat([d_emb, state], dim=-1)
        return self.net(x)