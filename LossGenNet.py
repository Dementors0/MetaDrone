import torch
from torch import nn
import torch.nn.functional as F

class LossGenNet(nn.Module):
    """
    升级版损失生成网络: 
    CNN (视觉) + MLP (状态) -> Fusion -> GRU (记忆) -> MLP (输出)
    """

    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        
        # 1. 增强视觉特征提取 (3层卷积)
        # Input: [B, 1, 12, 16]
        self.visual_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # -> [32, 12, 16]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),                              # -> [32, 6, 8]
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> [64, 6, 8]
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # -> [64, 3, 4]
            nn.LeakyReLU(0.1),
            
            nn.Flatten()  # 64 * 3 * 4 = 768
        )
        
        # 2. 状态特征映射 (把物理状态映射到高维空间)
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh() # 限制数值范围
        )

        # 计算融合后的维度: 视觉(768) + 状态(64)
        self.fusion_dim = 768 + 64

        # 3. 时序记忆单元 (GRU)
        self.gru = nn.GRUCell(self.fusion_dim, hidden_dim)

        # 4. 输出头 (多层 MLP)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 5),  # [Vel, Dir, Obs, Smooth, etc.]
            nn.Softmax(dim=-1)
        )

    def forward(self, depth_feat, state, hx=None):
        """
        参数:
            depth_feat: [B, 1, 12, 16] 深度图
            state: [B, state_dim] 物理状态
            hx: [B, hidden_dim] 上一步的隐状态 (如果是第一步则为 None)
        返回:
            weights: [B, 5]
            hx: [B, hidden_dim] 更新后的隐状态
        """
        # 1. 提取视觉特征
        v_emb = self.visual_net(depth_feat)
        
        # 2. 提取状态特征
        s_emb = self.state_proj(state)
        
        # 3. 特征拼接
        combined = torch.cat([v_emb, s_emb], dim=-1)
        
        # 4. GRU 记忆更新
        if hx is None:
            # 自动初始化为全0
            hx = torch.zeros(combined.size(0), self.gru.hidden_size, device=combined.device)
        
        hx = self.gru(combined, hx)
        
        # 5. 生成权重
        weights = self.head(hx)
        
        # 返回 weights 和 新的 hidden_state
        return weights, hx