import torch
from torch import nn

class LossGenNet(nn.Module):
    """
    升级版损失生成网络: 
    CNN (视觉) + MLP (状态) -> Token Fusion -> Transformer (时序记忆) -> MLP (输出)
    """

    def __init__(
        self,
        state_dim,
        hidden_dim=128,
        nhead=4,
        num_layers=2,
        max_seq_len=64,
        output_temperature=1.0,
        weight_floor=0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.output_temperature = float(max(output_temperature, 1e-3))
        self.weight_floor = float(min(max(weight_floor, 0.0), 0.249))
        
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
        self.visual_proj = nn.Linear(768, hidden_dim)
        
        # 2. 状态特征映射 (把物理状态映射到高维空间)
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh()
        )

        # 3. 时序编码器（因果 Mask，保证只看当前及历史）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

        # 4. 输出头
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 4),  # [Vel, Dir, Obs, Expl]
        )

    def forward(self, depth_feat, state, hx=None):
        """
        参数:
            depth_feat: [B, 1, 12, 16] 深度图
            state: [B, state_dim] 物理状态
            hx: [B, T_mem, hidden_dim] 历史记忆 token (如果是第一步则为 None)
        返回:
            weights: [B, 4]
            hx: [B, T_mem, hidden_dim] 更新后的记忆序列
        """
        # 1. 提取视觉特征
        v_emb = self.visual_net(depth_feat)
        v_emb = self.visual_proj(v_emb)
        
        # 2. 提取状态特征
        s_emb = self.state_proj(state)
        
        # 3. 生成当前 token
        current_token = self.pre_norm(v_emb + s_emb)

        # 4. 时序记忆更新
        if hx is None:
            seq = current_token.unsqueeze(1)
        else:
            seq = torch.cat([hx, current_token.unsqueeze(1)], dim=1)
            if seq.size(1) > self.max_seq_len:
                seq = seq[:, -self.max_seq_len:]

        seq_len = seq.size(1)
        x_in = seq + self.pos_emb[:, :seq_len]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x_in.device, dtype=torch.bool),
            diagonal=1,
        )
        x_out = self.transformer(x_in, mask=causal_mask)
        last_token = self.out_norm(x_out[:, -1])
        
        # 5. 生成权重（无约束输出，不做归一化）
        raw = self.head(last_token)
        weights = raw

        # 返回 weights 和 新的记忆序列
        return weights, seq
