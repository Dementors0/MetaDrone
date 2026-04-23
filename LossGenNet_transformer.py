import torch
from torch import nn

class LossGenNet(nn.Module):
    """
    升级版损失生成网络:
    CNN (视觉) + MLP (状态) -> Token Fusion -> Transformer (时序记忆) -> MLP (输出)
    输出为 proxy reference 的残差修正量（而非旧版 loss weights）。
    """

    def __init__(
        self,
        state_dim,
        geom_dim=19,
        progress_dim=8,
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
        # Kept only for backward compatibility with older checkpoints/configs.
        self.output_temperature = float(output_temperature)
        self.weight_floor = float(weight_floor)
        
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

        # 2.1 显式几何/风险特征映射
        self.geom_proj = nn.Sequential(
            nn.Linear(geom_dim, hidden_dim),
            nn.Tanh()
        )

        # 2.2 近期进展/卡住特征映射
        self.progress_proj = nn.Sequential(
            nn.Linear(progress_dim, hidden_dim),
            nn.Tanh()
        )

        # 2.3 多模态融合：concat -> projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
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
        # [delta_dir(3), delta_speed(1), delta_yaw(3), delta_margin(1)] -> total 8
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 8),
        )

    def forward(self, depth_feat, state, geom_feat, progress_feat, hx=None):
        """
        参数:
            depth_feat: [B, 1, 12, 16] 深度图
            state: [B, state_dim] 物理状态
            geom_feat: [B, geom_dim] 深度图几何/风险统计特征
            progress_feat: [B, progress_dim] 近期进展/卡住统计特征
            hx: [B, T_mem, hidden_dim] 历史记忆 token (如果是第一步则为 None)
        返回:
            delta_refs: [B, 8]
            hx: [B, T_mem, hidden_dim] 更新后的记忆序列
        """
        # 1. 提取视觉特征
        v_emb = self.visual_net(depth_feat)
        v_emb = self.visual_proj(v_emb)
        
        # 2. 提取状态特征
        s_emb = self.state_proj(state)

        # 3. 提取显式几何与进展特征
        g_emb = self.geom_proj(geom_feat)
        p_emb = self.progress_proj(progress_feat)
        
        # 4. 生成当前 token（多模态拼接后再融合）
        fused = torch.cat([v_emb, s_emb, g_emb, p_emb], dim=-1)
        current_token = self.pre_norm(self.fusion_proj(fused))

        # 5. 时序记忆更新
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
        
        # 6. 输出 reference 残差（范围约束在训练脚本里结合 naive reference 处理）
        delta_refs = self.head(last_token)

        # 返回 delta_refs 和 Transformer 编码后的记忆序列
        return delta_refs, x_out
