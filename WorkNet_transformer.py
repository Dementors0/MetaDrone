import torch
from torch import nn

def g_decay(x, alpha):
    return x * alpha + x.detach() * (1 - alpha)

class WorkNet(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4, d_model=192, nhead=6, num_layers=2, max_seq_len=64) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),  # 1, 12, 16 -> 32, 6, 8
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False), #  32, 6, 8 -> 64, 4, 6
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False), #  64, 4, 6 -> 128, 2, 4
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128 * 2 * 4, d_model, bias=False),
        )
        self.v_proj = nn.Linear(dim_obs, d_model)
        self.v_proj.weight.data.mul_(0.5)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        self.fc = nn.Linear(d_model, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)
        self.ln = nn.LayerNorm(d_model)

    def reset(self):
        pass

    def forward(self, x: torch.Tensor, v, hx=None):
        img_feat = self.stem(x)
        current_token = self.act(img_feat + self.v_proj(v))

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
        last_token = self.ln(x_out[:, -1])
        act = self.fc(self.act(last_token))
        hx = seq
        return act, None, hx


if __name__ == '__main__':
    WorkNet()
