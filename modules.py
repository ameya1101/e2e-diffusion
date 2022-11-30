import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return emb


class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_emb) -> None:
        super().__init__()

        sin_pos_emb = SinusoidalPosEmb(dim_emb)
        self.time_mlp = nn.Sequential(
            sin_pos_emb,
            nn.Linear(dim_emb, dim_emb),
            nn.GELU(),
            nn.Linear(dim_emb, dim_in),
        )

        self.layer = nn.Linear(dim_in, dim_out)

    def forward(self, x, time):
        t = self.time_mlp(time)
        x = x + t
        return self.layer(x)


class PointNet(nn.Module):
    def __init__(self, dim=3, residual=False) -> None:
        super().__init__()
        self.residual = residual

        time_dim = dim * 4
        self.layers = nn.ModuleList(
            [
                ConcatLinear(3, 64, time_dim),
                ConcatLinear(64, 128, time_dim),
                ConcatLinear(128, 256, time_dim),
                ConcatLinear(256, 128, time_dim),
                ConcatLinear(128, 64, time_dim),
                ConcatLinear(64, 3, time_dim),
            ]
        )

    def forward(self, x, time):
        x_ = x

        for layer in self.layers:
            x = layer(x, t)
            x = F.relu(x)

        if self.residual:
            return x + x_
        else:
            return x


if __name__ == "__main__":
    x = torch.randint(1, 10, size=(1, 4, 3))
    t = torch.tensor([3], dtype=torch.long)
    model = PointNet(residual=True)
    y = model(x, t)
    print(y)
    print(y.shape)
