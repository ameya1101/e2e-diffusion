import torch
import torch.nn as nn


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class TimeEmbedding(nn.Module):
    def __init__(self, num_embed) -> None:
        super(TimeEmbedding, self).__init__()
        self.num_embed = num_embed
        self.projections = self._gaussian_fourier_projection()

        self.mlp = nn.Sequential(
            nn.Linear(num_embed // 2, 2 * num_embed),
            nn.LeakyReLU(),
            nn.Linear(2 * num_embed, num_embed),
        )

    def forward(self, x):
        angles = x * self.projections.to(x.device)
        time_embedding = torch.concat([torch.sin(angles), torch.cos(angles)], dim=-1)
        time_embedding = self.mlp(time_embedding)
        return time_embedding

    def _gaussian_fourier_projection(self):
        half_dim = self.num_embed // 4
        emb = torch.log(torch.tensor(10000)) / (half_dim - 1)
        freq = torch.exp(
            -emb * torch.arange(start=0, end=half_dim, dtype=torch.float32)
        )
        return freq


class TransformerLayer(nn.Module):
    def __init__(self, projection_dim, num_heads, dropout=0.1) -> None:
        super(TransformerLayer, self).__init__()
        self.layernorm1 = nn.LayerNorm(projection_dim, eps=1e-6)
        self.mha = nn.MultiheadAttention(
            embed_dim=projection_dim // num_heads, num_heads=num_heads, dropout=dropout
        )
        self.layernorm2 = nn.LayerNorm(projection_dim, eps=1e-6)
        self.linear1 = nn.Linear(projection_dim, 2 * projection_dim)
        self.linear2 = nn.Linear(2 * projection_dim, projection_dim)

    def forward(self, x):
        x1 = self.layernorm1(x)
        attn_out, _ = self.mha(query=x1, key=x1, value=x1)
        x2 = attn_out + x
        x3 = self.layernorm2(x2)
        x3 = nn.functional.gelu(self.linear1(x3))
        x3 = nn.functional.gelu(self.linear2(x3))
        return x2 + x3


class DeepSetsAttention(nn.Module):
    def __init__(
        self,
        num_feats=3,
        num_heads=4,
        num_transformers=4,
        projection_dim=32,
    ) -> None:
        super(DeepSetsAttention, self).__init__()
        self.num_feats = num_feats
        self.num_heads = num_heads
        self.num_transformers = num_transformers
        self.projection_dim = projection_dim

        # time information is used as an additional feature for all particles
        self.time_mlp = nn.Sequential(
            nn.Linear(projection_dim, 2 * projection_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * projection_dim, projection_dim),
        )

        self.encoder_mlp = nn.Sequential(
            TimeDistributed(nn.Linear(projection_dim + num_feats, projection_dim)),
            TimeDistributed(nn.LeakyReLU())
        )
        self.time_linear = TimeDistributed(nn.Linear(projection_dim, projection_dim))

        self.transfomer_layers = nn.ModuleDict(
            {
                f"transformer_{i}": TransformerLayer(projection_dim=projection_dim, num_heads=num_heads) for i in range(num_transformers)
            }
        )

        self.layernorm = nn.LayerNorm(projection_dim, eps=1e-6)

        self.post_mlp = nn.Sequential(
            TimeDistributed(nn.Linear(2 * projection_dim, projection_dim)),
            TimeDistributed(nn.LeakyReLU()),
            TimeDistributed(nn.Linear(projection_dim, num_feats)),
        )

    def forward(self, x, time_embed):
        time = self.time_mlp(time_embed)  # (B, 64)
        time = torch.reshape(time, (-1, 1, time.shape[-1]))  # (B, 1, 64)
        time = torch.tile(time, (1, x.shape[1], 1))  # (B, N, 64)

        tdd = torch.concat([x, time], dim=-1)  # (B, N, 64 + 3)
        tdd = self.encoder_mlp(tdd)  # (B, N, 64)
        encoded_patches = TimeDistributed(self.time_linear)(tdd)
        for i in range(self.num_transformers):
            encoded_patches = self.transfomer_layers[f"transformer_{i}"](encoded_patches)

        representation = self.layernorm(encoded_patches)
        outputs = self.post_mlp(torch.concat([representation, tdd], dim=-1))  # (B, N, 3)
        return outputs
