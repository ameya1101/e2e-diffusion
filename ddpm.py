import math
from random import random

import torch
import torch.nn.functional as F
import torch.nn as nn

from tqdm.auto import tqdm
from .modules import ConcatSquishLinear

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=8e-3):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.99999)


class PointWiseNet(nn.Module):
    def __init__(self, context_dim, residual) -> None:
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = nn.ModuleList([
            ConcatSquishLinear(3, 128, context_dim + 3),
            ConcatSquishLinear(128, 256, context_dim + 3),
            ConcatSquishLinear(256, 512, context_dim + 3),
            ConcatSquishLinear(512, 256, context_dim + 3),
            ConcatSquishLinear(126, 128, context_dim + 3),
            ConcatSquishLinear(128, 3, context_dim + 3),
        ])

    def forward(self, x, beta, context):
        """
        Args:
            x: Point clouds at some timestep t, (B, N, d)
            beta: Time (B, )
            context: Shape embedding (B, F)
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)         # (B, 1, 1)
        context = context.view(batch_size, 1, 1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1) # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)
        
        if self.residual:
            return x + out
        else:
            return out


class PointDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        beta_schedule: str = "linear",
        timesteps: int = 1000
    ) -> None:
        super().__init__()

        self.model = model
        self.dim = model.dim

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise NotImplementedError(
                f"variance schedule {beta_schedule} not implemented"
            )

        alphas = 1.0 - betas
        # alphas_cumprod = alpha_bars
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )
        register_buffer("betas", betas)
        register_buffer("alphas", alphas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for forward diffusion q(x_t | x_{t - 1})
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recip1m_alphas_cumprod", torch.sqrt(1.0 / (1 - alphas_cumprod))
        )

        # calculations for posterior q(x_{t - 1} | x_t, x_0)
        # equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        posterior_variance = (
            betas * ((1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        )
        register_buffer("posterior_variance", posterior_variance)
        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )


    def get_loss(self, x0, context, t=None):
        """
        Args:
            x: Input point cloud, (B, N, d)
            context: Shape latent, (B, F)
        """
        batch_size, _, point_dim, device = x0.size(), x0.device
        if t == None:
            t = torch.randint(1, self.num_timesteps + 1, (batch_size,), device=device).long()

        beta = self.betas[t]
        c0 = self.sqrt_alphas_cumprod[t]             # (B, 1, 1)
        c1 = self.sqrt_one_minus_alphas_cumprod[t]   # (B, 1, 1)
        
        ε_rand = torch.randn_like(x0).view(-1, 1, 1)                              
        ε_theta = self.model(c0 * x + c1 * ε_rand,  beta=beta, context=context).view(-1, 1, 1)

        loss = F.mse_loss(ε_theta.view(-1, point_dim), ε_rand.view(-1, point_dim), reduction='mean')
        return loss

    @torch.no_grad()
    def sample(self, num_points, context, point_dim=3, return_traj=False):
        batch_size, device = context.size(0), context.device
        x_T = torch.randn([batch_size, num_points, point_dim]).to(device)
        traj = {self.num_timesteps: x_T}
        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            sigma = self.posterior_variance[t]
            c0 = self.sqrt_recip_alphas_cumprod
            c1 = self.betas[t] * self.sqrt_recip1m_alphas_cumprod[t]

            x_t = traj[t]
            beta = self.betas[[t] * batch_size]
            ε_theta = self.model(x_t, beta=beta, context=context)
            x_next = c0 * (x_t - c1 * ε_theta) +  sigma * z
            traj[t - 1] = x_next.detach()
            traj[t] = traj[t].cpu()
            if not return_traj:
                del traj[t]
            
        if return_traj:
            return traj
        else:
            return traj[0]

    def forward(self, x, *args, **kwargs):
        batch_size, num_points, dim, device = (*x.shape, x.device)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        return self.get_loss(x, t, *args, **kwargs)


if __name__ == '__main__':
    from modules import PointNet

    model = PointNet(residual=True)
    diffusion = PointDiffusion(model = model)
    x = torch.randn(1, 4, 3)
    print(diffusion(x))

    x = diffusion.sample(5)
    print(x)