import math
from random import random

import torch
import torch.nn.functional as F
import torch.nn as nn

from tqdm.auto import tqdm


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


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


class PointDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        beta_schedule: str = "linear",
        timesteps: int = 1000,
        loss_type: str = "l2",
    ) -> None:
        super().__init__()

        self.model = model
        self.loss_type = loss_type

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
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t - 1} | x_t, x_0)
        # equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
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

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_0, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t):
        pred_noise = self.model(x, t)
        x_0 = self.predict_start_from_noise(x, t, pred_noise)
        return pred_noise, x_0

    def p_mean_variance(self, x, t):
        _, x_0 = self.model_predictions(x, t)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_0=x_0, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_0

    def p_sample(self, x, t):
        batch_size, *_, device = *x.shape, x.device
        batched_times = torch.full((batch_size,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_0 = self.p_mean_variance(
            x=x, t=batched_times
        )
        noise = torch.randn_like(x) if t > 0.0 else 0.0  # no noise if t = 0
        pred_x = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_x, x_0

    def q_sample(self, x_0, t, noise=None):
        noise = torch.randn_like(x_0)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )

    @torch.no_grad()
    def sample(self, shape):
        batch, device = shape[0], self.betas.device
        x = torch.randn(shape, device=device)
        x_0 = None
        for t in tqdm(
            reversed(0, self.num_timesteps),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            x, x_0 = self.p_sample(x, t)
        return x

    def p_losses(self, x_0, t, noise=None):
        batch_size, _, dim = x_0.shape
        noise = torch.randn_like(x_0)

        # noise sample
        x = self.q_sample(x_0=x_0, t=t, noise=noise)

        # predict and take gradient step
        model_out = self.model(x, t)

        loss = self.loss_fn(model_out, noise, reduction="mean")
        return loss

    def forward(self, x, *args, **kwargs):
        batch_size, num_points, dim, device = (*x.shape, x.device)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)
