import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from modules import TimeEmbedding, DeepSetsAttention


class PointDiffusion(nn.Module):
    def __init__(self, num_deposits=300, model_config=None, device=None) -> None:
        super(PointDiffusion, self).__init__()

        if model_config is None:
            raise ValueError(f"Argument model_config cannot be {model_config}")
        self.config = model_config
        self.num_features = self.config["NUM_FEATS"]
        self.num_embed = self.config["NUM_EMBED"]
        self.num_steps = self.config["MAX_STEPS"]
        self.max_particles = num_deposits

        self.timesteps = (
            torch.arange(start=0, end=self.num_steps + 1, dtype=torch.float32, device=device)
            / self.num_steps
            + 8e-3
        )
        alphas = self.timesteps / (1 + 8e-3) * np.pi / 2.0
        alphas = torch.cos(alphas) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        self.betas = torch.clamp(betas, min=0, max=0.999)
        alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.concat(
            [torch.ones(1, dtype=torch.float32, device=device), self.alphas_cumprod[:-1]], dim=0
        )
        self.posterior_variance = (
            self.betas * (1 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

        self.time_embedder = TimeEmbedding(num_embed=self.num_embed)

        self.particle_model = DeepSetsAttention(
            num_feats=self.num_features,
            num_heads=1,
            num_transformers=8,
            projection_dim=self.num_embed,
        )

    def forward(self, x):
        random_t = torch.randint(low=0, high=self.num_steps, size=(x.shape[0], 1), device=x.device)
        alpha = torch.gather(
            torch.sqrt(self.alphas_cumprod), index=random_t.squeeze(), dim=0
        )
        sigma = torch.gather(
            torch.sqrt(1 - self.alphas_cumprod), index=random_t.squeeze(), dim=0
        )
        sigma = torch.clamp(sigma, min=1e-3, max=0.999)

        alpha_reshape = torch.reshape(alpha, shape=(-1, 1, 1))
        sigma_reshape = torch.reshape(sigma, shape=(-1, 1, 1))

        z = torch.randn_like(x)
        perturbed_x = alpha_reshape * x + z * sigma_reshape
        t_embedding = self.time_embedder(random_t)

        score = self.particle_model(perturbed_x, t_embedding)
        x_recon = alpha_reshape * x - sigma_reshape * score

        v = alpha_reshape * z - sigma_reshape * x
        loss = 0.7 * nn.functional.mse_loss(score, v) + 0.3 * nn.functional.mse_loss(x, x_recon)
        return loss


class DDIMSampler(nn.Module):
    def __init__(
        self,
        diffusion: PointDiffusion,
        num_samples: int,
        model: nn.Module = None,
        data_shape=None,
        device=None
    ) -> None:
        super(DDIMSampler, self).__init__()
        self.diffusion = diffusion
        self.model = model if model is not None else self.diffusion.particle_model
        self.num_samples = num_samples
        self.num_steps = self.diffusion.num_steps
        self.data_shape = (self.num_samples,) + data_shape
        self.device = device
        self.init_x = torch.randn(self.data_shape).to(device)

    def sample(self):
        x = self.init_x
        with torch.no_grad():
            for time_step in torch.arange(start=self.num_steps - 1, end=0, step=-1):
                batched_time_step = (
                    torch.ones((self.num_samples, 1), dtype=torch.int64) * time_step
                )
                z = torch.randn_like(x)

                alpha = torch.gather(
                    torch.sqrt(self.diffusion.alphas_cumprod),
                    index=batched_time_step.squeeze(),
                    dim=0,
                ).to(self.device)
                sigma = torch.gather(
                    torch.sqrt(1.0 - self.diffusion.alphas_cumprod),
                    index=batched_time_step.squeeze(),
                    dim=0,
                ).to(self.device)

                t_embedding = self.diffusion.time_embedder(batched_time_step.to(self.device))
                self.model.eval()
                score = self.model(x, t_embedding)
                alpha = torch.reshape(alpha, (-1, 1, 1))
                sigma = torch.reshape(sigma, (-1, 1, 1))
                x_recon = alpha * x - sigma * score
                p1 = torch.reshape(
                    torch.gather(
                        self.diffusion.posterior_mean_coef1,
                        index=batched_time_step.squeeze(),
                        dim=0,
                    ),
                    (-1, 1, 1),
                ).to(self.device)
                p2 = torch.reshape(
                    torch.gather(
                        self.diffusion.posterior_mean_coef2,
                        index=batched_time_step.squeeze(),
                        dim=0,
                    ),
                    (-1, 1, 1),
                ).to(self.device)
                mean = p1 * x_recon + p2 * x
                log_var = torch.reshape(
                    torch.gather(
                        torch.log(self.diffusion.posterior_variance),
                        index=batched_time_step.squeeze(),
                        dim=0,
                    ),
                    (-1, 1, 1),
                ).to(self.device)
                x = mean + torch.exp(0.5 * log_var) * z
            return mean


class EMA:
    def __init__(self, beta) -> None:
        self.beta = beta
        self.step = 0

    def update_model_average(self, ema_model: nn.Module, model: nn.Module):
        for current_param, ema_param in zip(model.parameters(), ema_model.parameters()):
            old_weight, new_weight = ema_param.data, current_param.data
            ema_param.data = self.update_average(old_weight, new_weight)

    def update_average(self, old, new):
        return old * self.beta + (1 - self.beta) * new

    def step_ema(
        self, ema_model: nn.Module, model: nn.Module, step_start_ema: int = 1000
    ):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model: nn.Module, model: nn.Module):
        ema_model.load_state_dict(model.state_dict())
