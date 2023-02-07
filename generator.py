import torch.nn as nn

from .modules import PointNet
from .ddpm import *
from .common import *


class PointCloudGenerator(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.encoder = PointNet(embed_dim=args.embed_dim)
        self.diffusion = PointDiffusion(
            model=PointWiseNet(context_dim=args.embed_dim, residual=args.residual),
            beta_schedule="linear",
            timesteps=args.num_steps
        )

    def get_loss(self, x, kl_weight=1.0, writer=None, epoch=None):
        """
        Args:
            x: Input point clouds, (B, N, d)
        """
        batch_size, _, _ = x.size()
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        log_pz = standard_normal_logprob(z).sum(dim=1)  # (B, ), Independence assumption
        entropy = gaussian_entropy(logvar=z_sigma)
        loss_prior = (-log_pz - entropy).mean()

        loss_recon = self.diffusion.get_loss(x, z)

        loss = kl_weight * loss_prior + loss_recon

        if writer is not None:
            writer.add_scalar('train/loss_entropy', -entropy.mean(), epoch)
            writer.add_scalar('train/loss_prior', -log_pz.mean(), epoch)
            writer.add_scalar('train/loss_recon', loss_recon.mean(), epoch)

        return loss

    def sample(self, z, num_points):
        """
        Args:
            z: Input latent, standard normal random samples, (B, F)
        """
        samples = self.diffusion.sample(num_points, context=z)
        return samples
