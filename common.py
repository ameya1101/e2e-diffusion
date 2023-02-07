import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(mean)
    return mean + eps * std


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1.0 + np.log(2 * np.pi))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2


def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            frac = (epoch - start_epoch) / (end_epoch - start_epoch)
            return (1 - frac) * 1.0 + frac * (end_lr / start_lr)

    return LambdaLR(optimizer, lr_lambda=lr_func)
