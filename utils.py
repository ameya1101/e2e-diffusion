import math

import torch
from torchvision import transforms as T


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(loader):
    while True:
        for data in loader:
            yield data


def has_int_sqrt(n):
    return (math.sqrt(n) ** 2) == n


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def max_crop(img, size=64):
    n_channels = img.shape[0]
    cropped_img = torch.zeros(n_channels, size, size)
    for channel in range(n_channels):
        x_max = (img[channel] == torch.max(img[channel])).nonzero()[0][1]
        y_max = (img[channel] == torch.max(img[channel])).nonzero()[0][1]
        cropped_img[channel] = T.functional.crop(
            img[channel], x_max - size // 2, y_max - size // 2, size, size
        )
    return cropped_img


def max_normalize(img):
    n_channels = img.shape[0]
    for channel in range(n_channels):
        img[channel] = img[channel] / torch.max(img[channel])
    return img
