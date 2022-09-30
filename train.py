import os
import torch.nn as nn
import torch.optim as optim
from models.modules import UNet
from models.ddpm import Diffusion
from tqdm import tqdm
from utils import save_images, EMA

def train(args):
    device = args.device
    dataloader = None
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=args.device)

    for epoch in range(args.epochs):
        for images in tqdm(dataloader):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    samples_images = diffusion.sample(model, n=images.shape[0])
    save_images(samples_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
             