import os
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
from tqdm import tqdm
from utils import save_images, setup_logging
import logging as logging
from models.ddpm import Diffusion
from models.modules import UNet
from dataset import JetDataset
from torch.utils.tensorboard import SummaryWriter
import argparse


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def train(args, dataloader, diffusion, model, optimizer, logger):
    device = args.device
    mse = nn.MSELoss()
    l = len(dataloader)
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, images in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

    samples_images = diffusion.sample(model, n=images.shape[0])
    save_images(samples_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args=[])
    args.run_name = "DDPM_Jet"
    args.epochs = 50
    args.batch_size = 12
    args.image_size = 64
    args.PATH = r"data/"
    args.device = "cuda"
    args.lr = 3e-4
    args.rank = [0, 1, 2]
    args.num_workers = 2

    setup_logging(args.run_name)

    parquet_files = [os.path.join(args.PATH, 'Boosted_Jets_Sample-0.snappy.parquet')]
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(64),
    ])
    dataset = JetDataset(parquet_files, transforms=transforms)

    train_size = 1000
    test_size = len(dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(dataset, [train_size, test_size])
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = UNet().to(args.device)
    model = torch.nn.DataParallel(model, device_ids=args.rank)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    diffusion = Diffusion(img_size=args.image_size, device=args.device)
    logger = SummaryWriter(os.path.join("runs", args.run_name)) 
    
    train(args, dataloader, diffusion, model, optimizer, logger)

if __name__ == '__main__':
    main()