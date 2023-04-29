from time import time
from statistics import mean
import warnings
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.optim.swa_utils import AveragedModel
import argparse
from utils import *
from diffusion import PointDiffusion

warnings.filterwarnings("ignore")


def train_step(
    model: PointDiffusion,
    device: str,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> float:
    model.train()
    epoch_t0 = time()
    losses = []
    for batch_idx, x in enumerate(dataloader):
        x = x.to(device)
        optimizer.zero_grad()
        loss = model(x)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            logger.info(
                f"Train Epoch: {epoch} [{batch_idx}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.6f})]\tLoss: {loss.item():.6f}"
            )
        losses.append(loss.item())
    logger.info(f"Epoch time: {time() - epoch_t0}s")
    logger.info(f"Epoch {epoch}: train loss = {mean(losses)}")
    return mean(losses)


if __name__ == "__main__":
    # TODO: Add Horovod compatibility

    # Setup command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.json",
        help="configuration file containing training hyperparameters",
    )
    parser.add_argument(
        "--data_path", default="./sample-data/", help="path containing training files"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument("--logging", type=eval, default=True, choices=[True, False])
    parser.add_argument("--log_root", type=str, default="./logs_gen")
    parser.add_argument("--ckpt_freq", type=int, default=30)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--sampling_freq", type=int, default=100)

    flags = parser.parse_args()
    config = load_json_file(flags.config)

    # Setup logging and checkpointing (if enabled)
    if flags.logging:
        log_dir = get_new_log_dir(
            flags.log_root,
            prefix="GEN_",
            postfix="_" + flags.tag if flags.tag is not None else "",
        )
        logger = get_logger("train", log_dir)
        ckpt_mgr = CheckPointManager(log_dir)
    else:
        logger = get_logger("train", None)
        ckpt_mgr = BlackHole()
    logger.info(flags)

    use_cuda = not flags.no_cuda and torch.cuda.is_available()
    logger.info(f"use_cuda={use_cuda}")
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using {device}")

    # Load dataset
    dataloader = get_dataloader(data_path=flags.data_path, batch_size=config["BATCH"])
    logger.info("Successfully loaded training events.")

    # Setup model and optimizers
    diffusion = PointDiffusion(
        num_deposits=config["NUM_DEPOSITS"], model_config=config
    ).to(device=device)
    total_trainable_params = sum(p.numel() for p in diffusion.parameters())
    logger.info(f"Total trainable parameters: {total_trainable_params}")
    logger.info(repr(diffusion))

    optimizer = Adam(diffusion.parameters(), lr=config["LR"])
    scheduler = StepLR(optimizer=optimizer, step_size=30)

    ema_avg = (
        lambda averaged_model_params, model_params, num_averaged: 0.005
        * averaged_model_params
        + 0.995 * model_params
    )
    ema_model = AveragedModel(diffusion, device=device, avg_fn=ema_avg)
    ema_start = config["EMA_START"]

    # Begin training loop
    logger.info("Training started.")
    for epoch in range(1, config["N_EPOCHS"] + 1):
        logger.info(f"---- Epoch {epoch} ----")
        train_loss = train_step(diffusion, device, dataloader, optimizer, epoch)
        if epoch > ema_start:
            ema_model.update_parameters()
        scheduler.step()
        if epoch % flags.ckpt_freq == 0 or epoch == (config["N_EPOCHS"] - 1):
            logging.info(f"Checkpointing at epoch {epoch}.")
            opt_states = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            ckpt_mgr.save(diffusion, flags, train_loss, others=opt_states, step=epoch)
            if epoch > ema_start:
                ckpt_mgr.save(
                    ema_model,
                    flags,
                    train_loss,
                    others=opt_states,
                    step=epoch,
                    tag="ema",
                )
