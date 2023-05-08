import time
from statistics import mean
import warnings
import torch
from torch.optim import Adamax
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel
import argparse
from utils import *
from diffusion import PointDiffusion

warnings.filterwarnings("ignore")


def train_step(
    flags,
    model: PointDiffusion,
    device: str,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> float:
    model.train()
    epoch_t0 = time.time()
    losses = []
    for batch_idx, x in enumerate(dataloader):
        x = x[0].to(device)  # weirdly dataloader returns list of batch objects
        optimizer.zero_grad()
        loss = model(x)
        loss.backward()
        optimizer.step()
        if flags.dry_run:
            exit(0)
        if batch_idx % 10 == 0:
            logger.info(
                f"Train Epoch: {epoch} [{batch_idx}/{len(dataloader)} ({100. * batch_idx / len(dataloader):.3f})%]\tLoss: {loss.item():.6f}"
            )
        losses.append(loss.item())
    logger.info(f"Epoch time: {time.time() - epoch_t0:.4f}s")
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
    parser.add_argument("--data_path", help="path containing training files")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument("--logging", type=eval, default=True, choices=[True, False])
    parser.add_argument("--log_root", type=str, default="./logs_gen")
    parser.add_argument("--ckpt_freq", type=int, default=30)
    parser.add_argument("--tag", type=str, default=None)

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
        num_deposits=config["NUM_DEPOSITS"], model_config=config, device=device
    ).to(device=device)
    total_trainable_params = sum(p.numel() for p in diffusion.parameters())
    logger.info(f"Total trainable parameters: {total_trainable_params}")
    logger.info(repr(diffusion))

    optimizer = Adamax(diffusion.parameters(), lr=config["LR"])
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config["N_EPOCHS"] * int(len(dataloader.dataset) / config["BATCH"]),
    )

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
        train_loss = train_step(flags, diffusion, device, dataloader, optimizer, epoch)
        if epoch > ema_start:
            ema_model.update_parameters(diffusion)
        scheduler.step()
        if (epoch - 1) % flags.ckpt_freq == 0 or epoch == (config["N_EPOCHS"] - 1):
            logger.info(f"Checkpointing at epoch {epoch}.")
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
