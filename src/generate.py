import os
from time import time
import warnings
import numpy as np
import torch
import argparse
from utils import *
from diffusion import PointDiffusion, DDIMSampler

warnings.filterwarnings("ignore")


def sample_events(config, sampler: DDIMSampler) -> torch.Tensor:
    start_t = time.time()
    transformed_samples = sampler.sample().cpu().numpy()
    end_t = time.time()
    print(f"Sampling {sampler.num_samples} events took {end_t - start_t} seconds.")
    raw_samples = reverse_preprocess(transformed_samples, config["NUM_DEPOSITS"])
    # raw_samples[:, 0:1] = np.clip(raw_samples[:, 0:1], 0, 124.0)
    return raw_samples


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
        "--save_path",
        default="./logs_gen/",
        help="path to save generataed samples",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="./logs_gen",
        help="path to the root directory of all log files",
    )
    parser.add_argument(
        "--expname",
        type=str,
        help="name of the experiment whose checkpoint is to be loaded",
    )
    flags = parser.parse_args()
    config = load_json_file(flags.config)

    log_dir = get_existing_log_dir(root=flags.log_root, local_path=flags.expname)
    ckpt_mgr = CheckPointManager(log_dir)

    use_cuda = not flags.no_cuda and torch.cuda.is_available()
    print(f"use_cuda={use_cuda}")
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using {device}")

    diffusion = PointDiffusion(num_deposits=300, model_config=config).to(device=device)
    state_dict = ckpt_mgr.load_best()["state_dict"]
    diffusion.load_state_dict(state_dict)

    print("Sampling events...")
    sampler = DDIMSampler(
        diffusion=diffusion,
        num_samples=5,
        data_shape=(config["NUM_DEPOSITS"], config["NUM_FEATS"]),
        device=device,
    ).to(device)
    events = sample_events(config, sampler)
    with open(os.path.join(flags.save_path, flags.expname, "samples.npy"), "wb") as f:
        np.save(f, events)
