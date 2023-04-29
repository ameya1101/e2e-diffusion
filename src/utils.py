import json, yaml
import os
import numpy as np
import torch
import logging
import logging.handlers
import time


class BlackHole(object):
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


class CheckPointManager(object):
    def __init__(self, save_dir, logger=BlackHole()) -> None:
        super().__init__()
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.ckpts = []
        self.logger = logger

        for f in os.listdir(self.save_dir):
            if f[:4] != "ckpt":
                continue
            _, score, epoch = f.split("_")
            epoch = epoch.split(".")[0]
            self.ckpts.append({"score": float(score), "file": f, "epoch": int(epoch)})

    def get_worst_ckpt_idx(self):
        idx = -1
        worst = float("-inf")
        for i, ckpt in enumerate(self.ckpts):
            if ckpt["score"] >= worst:
                idx = i
                worst = ckpt["score"]
        return idx if idx >= 0 else None

    def get_best_ckpt_idx(self):
        idx = -1
        best = float("inf")
        for i, ckpt in enumerate(self.ckpts):
            if ckpt["score"] <= best:
                idx = i
                best = ckpt["score"]
        return idx if idx >= 0 else None

    def get_latest_ckpt_idx(self):
        idx = -1
        latest_epoch = -1
        for i, ckpt in enumerate(self.ckpts):
            if ckpt["epoch"] > latest_epoch:
                idx = i
                latest_epoch = ckpt["epoch"]
        return idx if idx >= 0 else None

    def save(self, model, args, score, others=None, step=None, tag=None):
        if step is None:
            fname = (
                f"ckpt_{score:.3f}.pt" if tag is None else f"ckpt_{score:.3f}_tag.pt"
            )
        else:
            fname = (
                f"ckpt_{score:.3f}_{step}.pt"
                if tag is None
                else f"ckpt_{score:.3f}_{step}_tag.pt"
            )
        path = os.path.join(self.save_dir, fname)

        torch.save(
            {"args": args, "state_dict": model.state_dict(), "others": others}, path
        )

        self.ckpts.append({"score": score, "file": fname})

        return True

    def load_best(self):
        idx = self.get_best_ckpt_idx()
        if idx is None:
            raise IOError("No checkpoints found.")
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]["file"]))
        return ckpt

    def load_latest(self):
        idx = self.get_latest_ckpt_idx()
        if idx is None:
            raise IOError("No checkpoints found.")
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]["file"]))
        return ckpt

    def load_selected(self, file):
        ckpt = torch.load(os.path.join(self.save_dir, file))
        return ckpt


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s::%(name)s::%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, "log.txt"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_existing_log_dir(root="./logs", local_path=None):
    log_dir = os.path.join(root, local_path)
    if not os.exists(log_dir):
        raise FileNotFoundError(f"{log_dir} was not found.")
    return log_dir


def get_new_log_dir(root="./logs", postfix="", prefix=""):
    log_dir = os.path.join(
        root, prefix + time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime()) + postfix
    )
    os.makedirs(log_dir)
    return log_dir


def load_json_file(filename):
    JSON_PATH = os.path.join(filename)
    fstream = open(JSON_PATH)
    contents = yaml.safe_load(fstream)
    fstream.close()
    return contents


def save_json_file(PATH, data):
    with open(PATH, "w") as f:
        json.dump(data, f)


def _logit(z):
    alpha = 1e-6
    z = alpha + (1 - 2 * alpha) * z
    return np.ma.log(z / (1 - z))


def _reverse_logit(z):
    alpha = 1e-6
    exp = np.exp(z)
    z = exp / (1 + exp)
    return (z - alpha) / (1 - 2 * alpha)


def reverse_preprocess(deposits, num_deposits):
    data_dict = load_json_file(f"preprocessing_{num_deposits}.json")
    deposits = deposits.reshape(-1, deposits.shape[-1])
    deposits = deposits * data_dict["std_hit"] + data_dict["mean_hit"]
    deposits = _reverse_logit(deposits)
    deposits = (
        deposits * (np.array(data_dict["max_hit"] - data_dict["min_hit"]))
        + data_dict["min_hit"]
    )
    return deposits


def _preprocess(deposits: np.ndarray, save_json: bool = True):
    num_deposits = deposits.shape[1]  # shape: (M, N, F)
    num_feats = deposits.shape[2]
    deposits = deposits.reshape(-1, deposits.shape[-1])  # shape: (M * N, F)

    if save_json is True:
        data_dict = {
            "max_hit": np.max(deposits, axis=0).tolist(),
            "min_hit": np.min(deposits, axis=0).tolist(),
        }
        save_json_file(f"preprocessing_{num_deposits}.json", data_dict)
    else:
        data_dict = load_json_file(f"preprocessing_{num_deposits}.json")

    # Step 1: Normalize features to [0, 1]
    deposits = np.ma.divide(
        deposits - data_dict["min_hit"],
        np.array(data_dict["max_hit"]) - data_dict["min_hit"],
    )

    # Step 2: Convert features to logits
    deposits = _logit(deposits)

    # Step 3: Standardize features
    if save_json:
        data_dict["mean_hit"] = np.mean(deposits, axis=0).tolist()
        data_dict["std_hit"] = np.std(deposits, axis=0).tolist()
        save_json_file(f"preprocessing_{num_deposits}.json", data_dict)

    deposits = np.ma.divide(deposits - data_dict["mean_hit"], data_dict["std_hit"])

    deposits = deposits.reshape(-1, num_deposits, num_feats)
    return deposits.astype(np.float32)


def get_dataloader(data_path, batch_size=64, **kwargs):
    deposits = []
    event_files = os.listdir(data_path)
    event_files = [os.path.join(data_path, file) for file in event_files]

    for file in event_files:
        event = np.load(file)["ecal"]
        deposits.append(event)

    deposits = np.array(deposits)
    deposits = _preprocess(deposits)

    deposits_tensor = torch.from_numpy(deposits)
    dataset = torch.utils.data.TensorDataset(deposits_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, **kwargs)
    return loader
