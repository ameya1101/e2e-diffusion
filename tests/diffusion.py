import torch
from src.diffusion import *
from src.utils import *
import unittest


class DiffusionTest(unittest.TestCase):
    def __init__(self, methodName: str = "diffusionTest") -> None:
        super().__init__(methodName)
        config = load_json_file("config.json")
        self.diff = PointDiffusion(
            num_deposits=config["NUM_DEPOSITS"], model_config=config
        )
        self.sampler = DDIMSampler(
            diffusion=self.diff,
            num_samples=5,
            data_shape=(config["NUM_DEPOSITS"], config["NUM_FEATS"]),
        )

    def test_forwardpass(self):
        x = torch.randn(5, 10, 3)
        loss = self.diff.train_step(x)
        self.assertIsInstance(loss, torch.Tensor)

    def test_sampling(self):
        events = self.sampler.sample()
        self.assertEqual(events.shape, (5, 10, 3))


if __name__ == "__main__":
    unittest.main()
