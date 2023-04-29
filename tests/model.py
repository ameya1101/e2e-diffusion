import torch
from src.modules import *
import unittest


class ModuleTests(unittest.TestCase):
    def __init__(self, methodName: str = "moduleTest") -> None:
        super().__init__(methodName)
        self.time_embedder = TimeEmbedding(num_embed=16)
        self.model = DeepSetsAttention(
            num_feats=3, num_heads=1, num_transformers=5, projection_dim=16
        )

    def test_forwardpass(self):
        x = torch.randn(5, 10, 3)
        t = torch.randint(low=0, high=20, size=(x.shape[0], 1))

        t_embedding = self.time_embedder(t)
        self.assertEqual(t_embedding.shape, (5, 16))

        score = self.model(x, t_embedding)
        self.assertEqual(x.shape, score.shape)
