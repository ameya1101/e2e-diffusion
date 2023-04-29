import os
import numpy as np
from src.utils import _preprocess, reverse_preprocess, load_json_file
import unittest
import json


class TestDataProcessing(unittest.TestCase):
    def __init__(self, methodName: str = "processingTest") -> None:
        super().__init__(methodName)
        self.event_files = os.listdir("sample-data/")
        self.event_files = [
            os.path.join("sample-data/", file) for file in self.event_files
        ]
        self.deposits = []
        for file in self.event_files:
            event = np.load(file)["ecal"]
            self.deposits.append(event)
        self.deposits = np.array(self.deposits)

    def __del__(self):
        os.remove(f"preprocessing_{self.deposits.shape[1]}.json")

    def test_preprocessing(self):
        deposits_post = _preprocess(self.deposits, save_json=True)

        self.assertEqual(self.deposits.shape, deposits_post.shape)
        self.assertTrue(os.path.exists(f"preprocessing_{self.deposits.shape[1]}.json"))
        file = load_json_file(f"preprocessing_{self.deposits.shape[1]}.json")
        self.assertFalse(
            len(file["min_hit"]) == 0
            or len(file["max_hit"]) == 0
            or len(file["mean_hit"]) == 0
            or len(file["std_hit"]) == 0
        )


if __name__ == "__main__":
    unittest.main()
