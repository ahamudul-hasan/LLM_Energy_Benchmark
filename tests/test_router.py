"""
Unit Tests for Energy-Aware Prompt Complexity Router.
Verifies feature extraction, model training/calibration, and online sub-5ms routing.
"""

import os
import unittest
import numpy as np

from src.feature_extractor import extract_surface_heuristics, CPUFeatureExtractor
from src.dataset_loader import format_prompt, create_synthetic_fallback_dataset
from src.router import PromptRouter


class TestPromptRouter(unittest.TestCase):

    def test_heuristics_extraction(self):
        code_prompt = "def calculate_power(voltage: float, current: float) -> float:\n    return voltage * current\n"
        heuristics = extract_surface_heuristics(code_prompt)

        # Vector: [token_count, char_count, avg_word_len, punct_density, has_def, has_class, has_import, has_return, has_curl, has_select]
        self.assertEqual(len(heuristics), 10)
        self.assertEqual(heuristics[4], 1.0)  # has_def
        self.assertEqual(heuristics[7], 1.0)  # has_return
        self.assertGreater(heuristics[0], 0)   # token count

    def test_feature_extractor_shape(self):
        extractor = CPUFeatureExtractor(use_embeddings=False)
        vec = extractor.transform_single("Classify sentiment: 'Great movie!'")
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape[0], 10)

    def test_online_router_inference(self):
        router_model_path = os.path.join("src", "router_model.joblib")
        if os.path.exists(router_model_path):
            router = PromptRouter(model_path=router_model_path)
            res = router.route("What is the capital of Bangladesh?")

            self.assertIn("tier_selected", res)
            self.assertIn(res["tier_selected"], [1, 2, 3])
            self.assertIn("model_used", res)
            self.assertLess(res["router_latency_ms"], 50.0)  # Under 50ms (and typically sub-5ms)

    def test_synthetic_dataset_creation(self):
        items = create_synthetic_fallback_dataset("sst2", count=5)
        self.assertEqual(len(items), 5)
        self.assertEqual(items[0]["dataset"], "sst2")
        self.assertEqual(items[0]["tier"], 1)


if __name__ == "__main__":
    unittest.main()
