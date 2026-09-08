"""
Unit tests for src/measure_energy.py
"""

import unittest
import json
import time
from unittest.mock import MagicMock, patch
from src.measure_energy import (
    calculate_energy,
    measure_idle_power,
    cool_down_gpu,
    run_inference_and_measure
)


class TestMeasureEnergy(unittest.TestCase):

    def test_calculate_energy_trapezoidal(self):
        # Samples: (time, watts)
        samples = [
            (0.0, 10.0),
            (1.0, 20.0),  # dt=1, avg P=15 => 15 J
            (2.0, 20.0),  # dt=1, avg P=20 => 20 J
            (3.0, 10.0),  # dt=1, avg P=15 => 15 J
        ]
        total_energy = calculate_energy(samples)
        self.assertAlmostEqual(total_energy, 50.0, places=4)

    def test_measure_idle_power_mock(self):
        p_idle = measure_idle_power(seconds=0.2, sampling_interval=0.05, use_mock=True)
        self.assertTrue(18.0 <= p_idle <= 25.0)

    @patch("requests.post")
    def test_run_inference_and_measure_mock(self, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None

        chunks = [
            json.dumps({"response": "Green ", "done": False}).encode("utf-8"),
            json.dumps({
                "response": "computing is eco-friendly.",
                "done": True,
                "prompt_eval_count": 5,
                "eval_count": 4
            }).encode("utf-8")
        ]

        def delayed_iter():
            for c in chunks:
                time.sleep(0.1)  # Simulate streaming chunk delivery time
                yield c

        mock_response.iter_lines.side_effect = delayed_iter
        mock_post.return_value = mock_response

        result = run_inference_and_measure(
            prompt="What is green computing?",
            model="llama3.2:1b",
            mock=True,
            cool_down_seconds=0.1
        )

        self.assertNotIn("error", result)
        self.assertEqual(result["model_used"], "llama3.2:1b")
        self.assertEqual(result["prompt_token_count"], 5)
        self.assertEqual(result["output_token_count"], 4)
        self.assertEqual(result["response_text"], "Green computing is eco-friendly.")
        self.assertGreater(result["total_energy_joules"], 0.0)
        self.assertGreater(result["net_energy_joules"], 0.0)
        self.assertGreater(result["joules_per_token"], 0.0)


if __name__ == "__main__":
    unittest.main()
