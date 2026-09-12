"""
Online Prompt Complexity Router Engine.

Executes ultra-low latency host CPU routing (< 5 ms execution latency, < 0.05 J energy) per main.pdf Section 3.2.
Intercepts incoming queries, extracts CPU features, evaluates classifier probabilities,
applies confidence escalation thresholds (tau1, tau2), and dispatches request to the optimal model tier.
"""

import os
import time
import joblib
import numpy as np
from typing import Dict, Any, Optional

import sys
sys.path.insert(0, os.path.dirname(__file__))

from feature_extractor import CPUFeatureExtractor

TIER_MODEL_MAP = {
    1: "llama3.2:1b",
    2: "llama3.2:3b",
    3: "llama3.1:8b"
}


class PromptRouter:
    """Online prompt complexity router."""

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "router_model.joblib")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Router model checkpoint not found at {model_path}. Run train_router.py first.")

        payload = joblib.load(model_path)
        self.classifier = payload["classifier"]
        self.clf_name = payload["clf_name"]
        self.classes = payload["classes"]
        self.tau1 = payload["tau1"]
        self.tau2 = payload["tau2"]
        use_emb = payload.get("use_embeddings", False)

        self.extractor = CPUFeatureExtractor(use_embeddings=use_emb)

    def route(self, prompt: str) -> Dict[str, Any]:
        """
        Executes online prompt complexity routing decision.
        Returns dict containing predicted tier, selected model, execution latency (ms),
        posterior probabilities, and confidence threshold status.
        """
        start_t = time.perf_counter()

        # 1. Host CPU Feature Extraction
        x = self.extractor.transform_single(prompt).reshape(1, -1)

        # 2. Classifier Inference
        probs_raw = self.classifier.predict_proba(x)[0]
        probs = [0.0, 0.0, 0.0]
        for idx, c in enumerate(self.classes):
            probs[c - 1] = float(probs_raw[idx])

        p1, p2, p3 = probs[0], probs[1], probs[2]

        # 3. Confidence Threshold Decision Rules (main.pdf Section 3.2)
        if p1 >= self.tau1 and p1 >= p2 and p1 >= p3:
            selected_tier = 1
        elif (p1 + p2) >= self.tau2 and p2 >= p3:
            selected_tier = 2
        else:
            selected_tier = 3

        end_t = time.perf_counter()
        latency_ms = (end_t - start_t) * 1000.0

        # Estimated router energy overhead (< 0.05 J on CPU)
        energy_overhead_joules = (latency_ms / 1000.0) * 15.0  # Assumes 15W host CPU draw during feature extraction

        return {
            "tier_selected": selected_tier,
            "model_used": TIER_MODEL_MAP[selected_tier],
            "probs": [round(p, 4) for p in probs],
            "router_latency_ms": round(latency_ms, 3),
            "router_energy_joules": round(energy_overhead_joules, 5),
            "clf_backend": self.clf_name,
            "thresholds": {"tau1": self.tau1, "tau2": self.tau2}
        }


if __name__ == "__main__":
    router = PromptRouter()

    test_prompts = [
        ("Classify sentiment: 'The movie was wonderful!'", 1),
        ("Summarize the news article about green computing in 2 paragraphs.", 2),
        ("def solve_math(n):\n    # Calculate prime factors\n    return [i for i in range(2, n) if n % i == 0]\n", 3)
    ]

    print("--- Testing Online Prompt Router ---")
    for text, expected in test_prompts:
        res = router.route(text)
        print(f"\nPrompt: \"{text[:50]}...\"")
        print(f"Selected Tier: {res['tier_selected']} (Model: {res['model_used']}) | Expected: Tier {expected}")
        print(f"Probabilities: {res['probs']}")
        print(f"Router Overhead: {res['router_latency_ms']} ms | {res['router_energy_joules']} J")
