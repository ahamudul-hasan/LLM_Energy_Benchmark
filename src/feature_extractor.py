"""
Feature Extractor for Prompt Complexity Router.

Extracts hybrid feature representation x in R^d fusing:
1. Lexical and Syntactic Heuristics:
   - Token count, character count, avg word length, punctuation density
   - Syntax boolean flags: 'def ', 'class ', 'import ', 'return ', 'curl ', 'SELECT '
2. Dense Semantic Embeddings:
   - 384-dimensional sentence embedding using 'all-MiniLM-L6-v2' via sentence-transformers on host CPU.

Strictly adheres to Section 3.2 of main.pdf and Section 5 of AGENTS.md.
"""

import re
import time
import numpy as np
from typing import List, Union, Dict, Any

# Global cached embedding model instance
_MINILM_MODEL = None


def get_embedding_model():
    """Lazy loads sentence-transformers all-MiniLM-L6-v2 model on CPU."""
    global _MINILM_MODEL
    if _MINILM_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MINILM_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _MINILM_MODEL


def extract_surface_heuristics(prompt: str) -> List[float]:
    """
    Extracts lexical and syntactic features from prompt text.
    Returns feature list:
    [
        token_count,
        char_count,
        avg_word_len,
        punct_density,
        has_def,
        has_class,
        has_import,
        has_return,
        has_curl,
        has_select
    ]
    """
    words = prompt.split()
    token_count = float(len(words))
    char_count = float(len(prompt))
    avg_word_len = (char_count / token_count) if token_count > 0 else 0.0

    punct_count = sum(1 for c in prompt if c in ".,!?;:()[]{}\"'`")
    punct_density = (punct_count / char_count) if char_count > 0 else 0.0

    prompt_lower = prompt.lower()
    has_def = 1.0 if "def " in prompt else 0.0
    has_class = 1.0 if "class " in prompt else 0.0
    has_import = 1.0 if "import " in prompt else 0.0
    has_return = 1.0 if "return " in prompt else 0.0
    has_curl = 1.0 if "curl " in prompt_lower else 0.0
    has_select = 1.0 if "select " in prompt_lower else 0.0

    return [
        token_count,
        char_count,
        avg_word_len,
        punct_density,
        has_def,
        has_class,
        has_import,
        has_return,
        has_curl,
        has_select
    ]


class CPUFeatureExtractor:
    """Combines lexical heuristics and MiniLM dense embeddings into a single feature vector."""

    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings
        self.embedder = None
        if use_embeddings:
            try:
                self.embedder = get_embedding_model()
            except Exception as e:
                print(f"Warning: Failed to load sentence-transformers MiniLM: {e}. Falling back to heuristics-only.")
                self.use_embeddings = False

    def transform_single(self, prompt: str) -> np.ndarray:
        """Transforms a single prompt into a 1D feature array."""
        heuristics = extract_surface_heuristics(prompt)
        if self.use_embeddings and self.embedder is not None:
            emb = self.embedder.encode(prompt, convert_to_numpy=True)
            return np.hstack([heuristics, emb])
        else:
            return np.array(heuristics, dtype=np.float32)

    def transform_batch(self, prompts: List[str]) -> np.ndarray:
        """Transforms a batch of prompts into a 2D feature matrix (N, D)."""
        heuristics_list = [extract_surface_heuristics(p) for p in prompts]
        heuristics_arr = np.array(heuristics_list, dtype=np.float32)

        if self.use_embeddings and self.embedder is not None:
            embeddings_arr = self.embedder.encode(prompts, batch_size=32, convert_to_numpy=True)
            return np.hstack([heuristics_arr, embeddings_arr])
        else:
            return heuristics_arr


if __name__ == "__main__":
    extractor = CPUFeatureExtractor(use_embeddings=False)
    test_prompt = "def calculate_energy(p, t):\n    return p * t\n"
    start = time.perf_counter()
    vec = extractor.transform_single(test_prompt)
    latency_ms = (time.perf_counter() - start) * 1000.0

    print(f"Feature vector shape: {vec.shape}")
    print(f"Extraction latency: {latency_ms:.3f} ms")
    print(f"Heuristics sample: {vec[:10]}")
