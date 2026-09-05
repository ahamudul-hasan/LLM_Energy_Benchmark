# AGENTS.md — Guidelines for Benchmark & Test Implementation

This document defines the operational invariants, architectural constraints, and test generation rules for autonomous AI agents implementing or extending benchmarks, routing pipelines, and energy measurement scripts in this repository.

---

## 1. Project Context & Mission

This repository implements the experimental framework described in Chapter 3 of:
> **"A Comparative Analysis of Energy Efficiency Across Large Language Models"**  
> *Talha, Borno, Rabbi, Prianto (Department of CSE, United International University)*

The core goal is to evaluate whether a **lightweight, CPU-executed prompt-complexity router** can dynamically direct incoming queries to the smallest capable model tier (Tier 1, Tier 2, or Tier 3), thereby minimizing real-world GPU electrical energy consumption with minimal loss in output quality compared to a static 8B baseline.

---

## 2. Mandatory Measurement Invariants

Whenever implementing or modifying measurement code or test runs, you **MUST** strictly adhere to the following rules:

### A. High-Frequency NVML Power Sampling
1. Power $P(t)$ must be queried using `pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0` (in Watts).
2. Sampling must occur every **50 ms** (`time.sleep(0.05)`).
3. The sampler **must run in an isolated background thread or daemon** (`threading.Thread`) to prevent interference or thread contention with the LLM inference call.

### B. Baseline / Idle Power Profiling
1. Before every test run or evaluation batch, profile static GPU baseline power ($P_{\text{idle}}$) over a quiet resting period of at least **3.0 seconds**.
2. Never skip $P_{\text{idle}}$ measurement. Subtracting idle power is required to isolate **net dynamic inference energy**:
   $$E_{\text{net}} = E_{\text{total}} - (P_{\text{idle}} \times \Delta t_{\text{latency}})$$

### C. Trapezoidal Integration
Calculate total energy in Joules ($J$) using numerical trapezoidal integration across collected sample points $(t_k, P_k)$:
$$E_{\text{total}} = \sum_{k=1}^{N-1} \frac{P(t_k) + P(t_{k+1})}{2} \cdot (t_{k+1} - t_k)$$

### D. NVML Resource Management
Always wrap NVML operations in `try ... finally` blocks to ensure `pynvml.nvmlShutdown()` is called even if an exception occurs:
```python
from pynvml import nvmlInit, nvmlShutdown

nvmlInit()
try:
    # Run test harness
    ...
finally:
    nvmlShutdown()
```

### E. Cool-Down Delays
In batch benchmarks, always insert a cool-down period (e.g., **5.0 seconds**) between consecutive inferences to allow GPU voltage, clock rates, and temperatures to return to quiescent idle levels.

---

## 3. Model Tiers & Deployment Constraints

All models run locally via Ollama (or llama.cpp) within an **8 GB VRAM** boundary. Test configurations must target the designated model identifiers and quantization formats:

| Tier | Complexity Class | Primary Model Identifier | Format & Precision | Target Tasks |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | Low Complexity | `llama3.2:1b` | `Q8_0` (~1.3 GB VRAM) | Binary sentiment (SST-2), short fact extraction (SQuAD v2.0) |
| **Tier 2** | Medium Complexity | `llama3.2:3b` / `phi3.5:3.8b` | `Q4_K_M` (~2.6–3.1 GB VRAM) | Abstractive summarization (CNN/DailyMail), NLI (MNLI), JSON transformation |
| **Tier 3** | High Complexity | `llama3.1:8b` / `mistral:7b` | `Q4_K_M` (~5.2–5.8 GB VRAM) | Multi-step math (GSM8K), code synthesis (HumanEval), complex deduction |

---

## 4. Benchmark Workload Dataset Specifications

When constructing automated tests or benchmark suites, load or synthesize queries from the 6 target benchmarks (totaling 600 queries; 200 per complexity class):

### 1. Low-Complexity Subset (200 prompts)
- **SST-2 (Stanford Sentiment Treebank)**: 100 prompts. Single-sentence classification (`positive` / `negative`). Evaluation metric: **Accuracy**.
- **SQuAD v2.0**: 100 prompts. Closed-domain question-answering with context paragraph. Evaluation metric: **Exact Match (EM)** / Token F1.

### 2. Medium-Complexity Subset (200 prompts)
- **CNN/DailyMail**: 100 prompts. Multi-sentence news article summarization. Evaluation metric: **ROUGE-L**.
- **MNLI (Multi-genre NLI)**: 100 prompts. Premise-hypothesis entailment classification (`entailment`, `neutral`, `contradiction`). Evaluation metric: **Accuracy**.

### 3. High-Complexity Subset (200 prompts)
- **GSM8K**: 100 prompts. Multi-step grade school mathematical word problems requiring scratchpad / chain-of-thought calculation. Evaluation metric: **Accuracy** (final numerical answer match).
- **HumanEval**: 100 prompts. Python docstring-to-code functional implementation. Evaluation metric: **Pass@1** (executing against test suites).

---

## 5. Lightweight Prompt-Complexity Router Specifications

When implementing or testing the router:

1. **Host CPU Execution**: Feature extraction and inference for the router must execute strictly on host CPU (never on the GPU) to avoid competing for GPU memory bandwidth or compute pipelines.
2. **Feature Pipeline**:
   - **Surface Heuristics**: Token count, character length, presence of programming keywords (`def `, `class `, `return `, `import `, `curl `, `SELECT `).
   - **Semantic Embeddings**: 384-dimensional sentence embedding using `all-MiniLM-L6-v2` via `sentence-transformers`.
3. **Classifier**:
   - `sklearn.linear_model.LogisticRegression(C=1.0, max_iter=500)` or `sklearn.ensemble.RandomForestClassifier(n_estimators=100)`.
   - Output label: `1`, `2`, or `3` mapping directly to Tier 1, Tier 2, or Tier 3.
4. **Overhead Logging**:
   - Measure router latency $\Delta t_{\text{route}}$ (in milliseconds).
   - Target overhead: $\Delta t_{\text{route}} < 15\text{ ms}$ on typical modern CPU.

---

## 6. Required Evaluation Metrics

Ensure test runners record and compute all metrics defined in Section 3.5 of the paper:

1. **Energy Metrics**:
   - Total Workload Energy: $E_{\text{total}}$ in Joules / kiloJoules.
   - Dynamic Energy: $E_{\text{net}} = E_{\text{total}} - (P_{\text{idle}} \times \text{latency})$.
   - Joules per Output Token:
     $$\text{Joules/Token} = \frac{E_{\text{net}}}{N_{\text{output\_tokens}}}$$
   - Microarchitectural Energy Scaling Ratio:
     $$\eta = \frac{E_{\text{RTX 3060 Ti}}}{E_{\text{RTX 4060}}}$$
2. **Quality Metrics**:
   - Task performance score on ground truth ($Score$).
   - **Quality Preservation Ratio (QPR)** relative to the static 8B baseline:
     $$\text{QPR} = \frac{\text{Score}_{\text{Router}}}{\text{Score}_{\text{Tier 3 Static}}} \times 100\%$$
3. **Latency Profile**:
   - **Time-to-First-Token (TTFT)**: Track duration from request dispatch until arrival of first streamed chunk.
   - **End-to-End Latency**: Total seconds from dispatch until completion.
   - **Router Overhead**: $\Delta t_{\text{route}}$ in seconds or milliseconds.

---

## 7. Output Artifacts & Data Formats

Test scripts must record benchmark results into structured CSV or JSON files inside a dedicated directory (e.g., `results/`):

```json
{
  "test_id": "gsm8k_042",
  "tier_selected": 3,
  "model_used": "llama3.1:8b",
  "prompt_token_count": 86,
  "output_token_count": 142,
  "latency_seconds": 4.12,
  "ttft_seconds": 0.28,
  "router_overhead_seconds": 0.0084,
  "p_idle_watts": 21.3,
  "total_energy_joules": 182.4,
  "net_energy_joules": 94.6,
  "joules_per_token": 0.666,
  "quality_metric": "accuracy",
  "quality_score": 1.0
}
```

---

## 8. Non-NVIDIA Host Development Considerations

When developing, linting, or running unit tests on non-NVIDIA machines (e.g. macOS / Apple Silicon):
- NVML calls will fail because `nvidia-ml-py` requires an NVIDIA driver and hardware.
- Provide a mock/dummy power sampler mode (`MockPowerMonitor`) for unit testing router logic and dataset ingestion when `pynvml` cannot initialize.
- Mark hardware tests with an environment check:
  ```python
  try:
      import pynvml
      pynvml.nvmlInit()
      NVML_AVAILABLE = True
  except Exception:
      NVML_AVAILABLE = False
  ```
