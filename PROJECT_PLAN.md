# Energy-Aware LLM Routing — Complete Project Plan

**Course:** Academic Green Computing  
**Paper title:** *A Comparative Analysis of Energy Efficiency Across Large Language Models*  
**Team:** Fatin Israq Talha · Mehrab Sadekin Borno · Md Rakib Hasan Rabbi · Muhmmad Ahamadul Hasan Prianto  
**Affiliation:** Dept. of CSE, United International University, Dhaka  

---

## 0. Current Project Status

| Item | Status | Notes / Reference |
|---|---|---|
| Literature review | ✅ Done | 17 published references + TokenPowerBench (AAAI 2026) |
| Gap analysis table | ✅ Done | Table 1 in `main.pdf` (Direct NVML + Quantization + Serving-time routing) |
| Methodology section | ✅ Finalized | Sections 3.1, 3.2, 3.3 in `main.pdf` & `AGENTS.md` |
| Hardware testbed | ✅ Finalized | Platform A: RTX 4060 (Ada Lovelace 8GB) & Platform B: RTX 3060 Ti (Ampere 8GB) |
| Measurement Harness | ✅ Done | `src/measure_energy.py` (NVML 50ms sampling, idle baseline subtraction, trapezoidal integration) |
| Router Architecture | ✅ Finalized | Two-phase pipeline: Host CPU feature pipeline (heuristics + `all-MiniLM-L6-v2` ONNX) + Logistic Regression / Random Forest + $\tau_1, \tau_2$ FNR < 5% threshold calibration |
| Workload Benchmark | 🔶 In progress | 600 evaluation prompts across 6 datasets (SST-2, SQuAD v2.0, CNN/DailyMail, MNLI, GSM8K, HumanEval) |

---

## 1. Finalized Architecture & System Decisions

All core hardware, model tier, energy measurement, and routing decisions are locked as specified in Section 3 of `main.pdf` and [AGENTS.md](file:///e:/UIU/Trimester%2012/Green/Project/LLM_Energy_Benchmark/AGENTS.md):

### 1.1 Dual GPU Hardware Platforms (8 GB VRAM Constraint)
- **Platform A**: NVIDIA GeForce RTX 4060 (Ada Lovelace, 8 GB GDDR6, TSMC 4N, 115 W TDP, 3072 CUDA Cores).
- **Platform B**: NVIDIA GeForce RTX 3060 Ti (Ampere, 8 GB GDDR6, Samsung 8nm, 200 W TDP, 4864 CUDA Cores).
- **Purpose**: Enables evaluation of microarchitectural energy scaling ratio ($\eta = E_{\text{RTX 3060 Ti}} / E_{\text{RTX 4060}}$).

### 1.2 Model Candidate Pool & Quantization Tiers (Ollama / GGUF)
- **Tier 1 (Lightweight / Low Complexity)**: `Llama-3.2-1B-Instruct` served in `Q8_0` (~1.3 GB VRAM). Target tasks: Binary sentiment (SST-2), short fact extraction (SQuAD v2.0).
- **Tier 2 (Intermediate / Medium Complexity)**: `Llama-3.2-3B-Instruct` / `Phi-3.5-mini-3.8B` served in `Q4_K_M` (~2.6–3.1 GB VRAM). Target tasks: Abstractive summarization (CNN/DailyMail), NLI (MNLI).
- **Tier 3 (Heavyweight / High Complexity)**: `Llama-3.1-8B-Instruct` / `Mistral-7B-Instruct-v0.3` served in `Q4_K_M` (~5.2–5.8 GB VRAM). Target tasks: Math reasoning (GSM8K), code synthesis (HumanEval).

### 1.3 Host-CPU Prompt Complexity Estimation & Two-Phase Routing Mechanism
- **Execution Target**: Pure host CPU execution via ONNX Runtime & `scikit-learn` ($\Delta t_{\text{route}} < 5\text{ ms}$, $\Delta E_{\text{route}} < 0.05\text{ J}$), avoiding GPU context switches or VRAM contention.
- **Feature Vector $x \in \mathbb{R}^d$**:
  - *Lexical & Syntactic Heuristics*: Token length, character count, average word length, punctuation density, boolean syntax flags (`def`, `class`, `import`, `return`, `SELECT`, `curl`).
  - *Semantic Embeddings*: 384-dimensional dense sentence embeddings via `all-MiniLM-L6-v2` (22M params) on host CPU (< 3 ms execution).
- **Classification Backends**: Multi-class $L_2$-regularized multinomial Logistic Regression and shallow Random Forest ensemble producing posterior probabilities $\hat{p}_i = P(y = \text{Tier}_i \mid x)$ for $i \in \{1, 2, 3\}$.
- **Two-Phase Operational Pipeline** (`main.pdf` Section 3.2):
  1. **Offline Training & Calibration Phase**:
     - *Training Corpus*: 3,000 prompts sampled from training splits of the 6 target datasets.
     - *Ground Truth Labeling*: Empirical evaluation to assign $y \in \{1, 2, 3\}$ (minimal tier achieving correct execution/match).
     - *Training & CV*: 80/20 train-val split, 5-fold cross-validation optimizing macro-$F_1$ under $< 5\text{ ms}$ latency constraint.
     - *Threshold Calibration $(\tau_1, \tau_2)$*: Grid search calibration on validation split enforcing False Negative Rate $\text{FNR} < 5\%$ (preventing high-complexity prompts from misrouting to lower tiers).
  2. **Online Serving Phase**:
     - Extract CPU features $\to$ Evaluate classifier probabilities $\hat{p}_i \to$ Apply confidence thresholds $(\tau_1, \tau_2) \to$ Dispatch query to predicted tier $\hat{y} \in \{1, 2, 3\}$.

### 1.4 Power Measurement & Profiling Invariants (`pynvml`)
- **NVML Polling**: Decoupled background daemon thread querying board power $P(t)$ every 50 ms (20 Hz).
- **Idle Power Profiling**: Quiescent 3.0s baseline measurement ($P_{\text{idle}}$) prior to test runs. 5.0s cool-down delay between consecutive inferences.
- **Trapezoidal Integration & Dynamic Energy**:
  $$E_{\text{total}} = \sum_{k=1}^{N-1} \frac{P(t_k) + P(t_{k+1})}{2} \cdot (t_{k+1} - t_k)$$
  $$E_{\text{net}} = E_{\text{total}} - (P_{\text{idle}} \times \Delta t_{\text{latency}})$$

---

## 2. Environment & Repository Setup

### 2.1 Directory Structure
```
LLM_Energy_Benchmark/
├── AGENTS.md                    # Core operational rules & measurement invariants
├── PROJECT_PLAN.md              # Active master project plan
├── README.md                    # Repository documentation
├── requirements.txt             # Python dependencies
├── main.pdf                     # Target paper reference
├── Tier_1/                      # Tier 1 models & benchmarking configs
├── Tier_2/                      # Tier 2 models & benchmarking configs
├── Tier_3/                      # Tier 3 models & benchmarking configs
├── src/
│   ├── measure_energy.py        # NVML 50ms power profiler & dynamic energy integration
│   ├── dataset_loader.py        # Benchmark prompt ingestion & training split sampler
│   ├── feature_extractor.py     # CPU heuristics + MiniLM 384-d ONNX embedding pipeline
│   ├── train_router.py          # Offline router training, 5-fold CV, & tau threshold calibration
│   ├── router.py                # Online sub-5ms router inference engine
│   └── evaluate_router.py       # Benchmark runner (Router vs Static Tier 3 Baseline)
├── data/
│   ├── training/                # 3,000 prompt offline router training corpus & labels
│   ├── prompts/                 # 600 benchmark evaluation prompts (100 per dataset)
│   └── results/                 # Measurement CSV/JSON outputs & evaluation logs
└── tests/
    ├── test_power_monitor.py    # Mock & NVML profiler tests
    └── test_router.py           # Feature extraction & classification unit tests
```

### 2.2 Python Environment Setup
```bash
python -m venv venv
venv\Scripts\activate            # On Windows

pip install -r requirements.txt
# Dependencies: nvidia-ml-py, sentence-transformers, onnxruntime, scikit-learn, pandas, numpy, requests, pypdf
```

---

## 3. Phase-by-Phase Implementation Plan

### Phase 1 — Literature Review & Gap Analysis ✅ *Complete*
- [x] 17 published references verified and cited.
- [x] TokenPowerBench (AAAI 2026) integrated into Related Works.
- [x] Table 1 Gap Analysis in `main.pdf` finalized (Direct NVML power sampling + Quantization + Online serving-time complexity routing).

---

### Phase 2 — Model Candidate Pool & Deployment Infrastructure
**Goal:** Configure and verify Ollama / `llama.cpp` operational model tiers on GPU within the 8 GB VRAM budget.

- [ ] Verify local installation and API endpoints for target model tiers:
  - **Tier 1**: `llama3.2:1b` (`Q8_0`, ~1.3 GB VRAM).
  - **Tier 2**: `llama3.2:3b` / `phi3.5:3.8b` (`Q4_K_M`, ~2.6–3.1 GB VRAM).
  - **Tier 3**: `llama3.1:8b` / `mistral:7b` (`Q4_K_M`, ~5.2–5.8 GB VRAM).
- [ ] Sanity-check generation API latency and response structure across all model tiers.
- [ ] Log baseline memory footprints to ensure strict compliance with the 8 GB VRAM constraint.

**Deliverables:** 3 operational model tiers accessible via local inference wrapper.  
**Est. Time:** 1–2 days.

---

### Phase 3 — Benchmark Workload Dataset Ingestion
**Goal:** Prepare frozen evaluation benchmark (600 prompts) and offline training split (3,000 prompts).

- [ ] Ingest and sample prompt datasets (`src/dataset_loader.py`):
  - **Low Complexity (200 eval prompts)**: SST-2 (100 sentiment classification) + SQuAD v2.0 (100 closed-domain QA).
  - **Medium Complexity (200 eval prompts)**: CNN/DailyMail (100 abstractive summarization) + MNLI (100 entailment classification).
  - **High Complexity (200 eval prompts)**: GSM8K (100 math word problems) + HumanEval (100 Python docstring synthesis).
- [ ] Freeze evaluation subset to `data/prompts/<dataset>.jsonl` with fixed random seeds for strict reproducibility.
- [ ] Sample 3,000 training prompts from dataset training splits into `data/training/train_corpus.jsonl`.

**Deliverables:** Frozen 600-prompt evaluation benchmark and 3,000-prompt router training split.  
**Est. Time:** 2 days.

---

### Phase 4 — High-Frequency Energy Profiling Harness
**Goal:** Validate and refine the 50 ms NVML power sampler daemon and idle power baseline calculation.

- [ ] Verify `src/measure_energy.py` profiler implementation:
  - 50 ms background daemon thread (`pynvml.nvmlDeviceGetPowerUsage`).
  - Mandatory 3.0s quiescent pre-flight idle power profiling ($P_{\text{idle}}$).
  - 5.0s cool-down delay between consecutive benchmark runs.
  - Dynamic energy trapezoidal integration: $E_{\text{net}} = E_{\text{total}} - (P_{\text{idle}} \times \Delta t_{\text{latency}})$.
  - `MockPowerMonitor` fallback for non-NVIDIA host environments (`AGENTS.md` Section 8).
- [ ] Unit test power measurement harness on dummy workloads (`tests/test_power_monitor.py`).

**Deliverables:** Production-ready `src/measure_energy.py` supporting NVML power sampling and non-NVIDIA mock mode.  
**Est. Time:** 2 days.

---

### Phase 5 — Static Model Tier Performance Benchmarking
**Goal:** Run static baselines (Tier 1, Tier 2, Tier 3) across the 600 evaluation prompts to establish ground-truth quality and energy profiles.

- [ ] Run all 600 evaluation prompts on Tier 1 (`llama3.2:1b`), Tier 2 (`llama3.2:3b`), and Tier 3 (`llama3.1:8b`).
- [ ] Compute metrics per tier and task:
  - **Quality**: Accuracy (SST-2, MNLI, GSM8K), Exact Match / Token F1 (SQuAD v2.0), ROUGE-L (CNN/DailyMail), Pass@1 execution accuracy (HumanEval).
  - **Energy & Latency**: $E_{\text{total}}$ (kJ), $E_{\text{net}}$ (J), Joules/Token, TTFT (s), End-to-End Latency (s).
- [ ] Record results in `data/results/static_tier_baselines.csv`.

**Deliverables:** Benchmark results for static model tiers across all 6 datasets.  
**Est. Time:** 3–5 days compute time.

---

### Phase 6 — Router Offline Training & Threshold Calibration
**Goal:** Implement the two-phase prompt-complexity router training pipeline (`main.pdf` Section 3.2).

- [ ] **Empirical Ground-Truth Labeling**:
  - Evaluate the 3,000 training prompts on Tier 1, Tier 2, and Tier 3.
  - Assign label $y_i \in \{1, 2, 3\}$ corresponding to the lowest model tier capable of satisfying the ground-truth accuracy/evaluation criterion.
- [ ] **CPU Feature Extraction Pipeline (`src/feature_extractor.py`)**:
  - Implement surface heuristics (token count, char count, avg word length, punctuation density, boolean syntax flags `def`, `class`, `import`, `return`, `SELECT`, `curl`).
  - Implement 384-d sentence embedding extraction using `all-MiniLM-L6-v2` via CPU ONNX Runtime (< 3 ms latency).
- [ ] **Classifier Training & Cross-Validation (`src/train_router.py`)**:
  - Concatenate heuristic + dense embedding features ($x \in \mathbb{R}^d$).
  - Train multinomial Logistic Regression ($L_2$) and shallow Random Forest models using 80/20 train-validation split.
  - Perform 5-fold cross-validation optimizing macro-$F_1$ subject to $\Delta t_{\text{route}} < 5\text{ ms}$.
- [ ] **Confidence Threshold Calibration $(\tau_1, \tau_2)$**:
  - Perform grid search calibration on the 20% validation split.
  - Tune thresholds $\tau_1$ (Tier 1 $\to$ Tier 2) and $\tau_2$ (Tier 2 $\to$ Tier 3) to enforce False Negative Rate $\text{FNR} < 5\%$.
  - Save trained model weights and calibrated parameters $(\tau_1, \tau_2)$ to `src/router_model.joblib`.

**Deliverables:** Trained, low-latency CPU prompt complexity router and calibrated escalation thresholds.  
**Est. Time:** 3–4 days.

---

### Phase 7 — Online Router Integration & Benchmark Stream Evaluation
**Goal:** Evaluate the adaptive routing system against the static Tier 3 baseline on the 600 evaluation prompts.

- [ ] Implement `src/router.py` (online inference engine):
  - Extract CPU features $\to$ Classifier probabilities $\hat{p}_i \to$ Apply threshold logic $(\tau_1, \tau_2) \to$ Dispatch to target tier $\hat{y} \in \{1, 2, 3\}$.
  - Enforce sub-5 ms execution latency ($\Delta t_{\text{route}} < 5\text{ ms}$) and $< 0.05\text{ J}$ energy overhead.
- [ ] Implement `src/evaluate_router.py`:
  - Process the 600 held-out evaluation prompts through the Adaptive Router system.
  - Log tier selection distribution, total dynamic energy ($E_{\text{net}}$), Joules/Token, quality score, TTFT, $T_{\text{e2e}}$, and router overhead ($\Delta t_{\text{route}}$).
- [ ] Compute core comparative metrics (`main.pdf` Section 3.3 & `AGENTS.md` Section 6):
  - **Quality Preservation Ratio (QPR)**: $\text{QPR} = \frac{\text{Score}_{\text{Router}}}{\text{Score}_{\text{Tier 3 Static}}} \times 100\%$
  - **Energy Savings**: Total energy reduction relative to static Tier 3 baseline.
- [ ] Output structured results to `data/results/router_evaluation.csv` and `data/results/summary_metrics.json`.

**Deliverables:** End-to-end evaluation metrics proving adaptive router energy savings and quality retention.  
**Est. Time:** 2–3 days.

---

### Phase 8 — Cross-Platform Microarchitectural Scaling Analysis
**Goal:** Compare energy metrics across Platform A (RTX 4060) and Platform B (RTX 3060 Ti).

- [ ] Execute evaluation suite on Platform B (RTX 3060 Ti).
- [ ] Calculate Microarchitectural Energy Scaling Ratio:
  $$\eta = \frac{E_{\text{RTX 3060 Ti}}}{E_{\text{RTX 4060}}}$$
- [ ] Analyze process node efficiency gains (TSMC 4N vs Samsung 8nm) under static vs routed serving workloads.

**Deliverables:** Comparative cross-platform scaling ratio and microarchitectural analysis.  
**Est. Time:** 2 days.

---

### Phase 9 — Paper & Plan Synchronization
**Goal:** Verify all experimental results match the paper structure in `main.pdf`.

- [ ] Verify that all tables, metrics, equations, and architectural descriptions in `main.pdf` match experimental code outputs.
- [ ] Generate figures for paper: Energy vs. Accuracy per tier, Router tier distribution, and QPR vs. Static Tier 3 baseline.

**Deliverables:** Paper figures and synchronized experimental logs.  
**Est. Time:** 2 days.

---

## 4. Suggested Role Split

| Team Member | Primary Focus | Assigned Phases |
|---|---|---|
| Fatin Israq Talha | Model Tier Setup & Deployment Infrastructure | Phase 2, Phase 5 |
| Mehrab Sadekin Borno | High-Frequency Energy Harness & NVML Profiler | Phase 4, Phase 8 |
| Md Rakib Hasan Rabbi | Benchmark Workload & Dataset Pipeline | Phase 3, Phase 5 |
| Muhmmad Ahamadul Hasan Prianto | Two-Phase CPU Router Architecture & Evaluation | Phase 6, Phase 7, Phase 9 |

---

## 5. Operational Risk & Constraint Checklist

- [ ] **8 GB VRAM Ceiling**: Ensure no model allocation exceeds VRAM limits; utilize GGUF `Q8_0` for Tier 1 and `Q4_K_M` for Tiers 2 & 3.
- [ ] **Strict NVML Resource Safety**: Always wrap NVML calls in `try ... finally` blocks to execute `pynvml.nvmlShutdown()`.
- [ ] **Host CPU Router Isolation**: Ensure feature extraction (`all-MiniLM-L6-v2` ONNX) and classification execute strictly on host CPU cores without invoking CUDA kernels or GPU memory.
- [ ] **False Negative Minimization**: Calibrate thresholds $(\tau_1, \tau_2)$ until validation FNR $< 5\%$ to avoid routing high-complexity reasoning queries to lightweight tiers.
- [ ] **Quiescent Thermal State**: Maintain mandatory 3.0s baseline profiling ($P_{\text{idle}}$) and 5.0s cool-down between benchmark queries.
