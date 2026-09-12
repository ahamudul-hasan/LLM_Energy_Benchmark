# Energy-Aware LLM Routing — Complete Project Plan

**Course:** Academic Green Computing
**Paper title:** *A Comparative Analysis of Energy Efficiency Across Large Language Models*
**Team:** Fatin Israq Talha · Mehrab Sadekin Borno · Md Rakib Hasan Rabbi · Muhmmad Ahamadul Hasan Prianto
**Affiliation:** Dept. of CSE, United International University, Dhaka

---

## 0. Where things stand right now

| Item | Status |
|---|---|
| Literature review | ✅ Done — 17 published references + TokenPowerBench (AAAI 2026), all arXiv-only preprints removed |
| Gap analysis table | ✅ Done — 5 rows, matches trimmed reference list |
| Methodology section | ✅ Finalized — Defined in AGENTS.md and README.md |
| Hardware | ✅ Finalized — NVIDIA RTX 4060 (Ada Lovelace 8GB) & RTX 3060 Ti (Ampere 8GB) |
| Measurement Harness | ✅ Done — `src/measure_energy.py` (NVML 50ms sampling, idle power drop, trapezoidal integration) |
| Router design | ✅ Finalized — Host CPU feature pipeline (surface heuristics + `all-MiniLM-L6-v2`) + Logistic Regression / Random Forest |
| Workload Benchmark | 🔶 In progress — 600 prompts (200/complexity tier across 6 datasets: SST-2, SQuAD v2.0, CNN/DailyMail, MNLI, GSM8K, HumanEval) |

---

## 1. Finalized Architecture & System Decisions

All core hardware, model tier, and routing decisions are locked as specified in [AGENTS.md](file:///e:/UIU/Trimester%2012/Green/Project/LLM_Energy_Benchmark/AGENTS.md):

### 1.1 Dual GPU Hardware Platforms (8 GB VRAM Constraint)
- **Platform A**: NVIDIA GeForce RTX 4060 (Ada Lovelace, 8 GB GDDR6, TSMC 4N, 115 W TDP).
- **Platform B**: NVIDIA GeForce RTX 3060 Ti (Ampere, 8 GB GDDR6, Samsung 8nm, 200 W TDP).
- Enables evaluation of microarchitectural energy scaling ratio ($\eta = E_{\text{RTX 3060 Ti}} / E_{\text{RTX 4060}}$).

### 1.2 Model Candidate Pool & Quantization Tiers (Ollama / GGUF)
- **Tier 1 (Lightweight)**: `llama3.2:1b` (`Q8_0`, ~1.3 GB VRAM) — Target: Binary sentiment (SST-2), short fact extraction (SQuAD v2.0).
- **Tier 2 (Intermediate)**: `llama3.2:3b` / `phi3.5:3.8b` (`Q4_K_M`, ~2.6–3.1 GB VRAM) — Target: Multi-sentence summarization (CNN/DailyMail), NLI (MNLI).
- **Tier 3 (Heavyweight)**: `llama3.1:8b` / `mistral:7b` (`Q4_K_M`, ~5.2–5.8 GB VRAM) — Target: Math reasoning (GSM8K), code synthesis (HumanEval).

### 1.3 Host-CPU Prompt Complexity Router
- **Feature Extraction**: Token length, character count, code syntax flags (`def`, `class`, `import`, `curl`, SQL) + 384-d dense embeddings (`all-MiniLM-L6-v2`) on host CPU.
- **Classifier**: Multi-class Logistic Regression ($L_2$) or Random Forest predicting target tier $\hat{y} \in \{1, 2, 3\}$.
- **Target Overhead**: $\Delta t_{\text{route}} < 15\text{ ms}$.

### 1.4 Power Measurement Invariants (`pynvml`)
- **NVML Polling**: Background daemon thread sampling $P(t)$ every 50 ms (`time.sleep(0.05)`).
- **Idle Power**: Mandatory pre-flight GPU cool-down and 3.0s baseline resting power profiling ($P_{\text{idle}}$).
- **Net Energy**: Trapezoidal integration isolating dynamic energy: $E_{\text{net}} = E_{\text{total}} - (P_{\text{idle}} \times \Delta t_{\text{latency}})$.

---

## 2. Repository & environment setup (do this once, first)

### 2.1 Folder structure
```
green-computing-project/
├── README.md
├── requirements.txt
├── configs/
│   └── models.yaml              # model names, HF repo IDs, quant settings
├── src/
│   ├── quantize.py              # loads + quantizes a model
│   ├── measure_energy.py        # wraps inference in CodeCarbon + nvidia-smi polling
│   ├── run_benchmark.py         # runs a model×quant config against a task benchmark
│   ├── build_lookup_table.py    # turns raw results into the router's decision table
│   ├── router.py                # the router itself (rule-based v1, classifier v2)
│   └── evaluate_router.py       # router vs. baseline on a mixed prompt stream
├── data/
│   ├── prompts/                 # sampled benchmark subsets, one file per task
│   └── results/                 # raw energy + accuracy logs (CSV/JSON)
├── notebooks/
│   └── analysis.ipynb           # plots, energy-accuracy curves
└── paper/
    ├── main.tex
    ├── methodology.tex
    └── ref.bib
```

### 2.2 Environment
```bash
python -m venv venv
source venv/bin/activate          # or venv\Scripts\activate on Windows

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes auto-gptq autoawq
pip install codecarbon
pip install lm-eval                # lm-evaluation-harness
pip install pandas matplotlib seaborn scikit-learn
pip install nvidia-ml-py            # NVML bindings for power polling
```
`llama.cpp` (only needed if you go the GGUF/CPU route for any config): build separately per its README, or use the `llama-cpp-python` pip package.

### 2.3 Shared logging convention
Use one shared spreadsheet (Google Sheets or a `results/summary.csv` committed to the repo) with columns:
`model | quant_level | task | run_number | energy_kwh | co2_kg | latency_s | tokens_generated | accuracy_score | gpu_model | timestamp`

Every group member logs into the same sheet/file so Phase 6 analysis doesn't require merging four different formats later.

### 2.4 Git setup
```bash
git init
git add .
git commit -m "Initial project structure"
```
Add a `.gitignore` for `venv/`, `__pycache__/`, model weight caches, and large result files if they get big — keep only summarized CSVs in git, not raw model checkpoints.

---

## 3. Phase-by-phase plan

### Phase 1 — Literature Review ✅ *Complete*
- [x] 17 published references identified and verified
- [x] TokenPowerBench (AAAI 2026) added
- [x] Related Works narrative + Gap Analysis table updated and renumbered
- [x] `ref.bib` matches all in-text citations

---

### Phase 2 — Model & Quantization Pipeline
**Goal:** get all model×quantization configurations loading and generating coherent text.

- [ ] Finalize model list per decision 1.1/1.4 above
- [ ] Write `configs/models.yaml` listing each model's Hugging Face repo ID and target quant settings
- [ ] Implement `src/quantize.py`:
  - FP16: load directly via `transformers.AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float16)`
  - 8-bit: `load_in_8bit=True` via `bitsandbytes`
  - 4-bit: `load_in_4bit=True` (bitsandbytes NF4) **or** a calibrated GPTQ/AWQ checkpoint via `AutoGPTQ`/`AutoAWQ` if you want post-training calibration instead of on-the-fly quantization
- [ ] Sanity-check: for each of the 9 configs, generate a response to 2–3 hand-picked prompts and manually verify the output is coherent (not garbled — a common quantization failure mode)
- [ ] Record load time and peak VRAM usage per config (useful context for the paper, and an early warning if a config won't fit)

**Output:** 9 working model×quant configurations, each verified to produce sane output.
**Est. time:** 3–5 days (mostly download + debugging quant library quirks).

---

### Phase 3 — Task Benchmarks & Prompt Dataset
**Goal:** a fixed, reusable prompt set per task category, small enough to run 9× without burning the whole semester.

- [ ] Pick and download subsets (verify exact Hugging Face dataset path/config when implementing — names below are close but confirm on the HF Hub):
  - QA/reasoning: 100–300 examples from MMLU or ARC-Challenge
  - Math: 100–300 examples from GSM8K
  - Code: full HumanEval (164 problems — small enough to use in full)
  - Summarization: 100–200 examples from XSum or CNN/DailyMail
- [ ] Save each sampled subset to `data/prompts/<task>.jsonl` so every run uses an identical prompt set (critical for fair comparison)
- [ ] Decide and document the prompt template per task (few-shot vs. zero-shot, exact instruction wording) — keep this **identical** across all 9 model×quant configs
- [ ] Write a short `data/prompts/README.md` noting exact source, sampling method (e.g. random seed used), and template

**Output:** 4 frozen prompt files + documented prompting protocol.
**Est. time:** 2–3 days.

---

### Phase 4 — Energy Measurement Harness
**Goal:** a script that runs one config against one task and logs energy, latency, and token counts.

- [ ] Implement `src/measure_energy.py`:
  - Wrap the generation call with `codecarbon.EmissionsTracker` (`tracker.start()` / `tracker.stop()`)
  - In parallel, poll `nvidia-smi --query-gpu=power.draw --format=csv -l 1` (or NVML `nvmlDeviceGetPowerUsage`) in a background thread during the same call, and integrate power over time to get an independent joule estimate
  - If any config runs on CPU (GGUF/llama.cpp), swap in a Linux RAPL reader (`/sys/class/powercap/intel-rapl/`) instead
- [ ] Run every (model, quant, task) combination **3 times**, discard the first as a warm-up (cache/driver warm-up skews the first run), average the remaining 2 — or average all 3 if variance is low
- [ ] Append every run's numbers to `data/results/summary.csv` per the schema in §2.3
- [ ] Cross-check: for at least one config, compare CodeCarbon's kWh estimate against your direct NVML integration — note the discrepancy in the paper as a measurement-validity point

**Output:** `data/results/summary.csv` populated with energy + latency for all 9×4 = 36 (model, quant, task) combinations.
**Est. time:** 1–2 weeks of actual compute time (runs take real wall-clock time; budget for reruns if numbers look wrong).

---

### Phase 5 — Accuracy Evaluation
**Goal:** an accuracy score per (model, quant, task) combination, using the same prompt sets from Phase 3.

- [ ] Run each config through `lm-evaluation-harness` (or a custom scorer if a task needs one lm-eval doesn't cover well) against the frozen prompt sets
- [ ] Scoring per task:
  - QA/reasoning, math: exact-match accuracy
  - Code: `pass@1` (execute generated code against HumanEval's test cases)
  - Summarization: ROUGE-L
- [ ] Append accuracy scores to the same `summary.csv` rows from Phase 4 (join on model+quant+task)

**Output:** `summary.csv` fully populated — energy, latency, and accuracy for all 36 combinations.
**Est. time:** 3–5 days (can run in parallel with Phase 4 if you have benchmark scoring set up early, since it doesn't need the same GPU-exclusive measurement window).

---

### Phase 6 — Analysis
**Goal:** turn raw numbers into the energy–accuracy curves that justify the router's decisions.

- [ ] Load `summary.csv` into `notebooks/analysis.ipynb` with pandas
- [ ] For each task, plot energy (x-axis) vs. accuracy (y-axis) across the 9 model×quant configs — this is your key paper figure
- [ ] Identify the Pareto-optimal configs per task (the ones where no other config is both cheaper *and* more accurate)
- [ ] Estimate water impact: apply a published energy-to-water conversion factor (cite Li et al. [16] in your reference list) to your measured kWh totals — label this clearly as an estimate, not a direct measurement
- [ ] Save the per-task Pareto frontier as `data/results/lookup_table.csv` — this feeds directly into Phase 7

**Output:** energy-accuracy plots (paper figures) + `lookup_table.csv`.
**Est. time:** 3–4 days.

---

### Phase 7 — Router v1 (rule-based)
**Goal:** given a task category and an accuracy threshold, pick the cheapest config that meets it.

- [ ] Implement `src/router.py`:
  - Task classification: simple heuristic first — keyword/regex detection (code syntax → code task, numeric operators + "solve"/"calculate" → math, etc.) plus a length-based fallback. This doesn't need to be fancy for v1.
  - Configuration selection: look up the task's row in `lookup_table.csv`, pick the lowest-energy config whose accuracy ≥ your chosen threshold (e.g. within 5% of the best observed accuracy for that task)
- [ ] Unit-test the router on a handful of hand-written prompts per task to confirm it routes sensibly before the full evaluation

**Output:** working `router.py` that takes a prompt string and returns a (model, quant) choice.
**Est. time:** 2–3 days.

---

### Phase 8 — Router Evaluation
**Goal:** the headline result — energy saved vs. the "always use the biggest model" baseline.

- [ ] Build a held-out mixed prompt stream: sample additional prompts (not used in Phases 4–5) proportionally from all 4 task categories
- [ ] Run the stream through two systems:
  1. **Baseline:** every prompt → largest model, FP16
  2. **Router:** every prompt → router's chosen config
- [ ] Log total energy, mean accuracy, and mean latency for both
- [ ] Compute: % energy reduction, accuracy delta, latency delta

**Output:** the core results table/figure for the paper.
**Est. time:** 3–4 days.

---

### Phase 9 — Router v2 (stretch goal — only if time allows after Phase 8)
- [ ] Replace the keyword-based task classifier with a `scikit-learn` classifier (logistic regression or small decision tree) trained on labeled prompt examples (task category as the label)
- [ ] Features: token length, keyword presence flags, optionally sentence-embedding similarity to labeled examples per task
- [ ] Re-run Phase 8's evaluation with Router v2, compare against Router v1

**Output:** optional second results section showing whether a learned classifier beats the heuristic.
**Est. time:** 1 week, only attempt if Phases 1–8 are done with buffer remaining.

---

### Phase 10 — Paper Writing
Maps directly onto what you've built:

- [x] Abstract, Introduction — done
- [x] Related Works, Gap Analysis — done
- [ ] Methodology (Phases 2–7 above, in prose) — in progress, finalize once decisions in §1 are locked
- [ ] Experimental Setup (exact hardware, software versions, hyperparameters used)
- [ ] Results (Phase 6 plots, Phase 8 table)
- [ ] Discussion / Limitations — mention: CodeCarbon vs. direct NVML discrepancy, small benchmark subset sizes, single-hardware measurement, water estimate is derived not measured
- [ ] Conclusion
- [ ] Final reference/citation check against the paper body

**Est. time:** ongoing throughout, concentrated effort in the final 1–2 weeks.

---

### Phase 11 — Final Review & Submission
- [ ] Full internal read-through by all 4 members
- [ ] Check every figure/table is referenced in the text
- [ ] Check every in-text citation has a matching `ref.bib` entry and vice versa
- [ ] Format check against the course's submission rubric
- [ ] Compile a clean final PDF from Overleaf
- [ ] Prepare slides/demo if the course requires a presentation

---

## 4. Suggested role split

| Member | Suggested focus | Primary phases |
|---|---|---|
| _(assign)_ | Model & quantization pipeline | Phase 2 |
| _(assign)_ | Energy measurement harness (needs the measurement machine) | Phase 4 |
| _(assign)_ | Accuracy benchmarking | Phase 3, 5 |
| _(assign)_ | Router design, analysis, paper coordination | Phase 6, 7, 8, 10 |

Everyone contributes to Phase 10 (writing) and Phase 11 (review) regardless of primary role.

---

## 5. Risk checklist — flag these explicitly in the paper's Limitations

- [ ] CodeCarbon draws on TDP/grid-average estimates unless cross-validated against direct GPU power polling — note wherever the two diverge
- [ ] Small benchmark subsets (100–300 examples) trade statistical power for feasibility — say so explicitly
- [ ] All measurements from a single GPU model — results may not generalize to other hardware
- [ ] Water-impact numbers are derived estimates via a published conversion factor, not direct measurements
- [ ] If any config had to be dropped due to VRAM (per §1.1), state which and why
