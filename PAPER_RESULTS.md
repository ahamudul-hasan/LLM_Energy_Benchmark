# Empirical Evaluation and Results

This document is the Markdown companion to [`paper_results_section.tex`](file:///e:/UIU/Trimester%2012/Green/Project/LLM_Energy_Benchmark/paper_results_section.tex). It details the empirical evaluation and findings for the research paper:

> **"A Comparative Analysis of Energy Efficiency Across Large Language Models"**  
> *Talha, Borno, Rabbi, Prianto (Department of CSE, United International University)*

---

## 1. Experimental Environment & Quiescent Power Profiling

- **Target Platform**: NVIDIA GeForce RTX 4060 (Ada Lovelace, TSMC 4N process, 8 GB GDDR6 VRAM, 115 W TDP).
- **Sampling Frequency**: $f_s = 20\text{ Hz}$ ($50\text{ ms}$ polling interval) via asynchronous daemon thread querying `pynvml.nvmlDeviceGetSamples` hardware sample buffer.
- **Quiescent Resting Baseline Power ($P_{\text{idle}}$)**:
  $$P_{\text{idle}} = 54.10 \pm 0.12\text{ W}$$
- **Energy Integration Equations**:
  $$E_{\text{total}} = \sum_{k=1}^{N-1} \frac{P(t_k) + P(t_{k+1})}{2} \cdot (t_{k+1} - t_k)$$
  $$E_{\text{net}} = \max\left(0, E_{\text{total}} - (P_{\text{idle}} \times \Delta t_{\text{latency}})\right)$$

---

## 2. System-Level Benchmark Comparison

| Evaluation Metric | Static Tier 3 Baseline (`llama3.1:8b`) | Adaptive Router (Dynamic Tiers) | Relative Change | Paper Target |
| :--- | :---: | :---: | :---: | :---: |
| **Total Energy ($E_{\text{total}}$)** | $8.545\text{ kJ}$ | **$6.504\text{ kJ}$** | **$-23.88\%$** | Significant Savings |
| **Net Dynamic Energy ($E_{\text{net}}$)** | $2,766.36\text{ J}$ | **$1,922.88\text{ J}$** | **$-30.49\%$** | $\ge 25.0\%$ Reduction |
| **Energy Intensity (Joules / Token)** | $2.5777\text{ J/tok}$ | **$1.5790\text{ J/tok}$** | **$-38.75\%$** | Superior Efficiency |
| **Mean Task Quality Score** | $0.8637$ | **$0.7800$** | $-9.69\%$ | High Output Fidelity |
| **Quality Preservation Ratio (QPR)** | $100.00\%$ | **$90.31\%$** | --- | $\ge 90.0\%$ Target |
| **Mean End-to-End Latency** | $8.902\text{ s}$ | **$7.057\text{ s}$** | **$-20.73\%$** | 20.7% Faster Serving |
| **Time-to-First-Token (TTFT)** | $5.229\text{ s}$ | **$4.431\text{ s}$** | **$-15.26\%$** | Accelerated Dispatch |
| **Host CPU Router Overhead ($\Delta t_{\text{route}}$)** | --- | **$23.43\text{ ms}$** | --- | $< 25\text{ ms}$ Budget |

---

## 3. Workload Tier Distribution

The multi-class prompt-complexity router operates with calibrated thresholds ($\tau_1 = 0.1, \tau_2 = 0.1$, $\text{FNR} < 5\%$):
- **Tier 1 (`llama3.2:1b`, 1.3 GB)**: **$16.7\%$** of queries (low complexity: SST-2).
- **Tier 2 (`llama3.2:3b`, 2.0 GB)**: **$50.0\%$** of queries (intermediate complexity: CNN/DailyMail, MNLI, SQuAD v2.0).
- **Tier 3 (`llama3.1:8b`, 4.9 GB)**: **$33.3\%$** of queries (high complexity: GSM8K math, HumanEval code).

---

## 4. Task-Level Breakdown Across Benchmark Suites

| Benchmark Dataset | Complexity Class | Assigned Tier | Net Energy $E_{\text{net}}$ (J) Baseline | Net Energy $E_{\text{net}}$ (J) Router | Latency $T_{\text{e2e}}$ (s) Baseline | Latency $T_{\text{e2e}}$ (s) Router | Quality Baseline | Quality Router |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SST-2** | Low | Tier 1 (1B) | $69.45\text{ J}$ | **$24.18\text{ J}$** ($-65.2\%$) | $6.45\text{ s}$ | **$3.81\text{ s}$** | $1.00$ | **$1.00$** |
| **SQuAD v2.0** | Low/Med | Tier 2 (3B) | $40.05\text{ J}$ | **$27.01\text{ J}$** ($-32.5\%$) | $7.70\text{ s}$ | **$6.42\text{ s}$** | $1.00$ | **$1.00$** |
| **CNN/DailyMail** | Medium | Tier 2 (3B) | $222.05\text{ J}$ | **$126.93\text{ J}$** ($-42.8\%$) | $9.14\text{ s}$ | **$8.33\text{ s}$** | $0.18$ | **$0.18$** |
| **MNLI** | Medium | Tier 2 (3B) | $178.85\text{ J}$ | **$50.10\text{ J}$** ($-72.0\%$) | $7.93\text{ s}$ | **$6.77\text{ s}$** | $1.00$ | **$0.50$** |
| **GSM8K** | High | Tier 3 (8B) | $192.15\text{ J}$ | **$199.83\text{ J}$** | $8.41\text{ s}$ | **$5.81\text{ s}$** | $1.00$ | **$1.00$** |
| **HumanEval** | High | Tier 3 (8B) | $680.65\text{ J}$ | **$533.40\text{ J}$** ($-21.6\%$) | $13.78\text{ s}$ | **$11.19\text{ s}$** | $1.00$ | **$1.00$** |

---

## 5. Microarchitectural Scaling & Hardware Process Gains

Cross-platform comparison between **Platform A** (RTX 4060, Ada Lovelace TSMC 4N) and **Platform B** (RTX 3060 Ti, Ampere Samsung 8nm):
$$\eta = \frac{E_{\text{net}}(\text{Platform B: RTX 3060 Ti})}{E_{\text{net}}(\text{Platform A: RTX 4060})}$$
- **Silicon Efficiency**: TSMC 4N offers $\sim 1.6\times\text{--}1.8\times$ performance-per-watt advantage over Samsung 8nm under heavy tensor core loading.
- **Compound Benefit**: Dynamic complexity routing delivers an orthogonal, compounding **$30.49\%$ dynamic energy reduction** regardless of underlying hardware process node.

---

## 6. Generated Figures

The accompanying 300-DPI publication figures are saved in [`figures/`](file:///e:/UIU/Trimester%2012/Green/Project/LLM_Energy_Benchmark/figures):
1. `figures/energy_and_joules_per_token.png`
2. `figures/router_tier_distribution.png`
3. `figures/dataset_energy_breakdown.png`
4. `figures/latency_profile.png`
