# Empirical Evaluation & Cross-Platform Results

This document is the Markdown companion to [`paper_results_section.tex`](file:///e:/UIU/Trimester%2012/Green/Project/LLM_Energy_Benchmark/paper_results_section.tex). It details the empirical evaluation, dual-GPU benchmarking data, and microarchitectural scaling analysis across:

> **Platform A: NVIDIA GeForce RTX 4060** (Ada Lovelace, TSMC 4N process, 115W TDP, 8 GB VRAM)  
> **Platform B: NVIDIA GeForce RTX 3060 Ti** (Ampere, Samsung 8nm process, 200W TDP, 8 GB VRAM)

---

## 1. Hardware Environment & Quiescent Power Profiles

| Metric | Platform A: RTX 4060 | Platform B: RTX 3060 Ti | Microarchitectural Comparison |
| :--- | :---: | :---: | :---: |
| **Architecture** | Ada Lovelace | Ampere | Generational Leap |
| **Process Node** | TSMC 4N | Samsung 8nm | Advanced Lithography |
| **Rated TDP** | 115 W | 200 W | Lower Thermal Envelope on 4060 |
| **VRAM Capacity** | 8 GB GDDR6 | 8 GB GDDR6 | Identical 8 GB Budget |
| **Quiescent Resting Power ($P_{\text{idle}}$)** | **$54.10 \pm 0.12\text{ W}$** | **$14.76 \pm 0.08\text{ W}$** | Hardware Testbed Profile |

---

## 2. Cross-Platform System-Level Results Comparison

| Evaluation Metric | Platform A (RTX 4060) Baseline | Platform A (RTX 4060) Router | Platform A Savings | Platform B (RTX 3060 Ti) Baseline | Platform B (RTX 3060 Ti) Router | Platform B Savings |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Total Energy ($E_{\text{total}}$)** | $8.545\text{ kJ}$ | **$6.504\text{ kJ}$** | **$-23.88\%$** | $7.946\text{ kJ}$ | **$4.455\text{ kJ}$** | **$-43.93\%$** |
| **Net Dynamic Energy ($E_{\text{net}}$)** | $2,766.36\text{ J}$ | **$1,922.88\text{ J}$** | **$-30.49\%$** | $6,847.91\text{ J}$ | **$3,606.48\text{ J}$** | **$-47.33\%$** |
| **Mean Task Quality Score** | $0.8637$ | **$0.7800$** | $-9.69\%$ | $0.7827$ | **$0.6988$** | $-10.72\%$ |
| **Quality Preservation Ratio (QPR)**| $100.00\%$ | **$90.31\%$** | Target Met | $100.00\%$ | **$89.29\%$** | Target Met |
| **Mean End-to-End Latency** | $8.902\text{ s}$ | **$7.057\text{ s}$** | **$-20.73\%$** | $6.201\text{ s}$ | **$4.791\text{ s}$** | **$-22.74\%$** |
| **Mean TTFT** | $5.229\text{ s}$ | **$4.431\text{ s}$** | **$-15.26\%$** | $3.931\text{ s}$ | **$4.019\text{ s}$** | $+2.24\%$ |
| **Host CPU Router Overhead** | --- | **$23.43\text{ ms}$** | Negligible | --- | **$21.88\text{ ms}$** | Negligible |

---

## 3. Microarchitectural Energy Scaling Ratio ($\eta$)

To evaluate architectural silicon efficiency across generations, we calculate the empirical ratio:
$$\eta = \frac{E_{\text{net}}(\text{Platform B: RTX 3060 Ti})}{E_{\text{net}}(\text{Platform A: RTX 4060})}$$

- **Static 8B Baseline**:
  $$\eta_{\text{Static Baseline}} = \frac{6,847.91\text{ J}}{2,766.36\text{ J}} = \mathbf{2.475\times}$$
  The older Ampere Samsung 8nm architecture consumed nearly **$2.5\times$ more dynamic energy** for identical static serving workloads compared to Ada Lovelace TSMC 4N.
- **Adaptive Complexity Router**:
  $$\eta_{\text{Adaptive Router}} = \frac{3,606.48\text{ J}}{1,922.88\text{ J}} = \mathbf{1.876\times}$$
  Dynamic complexity routing compresses the microarchitectural disparity from $2.475\times$ to $1.876\times$.

> [!IMPORTANT]
> **Core Takeaway**: While the TSMC 4N process node delivers a $1.9\times\text{--}2.5\times$ hardware efficiency advantage, **dynamic complexity routing delivers an orthogonal, compounding 30.5% (4060) to 47.3% (3060 Ti) dynamic energy reduction on both platforms**.

---

## 4. Task-Level Breakdown Across Benchmarks

| Benchmark Dataset | Complexity | Tier | RTX 4060 Baseline | RTX 4060 Router | RTX 3060 Ti Baseline | RTX 3060 Ti Router | Quality 4060 (B / R) | Quality 3060 Ti (B / R) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SST-2** | Low | Tier 1 (1B) | $69.45\text{ J}$ | **$24.18\text{ J}$** ($-65.2\%$) | $318.70\text{ J}$ | **$232.50\text{ J}$** ($-27.0\%$) | $1.00$ / $1.00$ | $1.00$ / $1.00$ |
| **SQuAD v2.0** | Low/Med | Tier 2 (3B) | $40.05\text{ J}$ | **$27.01\text{ J}$** ($-32.5\%$) | $723.80\text{ J}$ | **$271.40\text{ J}$** ($-62.5\%$) | $1.00$ / $1.00$ | $1.00$ / $1.00$ |
| **CNN/DailyMail** | Medium | Tier 2 (3B) | $222.05\text{ J}$ | **$126.93\text{ J}$** ($-42.8\%$) | $542.90\text{ J}$ | **$390.10\text{ J}$** ($-28.1\%$) | $0.18$ / $0.18$ | $0.20$ / $0.19$ |
| **MNLI** | Medium | Tier 2 (3B) | $178.85\text{ J}$ | **$50.10\text{ J}$** ($-72.0\%$) | $269.80\text{ J}$ | $314.10\text{ J}$ | $1.00$ / $0.50$ | $0.50$ / $0.50$ |
| **GSM8K** | High | Tier 3 (8B) | $192.15\text{ J}$ | $199.83\text{ J}$ | $858.70\text{ J}$ | **$408.00\text{ J}$** ($-52.5\%$) | $1.00$ / $1.00$ | $1.00$ / $1.00$ |
| **HumanEval** | High | Tier 3 (8B) | $680.65\text{ J}$ | **$533.40\text{ J}$** ($-21.6\%$) | $710.10\text{ J}$ | **$187.10\text{ J}$** ($-73.7\%$) | $1.00$ / $1.00$ | $1.00$ / $0.50$ |

---

## 5. Generated Publication Figures

The figures are organized into dedicated subdirectories in [`figures/`](file:///e:/UIU/Trimester%2012/Green/Project/LLM_Energy_Benchmark/figures):

### Platform A: RTX 4060
- `figures/rtx4060/energy_and_joules_per_token.png`
- `figures/rtx4060/router_tier_distribution.png`
- `figures/rtx4060/dataset_energy_breakdown.png`
- `figures/rtx4060/latency_profile.png`

### Platform B: RTX 3060 Ti
- `figures/rtx3060ti/energy_and_joules_per_token.png`
- `figures/rtx3060ti/router_tier_distribution.png`
- `figures/rtx3060ti/dataset_energy_breakdown.png`
- `figures/rtx3060ti/latency_profile.png`

### Cross-Platform Scaling
- `figures/cross_platform/cross_platform_scaling_eta.png`
- `figures/cross_platform/cross_platform_joules_per_token.png`
