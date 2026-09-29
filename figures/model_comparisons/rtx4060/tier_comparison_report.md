# Head-to-Head Model Tier Comparison: Meta Llama vs. Alibaba Qwen 2.5

This evaluation analyzes the 2 candidate models per tier across all 3 complexity tiers under an 8 GB VRAM budget:

| Tier | Model | Family | Scale | Precision | VRAM | Dynamic Energy $E_{\text{net}}$ | Energy Intensity | Target Quality | Overall Quality | Latency | Quality / kJ | Recommendation |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Tier 1 | `Llama-3.2-1B` | Llama | 1.24B | Q8_0 | 1.3 GB | 62.27 J | 0.9741 J/tok | 75.0% | 69.4% | 5.216s | 11.14 | Candidate |
| Tier 1 | `Qwen-2.5-1.5B` | Qwen | 1.54B | Q4_K_M | 1.0 GB | 54.47 J | 0.5816 J/tok | 100.0% | 77.6% | 3.54s | 14.25 | Candidate |
| Tier 2 | `Llama-3.2-3B` | Llama | 3.21B | Q4_K_M | 2.0 GB | 118.34 J | 1.6576 J/tok | 32.4% | 77.5% | 10.881s | 6.55 | Candidate |
| Tier 2 | `Qwen-2.5-3B` | Qwen | 3.09B | Q4_K_M | 1.9 GB | 100.19 J | 0.6114 J/tok | 60.2% | 86.7% | 4.326s | 8.66 | Candidate |
| Tier 3 | `Llama-3.1-8B` | Llama | 8.03B | Q4_K_M | 4.9 GB | 209.13 J | 2.3903 J/tok | 100.0% | 86.3% | 10.741s | 4.13 | ⭐ **Recommended** |
| Tier 3 | `Qwen-2.5-7B` | Qwen | 7.61B | Q4_K_M | 4.7 GB | 286.41 J | 1.7958 J/tok | 100.0% | 87.0% | 12.573s | 3.04 | Candidate |