# Head-to-Head Model Tier Comparison: Meta Llama vs. Alibaba Qwen 2.5

This evaluation analyzes the 2 candidate models per tier across all 3 complexity tiers under an 8 GB VRAM budget:

| Tier | Model | Family | Scale | Precision | VRAM | Dynamic Energy $E_{\text{net}}$ | Energy Intensity | Target Quality | Overall Quality | Latency | Quality / kJ | Recommendation |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Tier 1 | `Llama-3.2-1B` | Llama | 1.24B | Q8_0 | 1.3 GB | 216.09 J | 7.3792 J/tok | 75.0% | 61.1% | 3.777s | 2.83 | ⭐ **Recommended** |
| Tier 1 | `Qwen-2.5-1.5B` | Qwen | 1.7B | Q4_K_M | 1.1 GB | 1329.05 J | 2.1906 J/tok | 75.0% | 61.2% | 11.051s | 0.46 | Candidate |
| Tier 2 | `Llama-3.2-3B` | Llama | 3.21B | Q4_K_M | 2.0 GB | 469.21 J | 17.6972 J/tok | 59.9% | 86.6% | 10.117s | 1.85 | ⭐ **Recommended** |
| Tier 2 | `Qwen-2.5-3B` | Qwen | 4.0B | Q4_K_M | 2.3 GB | 3065.06 J | 1.802 J/tok | 59.4% | 86.5% | 17.623s | 0.28 | Candidate |
| Tier 3 | `Llama-3.1-8B` | Llama | 8.03B | Q4_K_M | 4.9 GB | 1968.7 J | 11.1262 J/tok | 100.0% | 77.6% | 17.769s | 0.39 | ⭐ **Recommended** |
| Tier 3 | `Qwen-2.5-7B` | Qwen | 8.0B | Q4_K_M | 4.8 GB | 3508.47 J | 3.0841 J/tok | 75.0% | 78.7% | 21.862s | 0.22 | Candidate |