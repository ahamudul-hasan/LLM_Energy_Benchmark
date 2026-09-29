# Head-to-Head Model Tier Comparison: Meta Llama vs. Alibaba Qwen 2.5

This evaluation analyzes the 2 candidate models per tier across all 3 complexity tiers under an 8 GB VRAM budget:

| Tier | Model | Family | Scale | Precision | VRAM | Dynamic Energy $E_{\text{net}}$ | Energy Intensity | Target Quality | Overall Quality | Latency | Quality / kJ | Recommendation |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Tier 1 | `Llama-3.2-1B` | Llama | 1.24B | Q8_0 | 1.3 GB | 138.32 J | 5.4158 J/tok | 75.0% | 69.4% | 3.934s | 5.02 | ⭐ **Recommended** |
| Tier 1 | `Qwen-2.5-1.5B` | Qwen | 1.54B | Q4_K_M | 1.0 GB | 192.48 J | 15.9975 J/tok | 100.0% | 69.8% | 3.004s | 3.63 | Candidate |
| Tier 2 | `Llama-3.2-3B` | Llama | 3.21B | Q4_K_M | 2.0 GB | 159.92 J | 3.7253 J/tok | 34.0% | 78.0% | 10.458s | 4.88 | ⭐ **Recommended** |
| Tier 2 | `Qwen-2.5-3B` | Qwen | 3.09B | Q4_K_M | 1.9 GB | 208.11 J | 2.3682 J/tok | 61.3% | 87.1% | 3.579s | 4.19 | Candidate |
| Tier 3 | `Llama-3.1-8B` | Llama | 8.03B | Q4_K_M | 4.9 GB | 373.06 J | 2.0075 J/tok | 100.0% | 85.8% | 7.164s | 2.3 | ⭐ **Recommended** |
| Tier 3 | `Qwen-2.5-7B` | Qwen | 7.61B | Q4_K_M | 4.7 GB | 424.54 J | 4.1869 J/tok | 100.0% | 87.8% | 6.737s | 2.07 | Candidate |