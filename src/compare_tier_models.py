"""
Comprehensive Head-to-Head Model Comparison per Tier (Llama vs Qwen 2.5).

Analyzes the two models in each operational tier:
- Tier 1: Llama-3.2-1B vs Qwen-2.5-1.5B (Target: SST-2, SQuAD v2.0)
- Tier 2: Llama-3.2-3B vs Qwen-2.5-3B   (Target: CNN/DailyMail, MNLI)
- Tier 3: Llama-3.1-8B vs Qwen-2.5-7B   (Target: GSM8K, HumanEval)

Calculates:
- Net Dynamic Energy E_net (J)
- Joules per Output Token (J/tok)
- Task-Specific Quality & Overall Quality
- Latency & TTFT Profiles
- Quality-per-Joule & Energy-Delay Product
- Crowned Tier Champion / Optimal Model Selection
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(ROOT, "data", "results", "model_comparisons")

TIER_TARGET_TASKS = {
    1: ["sst2", "squad"],
    2: ["cnn_dailymail", "mnli"],
    3: ["gsm8k", "humaneval"],
}

MODEL_METADATA = {
    "llama3.2:1b": {"family": "Llama", "tier": 1, "params": "1.24B", "vram_gb": 1.3, "precision": "Q8_0", "label": "Llama-3.2-1B"},
    "qwen2.5:1.5b": {"family": "Qwen", "tier": 1, "params": "1.54B", "vram_gb": 1.0, "precision": "Q4_K_M", "label": "Qwen-2.5-1.5B"},
    "qwen3:1.7b": {"family": "Qwen", "tier": 1, "params": "1.7B", "vram_gb": 1.1, "precision": "Q4_K_M", "label": "Qwen-2.5-1.5B"},
    
    "llama3.2:3b": {"family": "Llama", "tier": 2, "params": "3.21B", "vram_gb": 2.0, "precision": "Q4_K_M", "label": "Llama-3.2-3B"},
    "qwen2.5:3b": {"family": "Qwen", "tier": 2, "params": "3.09B", "vram_gb": 1.9, "precision": "Q4_K_M", "label": "Qwen-2.5-3B"},
    "qwen3:4b": {"family": "Qwen", "tier": 2, "params": "4.0B", "vram_gb": 2.3, "precision": "Q4_K_M", "label": "Qwen-2.5-3B"},
    
    "llama3.1:8b": {"family": "Llama", "tier": 3, "params": "8.03B", "vram_gb": 4.9, "precision": "Q4_K_M", "label": "Llama-3.1-8B"},
    "qwen2.5:7b": {"family": "Qwen", "tier": 3, "params": "7.61B", "vram_gb": 4.7, "precision": "Q4_K_M", "label": "Qwen-2.5-7B"},
    "qwen3:8b": {"family": "Qwen", "tier": 3, "params": "8.0B", "vram_gb": 4.8, "precision": "Q4_K_M", "label": "Qwen-2.5-7B"},
}


def load_dataset(family: str, gpu_slug: str) -> pd.DataFrame:
    """Loads static baseline evaluations for a model family."""
    gpu_csv = os.path.join(RESULTS_ROOT, family, gpu_slug, "static_tier_baselines.csv")
    root_csv = os.path.join(RESULTS_ROOT, family, "static_tier_baselines.csv")

    # Prefer the file that contains all three tiers
    for path in [gpu_csv, root_csv]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if len(df["target_tier"].unique()) == 3:
                    return df
            except Exception:
                pass

    if os.path.exists(gpu_csv):
        return pd.read_csv(gpu_csv)
    elif os.path.exists(root_csv):
        return pd.read_csv(root_csv)
    else:
        raise FileNotFoundError(f"Could not find baseline results for {family} on {gpu_slug}")


def normalize_model_name(name: str) -> str:
    """Normalizes model identifier strings."""
    name_clean = str(name).strip().lower()
    for key in MODEL_METADATA:
        if key in name_clean or name_clean in key:
            return key
    return name_clean


def analyze_tier_models(gpu_slug: str = "rtx4060") -> pd.DataFrame:
    """Performs per-tier head-to-head comparison between Llama and Qwen models."""
    df_llama = load_dataset("llama", gpu_slug)
    df_qwen = load_dataset("qwen", gpu_slug)

    df_combined = pd.concat([df_llama, df_qwen], ignore_index=True)
    df_combined["norm_model"] = df_combined["model_used"].apply(normalize_model_name)

    records = []
    for tier in [1, 2, 3]:
        tier_data = df_combined[df_combined["target_tier"] == tier]
        models_in_tier = tier_data["norm_model"].unique()
        target_tasks = TIER_TARGET_TASKS[tier]

        for m in models_in_tier:
            meta = MODEL_METADATA.get(m, {
                "family": "Unknown", "tier": tier, "params": "N/A", "vram_gb": 0.0,
                "precision": "Q4", "label": m
            })
            m_sub = tier_data[tier_data["norm_model"] == m]
            
            # Target tasks specific to this tier
            target_sub = m_sub[m_sub["dataset"].isin(target_tasks)]
            target_quality = target_sub["quality_score"].mean() if len(target_sub) > 0 else m_sub["quality_score"].mean()

            # Output token throughput
            total_toks = m_sub["output_token_count"].sum()
            total_time = m_sub["latency_seconds"].sum()
            throughput = total_toks / total_time if total_time > 0 else 0.0

            mean_energy = m_sub["net_energy_joules"].mean()
            mean_jpt = m_sub["joules_per_token"].mean()
            mean_quality = m_sub["quality_score"].mean()
            mean_latency = m_sub["latency_seconds"].mean()
            mean_ttft = m_sub["ttft_seconds"].mean()

            # Quality per Joule & Energy Delay Product
            qpj = (mean_quality / mean_energy * 1000.0) if mean_energy > 0 else 0.0
            edp = mean_energy * mean_latency

            records.append({
                "tier": tier,
                "model_key": m,
                "model_name": meta["label"],
                "family": meta["family"],
                "params": meta["params"],
                "precision": meta["precision"],
                "vram_gb": meta["vram_gb"],
                "sample_count": len(m_sub),
                "mean_net_energy_j": round(mean_energy, 2),
                "mean_jpt": round(mean_jpt, 4),
                "target_task_quality": round(target_quality * 100.0, 1),
                "overall_quality": round(mean_quality * 100.0, 1),
                "mean_latency_s": round(mean_latency, 3),
                "mean_ttft_s": round(mean_ttft, 3),
                "throughput_tok_s": round(throughput, 1),
                "quality_per_kj": round(qpj, 2),
                "edp_j_s": round(edp, 2)
            })

    df_results = pd.DataFrame(records).sort_values(["tier", "family"])
    return df_results


def generate_comparison_plots(df: pd.DataFrame, output_dir: str, gpu_label: str):
    """Generates publication-quality head-to-head comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    tiers = [1, 2, 3]

    # --- Plot 1: Energy & Quality Head-to-Head ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(tiers))
    width = 0.35

    llama_rows = df[df["family"] == "Llama"].sort_values("tier")
    qwen_rows = df[df["family"] == "Qwen"].sort_values("tier")

    # Dynamic Energy
    b1 = ax1.bar(x - width/2, llama_rows["mean_net_energy_j"], width, label="Llama Family", color="#d9534f", edgecolor="black")
    b2 = ax1.bar(x + width/2, qwen_rows["mean_net_energy_j"], width, label="Qwen 2.5 Family", color="#337ab7", edgecolor="black")
    ax1.set_ylabel("Mean Net Dynamic Energy $E_{net}$ (Joules)")
    ax1.set_title(f"Dynamic Energy Consumption per Query\n({gpu_label})", fontweight="bold")
    ax1.set_xticks(x, [f"Tier 1 (1B)\nLlama vs Qwen", f"Tier 2 (3B)\nLlama vs Qwen", f"Tier 3 (8B/7B)\nLlama vs Qwen"])
    ax1.legend()

    for bar in list(b1) + list(b2):
        val = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, val + max(ax1.get_ylim()[1]*0.015, 1.0), f"{val:.1f} J", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Target Task Quality
    b3 = ax2.bar(x - width/2, llama_rows["target_task_quality"], width, label="Llama Family", color="#d9534f", edgecolor="black")
    b4 = ax2.bar(x + width/2, qwen_rows["target_task_quality"], width, label="Qwen 2.5 Family", color="#337ab7", edgecolor="black")
    ax2.set_ylabel("Target Task Quality Score (%)")
    ax2.set_title(f"Target Workload Quality Score\n(Domain-Assigned Tasks)", fontweight="bold")
    ax2.set_xticks(x, [f"Tier 1\n(SST-2, SQuAD)", f"Tier 2\n(CNN/DM, MNLI)", f"Tier 3\n(GSM8K, HumanEval)"])
    ax2.set_ylim(0, 115)
    ax2.legend()

    for bar in list(b3) + list(b4):
        val = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    p1 = os.path.join(output_dir, "tier_head_to_head_energy_quality.png")
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    print(f"Generated: {p1}")

    # --- Plot 2: Joules per Token & Latency ---
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5.5))

    b5 = ax3.bar(x - width/2, llama_rows["mean_jpt"], width, label="Llama Family", color="#f0ad4e", edgecolor="black")
    b6 = ax3.bar(x + width/2, qwen_rows["mean_jpt"], width, label="Qwen 2.5 Family", color="#5bc0de", edgecolor="black")
    ax3.set_ylabel("Energy Intensity (Joules / Generated Token)")
    ax3.set_title(f"Joules per Output Token (J/tok)\n({gpu_label})", fontweight="bold")
    ax3.set_xticks(x, [f"Tier 1\n(1B scale)", f"Tier 2\n(3B scale)", f"Tier 3\n(7B/8B scale)"])
    ax3.legend()

    for bar in list(b5) + list(b6):
        val = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, val + max(ax3.get_ylim()[1]*0.015, 0.05), f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    b7 = ax4.bar(x - width/2, llama_rows["mean_latency_s"], width, label="Llama Family", color="#f0ad4e", edgecolor="black")
    b8 = ax4.bar(x + width/2, qwen_rows["mean_latency_s"], width, label="Qwen 2.5 Family", color="#5bc0de", edgecolor="black")
    ax4.set_ylabel("Mean End-to-End Latency (Seconds)")
    ax4.set_title(f"Inference Latency Profile\n({gpu_label})", fontweight="bold")
    ax4.set_xticks(x, [f"Tier 1\n(1B scale)", f"Tier 2\n(3B scale)", f"Tier 3\n(7B/8B scale)"])
    ax4.legend()

    for bar in list(b7) + list(b8):
        val = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, val + max(ax4.get_ylim()[1]*0.015, 0.1), f"{val:.2f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    p2 = os.path.join(output_dir, "tier_head_to_head_joules_latency.png")
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    print(f"Generated: {p2}")


def generate_markdown_and_latex(df: pd.DataFrame, output_dir: str):
    """Exports structured comparative tables in Markdown and LaTeX format."""
    md_lines = [
        "# Head-to-Head Model Tier Comparison: Meta Llama vs. Alibaba Qwen 2.5",
        "",
        "This evaluation analyzes the 2 candidate models per tier across all 3 complexity tiers under an 8 GB VRAM budget:",
        "",
        "| Tier | Model | Family | Scale | Precision | VRAM | Dynamic Energy $E_{\\text{net}}$ | Energy Intensity | Target Quality | Overall Quality | Latency | Quality / kJ | Recommendation |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for tier in [1, 2, 3]:
        sub = df[df["tier"] == tier].sort_values("family")
        if len(sub) == 0:
            continue
        llama_rows = sub[sub["family"] == "Llama"]
        qwen_rows = sub[sub["family"] == "Qwen"]

        winner = "Llama"
        if len(llama_rows) > 0 and len(qwen_rows) > 0:
            llama_row = llama_rows.iloc[0]
            qwen_row = qwen_rows.iloc[0]
            winner = "Qwen 2.5" if qwen_row["quality_per_kj"] > llama_row["quality_per_kj"] else "Llama"

        for _, r in sub.iterrows():
            is_winner = (r["family"] == winner)
            rec_tag = "⭐ **Recommended**" if is_winner else "Candidate"
            md_lines.append(
                f"| Tier {r['tier']} | `{r['model_name']}` | {r['family']} | {r['params']} | {r['precision']} | {r['vram_gb']} GB | {r['mean_net_energy_j']} J | {r['mean_jpt']} J/tok | {r['target_task_quality']}% | {r['overall_quality']}% | {r['mean_latency_s']}s | {r['quality_per_kj']} | {rec_tag} |"
            )

    md_path = os.path.join(output_dir, "tier_comparison_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Generated Markdown Report: {md_path}")

    # LaTeX Table
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Head-to-Head Comparative Evaluation of Candidate Models per Tier (6 Models Total Across 8\,GB VRAM Budget).}",
        r"\label{tab:tier_model_comparison}",
        r"\small",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"\textbf{Tier} & \textbf{Candidate Model} & \textbf{Family} & \textbf{Params} & \textbf{VRAM} & \textbf{$E_{\text{net}}$ (J)} & \textbf{J/tok} & \textbf{Target Acc.} & \textbf{Latency (s)} & \textbf{Evaluation} \\",
        r"\midrule",
    ]

    for tier in [1, 2, 3]:
        sub = df[df["tier"] == tier].sort_values("family")
        for i, (_, r) in enumerate(sub.iterrows()):
            tier_str = f"Tier {r['tier']}" if i == 0 else ""
            latex_lines.append(
                f"{tier_str} & \\texttt{{{r['model_name']}}} & {r['family']} & {r['params']} & {r['vram_gb']}\\,GB & {r['mean_net_energy_j']} & {r['mean_jpt']} & {r['target_task_quality']}\\% & {r['mean_latency_s']} & Candidate \\\\"
            )
        latex_lines.append(r"\midrule")

    latex_lines[-1] = r"\bottomrule"
    latex_lines.extend([
        r"\end{tabular}",
        r"}",
        r"\end{table}"
    ])

    tex_path = os.path.join(output_dir, "tier_model_comparison_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))
    print(f"Generated LaTeX Table: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Tier Candidate Models")
    parser.add_argument("--gpu", default="rtx4060", choices=("rtx4060", "rtx3060ti"), help="GPU platform to analyze")
    args = parser.parse_args()

    gpu_label = "RTX 4060" if args.gpu == "rtx4060" else "RTX 3060 Ti"
    output_dir = os.path.join(ROOT, "figures", "model_comparisons", args.gpu)
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Analyzing Tier Models for {gpu_label} ---")
    df = analyze_tier_models(gpu_slug=args.gpu)
    print("\nSummary Results:")
    print(df[["tier", "model_name", "family", "mean_net_energy_j", "mean_jpt", "target_task_quality", "mean_latency_s", "quality_per_kj"]].to_string(index=False))

    generate_comparison_plots(df, output_dir, gpu_label)
    generate_markdown_and_latex(df, output_dir)
    print("\nTier Model Comparison Complete!")


if __name__ == "__main__":
    main()
