"""
Publication-Quality Figure Generator for LLM Energy Benchmark Paper.

Generates GPU-specific figures into:
  - figures/rtx4060/
  - figures/rtx3060ti/
And cross-platform comparative figures into:
  - figures/cross_platform/
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16
})


def generate_gpu_plots(summary: dict, df_b: pd.DataFrame, df_r: pd.DataFrame, out_dir: str, gpu_label: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n--- Generating figures for {gpu_label} in {out_dir} ---")

    # -------------------------------------------------------------------------
    # Figure 1: Net Dynamic Energy & Joules per Token
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    systems = ['Static Baseline\n(Llama-3.1-8B)', 'Adaptive Router\n(Dynamic Tiers)']
    net_energies = [summary["static_tier3_net_energy_joules"], summary["adaptive_router_net_energy_joules"]]
    colors_energy = ['#d9534f', '#5cb85c']

    bars1 = ax1.bar(systems, net_energies, color=colors_energy, width=0.5, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Net Dynamic Energy $E_{net}$ (Joules)')
    ax1.set_title(f'Dynamic Inference Energy ({gpu_label})\n({summary["energy_savings_percent"]}% Energy Reduction)', fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (max(net_energies)*0.02), f'{yval:.1f} J', ha='center', va='bottom', fontweight='bold')

    j_per_token = [summary["static_tier3_mean_joules_per_token"], summary["adaptive_router_mean_joules_per_token"]]
    colors_jpt = ['#f0ad4e', '#337ab7']

    bars2 = ax2.bar(systems, j_per_token, color=colors_jpt, width=0.5, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Energy Intensity (Joules / Generated Token)')
    jpt_red = round((j_per_token[0] - j_per_token[1]) / j_per_token[0] * 100.0, 1) if j_per_token[0] > 0 else 0
    ax2.set_title(f'Energy Efficiency per Token ({gpu_label})\n({jpt_red}% Reduction)', fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + (max(j_per_token)*0.02), f'{yval:.3f} J/tok', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "energy_and_joules_per_token.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Generated: {fig1_path}")

    # -------------------------------------------------------------------------
    # Figure 2: Router Tier Distribution Donut Chart
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    tier_labels = ['Tier 1 (1B)\nLow Complexity', 'Tier 2 (3B)\nMedium Complexity', 'Tier 3 (8B)\nHigh Complexity']
    tier_counts = [
        summary.get("tier_distribution_counts", {}).get("1", 0),
        summary.get("tier_distribution_counts", {}).get("2", 0),
        summary.get("tier_distribution_counts", {}).get("3", 0)
    ]
    colors_pie = ['#5cb85c', '#5bc0de', '#d9534f']
    explode = (0.05, 0.05, 0.05)

    wedges, texts, autotexts = ax.pie(
        tier_counts,
        explode=explode,
        labels=tier_labels,
        colors=colors_pie,
        autopct='%1.1f%%',
        pctdistance=0.75,
        startangle=140,
        wedgeprops=dict(width=0.45, edgecolor='black', linewidth=1.2)
    )

    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')

    ax.set_title(f'Online Routing Decisions ({gpu_label})\n(Total Queries: {summary["evaluation_prompts_count"]})', fontweight='bold')
    plt.tight_layout()
    fig2_path = os.path.join(out_dir, "router_tier_distribution.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Generated: {fig2_path}")

    # -------------------------------------------------------------------------
    # Figure 3: Per-Dataset Dynamic Energy Breakdown
    # -------------------------------------------------------------------------
    datasets = sorted(df_b['dataset'].unique())
    b_energy_per_ds = df_b.groupby('dataset')['net_energy_joules'].mean()
    r_energy_per_ds = df_r.groupby('dataset')['net_energy_joules'].mean()

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, [b_energy_per_ds[d] for d in datasets], width, label='Static Tier 3 (8B)', color='#d9534f', edgecolor='black', linewidth=1)
    rects2 = ax.bar(x + width/2, [r_energy_per_ds[d] for d in datasets], width, label='Adaptive Router', color='#5cb85c', edgecolor='black', linewidth=1)

    ax.set_ylabel('Mean Dynamic Energy $E_{net}$ (Joules)')
    ax.set_title(f'Dynamic Energy by Benchmark Task ({gpu_label})', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    fig3_path = os.path.join(out_dir, "dataset_energy_breakdown.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Generated: {fig3_path}")

    # -------------------------------------------------------------------------
    # Figure 4: Latency Profile (TTFT & End-to-End Latency)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    lat_metrics = ['Time-to-First-Token (TTFT)', 'End-to-End Latency']
    b_lat = [summary.get("static_tier3_mean_ttft_s", 0), summary.get("static_tier3_mean_latency_s", 0)]
    r_lat = [summary.get("adaptive_router_mean_ttft_s", 0), summary.get("adaptive_router_mean_latency_s", 0)]

    x_lat = np.arange(len(lat_metrics))
    rects1 = ax.bar(x_lat - width/2, b_lat, width, label='Static Tier 3 (8B)', color='#f0ad4e', edgecolor='black', linewidth=1)
    rects2 = ax.bar(x_lat + width/2, r_lat, width, label='Adaptive Router', color='#337ab7', edgecolor='black', linewidth=1)

    ax.set_ylabel('Duration (Seconds)')
    ax.set_title(f'Serving Latency Profile ({gpu_label})\n(Router Overhead: {summary.get("mean_router_overhead_ms", 0):.1f} ms)', fontweight='bold')
    ax.set_xticks(x_lat)
    ax.set_xticklabels(lat_metrics)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in rects1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{yval:.2f}s', ha='center', va='bottom', fontweight='bold')

    for bar in rects2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{yval:.2f}s', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    fig4_path = os.path.join(out_dir, "latency_profile.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Generated: {fig4_path}")


def generate_cross_platform_plots(summary_4060: dict, summary_3060: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n--- Generating Cross-Platform Comparative Figures in {out_dir} ---")

    # 1. Microarchitectural Scaling Ratio (eta)
    e_4060_net = summary_4060.get("adaptive_router_net_energy_joules", 1.0)
    e_3060_net = summary_3060.get("adaptive_router_net_energy_joules", 1.0)
    eta = round(e_3060_net / e_4060_net, 3) if e_4060_net > 0 else 1.0

    fig, ax = plt.subplots(figsize=(8, 5))
    plat_labels = ['Platform A: RTX 4060\n(TSMC 4N, 115W TDP)', 'Platform B: RTX 3060 Ti\n(Samsung 8nm, 200W TDP)']
    plat_energies = [e_4060_net, e_3060_net]
    colors_plat = ['#5cb85c', '#d9534f']

    bars = ax.bar(plat_labels, plat_energies, color=colors_plat, width=0.45, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Net Dynamic Energy $E_{net}$ (Joules)')
    ax.set_title(f'Cross-Platform Microarchitectural Scaling\nScaling Ratio $\\eta = E_{{3060Ti}} / E_{{4060}} = {eta:.2f}\\times$', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + (max(plat_energies)*0.02), f'{yval:.1f} J', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    fig_scale_path = os.path.join(out_dir, "cross_platform_scaling_eta.png")
    plt.savefig(fig_scale_path, dpi=300)
    plt.close()
    print(f"Generated: {fig_scale_path} (Scaling Ratio eta = {eta:.2f}x)")

    # 2. Joules per Token Cross-Platform Comparison
    jpt_4060_b = summary_4060.get("static_tier3_mean_joules_per_token", 0.0)
    jpt_4060_r = summary_4060.get("adaptive_router_mean_joules_per_token", 0.0)
    jpt_3060_b = summary_3060.get("static_tier3_mean_joules_per_token", 0.0)
    jpt_3060_r = summary_3060.get("adaptive_router_mean_joules_per_token", 0.0)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(2)
    width = 0.35

    rects1 = ax.bar(x - width/2, [jpt_4060_b, jpt_3060_b], width, label='Static Tier 3 (8B)', color='#d9534f', edgecolor='black', linewidth=1)
    rects2 = ax.bar(x + width/2, [jpt_4060_r, jpt_3060_r], width, label='Adaptive Router', color='#5cb85c', edgecolor='black', linewidth=1)

    ax.set_ylabel('Joules per Generated Token')
    ax.set_title('Cross-Platform Energy Efficiency (Joules / Token)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Platform A (RTX 4060)', 'Platform B (RTX 3060 Ti)'])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    fig_jpt_path = os.path.join(out_dir, "cross_platform_joules_per_token.png")
    plt.savefig(fig_jpt_path, dpi=300)
    plt.close()
    print(f"Generated: {fig_jpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Publication Figures")
    parser.add_argument("--gpu", type=str, default=None, help="GPU identifier (e.g. rtx4060 or rtx3060ti)")
    args = parser.parse_args()

    results_dir = os.path.join("data", "results")

    # Load primary summary
    summary_path = os.path.join(results_dir, "summary_metrics.json")
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found.")
        sys.exit(1)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    df_b = pd.read_csv(os.path.join(results_dir, "static_tier_baselines.csv"))
    df_r = pd.read_csv(os.path.join(results_dir, "router_evaluation.csv"))

    # Determine GPU slug
    gpu_name = summary.get("gpu_device", "")
    if args.gpu:
        gpu_slug = args.gpu.lower().replace(" ", "")
    elif "3060" in gpu_name:
        gpu_slug = "rtx3060ti"
    elif "4060" in gpu_name:
        gpu_slug = "rtx4060"
    else:
        gpu_slug = "rtx4060"

    gpu_label = "RTX 3060 Ti" if "3060" in gpu_slug else "RTX 4060"

    # 1. Generate in GPU-specific folder (e.g. figures/rtx4060/ or figures/rtx3060ti/)
    target_out_dir = os.path.join("figures", gpu_slug)
    generate_gpu_plots(summary, df_b, df_r, target_out_dir, gpu_label)

    # 2. Also update top-level figures/ folder for convenience
    generate_gpu_plots(summary, df_b, df_r, "figures", gpu_label)

    # 3. Check if cross-platform data exists
    summary_4060_path = os.path.join(results_dir, "rtx4060", "summary_metrics.json")
    if not os.path.exists(summary_4060_path):
        summary_4060_path = os.path.join(results_dir, "summary_metrics_rtx4060.json")

    summary_3060_path = os.path.join(results_dir, "rtx3060ti", "summary_metrics.json")
    if not os.path.exists(summary_3060_path):
        summary_3060_path = os.path.join(results_dir, "summary_metrics_rtx3060ti.json")

    # If current run is 3060 Ti, use current summary as 3060 Ti data
    if "3060" in gpu_slug:
        s3060 = summary
    elif os.path.exists(summary_3060_path):
        with open(summary_3060_path, "r", encoding="utf-8") as f:
            s3060 = json.load(f)
    else:
        s3060 = None

    if os.path.exists(summary_4060_path) and s3060 is not None:
        with open(summary_4060_path, "r", encoding="utf-8") as f:
            s4060 = json.load(f)
        cross_out_dir = os.path.join("figures", "cross_platform")
        generate_cross_platform_plots(s4060, s3060, cross_out_dir)

    print("\n--- All plotting workflows complete! ---")


if __name__ == "__main__":
    main()
