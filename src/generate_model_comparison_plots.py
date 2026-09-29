"""Generate RTX 3060 Ti plots for the multi-model benchmark matrix."""

import json
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
})


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(ROOT, "data", "results", "model_comparisons")
GPU_SLUG = "rtx3060ti"
GPU_LABEL = "RTX 3060 Ti"

PROFILES = {
    "llama": "Llama family",
    "gemma-phi-mistral": "Gemma / Phi / Mistral",
    "qwen": "Qwen family",
}
COLORS = {"Llama family": "#d9534f", "Gemma / Phi / Mistral": "#5bc0de", "Qwen family": "#337ab7"}


def load_runs(gpu_slug):
    runs = []
    for profile, label in PROFILES.items():
        profile_dir = os.path.join(RESULTS_ROOT, profile)
        if not os.path.exists(profile_dir):
            continue
        sum_path = os.path.join(profile_dir, "summary_metrics.json")
        if not os.path.exists(sum_path):
            continue
        with open(sum_path, encoding="utf-8") as handle:
            summary = json.load(handle)

        static_path = os.path.join(profile_dir, gpu_slug, "static_tier_baselines.csv")
        if not os.path.exists(static_path):
            static_path = os.path.join(profile_dir, "static_tier_baselines.csv")

        routed_path = os.path.join(profile_dir, gpu_slug, "router_evaluation.csv")
        if not os.path.exists(routed_path):
            routed_path = os.path.join(profile_dir, "router_evaluation.csv")

        if os.path.exists(static_path) and os.path.exists(routed_path):
            static = pd.read_csv(static_path)
            routed = pd.read_csv(routed_path)
            runs.append({"profile": label, "summary": summary, "static": static, "routed": routed})
    return runs


def add_labels(axis, bars, suffix=""):
    for bar in bars:
        value = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(axis.get_ylim()[1] * 0.015, 0.01),
            f"{value:.2f}{suffix}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )


def plot_family_energy(runs):
    names = [run["profile"] for run in runs]
    x = np.arange(len(names))
    width = 0.35
    static_energy = [run["summary"]["static_tier3_net_energy_joules"] for run in runs]
    routed_energy = [run["summary"]["adaptive_router_net_energy_joules"] for run in runs]
    static_jpt = [run["summary"]["static_tier3_mean_joules_per_token"] for run in runs]
    routed_jpt = [run["summary"]["adaptive_router_mean_joules_per_token"] for run in runs]

    fig, (ax_energy, ax_jpt) = plt.subplots(1, 2, figsize=(14, 6))
    bars = ax_energy.bar(x - width / 2, static_energy, width, label="Static Tier 3", color="#d9534f", edgecolor="black")
    bars_r = ax_energy.bar(x + width / 2, routed_energy, width, label="Adaptive Router", color="#5cb85c", edgecolor="black")
    ax_energy.set_ylabel("Net Dynamic Energy $E_{net}$ (Joules)")
    ax_energy.set_title(f"Dynamic Inference Energy\n({GPU_LABEL})", fontweight="bold")
    ax_energy.set_xticks(x, names)
    ax_energy.legend()
    add_labels(ax_energy, bars, " J")
    add_labels(ax_energy, bars_r, " J")

    bars = ax_jpt.bar(x - width / 2, static_jpt, width, label="Static Tier 3", color="#f0ad4e", edgecolor="black")
    bars_r = ax_jpt.bar(x + width / 2, routed_jpt, width, label="Adaptive Router", color="#337ab7", edgecolor="black")
    ax_jpt.set_ylabel("Joules per Generated Token")
    ax_jpt.set_title(f"Energy Efficiency per Token\n({GPU_LABEL})", fontweight="bold")
    ax_jpt.set_xticks(x, names)
    add_labels(ax_jpt, bars, " J/tok")
    add_labels(ax_jpt, bars_r, " J/tok")
    fig.tight_layout()
    path = os.path.join(OUTPUT_ROOT, "model_family_energy_comparison.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Generated: {path}")


def plot_all_models(runs):
    records = []
    for run in runs:
        for tier, group in run["static"].groupby("target_tier"):
            records.append({
                "model": group["model_used"].iloc[0],
                "tier": int(tier),
                "energy": group["net_energy_joules"].mean(),
                "quality": group["quality_score"].mean() * 100,
            })
    frame = pd.DataFrame(records).sort_values(["tier", "model"])
    colors = ["#5cb85c" if tier == 1 else "#5bc0de" if tier == 2 else "#d9534f" for tier in frame["tier"]]

    fig, (ax_energy, ax_quality) = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(frame))
    bars = ax_energy.bar(x, frame["energy"], color=colors, edgecolor="black")
    ax_energy.set_ylabel("Mean Net Dynamic Energy (Joules)")
    ax_energy.set_title(f"Static Energy by Model\n({GPU_LABEL})", fontweight="bold")
    ax_energy.set_xticks(x, frame["model"], rotation=35, ha="right")
    add_labels(ax_energy, bars, " J")

    bars = ax_quality.bar(x, frame["quality"], color=colors, edgecolor="black")
    ax_quality.set_ylabel("Mean Quality Score (%)")
    ax_quality.set_ylim(0, 105)
    ax_quality.set_title("Static Quality by Model\n(12-prompt validation run)", fontweight="bold")
    ax_quality.set_xticks(x, frame["model"], rotation=35, ha="right")
    add_labels(ax_quality, bars, "%")
    fig.tight_layout()
    path = os.path.join(OUTPUT_ROOT, "all_models_energy_quality.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Generated: {path}")


def plot_latency_quality(runs):
    names = [run["profile"] for run in runs]
    x = np.arange(len(names))
    width = 0.35
    static_latency = [run["summary"]["static_tier3_mean_latency_s"] for run in runs]
    routed_latency = [run["summary"]["adaptive_router_mean_latency_s"] for run in runs]
    qpr = [run["summary"]["quality_preservation_ratio_qpr"] for run in runs]

    fig, (ax_latency, ax_qpr) = plt.subplots(1, 2, figsize=(14, 6))
    bars = ax_latency.bar(x - width / 2, static_latency, width, label="Static Tier 3", color="#f0ad4e", edgecolor="black")
    bars_r = ax_latency.bar(x + width / 2, routed_latency, width, label="Adaptive Router", color="#337ab7", edgecolor="black")
    ax_latency.set_ylabel("End-to-End Latency (Seconds)")
    ax_latency.set_title(f"Serving Latency by Model Family\n({GPU_LABEL})", fontweight="bold")
    ax_latency.set_xticks(x, names)
    ax_latency.legend()
    add_labels(ax_latency, bars, "s")
    add_labels(ax_latency, bars_r, "s")

    bars = ax_qpr.bar(x, qpr, color=[COLORS[name] for name in names], edgecolor="black")
    ax_qpr.axhline(100, color="black", linestyle="--", linewidth=1, label="100% baseline")
    ax_qpr.set_ylabel("Quality Preservation Ratio (%)")
    ax_qpr.set_title(f"Router Quality Preservation\n({GPU_LABEL})", fontweight="bold")
    ax_qpr.set_xticks(x, names)
    ax_qpr.legend()
    add_labels(ax_qpr, bars, "%")
    fig.tight_layout()
    path = os.path.join(OUTPUT_ROOT, "model_family_latency_quality.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Generated: {path}")


def plot_router_distribution(runs):
    labels = [run["profile"] for run in runs]
    tier_values = np.array([
        [run["summary"]["tier_distribution_counts"].get(str(tier), 0) for tier in (1, 2, 3)]
        for run in runs
    ])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(labels))
    for tier, color in zip((1, 2, 3), ("#5cb85c", "#5bc0de", "#d9534f")):
        values = tier_values[:, tier - 1]
        ax.bar(x, values, bottom=bottom, label=f"Tier {tier}", color=color, edgecolor="black")
        bottom += values
    ax.set_ylabel("Number of Routed Queries")
    ax.set_title(f"Online Routing Decisions by Model Family\n({GPU_LABEL})", fontweight="bold")
    ax.set_xticks(x, labels)
    ax.legend()
    for index, total in enumerate(bottom):
        ax.text(index, total + 0.15, f"n={int(total)}", ha="center", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_ROOT, "model_family_router_distribution.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Generated: {path}")


def main():
    global GPU_SLUG, GPU_LABEL, OUTPUT_ROOT
    parser = argparse.ArgumentParser(description="Generate multi-model comparison plots for one GPU")
    parser.add_argument("--gpu", default="rtx3060ti", choices=("rtx3060ti", "rtx4060"))
    args = parser.parse_args()
    GPU_SLUG = args.gpu
    GPU_LABEL = "RTX 3060 Ti" if GPU_SLUG == "rtx3060ti" else "RTX 4060"
    output_root = os.path.join(ROOT, "figures", "model_comparisons", GPU_SLUG)
    OUTPUT_ROOT = output_root
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    runs = load_runs(GPU_SLUG)
    plot_family_energy(runs)
    plot_all_models(runs)
    plot_latency_quality(runs)
    plot_router_distribution(runs)
    print(f"{GPU_LABEL} model comparison plots complete.")


if __name__ == "__main__":
    main()