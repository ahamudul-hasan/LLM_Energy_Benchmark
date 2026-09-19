"""
Benchmark Evaluation Stream Runner for Adaptive Router vs Static Tier 3 Baseline.

Evaluates system performance across the 600-prompt evaluation benchmark per main.pdf Section 3.3:
1. Static Tier 3 Baseline: Directs all 600 prompts to Llama-3.1-8B-Instruct (Q4_K_M).
2. Adaptive Router: Dynamically routes prompts to Tier 1, 2, or 3 via host CPU complexity estimation.

Computes core paper metrics:
- Dynamic Net Energy E_net (Joules) & Total Energy E_total (kJ)
- Joules per Generated Output Token
- Quality Preservation Ratio (QPR = Score_Router / Score_Tier3 * 100%)
- Latency Profile: TTFT, End-to-End Latency, Router Overhead (Delta t_route)
"""

import os
import glob
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List

import sys
sys.path.insert(0, os.path.dirname(__file__))

from dataset_loader import load_jsonl
from measure_energy import run_inference_and_measure, measure_idle_power
from router import PromptRouter


def score_response(dataset_name: str, response: str, target: str) -> float:
    """Computes task quality score (Accuracy, EM, token F1/ROUGE proxy, or code pass proxy)."""
    import re
    response_clean = response.strip().lower()
    target_clean = target.strip().lower()

    if dataset_name in ["sst2", "mnli"]:
        return 1.0 if target_clean in response_clean else 0.0

    elif dataset_name == "squad":
        # Check if target answer string is contained in the concise QA response
        if not target_clean:
            return 1.0
        return 1.0 if target_clean in response_clean else 0.0

    elif dataset_name == "cnn_dailymail":
        # Token F1 proxy for abstractive summarization
        target_words = target_clean.split()
        resp_words = response_clean.split()
        if not target_words or not resp_words:
            return 0.0
        target_set = set(target_words)
        resp_set = set(resp_words)
        overlap = len(target_set.intersection(resp_set))
        if overlap == 0:
            return 0.0
        p = overlap / len(resp_set)
        r = overlap / len(target_set)
        return (2.0 * p * r) / (p + r)

    elif dataset_name == "gsm8k":
        # Extract numerical answer from target (text following '####')
        target_num = target.split("####")[-1].strip().replace(",", "") if "####" in target else target.strip()
        # Direct match on #### target
        if f"#### {target_num}" in response or f"####{target_num}" in response:
            return 1.0
        # Also check last extracted number in the generated response
        resp_nums = re.findall(r"[-+]?\d*\.?\d+", response.replace(",", ""))
        if resp_nums and resp_nums[-1] == target_num:
            return 1.0
        return 0.0

    elif dataset_name == "humaneval":
        # Docstring to functional code pass proxy
        if "def " in response and ("return " in response or "yield " in response):
            return 1.0
        return 0.0

    return 0.5


def run_benchmark_eval(
    mock: bool = True,
    limit_per_ds: int = 10,
    cool_down: float = 1.0,
    eval_all_tiers: bool = False
) -> Dict[str, Any]:
    print("--- Starting Benchmark Evaluation Stream (Router vs Static Baselines) ---")
    print(f"Config: mock={mock}, limit_per_ds={limit_per_ds}, cool_down={cool_down}s, eval_all_tiers={eval_all_tiers}")

    router = PromptRouter()
    prompt_files = sorted(glob.glob(os.path.join("data", "prompts", "*.jsonl")))

    all_prompts = []
    for pf in prompt_files:
        items = load_jsonl(pf)[:limit_per_ds]
        all_prompts.extend(items)

    print(f"Loaded {len(all_prompts)} evaluation prompts across {len(prompt_files)} benchmark suites.")

    # 1. Profile quiescent batch idle power over at least 3.0s (AGENTS.md Section 2.B)
    print("Measuring baseline GPU idle power (3.0s quiescent resting period)...")
    batch_p_idle = measure_idle_power(seconds=3.0, use_mock=mock)
    print(f"Quiescent baseline idle power: {batch_p_idle:.2f} W")

    results_baseline = []
    results_router = []
    results_tier1 = []
    results_tier2 = []

    out_dir = os.path.join("data", "results")
    os.makedirs(out_dir, exist_ok=True)

    for idx, item in enumerate(all_prompts):
        prompt_text = item["prompt"]
        ds_name = item["dataset"]
        target = item["target"]

        # 1. Static Tier 3 Baseline Run (Llama-3.1-8B)
        res_b = run_inference_and_measure(
            prompt=prompt_text,
            model="llama3.1:8b",
            mock=mock,
            simulate=mock,
            cool_down_seconds=cool_down,
            p_idle=batch_p_idle
        )
        res_b["dataset"] = ds_name
        res_b["target_tier"] = 3
        res_b["quality_score"] = score_response(ds_name, res_b.get("response_text", ""), target)
        results_baseline.append(res_b)

        # Optional: Static Tier 1 and Tier 2 Baselines (Phase 5)
        if eval_all_tiers:
            res_t1 = run_inference_and_measure(
                prompt=prompt_text,
                model="llama3.2:1b",
                mock=mock,
                simulate=mock,
                cool_down_seconds=cool_down,
                p_idle=batch_p_idle
            )
            res_t1["dataset"] = ds_name
            res_t1["target_tier"] = 1
            res_t1["quality_score"] = score_response(ds_name, res_t1.get("response_text", ""), target)
            results_tier1.append(res_t1)

            res_t2 = run_inference_and_measure(
                prompt=prompt_text,
                model="llama3.2:3b",
                mock=mock,
                simulate=mock,
                cool_down_seconds=cool_down,
                p_idle=batch_p_idle
            )
            res_t2["dataset"] = ds_name
            res_t2["target_tier"] = 2
            res_t2["quality_score"] = score_response(ds_name, res_t2.get("response_text", ""), target)
            results_tier2.append(res_t2)

        # 2. Adaptive Router Run (Phase 7)
        r_info = router.route(prompt_text)
        res_r = run_inference_and_measure(
            prompt=prompt_text,
            model=r_info["model_used"],
            mock=mock,
            simulate=mock,
            cool_down_seconds=cool_down,
            p_idle=batch_p_idle
        )
        res_r["dataset"] = ds_name
        res_r["tier_selected"] = r_info["tier_selected"]
        res_r["router_latency_ms"] = r_info["router_latency_ms"]
        res_r["quality_score"] = score_response(ds_name, res_r.get("response_text", ""), target)
        results_router.append(res_r)

        print(
            f"[{idx + 1:3d}/{len(all_prompts)}] {ds_name:12s} | "
            f"T3 (8B): {res_b.get('latency_seconds', 0):.2f}s, {res_b.get('net_energy_joules', 0):.1f}J, Q={res_b.get('quality_score', 0):.2f} | "
            f"Router (T{r_info['tier_selected']} {r_info['model_used']}): {res_r.get('latency_seconds', 0):.2f}s, {res_r.get('net_energy_joules', 0):.1f}J, Q={res_r.get('quality_score', 0):.2f}"
        )

        # Checkpoint every 5 prompts or on last prompt
        if (idx + 1) % 5 == 0 or (idx + 1) == len(all_prompts):
            all_baseline_rows = results_baseline.copy()
            if eval_all_tiers:
                all_baseline_rows = results_tier1 + results_tier2 + results_baseline
            pd.DataFrame(all_baseline_rows).to_csv(os.path.join(out_dir, "static_tier_baselines.csv"), index=False)
            pd.DataFrame(results_router).to_csv(os.path.join(out_dir, "router_evaluation.csv"), index=False)

    df_b = pd.DataFrame(results_baseline)
    df_r = pd.DataFrame(results_router)

    # Compute Summary Aggregates
    e_total_b_kj = df_b["total_energy_joules"].sum() / 1000.0
    e_total_r_kj = df_r["total_energy_joules"].sum() / 1000.0

    e_net_b_j = df_b["net_energy_joules"].sum()
    e_net_r_j = df_r["net_energy_joules"].sum()

    quality_b = df_b["quality_score"].mean()
    quality_r = df_r["quality_score"].mean()

    qpr = (quality_r / quality_b * 100.0) if quality_b > 0 else 100.0
    energy_savings_pct = ((e_net_b_j - e_net_r_j) / e_net_b_j * 100.0) if e_net_b_j > 0 else 0.0

    # Per-tier breakdowns
    tier_counts = df_r["tier_selected"].value_counts().to_dict()
    tier_percentages = {f"Tier_{k}_pct": round(v / len(df_r) * 100.0, 1) for k, v in tier_counts.items()}

    summary = {
        "evaluation_prompts_count": len(all_prompts),
        "quiescent_p_idle_watts": round(batch_p_idle, 2),
        "static_tier3_total_energy_kj": round(e_total_b_kj, 3),
        "adaptive_router_total_energy_kj": round(e_total_r_kj, 3),
        "static_tier3_net_energy_joules": round(e_net_b_j, 2),
        "adaptive_router_net_energy_joules": round(e_net_r_j, 2),
        "energy_savings_percent": round(energy_savings_pct, 2),
        "static_tier3_mean_joules_per_token": round(df_b["joules_per_token"].mean(), 4),
        "adaptive_router_mean_joules_per_token": round(df_r["joules_per_token"].mean(), 4),
        "static_tier3_quality_score": round(quality_b, 4),
        "adaptive_router_quality_score": round(quality_r, 4),
        "quality_preservation_ratio_qpr": round(qpr, 2),
        "mean_router_overhead_ms": round(df_r["router_latency_ms"].mean(), 3),
        "static_tier3_mean_latency_s": round(df_b["latency_seconds"].mean(), 3),
        "adaptive_router_mean_latency_s": round(df_r["latency_seconds"].mean(), 3),
        "static_tier3_mean_ttft_s": round(df_b["ttft_seconds"].mean(), 3),
        "adaptive_router_mean_ttft_s": round(df_r["ttft_seconds"].mean(), 3),
        "tier_distribution_counts": tier_counts,
        "tier_distribution_percentages": tier_percentages
    }

    gpu_name = "Mock / CPU"
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(h)
    except Exception:
        pass

    summary["gpu_device"] = gpu_name

    print("\n================ EVALUATION RESULTS SUMMARY ================")
    print(json.dumps(summary, indent=2))
    print("============================================================\n")

    with open(os.path.join(out_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2))

    # Also save to GPU-specific directory (e.g. data/results/rtx4060/ or data/results/rtx3060ti/)
    gpu_slug = gpu_name.lower().replace("nvidia", "").replace("geforce", "").replace(" ", "").replace("-", "")
    if not gpu_slug or "mock" in gpu_slug:
        gpu_slug = "mock"

    gpu_out_dir = os.path.join(out_dir, gpu_slug)
    os.makedirs(gpu_out_dir, exist_ok=True)

    with open(os.path.join(gpu_out_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2))
    df_b.to_csv(os.path.join(gpu_out_dir, "static_tier_baselines.csv"), index=False)
    df_r.to_csv(os.path.join(gpu_out_dir, "router_evaluation.csv"), index=False)
    print(f"GPU-specific results archived to: {gpu_out_dir}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Adaptive Router vs Static Baseline Evaluation")
    parser.add_argument("--mock", action="store_true", help="Force mock simulation mode")
    parser.add_argument("--limit-per-ds", type=int, default=10, help="Number of prompts per dataset to evaluate")
    parser.add_argument("--cool-down", type=float, default=1.0, help="Cool down period between inferences (seconds)")
    parser.add_argument("--eval-all-tiers", action="store_true", help="Also evaluate static Tier 1 and Tier 2 baselines")
    args = parser.parse_args()

    run_benchmark_eval(
        mock=args.mock,
        limit_per_ds=args.limit_per_ds,
        cool_down=args.cool_down,
        eval_all_tiers=args.eval_all_tiers
    )
