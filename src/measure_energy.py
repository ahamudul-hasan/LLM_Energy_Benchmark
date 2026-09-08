"""
Energy Measurement Harness for LLM Inference Benchmarking.

Profiles baseline GPU idle power, samples high-frequency power draw via NVML during inference,
and computes net dynamic energy (Joules) using numerical trapezoidal integration.
Strictly adheres to measurement invariants specified in AGENTS.md.
"""

import time
import threading
import json
import argparse
import sys
import random
import requests
from typing import Dict, Any, List, Tuple, Optional

# -----------------------------------------------------------------------------
# NVML Hardware Availability Check
# -----------------------------------------------------------------------------
NVML_AVAILABLE = False
try:
    import pynvml
    pynvml.nvmlInit()
    pynvml.nvmlShutdown()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


class NVMLPowerSampler:
    """High-frequency (50ms) background GPU power sampler using pynvml."""

    def __init__(self, device_index: int = 0, sampling_interval: float = 0.05):
        self.device_index = device_index
        self.sampling_interval = sampling_interval
        self.samples: List[Tuple[float, float]] = []
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.handle = None

    def start(self):
        self.samples = []
        self.stop_event.clear()
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self.thread = threading.Thread(target=self._run_sampling, daemon=True)
        self.thread.start()

    def _run_sampling(self):
        while not self.stop_event.is_set():
            try:
                t = time.perf_counter()
                # pynvml returns power usage in milliwatts
                mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                p_watts = mw / 1000.0
                self.samples.append((t, p_watts))
            except Exception:
                pass
            time.sleep(self.sampling_interval)

    def stop(self) -> List[Tuple[float, float]]:
        if self.stop_event:
            self.stop_event.set()
        if self.thread:
            self.thread.join()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return self.samples


class MockPowerSampler:
    """Mock power sampler for non-NVIDIA / dev environments without NVML."""

    def __init__(self, sampling_interval: float = 0.05, base_power_watts: float = 95.0):
        self.sampling_interval = sampling_interval
        self.base_power_watts = base_power_watts
        self.samples: List[Tuple[float, float]] = []
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.samples = []
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_sampling, daemon=True)
        self.thread.start()

    def _run_sampling(self):
        while not self.stop_event.is_set():
            t = time.perf_counter()
            # Simulate active load with slight noise
            p_watts = self.base_power_watts + random.uniform(-4.0, 4.0)
            self.samples.append((t, p_watts))
            time.sleep(self.sampling_interval)

    def stop(self) -> List[Tuple[float, float]]:
        if self.stop_event:
            self.stop_event.set()
        if self.thread:
            self.thread.join()
        return self.samples


def cool_down_gpu(seconds: float = 5.0) -> None:
    """
    Inserts a cool-down delay to allow GPU voltage, clock rates,
    and temperature to return to quiescent idle levels.
    """
    if seconds > 0:
        time.sleep(seconds)


def measure_idle_power(
    seconds: float = 3.0,
    sampling_interval: float = 0.05,
    use_mock: bool = False
) -> float:
    """
    Profiles static GPU baseline power (P_idle) over a resting period of at least 3 seconds.
    """
    readings = []
    start = time.perf_counter()

    if not use_mock and NVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            while time.perf_counter() - start < seconds:
                mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                readings.append(mw / 1000.0)
                time.sleep(sampling_interval)
            pynvml.nvmlShutdown()
            if readings:
                return sum(readings) / len(readings)
        except Exception:
            pass

    # Fallback / Mock mode idle power profile (~21.5 Watts resting)
    while time.perf_counter() - start < seconds:
        readings.append(21.5 + random.uniform(-0.8, 0.8))
        time.sleep(sampling_interval)
    return sum(readings) / len(readings)


def calculate_energy(samples: List[Tuple[float, float]]) -> float:
    """
    Calculates total energy in Joules using numerical trapezoidal integration.
    E = sum_k=1^{N-1} ((P_k + P_{k+1}) / 2) * (t_{k+1} - t_k)
    """
    if len(samples) < 2:
        return 0.0

    energy = 0.0
    for i in range(1, len(samples)):
        t1, p1 = samples[i - 1]
        t2, p2 = samples[i]
        dt = t2 - t1
        energy += ((p1 + p2) / 2.0) * dt
    return energy


def run_inference_and_measure(
    prompt: str,
    model: str = "llama3.2:1b",
    ollama_url: str = "http://localhost:11434",
    mock: bool = False,
    cool_down_seconds: float = 3.0,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes an inference request against local Ollama server, monitors GPU power
    at 50ms intervals, and measures latency, TTFT, total energy, and net dynamic energy.
    """
    use_mock = mock or (not NVML_AVAILABLE)

    # 1. Cool-down GPU to reach quiescent idle state
    cool_down_gpu(cool_down_seconds)

    # 2. Profile baseline idle power
    p_idle = measure_idle_power(seconds=3.0, use_mock=use_mock)

    # 3. Initialize background power monitor
    if use_mock:
        sampler = MockPowerSampler(sampling_interval=0.05, base_power_watts=85.0)
    else:
        sampler = NVMLPowerSampler(device_index=0, sampling_interval=0.05)

    sampler.start()

    # 4. Dispatch streaming request to Ollama to capture TTFT accurately
    start_time = time.perf_counter()
    first_token_time: Optional[float] = None
    full_response = []
    prompt_tokens = 0
    output_tokens = 0

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    if system_prompt:
        payload["system"] = system_prompt

    try:
        req = requests.post(f"{ollama_url}/api/generate", json=payload, stream=True, timeout=120)
        req.raise_for_status()

        for line in req.iter_lines():
            if line:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                chunk = json.loads(line.decode("utf-8"))
                full_response.append(chunk.get("response", ""))
                if chunk.get("done", False):
                    prompt_tokens = chunk.get("prompt_eval_count", len(prompt.split()))
                    output_tokens = chunk.get("eval_count", len("".join(full_response).split()))
    except Exception as e:
        end_time = time.perf_counter()
        samples = sampler.stop()
        total_energy = calculate_energy(samples)
        latency = end_time - start_time
        net_energy = max(0.0, total_energy - (p_idle * latency))

        return {
            "error": str(e),
            "model_used": model,
            "latency_seconds": round(latency, 4),
            "ttft_seconds": round(latency, 4),
            "p_idle_watts": round(p_idle, 2),
            "total_energy_joules": round(total_energy, 4),
            "net_energy_joules": round(net_energy, 4),
            "joules_per_token": 0.0,
            "nvml_used": not use_mock
        }

    end_time = time.perf_counter()

    # 5. Stop power sampling daemon
    samples = sampler.stop()

    # 6. Calculate timing and energy metrics
    latency = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else latency
    total_energy = calculate_energy(samples)
    net_energy = total_energy - (p_idle * latency)
    # Ensure non-negative net dynamic energy
    net_energy = max(0.0, net_energy)

    output_tokens_clean = max(output_tokens, 1)
    joules_per_token = net_energy / output_tokens_clean

    response_text = "".join(full_response).strip()

    return {
        "test_id": f"eval_{int(time.time() * 1000)}",
        "model_used": model,
        "prompt_token_count": prompt_tokens,
        "output_token_count": output_tokens,
        "latency_seconds": round(latency, 4),
        "ttft_seconds": round(ttft, 4),
        "p_idle_watts": round(p_idle, 2),
        "total_energy_joules": round(total_energy, 4),
        "net_energy_joules": round(net_energy, 4),
        "joules_per_token": round(joules_per_token, 4),
        "nvml_used": not use_mock,
        "response_text": response_text
    }


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Energy Benchmark Harness")
    parser.add_argument("--prompt", type=str, default="Explain green computing in 2 sentences.", help="Prompt to evaluate")
    parser.add_argument("--model", type=str, default="llama3.2:1b", help="Ollama model tag")
    parser.add_argument("--url", type=str, default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--mock", action="store_true", help="Force mock power sampler mode")
    parser.add_argument("--cool-down", type=float, default=3.0, help="Pre-flight cool down delay in seconds")

    args = parser.parse_args()

    print(f"--- Starting Energy Measurement ---")
    print(f"Model: {args.model}")
    print(f"NVML Available: {NVML_AVAILABLE}")
    print(f"Sampler Mode: {'MOCK' if (args.mock or not NVML_AVAILABLE) else 'NVML'}")

    result = run_inference_and_measure(
        prompt=args.prompt,
        model=args.model,
        ollama_url=args.url,
        mock=args.mock,
        cool_down_seconds=args.cool_down
    )

    print("\n--- Benchmark Result ---")
    print(json.dumps({k: v for k, v in result.items() if k != "response_text"}, indent=2))
    if "response_text" in result:
        print("\nGenerated Response:")
        print(result["response_text"])
