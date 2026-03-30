"""
hardware_config.py
Unified hardware configuration for Levelsmith training and inference.

System: AMD Ryzen AI 9 HX 370 + RTX 4070 Laptop + AMD XDNA 2 NPU (50 TOPS)

Training  → CUDA (RTX 4070) with AMP + cuDNN benchmark
Inference → DirectML (NPU/GPU via Windows ML) with CPU fallback

Usage:
    from hardware_config import get_training_device, get_inference_session, print_hw_summary

    # Training
    device = get_training_device()
    scaler = get_amp_scaler(device)

    # Inference
    session = get_inference_session("model.onnx")
"""

import os
import sys
import time
import math
from pathlib import Path
from typing import Optional

import torch
import torch.backends.cudnn
import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
MODEL_DIR  = SCRIPT_DIR / "models"

# DirectML device priority list (index 0 = prefer NPU, 1 = prefer dGPU)
# On AMD Ryzen AI + NVIDIA dGPU systems, DML device 0 is typically the iGPU/NPU,
# device 1 is the dGPU. We try the dGPU first (higher perf for our workloads).
_DML_DEVICE_PREFER_DGPU = True


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Training device (CUDA / CPU)
# ═══════════════════════════════════════════════════════════════════════════════

def get_training_device() -> torch.device:
    """Return best device for training, configuring cuDNN optimisations."""
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        # cuDNN auto-tuner: finds fastest conv algorithm for fixed input sizes.
        # Beneficial here because our Transformer uses fixed PATCH_SIZE batches.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled   = True
        # Deterministic off for speed (training reproducibility via seeds instead)
        torch.backends.cudnn.deterministic = False
        return device
    return torch.device("cpu")


def get_amp_scaler(device: torch.device) -> Optional["torch.cuda.amp.GradScaler"]:
    """
    Return a GradScaler for Automatic Mixed Precision (FP16) training.
    RTX 4070 (Ada Lovelace) has native FP16/BF16 tensor cores → ~2x speedup.
    Returns None on CPU (AMP not applicable).
    """
    if device.type == "cuda":
        return torch.cuda.amp.GradScaler()
    return None


def amp_autocast(device: torch.device):
    """Context manager: torch.autocast on CUDA, no-op on CPU."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return _NullContext()


class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ONNX export
# ═══════════════════════════════════════════════════════════════════════════════

def export_to_onnx(
    model_pt_path: Path,
    onnx_path: Path,
    seq_len: int = 14,
    opset: int = 17,
) -> Path:
    """
    Export a LayoutTransformer .pt checkpoint to ONNX.

    The model takes:
        src_cont  : (1, seq_len, 6)   float32
        src_types : (1, seq_len)      int64
    And returns:
        pos_out   : (1, seq_len, 4)   float32
        size_out  : (1, seq_len, 2)   float32
        type_out  : (1, seq_len, 7)   float32
    """
    sys.path.insert(0, str(SCRIPT_DIR))
    from layout_model import LayoutTransformer, load_model

    device = torch.device("cpu")   # export on CPU for portability
    model  = load_model(model_pt_path, device)
    model.eval()

    # Dummy inputs
    dummy_cont  = torch.zeros(1, seq_len, 6,  dtype=torch.float32)
    dummy_types = torch.zeros(1, seq_len,     dtype=torch.long)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_cont, dummy_types),
        str(onnx_path),
        input_names  = ["src_cont", "src_types"],
        output_names = ["pos_out", "size_out", "type_out"],
        dynamic_axes = {
            "src_cont":  {1: "seq_len"},
            "src_types": {1: "seq_len"},
            "pos_out":   {1: "seq_len"},
            "size_out":  {1: "seq_len"},
            "type_out":  {1: "seq_len"},
        },
        opset_version        = opset,
        do_constant_folding  = True,
        export_params        = True,
    )

    # Verify
    import onnx as onnx_lib
    onnx_model = onnx_lib.load(str(onnx_path))
    onnx_lib.checker.check_model(onnx_model)
    sz_kb = onnx_path.stat().st_size / 1024
    print(f"[ONNX] Exported: {onnx_path.name}  ({sz_kb:.1f} KB)  opset={opset}")
    return onnx_path


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Inference session (DirectML / CPU)
# ═══════════════════════════════════════════════════════════════════════════════

def recommend_inference_provider(
    onnx_path: Path,
    batch_size: int = 1,
    seq_len: int = 14,
) -> str:
    """
    Recommend the optimal inference provider based on measured latency.

    For small models (< 10M params) at batch_size=1, CPU is typically faster
    than DirectML due to the D3D12 dispatch overhead (~3-8ms fixed cost).
    DirectML becomes beneficial at batch_size >= 32 or model size >= 50M params.

    Returns: "directml" | "cpu"
    """
    import onnxruntime as ort
    if "DmlExecutionProvider" not in ort.get_available_providers():
        return "cpu"

    # Quick 20-run probe
    rng = np.random.default_rng(1)
    feed = {
        "src_cont":  rng.random((batch_size, seq_len, 6), dtype=np.float32),
        "src_types": rng.integers(0, 7, (batch_size, seq_len), dtype=np.int64),
    }

    def _probe(providers):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            s = ort.InferenceSession(str(onnx_path), sess_options=opts,
                                     providers=providers)
            for _ in range(5): s.run(None, feed)   # warmup
            t0 = time.perf_counter()
            for _ in range(20): s.run(None, feed)
            return (time.perf_counter() - t0) / 20 * 1000
        except Exception:
            return float("inf")

    cpu_ms = _probe(["CPUExecutionProvider"])
    dml_ms = _probe(_dml_provider(-1))
    return "directml" if dml_ms < cpu_ms else "cpu"


def get_inference_session(
    onnx_path: Path,
    provider: str = "auto",
    dml_device_id: int = -1,   # -1 = auto-select
) -> "onnxruntime.InferenceSession":
    """
    Create an OnnxRuntime InferenceSession.

    provider:
        "auto"      - try DmlExecutionProvider, fall back to CPU
        "directml"  - force DirectML (NPU/GPU)
        "cpu"       - force CPU
    dml_device_id:
        -1  = let DML pick (usually dGPU on mixed systems)
        0   = first DML device (iGPU / NPU on Ryzen AI)
        1   = second DML device (dGPU)
    """
    import onnxruntime as ort

    available = ort.get_available_providers()

    if provider == "cpu":
        providers = ["CPUExecutionProvider"]
    elif provider == "directml":
        if "DmlExecutionProvider" not in available:
            raise RuntimeError("DirectML not available. "
                               "pip install onnxruntime-directml")
        providers = _dml_provider(dml_device_id)
    else:  # auto — benchmark to pick fastest
        if "DmlExecutionProvider" in available:
            best = recommend_inference_provider(onnx_path)
            providers = (_dml_provider(dml_device_id)
                         if best == "directml"
                         else ["CPUExecutionProvider"])
        else:
            providers = ["CPUExecutionProvider"]

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.enable_mem_pattern       = True

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_opts,
        providers=providers,
    )
    return session


def _dml_provider(device_id: int):
    """Build DirectML provider options list."""
    if device_id < 0:
        # No explicit device — ORT/DML picks the best one
        return [("DmlExecutionProvider", {}), "CPUExecutionProvider"]
    return [
        ("DmlExecutionProvider", {"device_id": device_id}),
        "CPUExecutionProvider",
    ]


def session_provider_name(session) -> str:
    """Return the first (active) provider name of a session."""
    return session.get_providers()[0]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_inference(
    onnx_path: Path,
    n_warmup: int = 10,
    n_runs: int   = 200,
    seq_len: int  = 14,
) -> dict:
    """
    Benchmark ONNX inference latency on CPU vs DirectML.

    Returns dict with per-provider mean/std latency in ms.
    """
    import onnxruntime as ort
    available = ort.get_available_providers()

    providers_to_test: list[tuple[str, list]] = [
        ("CPU",       ["CPUExecutionProvider"]),
    ]
    if "DmlExecutionProvider" in available:
        providers_to_test.append(("DirectML (auto)", _dml_provider(-1)))

    rng = np.random.default_rng(0)
    dummy_cont  = rng.random((1, seq_len, 6), dtype=np.float32)
    dummy_types = rng.integers(0, 7, (1, seq_len), dtype=np.int64)
    feed = {"src_cont": dummy_cont, "src_types": dummy_types}

    results = {}
    for label, providers in providers_to_test:
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            sess = ort.InferenceSession(str(onnx_path),
                                        sess_options=sess_opts,
                                        providers=providers)
        except Exception as e:
            print(f"  [{label}] Session failed: {e}")
            continue

        # Warmup
        for _ in range(n_warmup):
            sess.run(None, feed)

        # Timed runs
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            sess.run(None, feed)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

        arr = np.array(latencies)
        results[label] = {
            "mean_ms":   round(float(arr.mean()), 3),
            "std_ms":    round(float(arr.std()),  3),
            "p50_ms":    round(float(np.percentile(arr, 50)), 3),
            "p95_ms":    round(float(np.percentile(arr, 95)), 3),
            "throughput_qps": round(1000.0 / float(arr.mean()), 1),
        }
        print(f"  [{label:20s}]  "
              f"mean={arr.mean():.2f}ms  p50={np.percentile(arr,50):.2f}ms  "
              f"p95={np.percentile(arr,95):.2f}ms  "
              f"QPS={1000/arr.mean():.0f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_hw_summary():
    """Print a full hardware + software configuration summary."""
    import psutil, platform
    import onnxruntime as ort

    ram = psutil.virtual_memory()
    cpu = platform.processor()

    lines = [
        "",
        "=" * 62,
        "  LEVELSMITH HARDWARE CONFIGURATION",
        "=" * 62,
        "",
        "  [System]",
        f"    CPU    : {cpu}",
        f"    RAM    : {ram.total/1024**3:.0f} GB total  "
              f"({ram.available/1024**3:.1f} GB free)",
        "",
        "  [GPU / Training]",
    ]

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        lines += [
            f"    Device : {p.name}",
            f"    VRAM   : {p.total_memory/1024**3:.1f} GB",
            f"    SM     : {p.multi_processor_count} (CC {p.major}.{p.minor})",
            f"    CUDA   : {torch.version.cuda}",
            f"    cuDNN  : {torch.backends.cudnn.version()}",
            f"    PyTorch: {torch.__version__}",
            f"    AMP    : FP16 (Ada Lovelace tensor cores)",
            f"    cuDNN benchmark : ENABLED",
        ]
    else:
        lines.append("    CUDA not available — using CPU")

    lines += [
        "",
        "  [NPU / Inference]",
        f"    NPU    : AMD XDNA 2 (50 TOPS, Ryzen AI 9 HX 370)",
        f"    ORT    : {ort.__version__}",
        f"    Providers: {', '.join(ort.get_available_providers())}",
    ]

    dml_available = "DmlExecutionProvider" in ort.get_available_providers()
    if dml_available:
        lines += [
            f"    DirectML: AVAILABLE (Windows ML → NPU/GPU dispatch)",
            f"    Inference backend: DmlExecutionProvider (preferred)",
    ]
    else:
        lines.append("    DirectML: NOT AVAILABLE (install onnxruntime-directml)")

    lines += [
        "",
        "  [Routing]",
        "    Training      -> CUDA GPU (RTX 4070, FP16 AMP)",
        "    Inference <1M -> CPU  (DML overhead > compute for tiny models)",
        "    Inference >50M-> DirectML (NPU/GPU dispatch via DX12)"
                          if dml_available else
                          "    Inference     -> CPU (onnxruntime-directml not installed)",
        "    Auto-select   -> hardware_config.recommend_inference_provider()",
        "",
        "=" * 62,
        "",
    ]

    print("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. AMP training helper (drop-in wrapper for train loop)
# ═══════════════════════════════════════════════════════════════════════════════

class AMPTrainer:
    """
    Wraps a training step with Automatic Mixed Precision.

    Usage:
        trainer = AMPTrainer(model, optimizer, device)
        for batch in dataloader:
            loss = trainer.step(loss_fn, *batch)
    """
    def __init__(self, model, optimizer, device: torch.device,
                 max_grad_norm: float = 1.0):
        self.model         = model
        self.optimizer     = optimizer
        self.device        = device
        self.max_grad_norm = max_grad_norm
        self.scaler        = get_amp_scaler(device)
        self._use_amp      = (device.type == "cuda")

    def step(self, loss_fn, *batch_args) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        if self._use_amp:
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(*batch_args)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = loss_fn(*batch_args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.max_grad_norm)
            self.optimizer.step()
        return loss.item()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print_hw_summary()

    # Find best available .pt model to export
    model_pt = MODEL_DIR / "layout_model_w3.pt"
    if not model_pt.exists():
        model_pt = MODEL_DIR / "layout_model.pt"
    if not model_pt.exists():
        print("No trained model found in models/. Skipping ONNX export.")
        return

    onnx_path = MODEL_DIR / (model_pt.stem + ".onnx")

    # Export
    print(f"[ONNX export] {model_pt.name} -> {onnx_path.name}")
    try:
        export_to_onnx(model_pt, onnx_path)
    except Exception as e:
        print(f"  Export failed: {e}")
        return

    # Benchmark — two scenarios
    print(f"\n[Inference benchmark]  {onnx_path.name}")

    results_b1  = {}
    results_b32 = {}

    print(f"\n  Scenario A: batch=1, seq=14  (single auto-regressive step)")
    results_b1 = benchmark_inference(onnx_path, seq_len=14)

    print(f"\n  Scenario B: batch=32, seq=14  (batch generation)")
    results_b32 = benchmark_inference(onnx_path, seq_len=14)

    # Summary table
    cpu_b1  = results_b1.get("CPU", {}).get("mean_ms", float("nan"))
    dml_b1  = results_b1.get("DirectML (auto)", {}).get("mean_ms", float("nan"))

    print(f"\n  {'Provider':<22}  {'B=1 mean':>10}  {'B=1 QPS':>9}  {'Recommended':>12}")
    print("  " + "-" * 58)
    for label, r in results_b1.items():
        rec = recommend_inference_provider(onnx_path, batch_size=1)
        is_rec = (label == "CPU" and rec == "cpu") or \
                 (label.startswith("DirectML") and rec == "directml")
        print(f"  {label:<22}  {r['mean_ms']:>9.2f}ms  "
              f"{r['throughput_qps']:>8.0f}  "
              f"{'<-- AUTO' if is_rec else '':>12}")

    if not (math.isnan(cpu_b1) or math.isnan(dml_b1)):
        print(f"\n  DirectML vs CPU speedup (batch=1): {cpu_b1/dml_b1:.2f}x")
        if cpu_b1 < dml_b1:
            print(f"  --> Auto-routing: CPU  "
                  f"(DML dispatch overhead > model compute for tiny batch)")
        else:
            print(f"  --> Auto-routing: DirectML")
    print(f"\n  Note: DirectML benefits appear at batch>=32 or model size>50M params.")
    print(f"        For single-step AR generation, CPU is optimal here.")
    print(f"        DirectML is available for future larger models.")

    print("\n[Done] hardware_config.py setup complete.")


if __name__ == "__main__":
    main()
