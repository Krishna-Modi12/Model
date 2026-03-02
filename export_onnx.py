"""
export_onnx.py — Export Face Shape Model to ONNX & TorchScript
================================================================

Environment Version Notes (Windows, Python 3.10, PyTorch 2.5.1+cu118):
  WORKING:  onnx==1.14.1  +  onnxruntime==1.19.2
  BROKEN:   onnx>=1.16    →  conflicts with ml_dtypes==0.3.1 installed by TF
  BROKEN:   onnx==1.16.x  →  DLL load error on this Python/Windows build

  Root cause: tensorflow-intel 2.15.0 pins ml_dtypes~=0.2.0 but installs
  0.3.1. onnx>=1.16 then requires ml_dtypes>=0.5.0 and calls
  ml_dtypes.float4_e2m1fn which doesn't exist in 0.3.1 → AttributeError.

  Fix applied: pin to onnx==1.14.1 (doesn't use ml_dtypes at all) and
  onnxruntime==1.19.2 (resolves DLL init failure on numpy 2.x).
  Add to requirements.txt if rebuilding:
    onnx==1.14.1
    onnxruntime==1.19.2
================================================================
Exports the trained PyTorch Lightning model to deployment-friendly formats.

Usage:
    python export_onnx.py
    python export_onnx.py --checkpoint "path/to/model.ckpt"

Confirmed forward signature (from predict.py):
    model(image_tensor, geo_ratios) -> output.face_shape_logits
    - image_tensor : Tensor[1, 3, 224, 224]  float32
    - geo_ratios   : Tensor[1, 15]           float32
    - output       : .face_shape_logits -> Tensor[1, 5]

NOTE: This script intentionally does NOT import FaceAnalysisLightningModule.
The Lightning import chain (torchvision -> torch._dynamo) conflicts with the
TensorFlow ml_dtypes package present in this environment. Instead we load the
model weights directly from the .ckpt file (which is a plain torch save dict).
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch

# ── Step 1: Deterministic mode & disable gradients ──────────
torch.manual_seed(42)
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass
torch.set_grad_enabled(False)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import only the bare model class (no Lightning, no torchvision dep chain)
from src.models.face_analysis_model import FaceAnalysisModel

# ── Constants ───────────────────────────────────────────────
DEFAULT_CHECKPOINT = str(
    PROJECT_ROOT / "checkpoints" / "finetune_matched_v3"
    / "finetune_v3_epoch=23_val_f1=0.7616.ckpt"
)
EXPORT_DIR = PROJECT_ROOT / "exported"
ONNX_PATH = EXPORT_DIR / "face_shape_model.onnx"
TORCHSCRIPT_PATH = EXPORT_DIR / "face_shape_model.pt"

# Forward signature constants (confirmed from predict.py)
IMAGE_SHAPE = (1, 3, 224, 224)
RATIOS_SHAPE = (1, 15)
NUM_CLASSES = 5


def print_separator(title: str = ""):
    print(f"\n{'=' * 56}")
    if title:
        print(f"  {title}")
        print(f"{'=' * 56}")


def get_file_size_mb(path: str) -> float:
    """Return file size in megabytes."""
    return os.path.getsize(path) / (1024 * 1024)


# ── Step 2: Load the model ──────────────────────────────────
def load_model(checkpoint_path: str) -> torch.nn.Module:
    """
    Load trained weights from a Lightning checkpoint without importing Lightning.

    Lightning .ckpt files are standard torch save dicts with keys:
      - 'state_dict'   : model weights prefixed with 'model.'
      - 'hyper_parameters', 'epoch', 'global_step', etc.

    We instantiate FaceAnalysisModel with the same config used during training,
    then strip the 'model.' prefix and load only the model weights.
    """
    print(f"  Loading checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"[FATAL] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load raw checkpoint dict
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract hyperparameters saved by Lightning (if present)
    hparams = ckpt.get("hyper_parameters", {})
    print(f"  Checkpoint epoch : {ckpt.get('epoch', 'unknown')}")
    print(f"  Hyper-parameters : {list(hparams.keys()) if hparams else 'none saved'}")

    # Instantiate model with the same config used during training
    model = FaceAnalysisModel(
        backbone="efficientnet_b4",
        pretrained=False,       # weights come from checkpoint, not ImageNet
        dropout=0.5,
        geometric_features=15,
        num_classes=5,
        freeze_backbone=False,  # export with full backbone
    )

    # Strip 'model.' prefix added by LightningModule
    state_dict = ckpt["state_dict"]
    model_state = {
        k.removeprefix("model."): v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }

    missing, unexpected = model.load_state_dict(model_state, strict=True)
    if missing:
        print(f"  [WARNING] Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  [WARNING] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    model.cpu()
    model.eval()
    print("  Model weights loaded successfully on CPU (eval mode)")
    return model


# ── Wrapper to extract only logits from ModelOutput ─────────
class LogitsWrapper(torch.nn.Module):
    """
    Wraps FaceAnalysisModel to return only the face_shape_logits tensor.
    ONNX export requires plain tensor outputs, not dataclass/namedtuple.
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, ratios: torch.Tensor) -> torch.Tensor:
        output = self.model(image, ratios)
        return output.face_shape_logits


# ── Step 3: Create dummy inputs ─────────────────────────────
def create_dummy_inputs():
    """
    Create dummy input tensors matching the exact forward signature.
    Always batch_size=1 for export and validation.
    """
    dummy_image = torch.randn(*IMAGE_SHAPE, dtype=torch.float32)
    dummy_ratios = torch.randn(*RATIOS_SHAPE, dtype=torch.float32)

    print(f"  Dummy image  shape: {list(dummy_image.shape)}  dtype: {dummy_image.dtype}")
    print(f"  Dummy ratios shape: {list(dummy_ratios.shape)}  dtype: {dummy_ratios.dtype}")

    return dummy_image, dummy_ratios


# ── Step 4: Export to ONNX ──────────────────────────────────
def export_onnx(wrapper: torch.nn.Module, dummy_image, dummy_ratios) -> bool:
    """Export model to ONNX format with dynamic batch axis."""
    print_separator("Step 4: ONNX Export")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {
        "image":  {0: "batch"},
        "ratios": {0: "batch"},
        "logits": {0: "batch"},
    }

    for opset in [17, 14]:
        try:
            print(f"  Exporting with opset_version={opset}...")
            torch.onnx.export(
                wrapper,
                (dummy_image, dummy_ratios),
                str(ONNX_PATH),
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=["image", "ratios"],
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
            )
            print(f"  ONNX exported successfully: {ONNX_PATH}")
            print(f"  File size: {get_file_size_mb(ONNX_PATH):.1f} MB")

            # Print discovered ONNX graph I/O
            import onnx
            onnx_model = onnx.load(str(ONNX_PATH))
            onnx.checker.check_model(onnx_model)
            print("\n  ONNX Graph I/O:")
            for inp in onnx_model.graph.input:
                shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
                print(f"    Input : name={inp.name:<8} shape={shape}")
            for out in onnx_model.graph.output:
                shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
                print(f"    Output: name={out.name:<8} shape={shape}")

            return True

        except Exception as e:
            if opset == 17:
                print(f"  [WARNING] opset 17 failed: {e}")
                print(f"  Retrying with opset 14...")
            else:
                print(f"  [FATAL] ONNX export failed with both opset 17 and 14: {e}")
                sys.exit(1)

    return False


# ── Step 5: Validate ONNX ───────────────────────────────────
def validate_onnx(wrapper, dummy_image, dummy_ratios) -> float:
    """
    Compare raw logits from PyTorch vs ONNX Runtime.
    Returns max absolute difference.
    """
    print_separator("Step 5: ONNX Validation")
    import onnxruntime as ort

    # PyTorch reference output (raw logits, NOT softmax)
    with torch.no_grad():
        pt_logits = wrapper(dummy_image, dummy_ratios).numpy()

    # ONNX Runtime output
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(
        ["logits"],
        {
            "image": dummy_image.numpy(),
            "ratios": dummy_ratios.numpy(),
        }
    )[0]

    max_diff = float(np.max(np.abs(pt_logits - onnx_logits)))
    print(f"  PyTorch logits : {pt_logits.flatten()}")
    print(f"  ONNX logits    : {onnx_logits.flatten()}")
    print(f"  Max abs diff   : {max_diff:.2e}")

    if max_diff < 1e-5:
        print("  ONNX Validation: PASSED")
    else:
        print("  ONNX Validation: FAILED")
        print(f"  [FATAL] Max difference {max_diff:.2e} exceeds tolerance 1e-5")
        sys.exit(1)

    return max_diff


# ── Step 6: Export to TorchScript ───────────────────────────
def export_torchscript(wrapper, dummy_image, dummy_ratios) -> float:
    """
    Export using torch.jit.trace() (no conditional branching detected).
    Falls back to torch.jit.script() if tracing fails.
    Returns max absolute difference against PyTorch reference.
    """
    print_separator("Step 6: TorchScript Export")

    # PyTorch reference
    with torch.no_grad():
        pt_logits = wrapper(dummy_image, dummy_ratios).numpy()

    # Try trace first (confirmed safe — no dynamic branching in forward)
    try:
        print("  Using torch.jit.trace() (no conditional branching detected)...")
        traced = torch.jit.trace(wrapper, (dummy_image, dummy_ratios))
        print("  Tracing succeeded")
    except Exception as e:
        print(f"  [WARNING] torch.jit.trace() failed: {e}")
        print("  Falling back to torch.jit.script()...")
        try:
            traced = torch.jit.script(wrapper)
            print("  torch.jit.script() succeeded")
        except Exception as e2:
            print(f"  [FATAL] Both trace and script failed: {e2}")
            sys.exit(1)

    # Save
    traced.save(str(TORCHSCRIPT_PATH))
    print(f"  TorchScript saved: {TORCHSCRIPT_PATH}")
    print(f"  File size: {get_file_size_mb(TORCHSCRIPT_PATH):.1f} MB")

    # Validate
    with torch.no_grad():
        ts_logits = traced(dummy_image, dummy_ratios).numpy()

    max_diff = float(np.max(np.abs(pt_logits - ts_logits)))
    print(f"  Max abs diff   : {max_diff:.2e}")

    if max_diff < 1e-5:
        print("  TorchScript Validation: PASSED")
    else:
        print("  TorchScript Validation: FAILED")
        print(f"  [FATAL] Max difference {max_diff:.2e} exceeds tolerance 1e-5")
        sys.exit(1)

    return max_diff


# ── Step 7: Benchmark ───────────────────────────────────────
def benchmark(wrapper, dummy_image, dummy_ratios):
    """
    Run fair benchmarks: 10 warm-up + 100 timed passes for each backend.
    All on CPU with torch.no_grad().
    """
    print_separator("Step 7: Inference Benchmark (CPU)")
    print(f"  CPU threads: {torch.get_num_threads()}")
    print(f"  Warm-up: 10 passes  |  Timed: 100 passes\n")

    import onnxruntime as ort

    results = {}

    # ── PyTorch ──
    for _ in range(10):
        wrapper(dummy_image, dummy_ratios)

    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        wrapper(dummy_image, dummy_ratios)
        times.append((time.perf_counter() - t0) * 1000)
    pt_avg = np.mean(times)
    pt_std = np.std(times)
    results["PyTorch"] = (pt_avg, pt_std)
    print(f"  PyTorch     : {pt_avg:.1f}ms avg +/- {pt_std:.1f}ms")

    # ── ONNX Runtime ──
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    feed = {"image": dummy_image.numpy(), "ratios": dummy_ratios.numpy()}

    for _ in range(10):
        session.run(["logits"], feed)

    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        session.run(["logits"], feed)
        times.append((time.perf_counter() - t0) * 1000)
    onnx_avg = np.mean(times)
    onnx_std = np.std(times)
    speedup_onnx = pt_avg / onnx_avg
    results["ONNX"] = (onnx_avg, onnx_std, speedup_onnx)
    print(f"  ONNX        : {onnx_avg:.1f}ms avg +/- {onnx_std:.1f}ms  ({speedup_onnx:.1f}x faster)")

    # ── TorchScript ──
    traced = torch.jit.load(str(TORCHSCRIPT_PATH), map_location="cpu")
    traced.eval()

    for _ in range(10):
        traced(dummy_image, dummy_ratios)

    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        traced(dummy_image, dummy_ratios)
        times.append((time.perf_counter() - t0) * 1000)
    ts_avg = np.mean(times)
    ts_std = np.std(times)
    speedup_ts = pt_avg / ts_avg
    results["TorchScript"] = (ts_avg, ts_std, speedup_ts)
    print(f"  TorchScript : {ts_avg:.1f}ms avg +/- {ts_std:.1f}ms  ({speedup_ts:.1f}x faster)")

    return results


# ── Step 8: Export Summary ──────────────────────────────────
def print_summary(checkpoint_path, onnx_diff, ts_diff, bench_results):
    """Print the final export summary."""
    ckpt_size = get_file_size_mb(checkpoint_path)
    onnx_size = get_file_size_mb(ONNX_PATH)
    ts_size   = get_file_size_mb(TORCHSCRIPT_PATH)
    onnx_speedup = bench_results["ONNX"][2]

    print_separator("Export Summary")
    print(f"  ONNX Model     : exported/face_shape_model.onnx")
    print(f"  TorchScript    : exported/face_shape_model.pt")
    print(f"  Original .ckpt : {ckpt_size:.1f} MB")
    print(f"  ONNX Size      : {onnx_size:.1f} MB")
    print(f"  TorchScript    : {ts_size:.1f} MB")
    print(f"  ONNX Speedup   : {onnx_speedup:.1f}x faster than PyTorch")
    print(f"  ONNX Match     : PASSED (max diff: {onnx_diff:.2e})")
    print(f"  TS Match       : PASSED (max diff: {ts_diff:.2e})")
    print(f"{'=' * 56}")
    print(f"  Note: TorchScript size is expected to be similar to the .ckpt.")
    print(f"  Note: ONNX strips optimizer state, so it is typically smaller.")
    print(f"{'=' * 56}\n")


# ── Main ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Export Face Shape Model to ONNX & TorchScript")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to Lightning checkpoint")
    args = parser.parse_args()

    # Print discovered signature first (as required)
    print("\nDiscovered forward signature: forward(image, ratios) -> logits")
    print(f"  image  : Tensor[1, 3, 224, 224]  float32")
    print(f"  ratios : Tensor[1, 15]           float32")
    print(f"  logits : Tensor[1, 5]            float32")

    # Step 2: Load model
    print_separator("Step 2: Load Model")
    raw_model = load_model(args.checkpoint)
    wrapper = LogitsWrapper(raw_model)
    wrapper.eval()

    # Step 3: Create dummy inputs
    print_separator("Step 3: Dummy Inputs")
    dummy_image, dummy_ratios = create_dummy_inputs()

    # Step 4: ONNX export
    export_onnx(wrapper, dummy_image, dummy_ratios)

    # Step 5: Validate ONNX
    onnx_diff = validate_onnx(wrapper, dummy_image, dummy_ratios)

    # Step 6: TorchScript export + validate
    ts_diff = export_torchscript(wrapper, dummy_image, dummy_ratios)

    # Step 7: Benchmark
    bench_results = benchmark(wrapper, dummy_image, dummy_ratios)

    # Step 8: Summary
    print_summary(args.checkpoint, onnx_diff, ts_diff, bench_results)

    print("All exports completed successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()
