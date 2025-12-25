import os
import time
import torch
import argparse
from fastervit import create_model
from torch.serialization import add_safe_globals  # <-- NEW

# --------------------------------------------------------------
# Allow argparse.Namespace in checkpoints (PyTorch ≥ 2.6)
# Do this ONLY if you trust the checkpoint source (e.g., NVLabs).
# --------------------------------------------------------------
add_safe_globals([argparse.Namespace])

# --------------------------------------------------------------
# Auto-select device (GPU if available)
# --------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
print("torch.cuda.is_available():", torch.cuda.is_available())

# --------------------------------------------------------------
# Throughput settings
# --------------------------------------------------------------
BATCH_SIZE = 16
WARMUP_ITERS = 5
MEASURE_ITERS = 20
IMAGE_SIZE = 224

WEIGHTS_DIR = r"C:\Users\Rebecca Fernando\FastVit_experiments\weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def benchmark(model_name):
    print(f"\n=== Benchmarking {model_name} ===")

    model_path = os.path.join(WEIGHTS_DIR, f"{model_name}.pth.tar")
    print("  → Using weights:", model_path)

    model = create_model(
        model_name,
        pretrained=True,
        model_path=model_path
    ).to(device)

    model.eval()
    dummy = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(dummy)

    # Measure
    t0 = time.time()
    with torch.no_grad():
        for _ in range(MEASURE_ITERS):
            _ = model(dummy)
    t1 = time.time()

    elapsed = t1 - t0
    total_images = BATCH_SIZE * MEASURE_ITERS
    throughput = total_images / elapsed

    print(f"Processed {total_images} images in {elapsed:.3f}s")
    print(f"Throughput: {throughput:.2f} images/sec")


if __name__ == "__main__":
    for name in [
        "faster_vit_0_224", "faster_vit_1_224", "faster_vit_2_224",
        "faster_vit_3_224", "faster_vit_4_224", "faster_vit_5_224",
        "faster_vit_6_224"
    ]:
        benchmark(name)
