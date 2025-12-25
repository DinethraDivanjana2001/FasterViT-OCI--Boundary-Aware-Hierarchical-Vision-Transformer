import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from timm.data import resolve_data_config, create_transform
from fastervit import create_model

# ==========================
# 1) Config
# ==========================

# Use CPU because RTX 5060 CUDA is not supported by this PyTorch build
device = "cpu"
print("Using device:", device)

# Path to your small ImageNet-1K subset (validation-style)
IMAGENET_ROOT = r"C:\Users\Rebecca Fernando\FastVit_experiments\imagenet1kvaldataset"

# Where FasterViT weights are stored
WEIGHTS_DIR = r"C:\Users\Rebecca Fernando\FastVit_experiments\weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Which model to evaluate
MODEL_NAME = "faster_vit_0_224"   # you can change to 1,2,... later

BATCH_SIZE = 16        # keep small for CPU
NUM_WORKERS = 4        # adjust if needed
MAX_SAMPLES = 1000     # set None to use all images, or a number like 1000/5000/10000


# ==========================
# 2) Load model + transforms
# ==========================

def load_model(model_name):
    model_path = os.path.join(WEIGHTS_DIR, f"{model_name}.pth.tar")
    print(f"\nLoading model: {model_name}")
    print("  → Using weights:", model_path)

    model = create_model(
        model_name,
        pretrained=True,
        model_path=model_path,
    ).to(device)

    model.eval()

    # Use timm to get the correct preprocessing for this model
    cfg = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**cfg)

    print("  Data config:", cfg)
    return model, transform


# ==========================
# 3) Build dataset + dataloader
# ==========================

def build_dataloader(transform):
    print("\nBuilding dataset from:", IMAGENET_ROOT)
    dataset = datasets.ImageFolder(IMAGENET_ROOT, transform=transform)

    print("  Total images in folder:", len(dataset))
    if MAX_SAMPLES is not None and MAX_SAMPLES < len(dataset):
        print(f"  Using subset of first {MAX_SAMPLES} images for evaluation.")
        dataset = Subset(dataset, range(MAX_SAMPLES))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    num_classes = len(dataset.dataset.classes) if isinstance(dataset, Subset) else len(dataset.classes)
    print("  Num classes:", num_classes)
    return loader, num_classes


# ==========================
# 4) Evaluation (Top-1 / Top-5)
# ==========================

def accuracy_topk(output, target, topk=(1,)):
    """Compute top-k accuracies."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top-k predictions
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # shape: [maxk, batch_size]
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k * 100.0 / batch_size).item())
        return res


def evaluate(model, loader):
    print("\nStarting evaluation...")
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)  # [B, 1000]
            top1, top5 = accuracy_topk(outputs, targets, topk=(1, 5))

            batch_size = images.size(0)
            total_top1 += top1 * batch_size
            total_top5 += top5 * batch_size
            total_samples += batch_size

    top1_avg = total_top1 / total_samples
    top5_avg = total_top5 / total_samples
    return top1_avg, top5_avg, total_samples


# ==========================
# 5) Main
# ==========================

if __name__ == "__main__":
    # 1. Load model + its preprocessing
    model, transform = load_model(MODEL_NAME)

    # 2. Build dataloader from your subset
    loader, num_classes = build_dataloader(transform)

    # 3. Evaluate
    top1, top5, n = evaluate(model, loader)

    print("\n========== EVALUATION RESULT ==========")
    print(f"Model:     {MODEL_NAME}")
    print(f"Samples:   {n}")
    print(f"Top-1 Acc: {top1:.2f}%")
    print(f"Top-5 Acc: {top5:.2f}%")
    print("=======================================")
