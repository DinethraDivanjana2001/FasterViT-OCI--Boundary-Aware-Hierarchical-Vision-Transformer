import os
import torch
from fastervit import create_model

# Your laptop GPU (RTX 5060) is not supported by current PyTorch -> force CPU
device = "cpu"
print("Using device:", device)

# Define where weights are saved
WEIGHTS_DIR = r"C:\Users\Rebecca Fernando\FastVit_experiments\weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Models for Experiment 1 (ImageNet-1K pretrained)
models_to_load = [
    "faster_vit_0_224",
    "faster_vit_1_224",
    "faster_vit_2_224",
    "faster_vit_3_224",
    "faster_vit_4_224",
    "faster_vit_5_224",
    "faster_vit_6_224",
]


for model_name in models_to_load:
    print(f"\n===== Loading {model_name} =====")
    
    # Weight file path for this model
    model_path = os.path.join(WEIGHTS_DIR, f"{model_name}.pth.tar")
    print("  → Weight file:", model_path)

    # Load pretrained FasterViT
    model = create_model(
        model_name,
        pretrained=True,
        model_path=model_path,
    ).to(device)

    model.eval()

    # Dummy input (ImageNet resolution)
    x = torch.randn(1, 3, 224, 224).to(device)

    with torch.no_grad():
        y = model(x)

    print(f"  Output shape: {y.shape}")
    print("  ✓ Loaded and forward-pass completed")
