import os
import torch
from fastervit import create_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

WEIGHTS_DIR = r"C:\Users\Rebecca Fernando\FastVit_experiments\weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

for name in ["faster_vit_0_224", "faster_vit_1_224", "faster_vit_2_224"]:
    print(f"\nLoading model: {name}")
    model_path = os.path.join(WEIGHTS_DIR, f"{name}.pth.tar")
    print("  -> Using weight file:", model_path)

    model = create_model(
        name,
        pretrained=True,
        model_path=model_path
    ).to(device)

    model.eval()
    x = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        y = model(x)
    print(f"  Output shape for {name}:", y.shape)
