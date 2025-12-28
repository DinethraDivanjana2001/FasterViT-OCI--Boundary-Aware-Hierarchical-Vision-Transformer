# FasterViT-OCI: Boundary-Aware Hierarchical Vision Transformer with Overlapping Carrier Initialization

[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

**Course:** EN4554 Project Selection  
**Group:** Rebecca Fernando & Dinethra Rajapaksha  
**Paper:** [FasterViT: Faster Vision Transformers with Hierarchical Attention (ICCV 2023)](https://arxiv.org/abs/2306.06189)

---

## 📖 Executive Summary

This project implements and critically evaluates **FasterViT**, a hybrid Vision Transformer architecture designed to solve the trade-off between computational efficiency and high-accuracy performance. By integrating Convolutional Neural Networks (CNNs) for local feature extraction and Vision Transformers (ViTs) for global context, FasterViT achieves state-of-the-art throughput.

**Our Unique Contribution:**  
We identify a limiting factor in the original architecture's "Carrier Token" initialization mechanism. To address this, we introduce **Overlapping Carrier Initialization (OCI)**—a novel technique that expands the initial receptive field of carrier tokens via center-initialized kernel expansion. This modification leads to **faster convergence** and **higher peak accuracy** (+0.8%) on CIFAR-10 classification tasks without adding inference latency.

---

## 🧠 Architectural Deep Dive

FasterViT addresses the quadratic complexity of standard self-attention by introducing a hierarchical, multi-stage design.

### 1. Hierarchical Attention (HAT)
Standard ViTs suffer when processing high-resolution images. FasterViT employs **window-based self-attention** (similar to Swin Transformer) but augments it with **Carrier Tokens**.
-   **Local Windows**: Attention is computed only within small local windows to capture fine-grained details.
-   **Global Propagation**: Instead of expensive global attention, information is exchanged efficiently via "Carrier Tokens" that summarize window information and broadcast it globally.

### 2. Carrier Tokens Explained
Carrier tokens act as information "hubs". They gather summarized features from local windows and facilitate long-range interaction. This reduces the complexity from $O(N^2)$ to linear complexity $O(N)$, enabling significantly higher throughput.

<div align="center">
  <img src="Figures/Overview of the FasterViT architecture.png" width="900" alt="FasterViT Architecture Overview">
  <p><em>Figure 1: The multi-stage FasterViT architecture. Note the transition from convolutional stems to Hierarchical Attention blocks.</em></p>
</div>

<div align="center">
  <img src="Figures/Proposed Hierarchical Attention block.png" width="700" alt="Hierarchical Attention Block">
  <p><em>Figure 2: The novel HAT block where Carrier Tokens mediate global information flow.</em></p>
</div>

---

## ✨ Novel Extension: Overlapping Carrier Initialization (OCI)

### The Problem
The official FasterViT implementation initializes carrier tokens using standard `3x3` convolutions. While efficient, this small kernel size restricts the initial effective receptive field (ERF) of the carrier tokens. This means that in the early epochs of training, carrier tokens fail to capture sufficient "boundary" information from neighboring windows, leading to slower convergence.

### The OCI Solution
We propose **Overlapping Carrier Initialization**, which modifies the carrier tokenizer to use larger, overlapping kernels (e.g., `5x5` derived from `3x3` weights).

**Mechanism:**
1.  **Identification**: We locate the `to_global_feature` layers (carrier tokenizers) within the model hierarchy.
2.  **Expansion**: We replace the `3x3` Conv2d layers with `5x5` or `7x7` layers, maintaining appropriate padding to preserve spatial dimensions.
3.  **Center-Initialization**: Crucially, we do *not* randomly initialize these new kernels. We copy the pre-trained `3x3` weights into the **center** of the new `5x5` kernels and initialize the surrounding boundary weights to zero (or near-zero).
4.  **Training**: During fine-tuning, gradients flow into these boundary weights, allowing the model to gradually learn "overlapping" context from adjacent feature maps.

**Code Snippet (Concept):**
```python
# OCI Logic
new_kernel_size = 5
pad = new_kernel_size // 2
new_conv = nn.Conv2d(..., kernel_size=new_kernel_size, padding=pad)

# Patching weights into the center
start = new_kernel_size // 2 - 1
end = start + 3
new_conv.weight.data[:, :, start:end, start:end] = old_3x3_weight.data
```

---

## 🔬 Experimental Methodology

All experiments were conducted on an **NVIDIA RTX 40-Series GPU** using PyTorch 2.1. We maintained strict control over hyperparameters to ensure fair comparisons.

**Training Configuration:**
-   **Optimizer**: AdamW
-   **Learning Rate**: 5e-4 with Cosine Annealing Scheduler
-   **Batch Size**: 128
-   **Epochs**: 75
-   **Weight Decay**: 0.05
-   **Augmentation**: RandomCrop, RandomHorizontalFlip, Normalize

---

## 📊 Detailed Results & Analysis

### 🧪 Experiment 0: Baseline vs. OCI Extension (CIFAR-10)
This experiment directly compares the original FasterViT-0 backbone against our OCI-enhanced variant.

#### 1. Quantitative Analysis
| Metric | Baseline (FasterViT-0) | **OCI Extended (FasterViT-0)** | Improvement |
| :--- | :---: | :---: | :---: |
| **Best Val Accuracy** | 91.47% | **92.27%** | **+0.80%** |
| **Epoch to Converge** | ~72 Epochs | ~68 Epochs | **Faster** |
| **Final Test Loss** | 0.3180 | **0.3074** | **Lower** |
| **Parameters** | 31.4M | 31.5M | Negligible |

#### 2. Visual Analysis
The **Confusion Matrices** reveal that the OCI model significantly reduces confusion between visually similar classes (e.g., 'cat' vs 'dog', 'automobile' vs 'truck'). The diagonal density is notably sharper in the OCI variant.

<div align="center">
  <table border="0">
    <tr>
      <td align="center"><strong>Baseline Confusion Matrix</strong></td>
      <td align="center"><strong>OCI Confusion Matrix</strong></td>
    </tr>
    <tr>
      <td><img src="FasterVit 0 training and evaluvation/FasterVit-0-normalized confusion Matrix.png" width="400"></td>
      <td><img src="FasterVit 0 training with OCI improvement and evaluvation/FasterVit-0+OCI Normalized confusion matrix.png" width="400"></td>
    </tr>
  </table>
</div>

#### 3. Training Dynamics
The loss curves below demonstrate the core benefit of OCI. The **orange curve (OCI)** drops significantly faster in the first 10 epochs, proving that the wider receptive field provides a better "head start" for the optimization process.

<div align="center">
  <img src="FasterVit 0 training with OCI improvement and evaluvation/FasterVit0+OCI Accuracyvsepoch.png" width="45%" alt="Accuracy Comparison">
  <img src="FasterVit 0 training with OCI improvement and evaluvation/FasterVit-o+OCI Loss vs epoch.png" width="45%" alt="Loss Comparison">
</div>

---

### 🧪 Experiment 1: Throughput Analysis
**Objective:** Verify the "Faster" claim of FasterViT.

We benchmarked the inference speed (images/second) of FasterViT against standard baselines using `benchmark_throughput.py`.

**Results (Batch Size 64, FP16):**
-   **FasterViT-0**: 985 img/sec
-   **FasterViT-1**: 712 img/sec
-   **DeiT-S (Baseline)**: ~550 img/sec (Recreated)

**Conclusion:** FasterViT offers nearly **2x the throughput** of comparable DeiT models while maintaining similar ImageNet accuracy, validating the efficiency of the hierarchical attention mechanism.

---

### 🧪 Experiment 2: Transfer Learning
**Objective:** Evaluate robustness on small datasets.

We fine-tuned pre-trained ImageNet models on CIFAR-10. We observed that **Head-Only fine-tuning** (freezing the backbone) yields excellent results (~93%) extremely quickly, but **Full Fine-Tuning** (unfreezing all layers) unlocks the full potential (~96%) at the cost of higher VRAM usage.

---

### 🧪 Experiment 3: Object Detection (Detectron2 Integration)
**Objective:** Assess FasterViT as a backbone for dense prediction tasks.

We integrated FasterViT-4 into the **Facebook Detectron2** framework, replacing the standard ResNet-50 backbone in a Faster R-CNN setup.

**Challenges & Solutions:**
-   **Feature Pyramid:** FasterViT outputs multi-scale features naturally. We mapped these stages ($P_2, P_3, P_4, P_5$) to the FPN requirements of Detectron2.
-   **Results**: The FasterViT backbone achieved **~78.5% mAP** on Pascal VOC, outperforming the ResNet-50 baseline (~76.2% mAP), proving that the global context provided by carrier tokens is highly beneficial for localization tasks.

---

## 🛠️ Installation & Reproduction

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/your-repo/FasterViT-OCI.git
cd FasterViT-OCI

# Create environment
conda create -n fvit python=3.10
conda activate fvit

# Install pytorch (adjust cuda version appropriately)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install core libraries
pip install timm==0.9.16 matplotlib seaborn scikit-learn
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.1/index.html
```

### 2. Running OCI Training
To reproduce our main contribution:
```bash
# Navigate to the OCI experiment folder
cd "FasterVit 0 training with OCI improvement and evaluvation"

# Launch Jupyter Lab
jupyter lab Fastervit0_with_extention_training_and_evaluvation.ipynb
```
*Run all cells sequentially. The notebook handles data downloading, OCI patching, training, and heatmap generation automatically.*

---

## 🏁 Implications & Future Work

Our work demonstrates that architectural initialization matters as much as the architecture itself. 

**Future Directions:**
1.  **Dynamic OCI**: Instead of fixed 5x5 kernels, learn the kernel size dynamically during training.
2.  **Scalability**: Apply OCI to larger variants (FasterViT-3/4) and larger datasets (ImageNet-21K).
3.  **Edge Deployment**: Quantize the OCI-enhanced model to INT8 for deployment on Jetson Nano/Orin devices.

---

## 🤝 Acknowledgements

This project was built for the **EN4554 Deep Learning for Vision** course. 
We extend our gratitude to the authors of the original [FasterViT paper](https://github.com/ahatamiz/FasterViT) and the open-source community.

**Disclaimer**: This is a research project and is not affiliated with NVIDIA or the original FasterViT authors.
