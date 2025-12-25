# Experiment 1 — FasterViT ImageNet-1K Throughput Replication

This folder contains all scripts related to **Experiment 1** from the FasterViT paper:

> **“Image classification on ImageNet-1K + throughput comparison across model variants.”**

Since training FasterViT on full ImageNet requires multi-GPU clusters (A100s),  
this experiment focuses on the part that *is feasible on local hardware*:

### ⭐ **Goal:**  
Replicate the **inference throughput** results (images/sec) for:

- `faster_vit_0_224`
- `faster_vit_1_224`
- `faster_vit_2_224`

using the official **pretrained ImageNet-1K weights**.

This matches the “throughput” component of the FasterViT paper’s Experiment 1.

---

## 1. Why Throughput Replication?

The FasterViT paper evaluates two main metrics:

1. **Top-1 accuracy** (requires full ImageNet-1K training → not feasible locally)
2. **Throughput** (forward-pass speed measured in images/sec → feasible)

Throughput directly reflects the model’s efficiency and scalability.  
FasterViT introduces carrier tokens + hierarchical attention to improve speed,  
so **reproducing throughput is the most important part of Experiment 1**.

---

## 2. Hardware Notes

Your laptop GPU (RTX 5060 Laptop) reports:

