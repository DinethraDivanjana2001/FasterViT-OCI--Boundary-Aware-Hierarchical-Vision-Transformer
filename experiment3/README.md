# Experiment 3 — Detectron2 + FasterViT Backbones (Pascal VOC)

This folder trains and compares Detectron2 Faster R-CNN detectors on Pascal VOC using (1) the standard ResNet-50 C4 backbone and (2) a custom FasterViT-4 C4-style backbone, plus throughput and inference demos.

## Files
- [experiment3/experiment3_V1.ipynb](experiment3/experiment3_V1.ipynb) — Full pipeline: data download (VOC via Kaggle, optional COCO sample zip), baseline Faster R-CNN R50-C4 training/eval, throughput, then swapping in FasterViT-4 (ImageNet-21K weights) as a Detectron2 backbone, re-training, eval, throughput, and inference demo on a sample image.
- [experiment3/experiment3_V2.ipynb](experiment3/experiment3_V2.ipynb) — Same workflow as V1 with minor cleanup; keep kaggle.json upload, Drive mounts, and FasterViT wrapper identical.

## Requirements
- GPU (Colab-style scripts assume CUDA).
- Kaggle API key (`kaggle.json`) to download VOC 2007/2012; notebook uploads and moves it to `~/.kaggle/` with 600 perms.
- Python packages installed in-notebook: `torch`, `torchvision`, `detectron2` (from GitHub), `pycocotools`, `kaggle`, `fastervit>=0.9.8`.
- FasterViT checkpoint downloaded in-notebook: `fastervit_4_21k_224_w14.pth.tar` (Hugging Face link already in cells).

## What the notebooks do
- **Data**: download Pascal VOC 2007+2012, register datasets, optional COCO mini zip.
- **Baseline detector**: Detectron2 Faster R-CNN with ResNet-50 C4 backbone; short training run (5k iters, small batch), evaluation on VOC2007 test, throughput measurement with dummy inputs.
- **FasterViT backbone**: custom Detectron2 backbone wrapper that uses FasterViT-4 (any-res) up to level1, projects to 1024 channels, and plugs into the C4 head; trains on the same config as baseline for a fair comparison; evaluation and throughput.
- **Inference demo**: loads saved weights from Google Drive, runs prediction on a sample image (`darknet dog`), visualizes detections.

## How to run (Colab-style)
1) Upload `kaggle.json` when prompted; confirm it lands in `~/.kaggle/` with 600 perms.  
2) Run install cells (PyTorch, Detectron2, pycocotools, fastervit).  
3) Execute dataset download/register cells.  
4) Train baseline R50-C4; checkpoints saved under Drive path `detectron2_voc/output_voc_resnet_c4`.  
5) Train FasterViT-C4; checkpoints saved under `detectron2_voc/output_voc_fastervit_c4`.  
6) Run eval cells for VOC2007 test; compare mAP and throughput prints.  
7) Run inference cells to visualize detections (ensure correct weight paths if you changed output dirs).

## Tips
- If not on Colab, adjust paths (`/content/...` and Drive mounts) to local paths and ensure CUDA is available.
- If Detectron2 registry complains about duplicate backbones, the notebook already deletes existing `FasterViT_C4Backbone`; re-run that cell after restarts.
- Throughput cells use dummy tensors (3×800×800, batch 4); lower batch or resolution to fit smaller GPUs.
- Keep the FasterViT weight path in `cfg.MODEL.FASTERVIT.WEIGHTS` aligned with where the `.pth.tar` file is downloaded.
