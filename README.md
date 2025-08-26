# OWOBJ: Open-World Objectness Modeling Unifies Novel Object Detection (CVPR 2025)

📂 [Repository](https://github.com/AI4Math-ShanZhang/OWOBJ) | 📄 [Paper (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Open-World_Objectness_Modeling_Unifies_Novel_Object_Detection_CVPR_2025_paper.pdf)

OWOBJ introduces an open-world objectness modeling framework that unifies novel object detection across benchmarks and settings. Building on transformer-based detectors with an explicit objectness formulation, OWOBJ improves unknown-object recall while preserving known-class detection.

This repository provides the official PyTorch implementation and training/evaluation recipes for instantiating our OWOBJ within the base model [PROB](https://github.com/orrzohar/PROB).

## ✨ Highlights
- Unified objectness modeling for open-world detection
- Strong unknown recall with robust known mAP
- Training/evaluation recipes adapted from PROB for fast reproduction

## 📦 Installation

We validate with Ubuntu 20.04/22.04, CUDA 11.3+, Python 3.10, and PyTorch 1.12.0 (cu113). Follow PROB’s setup guidance for maximum compatibility.

```bash
# Create environment
conda create --name owobj python==3.10.4 -y
conda activate owobj

# Python deps
pip install -r requirements.txt

# PyTorch (CUDA 11.3 build)
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 \
  --extra-index-url https://download.pytorch.org/whl/cu113
```

### Backbone Features

Download the DINO ResNet-50 backbone and place it in the `models` directory:

- [`dino_resnet50_pretrain.pth`](https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth)

### Compile CUDA ops
This project uses Deformable-DETR operators (same as PROB). Build them before training:
```bash
cd ./models/ops
sh ./make.sh
# optional sanity check
python test.py
```

## 📂 Data Structure

We follow the VOC-style layout used in PROB ([PROB repo](`https://github.com/orrzohar/PROB`)).

```
OWOBJ/
└── data/
    └── OWOD/
        ├── JPEGImages
        ├── Annotations
        └── ImageSets
            ├── OWDETR
            ├── TOWOD
            └── VOC2007
```
<code_block_to_apply_changes_from>
```
data/coco/
├── annotations/
├── train2017/
└── val2017/
```

2) Move all images from `train2017/` and `val2017/` into `data/OWOD/JPEGImages/`.

3) Convert COCO json annotations to VOC xml using `datasets/coco2voc.py`.

4) Download PASCAL VOC 2007 & 2012 (trainval 2007/2012 and test 2007), untar, then:
- Move all images to `data/OWOD/JPEGImages/`
- Move all annotations to `data/OWOD/Annotations/`

Note: OWOBJ follows VOC format for loading and evaluation, consistent with PROB.

## 🚀 Training

Single-node (4 GPUs) training:
```bash
bash ./run.sh
```

You can enable or switch configurations by editing `run.sh` and the scripts in `configs/`:
- `M_OWOD_BENCHMARK.sh`: training for tasks 1–4 on the MOWOD benchmark
- `EVAL_M_OWOD_BENCHMARK.sh`: evaluation on MOWOD (tasks 1–4)
- Other variants mirror PROB’s scripts and can be adapted similarly

If needed, grant execute permissions in `configs/` and `tools/`:
```bash
chmod +x configs/*.sh tools/*.sh
```

## 📈 Evaluation

```bash
bash ./run_eval.sh
```
- Place trained weights under `exps/` matching the expected directory structure (similar to PROB).

Example:
```
exps/
├── MOWODB/
     └── OWOBJ/ (t1.ph - t4.ph)

```
For additional training and evaluation details, please refer to the [PROB repository](https://github.com/orrzohar/PROB)..
## 📄 License
Apache-2.0

## 📝 Citation

If you use OWOBJ, please cite:
```bibtex
@inproceedings{Zhang_2025_CVPR_OWOBJ,
  author    = {Zhang, Shan and ...},
  title     = {Open-World Objectness Modeling Unifies Novel Object Detection},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
```

## 🙏 Acknowledgements
OWOBJ builds on excellent prior work including PROB, OW-DETR, Deformable DETR, DETReg, and OWOD. If you found OWOBJ useful please consider citing these works as well.
