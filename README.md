# Machine Learning & Computer Vision (MLCV) Workspace

A central repository dedicated to tracking my learning journey, implementations, and core projects in **Machine Learning and Computer Vision**. This workspace serves as a hands-on laboratory for experimenting with modern architectures, custom training loops, and data pipelines.

---

## 📂 Repository Structure

```text
mlcv_ws/
└── src/
    ├── datasets/            # Datasets used for local training (ignored by Git)
    ├── scripts/             # Core scripts for data processing and model routines
    │   ├── cifar_exp1/      # Experiment 1: ViT vs. ConvNeXt-Tiny on CIFAR-10
    │   ├── training/        # Model training and optimization loops
    │   └── inference/       # Evaluation and prediction scripts
    ├── notebooks/           # Exploratory Data Analysis (EDA) and prototyping
    └── requirements.txt     # Dependencies and environment setup
```

Each experiment lives in its own folder under `src/scripts/` (`cifar_exp1`, `cifar_exp2`, ...) so that its code, report and outputs stay together.

---

## 🛠️ Tech Stack & Tools

* **Core Frameworks:** PyTorch / TensorFlow
* **Computer Vision:** OpenCV, Albumentations, Torchvision
* **Data & Math:** NumPy, Pandas, Scikit-Learn
* **Visualization:** Matplotlib, Pillow
* **Experiment Tracking:** Weights & Biases / TensorBoard

---

## 🧪 Experiments

| # | Folder | Topic | Models | Status |
|---|--------|-------|--------|--------|
| 1 | [`src/scripts/cifar_exp1/`](src/scripts/cifar_exp1/) | CIFAR-10 image classification and model interpretability | Vision Transformer, ConvNeXt-Tiny | ✅ Done |

### Experiment 1: CIFAR-10 with a Vision Transformer and ConvNeXt-Tiny

**Goal:** train two very different architectures on CIFAR-10, push accuracy as high as possible, and then look inside the trained models to see what they learn.

**Dataset:** CIFAR-10 (50,000 labelled 32×32 images, 10 classes), split 40,000 train / 10,000 validation with stratified sampling (1,000 validation images per class, seed 42).

**Starting point (baselines):**

| Model | Validation accuracy |
|-------|---------------------|
| Linear classifier (Keras) | about 43% |
| Small MLP (Keras) | about 47% |

**Part A: Vision Transformer (trained from scratch)**

* Patch size 4, embedding dim 192, 6 heads, 6 transformer blocks, MLP dim 384 (about 1.8 M parameters).
* AdamW with warmup and cosine decay, dropout, label smoothing, random crop and flip augmentation.
* A `--sweep` mode adds one fix at a time (learning rate, patch size, size of the model, augmentation, regularization) to show what each change gains.
* **Visualizer** (`visualize.py`): runs one image through the saved model and plots the learned patch filters, patch-conv feature maps, token features after every block, per-head CLS attention maps, and attention rollout.

**Part B: ConvNeXt-Tiny (pure CNN, ImageNet-pretrained and fine-tuned)**

* 32×32 images upscaled to 224×224, fine-tuned for 10 epochs with AdamW and label smoothing.
* Other backbones can be swapped in with `--arch` (`efficientnet_b0`, `resnet50`, or `resnet18_cifar` for a from-scratch baseline).
* **Visualizer** (`cnn_visualize.py`): hooks every `Conv2d` layer (22 in ConvNeXt-Tiny) and shows the image after each convolution with the trained weights, plus an overview of the whole network and the first-layer kernels.

**Results** (update after running):

| Model | Best validation accuracy |
|-------|--------------------------|
| ViT (from scratch) | _fill in_ |
| ConvNeXt-Tiny (fine-tuned) | _fill in_ |

**Full write-up:** see [`CIFAR10_ViT_and_ConvNeXt_Report.md`](src/scripts/cifar_exp1/CIFAR10_ViT_and_ConvNeXt_Report.md) for the architecture diagrams, tensor shapes through every operation, and an explanation of each visualization figure.

**Files in `src/scripts/cifar_exp1/`:**

| File | Purpose |
|------|---------|
| `model.py`, `train.py`, `visualize.py` | ViT model, training/sweep script, visualizer |
| `cnn_model.py`, `cnn_train.py`, `cnn_visualize.py` | CNN model builders, fine-tuning script, visualizer |
| `CIFAR10_ViT_and_ConvNeXt_Report.md` | Detailed report with figures |

Generated files (`cifar10_train.pt`, `vit_best.pt`, `cnn_best.pt`, and the `viz/` and `viz_cnn/` image folders) are outputs of the scripts below and are best kept out of Git (see the note under Installation).

**Run it:**

```bash
cd src/scripts/cifar_exp1

# 0. One-time: convert the CIFAR-10 PNGs to cifar10_train.pt
#    (this is the data-conversion cell in the ViT notebook)

# Vision Transformer
python train.py --sweep                  # optional: compare settings
python train.py                          # trains and saves vit_best.pt
python visualize.py --ckpt vit_best.pt --image cat.png

# ConvNeXt-Tiny
python cnn_train.py                      # fine-tunes and saves cnn_best.pt
python cnn_visualize.py --ckpt cnn_best.pt --image cat.png
python cnn_visualize.py --ckpt cnn_best.pt --list_layers
```

---

## 🚀 Projects & Learning Focus

* **Foundational ML:** Regression, Classification, Clustering, and Feature Engineering.
* **Deep Learning:** Custom CNN architectures, Transfer Learning, and Optimization techniques.
* **Computer Vision:** Object Detection, Semantic Segmentation, and Image Augmentation pipelines.
* **Model Interpretability:** Visualizing patch embeddings, attention maps and convolutional feature maps (Experiment 1).

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd mlcv_ws/src
   ```

2. **Set up a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

> **Note:** keep datasets and large generated files out of Git by adding them to `.gitignore`:
> ```text
> datasets/
> *.pt
> viz/
> viz_cnn/
> ```

---

## 🗺️ Adding a New Experiment

1. Create `src/scripts/<name>/` (for example `cifar_exp2`) with its scripts and a short report.
2. Add a row to the **Experiments** table above, with a short summary section like Experiment 1.
3. Record the final numbers in the results table once the run is finished.

---

## 📝 License
This repository is open-source and available under the **MIT License**.