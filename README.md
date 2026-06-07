# 🌊 Plastic Pollution Detection from Images

> Binary image classification — **Plastic vs Non-Plastic** — using a Custom CNN, EfficientNetB0, and MobileNetV2 on the TrashNet dataset.

**Author:** Tejas Gowda  
**Program:** M.Sc. Computer Science — University of Europe for Applied Sciences, Potsdam  
**Supervisor:** Shan Faiz  
**Course:** Pattern Recognition — Phase 2 Project  
**Date:** June 2026

---

## 📋 Project Overview

This project builds and compares three deep learning models to detect plastic pollution from waste images:

| Model | Type | Test Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|---|
| Custom CNN | From scratch | 84.2% | 43.4% | 0.8516 |
| EfficientNetB0 | Transfer Learning | 80.8% | 0.0% | 0.4248 |
| **MobileNetV2** ★ | **Transfer Learning** | **94.2%** | **85.3%** | **0.9792** |

**MobileNetV2 is the best-performing model** with 94.2% accuracy, 87.7% recall, and AUC-ROC of 0.9792.

---

## 📂 Repository Structure

```
plastic-pollution-detection/
│
├── Final_Phase_2.ipynb          # Main Kaggle notebook — full pipeline
├── README.md                    # This file
│
├── figures/                     # All saved output figures
│   ├── 01_class_distribution.png
│   ├── 02_sample_images.png
│   ├── 03_augmentation.png
│   ├── 04_cnn_curves.png
│   ├── 05_efficientnet_curves.png
│   ├── 06_mobilenet_curves.png
│   ├── 07_confusion_matrices.png
│   ├── 08_roc_curves.png
│   ├── 09_model_comparison.png
│   ├── 10_gradcam.png
│   └── 11_error_analysis.png
│
└── models/                      # Saved trained models
    ├── custom_cnn_final.h5
    ├── efficientnetb0_final.h5
    └── mobilenetv2_final.h5
```

---

## 🗃️ Dataset

**TrashNet** by Gary Thung & Mindy Yang (Stanford University)

- **Kaggle:** https://www.kaggle.com/datasets/feyzazkefe/trashnet
- **Total images:** 2,527
- **Original classes:** cardboard, glass, metal, paper, plastic, trash
- **Binary labels used:** plastic (482 images) vs non-plastic (2,045 images)
- **Split:** 70% Train / 15% Validation / 15% Test (stratified)

> ⚠️ The dataset is not included in this repository. Add it via Kaggle (see instructions below).

---

## 🚀 How to Run

### Option 1 — Run on Kaggle (Recommended)

1. Go to [kaggle.com](https://kaggle.com) → **Create** → **New Notebook**
2. Upload `Final_Phase_2.ipynb` via **File → Import Notebook**
3. Add the dataset: **+ Add Input** → search `feyzazkefe/trashnet` → **Add**
4. Enable GPU: right panel → **Session options** → Accelerator → **GPU T4**
5. Click **Run All** — full training takes ~20–40 minutes on GPU

### Option 2 — Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/plastic-pollution-detection.git
cd plastic-pollution-detection

# 2. Install dependencies
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn pillow opencv-python

# 3. Download the dataset from Kaggle
#    Place it at: ./dataset-resized/ (cardboard/, glass/, metal/, paper/, plastic/, trash/)

# 4. Update BASE_DIR in the notebook:
#    BASE_DIR = Path('./dataset-resized')

# 5. Run the notebook
jupyter notebook Final_Phase_2.ipynb
```

---

## 🔧 Requirements

```
tensorflow >= 2.10
numpy
pandas
matplotlib
seaborn
scikit-learn
pillow
opencv-python
```

Install all at once:
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow opencv-python
```

---

## 🏗️ Model Architectures

### Custom CNN
- 4 convolutional blocks: 32 → 64 → 128 → 256 filters
- Each block: Conv2D × 2 → BatchNorm → ReLU → MaxPool → Dropout
- Head: GlobalAveragePooling → Dense(256, L2) → Dropout(0.5) → Dense(1, Sigmoid)
- ~3.2M parameters

### EfficientNetB0 (Transfer Learning)
- Pretrained on ImageNet (1.2M images, 1,000 classes)
- Phase 1: freeze base, train head (5 epochs, LR=1e-3)
- Phase 2: unfreeze last 20 layers, fine-tune (LR=1e-5)
- ~4.1M parameters

### MobileNetV2 (Transfer Learning) ★ Best
- Pretrained on ImageNet, depthwise separable convolutions
- Phase 1: freeze base, train head (5 epochs, LR=1e-3)
- Phase 2: unfreeze last 30 layers, fine-tune (LR=1e-5)
- ~2.3M parameters

---

## 📊 Results

### Confusion Matrices
Custom CNN classifies non-plastic well (297/307) but misses most plastic (only 23/73). EfficientNetB0 predicted everything as non-plastic. MobileNetV2 correctly identifies 294 non-plastic and 64 plastic items.

### Key Findings
- **MobileNetV2** achieves 94.2% accuracy, 87.7% recall, F1-score 85.3%, AUC 0.9792
- **Custom CNN** has high precision (69.7%) but very low recall (31.5%) due to class imbalance
- **EfficientNetB0** failed completely — fine-tuning LR of 1e-5 too aggressive for this dataset size
- **Grad-CAM** shows the CNN focuses on object contours and reflective surfaces for plastic detection
- Transparent plastic bottles are the hardest to classify — visually similar to glass

---

## 🔥 Grad-CAM Visualisations

Grad-CAM heatmaps show which image regions influence the model's prediction most. Red/warm areas = high activation for plastic class. The visualisations confirm the CNN learns plastic-relevant features but struggles with small or transparent items.

---

## 🔮 Future Work

- Apply class weighting or SMOTE oversampling to address the 1:4 class imbalance
- Lower EfficientNetB0 fine-tuning LR to 1e-6 for better convergence
- Collect more diverse real-world outdoor plastic pollution images
- Evaluate on datasets with environmental/field photography context
- Experiment with ensemble methods combining CNN and MobileNetV2 predictions

---

## 📚 References

- Thung, G., & Yang, M. (2016). *Classification of Trash for Recyclability Status.* Stanford CS229.
- Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* ICML 2019.
- Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018.
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations via Gradient-based Localization.* ICCV 2017.
- Chollet, F. (2021). *Deep Learning with Python, 2nd Edition.* Manning Publications.

---

## 📄 License

This project is submitted as academic coursework for the Pattern Recognition course at the University of Europe for Applied Sciences. All code is original work by Tejas Gowda.
