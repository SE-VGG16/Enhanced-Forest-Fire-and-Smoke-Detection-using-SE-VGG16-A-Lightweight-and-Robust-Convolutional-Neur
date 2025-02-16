# 🔥 Enhanced Forest Fire and Smoke Detection using SE-VGG16: A Lightweight and Robust Convolutional Neural Network  
Our paper is currently under submission, and detailed data information will be made public after its publication

🚀 **SE-VGG16** is a deep learning model based on **VGG16**, enhanced with **Squeeze-and-Excitation Blocks (SEBlocks)** to improve feature representation. This model efficiently detects **forest fires and smoke** in **real-time** with high accuracy, making it ideal for **UAV monitoring, satellite analysis, and wildfire prevention systems**.  

## 🌲 Why SE-VGG16?  
✔️ **State-of-the-Art Accuracy** – Achieves **98% accuracy**, **98.5% recall**, and **98% F1-score**, outperforming existing methods.  
✔️ **Lightweight & Efficient** – Optimized for **real-time detection** with UAVs and edge devices.  
✔️ **Enhanced Feature Extraction** – Uses **SEBlocks** to **improve sensitivity to fire features** in complex environments.  
✔️ **Robust to Challenging Conditions** – Works effectively **under dense vegetation, varying lighting, and smoke occlusion**.  
📌 Key Features:
🔹 SEBlocks integration for enhanced feature recalibration.
🔹 Trained on a diverse Kaggle dataset of fire and non-fire images.
🔹 Supports real-time inference on resource-constrained devices.
---

## 📂 Repository Structure  
- `model/` – Contains the **SE-VGG16** architecture.  
- `dataset/` – Dataset preprocessing and augmentation.  
- `training/` – Training pipeline with **cross-entropy loss and Adam optimizer**.  
- `evaluation/` – Computes accuracy, precision, recall, and F1-score.  
- `saving/` – Saves pretrained weights for transfer learning.  
- `testing/` – Loads the trained model and makes predictions on test images.  

---

## 🛠 Installation  

1️⃣ **Clone the repository:**  
```bash
git clone https://github.com/SE-VGG16/SE-VGG16-A-High-Performance-Model-for-Forest-Fire-and-Smoke-Detection.git
cd SE-VGG16
2️⃣ **Install dependencies:**
```bash
pip install -r requirements.txt


