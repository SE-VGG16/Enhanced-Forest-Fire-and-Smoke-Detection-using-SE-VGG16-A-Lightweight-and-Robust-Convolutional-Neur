🔥 Enhanced Forest Fire and Smoke Detection using SE-VGG16: A Lightweight and Robust Convolutional Neural Network  

---

## 🌟 Introduction  
SE-VGG16 is a deep learning-based model that enhances **VGG16** with **Squeeze-and-Excitation (SE) Blocks** to improve fire and smoke detection. The model is optimized for **real-time UAV-based detection**, **low-resource environments**, and **high detection accuracy**.  

✔ **98% Accuracy**  
✔ **Real-time performance for UAVs**  
✔ **Lightweight & scalable architecture**  
✔ **Works under dense vegetation and smoke occlusion**  

📄 **Paper:** [Coming Soon]  
📂 **Dataset:** [Kaggle Forest Fire Dataset](https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images)  


---

## 🏗 Architecture  
SE-VGG16 is based on **VGG16**, modified with **Squeeze-and-Excitation Blocks (SEBlocks)** to enhance feature recalibration:  



✔ **Feature recalibration with SEBlocks**  
✔ **Improved small-scale fire detection**  
✔ **Optimized for high-precision classification**  

---

## 🛠 Installation  

### Clone the repository  
\`\`\`bash
git clone https://github.com/SE-VGG16/SE-VGG16-A-High-Performance-Model-for-Forest-Fire-and-Smoke-Detection.git
cd SE-VGG16
\`\`\`

### Install dependencies  
\`\`\`bash
pip install -r requirements.txt
\`\`\`

---

## 📂 Dataset Preparation  
SE-VGG16 is trained on a **fire/non-fire classification dataset**. Follow these steps:  
1. **Download Dataset**: [Kaggle Dataset](https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images)  
2. **Organize Data:**  
\`\`\`
dataset/
  ├── train/
  │   ├── fire/
  │   ├── non-fire/
  ├── val/
  │   ├── fire/
  │   ├── non-fire/
  ├── test/
      ├── fire/
      ├── non-fire/
\`\`\`
3. **Preprocess the Dataset**  
\`\`\`bash
python dataset_preprocessing.py --input dataset/ --output processed_data/
\`\`\`

---

## 🎯 Training  
Run the following command to **train SE-VGG16**:  
\`\`\`bash
python train.py --epochs 10 --batch-size 32 --lr 0.001 --save-path best_model.pth
\`\`\`
✔ Uses **Cross-Entropy Loss**  
✔ Optimized with **Adam Optimizer**  
✔ Supports **GPU acceleration**  

---

## 📊 Evaluation  
Evaluate the trained model using the test dataset:  
\`\`\`bash
python evaluate.py --model best_model.pth --data test/
\`\`\`
✔ **Accuracy, Precision, Recall, F1-Score**  
✔ Generates a **detailed classification report**  

---

## 🧪 Testing & Inference  
Test the model on a **single image**:  
\`\`\`bash
python test.py --model best_model.pth --image sample.jpg
\`\`\`
✔ Output: `"🔥 Fire Detected"` or `"✅ Normal"`  

---

## 📌 Pretrained Models  
Pre-trained SE-VGG16 models are available:  
📥 **[Download Here](https://github.com/SE-VGG16/weights/)**  

---

## 🚀 Performance Comparison  
### 🔥 SE-VGG16 outperforms SOTA models!  
| Model            | Accuracy | Recall | Precision | F1-Score |
|-----------------|---------|--------|-----------|---------|
| ResNet-50       | 88%     | 86%    | 84.6%     | 85.3%  |
| VGG-16          | 89%     | 87.2%  | 85%       | 86.1%  |
| DenseNet121     | 87%     | 85.7%  | 82%       | 83.8%  |
| **SE-VGG16 (Ours)** | **98%** | **98.5%** | **97%** | **98%** |

📌 SE-VGG16 achieves **98% accuracy** while maintaining a **lightweight architecture**!

---

## 💾 Saving & Loading the Model  
Save the trained model:  
\`\`\`python
torch.save(model.state_dict(), "best_model.pth")
\`\`\`
Load the trained model for inference:  
\`\`\`python
model.load_state_dict(torch.load("best_model.pth"))
\`\`\`

---

## 📝 Citation  
If you use **SE-VGG16** in your research, please cite:  
📄 **Enhanced Forest Fire and Smoke Detection using SE-VGG16**  
\`\`\`bibtex
@article{SE-VGG16,
  title={Enhanced Forest Fire and Smoke Detection using SE-VGG16},
  author={Akmalbek Abdusalomov, Sabina Umirzakova, et al.},
  journal={Journal of AI & Computer Vision},
  year={2025}
}
\`\`\`

---

## 🎯 Future Work  
✔️ **Deploying SE-VGG16 on UAVs**  
✔️ **Enhancing temporal fire detection with video streams**  
✔️ **Developing an edge-device optimized version**  

---

## 🤝 Contributors  
👨‍💻 **Akmalbek Abdusalomov**  
👩‍💻 **Sabina Umirzakova**  
👨‍💻 **Komil Tashev**  
👩‍💻 **Guzalxon Belalova**  

📬 For inquiries: [Contact Us](mailto:research@se-vgg16.org)  

---

## 📌 License  
This project is released under the **MIT License**.  

---

🚀 **Let's make wildfire detection smarter, faster, and more efficient!** 🔥  
"""


