# Garbage Classification Using CNN & Transfer Learning (PyTorch)

This project builds two models for classifying waste images into six categories — **cardboard, glass, metal, paper, plastic,** and **trash** — using:
1. A **custom Convolutional Neural Network (CNN)** built from scratch  
2. **Transfer Learning with ResNet50**, leveraging pretrained ImageNet weights for enhanced feature extraction

The project covers the entire deep-learning workflow: data preprocessing, model architecture, training, evaluation, and inference.

---

## 🗂 Project Structure

## 🔧 Tech Stack

- Python
- PyTorch
- Torchvision (ResNet50 models)
- NumPy / Pandas
- Matplotlib
- Jupyter Notebook / Google Colab

---

## 🧹 Data Preprocessing & Augmentation

All images were:
- Resized to **256×256**
- Normalized
- Augmented (training only) using:
  - `RandomHorizontalFlip`
  - `RandomResizedCrop(256)`
  - `ColorJitter`
  - `ToTensor()`

Validation & test used:
- `Resize(256)`
- `CenterCrop(256)`
- `ToTensor()`

---

## 🧠 Model Architecture

### ✔️ Transfer Learning with ResNet50
- Loaded a **pretrained ResNet50** model (`pretrained=True`)
- Replaced the final fully connected layer:
  ```python
  num_ftrs = self.network.fc.in_features
  self.network.fc = nn.Linear(num_ftrs, 6)

---

## 🚀 Training Details

- **Optimizer:** Adam  
- **Loss:** CrossEntropyLoss  
- **Epochs: 9**    
- GPU acceleration when available  
- Custom training/validation loops

---

## 📊 Results

### **Custom CNN Results**
| Metric | Value |
|-------|--------|
| Validation Accuracy | **96%** |
| Validation Loss | 0.1505 |
| Test Accuracy | **93%** |
| Test Loss | 0.0032 |

### **ResNet50 Transfer Learning Results**
- Achieved **higher accuracy and faster convergence**  
- Extracted stronger high-level features due to ImageNet pretraining  
- Showed improved generalization on complex, noisy garbage images  


---

## 📈 Visualizations

Includes:
- Training/validation accuracy curves  
- Training/validation loss curves  
- Confusion-style performance inspection  
- Sample predictions  

---

## 🔍 Inference

A real-time prediction function is included for both models:

1. Load input image  
2. Apply preprocessing  
3. Perform inference with The Model  
4. Return predicted category  

---


## 🏁 Conclusion

Using transfer learning significantly boosts accuracy, improves generalization, and reduces training time — making the system more practical for production-level waste-sorting applications.

---
