# 🌿 Plant Disease Detection System using Deep Learning

A **Deep Learning–based Plant Disease Detection System** that classifies plant leaf images into disease categories.  
The system uses **MobileNetV2** with transfer learning and is deployed as a **single-file Streamlit web application**.

---

## 📌 Project Overview

Plant diseases can severely impact agricultural productivity. Early and accurate detection enables timely treatment and helps reduce crop loss.  
This project uses **Convolutional Neural Networks (CNNs)** to automatically identify plant diseases from leaf images.

- **Best Model:** MobileNetV2  
- **Deployment:** Streamlit  
- **Input:** Leaf image  
- **Output:** Predicted plant disease class  

---

## 🎯 Objectives

- Develop an accurate plant disease classification model  
- Evaluate multiple pre-trained CNN architectures  
- Select the best-performing model based on test accuracy  
- Deploy the solution using an interactive Streamlit interface  

---

## 📂 Dataset Description

- Image-based plant disease dataset  
- Split into training, validation, and testing sets  
- Each class represents a specific plant disease  

### Dataset Structure
dataset/
├── train/
├── validation/
└── test/

---

## 🧠 Deep Learning Models Evaluated

- MobileNetV2  
- ResNet50  
- EfficientNetB0  

Transfer learning was applied by freezing base layers and training custom classification layers.

---

## 📈 Model Performance

| Model | Test Accuracy |
|------|--------------|
| **MobileNetV2** | ⭐ Best Model |
| ResNet50 | 39.74% |
| EfficientNetB0 | 33.77% |

MobileNetV2 achieved the best performance and was selected for deployment.

---

## ⚙️ Training Configuration

- Image Size: 224 × 224  
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Batch Size: 32  
- Epochs: 15  
- Data Augmentation: Rotation, Zoom, Horizontal Flip  

---

## 🌐 Streamlit Web Application

The Streamlit app enables users to:

- Upload a plant leaf image  
- Instantly receive disease predictions  
- Interact with a simple and responsive UI  

The complete workflow (loading model, preprocessing, prediction) is implemented in a **single Python file**.

---

## 🛠 Technologies Used

- Python  
- TensorFlow & Keras  
- Streamlit  
- NumPy  
- Pillow (PIL)  

---

## 📂 Project Structure

plant-disease-detection/

├── app.py
├── best_plant_disease_model.h5
├── requirements.txt
└── README.md

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection
pip install -r requirements.txt
streamlit run app.py
