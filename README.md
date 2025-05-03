# ♻️ Plastic Waste Classification using CNN

### 📌 Edunet-Shell Skills4Future AICTE Internship Project

---

## 🎯 Project Aim

The aim of this project is to build a **deep learning-based image classification system** using **Convolutional Neural Networks (CNN)** that automatically classifies plastic waste into two categories: **Recyclable** and **Organic**.  
It also includes a **Streamlit web app** for real-time waste classification, promoting automated waste segregation and environmental awareness.

---

## 🗂️ Dataset

The dataset consists of labeled images organized into the following folders:
- **TRAIN/** – Images used for training the model.
- **TEST/** – Images used for testing and evaluation.

---

## 🛠️ Tools & Technologies

- **Programming Language**: Python  
- **Libraries & Frameworks**:
  - TensorFlow & Keras (for model building)
  - OpenCV (for image processing)
  - Pandas (for data handling)
  - Matplotlib (for data visualization)
  - Streamlit (for building the web app)

---

## 📅 Weekly Progress

### ✅ Week 1: Dataset Preparation
- Downloaded and organized dataset into `TRAIN/` and `TEST/` folders.
- Preprocessed images using OpenCV and converted them to RGB format.
- Created a DataFrame containing image data and corresponding labels.
- Visualized label distribution using pie charts.
- Created `waste_classification.ipynb` notebook for development.

---

### ✅ Week 2: CNN Model Development
- Built a CNN model with the following architecture:
  - **Conv2D (32, 64, 128 filters)** → ReLU → MaxPooling
  - **Flatten** → Dense(256) → Dropout(0.5) → Dense(64) → Dropout(0.5)
  - **Output Layer**: Dense(2) with softmax activation
- Compiled using:
  - **Loss Function**: Binary Cross-Entropy
  - **Optimizer**: Adam
  - **Metrics**: Accuracy
- Trained the model for **15 epochs** with **batch size 64**
- Visualized training and validation accuracy/loss

---

### ✅ Week 3: Evaluation & Deployment
- Implemented `predict_fun(img)` for classifying uploaded images.
- Evaluated model using test images and accuracy plots.
- Developed a **Streamlit web app (`Waste_classification.py`)**:
  - Users can upload an image.
  - App displays prediction: **Recyclable** or **Organic**.

---

## 📊 Results & Observations

- The model shows good performance and stable training curves.
- Loss and accuracy plots indicate minimal overfitting.
- Some misclassifications suggest scope for fine-tuning.
- Streamlit app provides an interactive and easy-to-use interface.

---

## 🚀 How to Run the App

1. Clone the repository.
2. Install dependencies using:
