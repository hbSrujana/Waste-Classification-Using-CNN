# ♻️ Plastic Waste Classification using CNN

### 📌 Edunet-Shell Skills4Future AICTE Internship Project

---

## 📝 Description

This project aims to build an intelligent image classification system using **Convolutional Neural Networks (CNN)** to classify plastic waste into two categories: **Recyclable** and **Organic**.  
A **Streamlit-based web application** is also integrated for real-time classification, supporting automated waste segregation and environmental sustainability.

---

## 🗂️ Dataset

The dataset contains two subfolders:

- `TRAIN/` – for training images  
- `TEST/` – for testing images

Each image is labeled as either **Recyclable** or **Organic**.

---

## ⚙️ Tools & Technologies

- **Language:** Python  
- **Libraries & Frameworks:**
  - TensorFlow & Keras – Model building
  - OpenCV – Image processing
  - Pandas – Data handling
  - Matplotlib – Data visualization
  - Streamlit – Web app deployment

---

## 📅 Weekly Progress

### 📍 Week 1: Dataset Preparation

- Downloaded and organized dataset into `TRAIN/` and `TEST/` folders.
- Preprocessed images using OpenCV and converted them to RGB.
- Created a DataFrame with images and labels.
- Visualized label distribution using pie charts.
- Created `waste_classification.ipynb` notebook for model development.

---

### 📍 Week 2: CNN Model Development

- Built a CNN model with the following architecture:
  - 3 Convolutional Layers (32, 64, 128 filters) with ReLU and MaxPooling
  - Flatten layer
  - Dense Layer (256 neurons, ReLU, Dropout 0.5)
  - Dense Layer (64 neurons, ReLU, Dropout 0.5)
  - Output Layer (2 neurons, Softmax activation)
- Compiled with:
  - **Loss Function:** Binary Crossentropy
  - **Optimizer:** Adam
  - **Batch Size:** 64
  - **Epochs:** 15
- Visualized training and validation accuracy and loss.

---

### 📍 Week 3: Evaluation & Deployment

- Evaluated the model using test data.
- Developed a **Streamlit web app (`Waste_classification.py`)**:
  - Users can upload an image.
  - Model returns prediction: *Recyclable* or *Organic*.

---

## 📊 Results

- The model performs well with stable accuracy and loss.
- The Streamlit app provides a user-friendly interface for real-time waste classification.
