# 🖼️ CIFAR-10 Image Classification Using MobileNetV2

An end-to-end deep learning project that classifies images into 10 different categories using **Transfer Learning with MobileNetV2**.

The project includes dataset preprocessing, model training, evaluation, performance analysis, and deployment using a Streamlit web application.

---

## 🚀 Live Demo

(Add your Streamlit Cloud link here)

---

# 📌 Project Overview

Image classification is a fundamental computer vision task where a model learns to identify and categorize images into predefined classes.

In this project, the **CIFAR-10 dataset** is used to train an image classification model using **MobileNetV2**, a pretrained convolutional neural network originally trained on ImageNet.

Transfer learning is used to reuse the pretrained MobileNetV2 features while training a new classification head specifically for the 10 CIFAR-10 classes.

The trained model is deployed using Streamlit, allowing users to upload an image and receive a predicted class along with the prediction confidence.

---

# 🎯 Objectives

- Understand the fundamentals of image classification.
- Preprocess image data for deep learning.
- Apply transfer learning using MobileNetV2.
- Train a classifier for the CIFAR-10 dataset.
- Evaluate model performance using multiple metrics.
- Visualize classification performance using a confusion matrix.
- Deploy the trained model using Streamlit.

---

# 📂 Dataset

The project uses the **CIFAR-10 dataset**.

The dataset contains **60,000 RGB images** of size 32 × 32 belonging to 10 different classes.

- 50,000 training images
- 10,000 testing images

## Classes

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

# 🔍 Data Preprocessing

The images were prepared before being passed to MobileNetV2.

The following preprocessing steps were performed:

- Converted images to `float32`.
- Resized images from **32 × 32** to **96 × 96**.
- Applied MobileNetV2-specific preprocessing.
- Split the training data into training and validation sets.
- Created efficient TensorFlow datasets using `tf.data`.
- Batched the data with a batch size of 32.
- Used prefetching to improve data pipeline performance.

---

# 🛠️ Technologies Used

## Programming Language

- Python

## Libraries

- TensorFlow
- Keras
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Deep Learning

- Convolutional Neural Networks
- Transfer Learning
- MobileNetV2

## Deployment

- Streamlit

---

# ⚙️ Machine Learning Workflow

```text
CIFAR-10 Dataset
        ↓
Data Preprocessing
        ↓
Train-Validation Split
        ↓
Image Resizing
        ↓
MobileNetV2 Preprocessing
        ↓
Pretrained MobileNetV2
        ↓
Freeze Pretrained Layers
        ↓
Add Custom Classification Head
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Streamlit Deployment
