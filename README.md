# 🖼️ CIFAR-10 Image Classification Using MobileNetV2

An image classification project that uses **Transfer Learning with MobileNetV2** to classify CIFAR-10 images into 10 different classes.

The project covers image preprocessing, model training, evaluation, and deployment using **Streamlit**.

---

## 📌 About the Project

The **CIFAR-10 dataset** contains 60,000 RGB images belonging to 10 classes:

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

The original images are **32 × 32 pixels**. Since MobileNetV2 works with larger input images, the images were resized to **96 × 96** before being passed to the model.

A pretrained **MobileNetV2** model with ImageNet weights was used as the feature extractor. Its pretrained layers were frozen, and a custom classification head was added for the 10 CIFAR-10 classes.

---

## 🧠 Model Architecture

```text
MobileNetV2
     ↓
Global Average Pooling
     ↓
Dense (128, ReLU)
     ↓
Dropout (0.3)
     ↓
Dense (10, Softmax)
```

### Model Details

- **Base Model:** MobileNetV2
- **Pretrained Weights:** ImageNet
- **Input Size:** 96 × 96 × 3
- **Batch Size:** 32
- **Epochs:** 5
- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Trainable Parameters:** 165,258
- **Total Parameters:** 2,423,242

---

## 📊 Results

The model achieved a **final test accuracy of 86.34%**.

---

## 🖥️ Streamlit Application

The trained model was integrated into a **Streamlit web application**.

The application allows users to:

- Upload an image
- Process the image
- Predict its CIFAR-10 class
- Display the prediction confidence

### Example

```text
Actual: Cat
Predicted: Cat
Confidence: 91.20%
```

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pillow
- Streamlit

---

## 📂 Project Structure

```text
CIFAR10-Image-Classification/
│
├── app.py
├── cifar10_mobilenetv2.weights.h5
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## ▶️ Run the Project

Clone the repository:

```bash
git clone https://github.com/SwathiAdireddy/cifar10-image-classification.git
```

Go to the project directory:

```bash
cd cifar10-image-classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📚 Key Learning

This project helped me understand:

- Image classification using deep learning
- CNN-based feature extraction
- Transfer learning with MobileNetV2
- Image preprocessing
- Model evaluation
- Confusion matrix analysis
- Saving and loading model weights
- Building a Streamlit ML application

---

## 👩‍💻 Author

**Swathi**
