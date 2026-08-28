import streamlit as st
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Page configuration
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="centered"
)


# CIFAR-10 class names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# Loading the pretrained MobileNetV2 base model
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(96, 96, 3)
)

# Freezing the pretrained layers
base_model.trainable = False


# Recreating the trained model architecture
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])


# Loading the trained weights
model.load_weights("cifar10_mobilenetv2.weights.h5")


# Application title
st.title("CIFAR-10 Image Classifier")

st.write(
    "Upload an image and the MobileNetV2 model will predict "
    "which CIFAR-10 class it belongs to."
)


# Image uploader
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Reading the uploaded image
    image = tf.keras.utils.load_img(
        uploaded_file,
        color_mode="rgb"
    )

    # Displaying the uploaded image
    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )


    # Converting image to NumPy array
    image_array = tf.keras.utils.img_to_array(image)


    # Resizing image to the size expected by MobileNetV2
    image_array = tf.image.resize(
        image_array,
        (96, 96)
    )


    # Applying MobileNetV2 preprocessing
    image_array = preprocess_input(
        image_array
    )


    # Adding batch dimension
    image_array = tf.expand_dims(
        image_array,
        axis=0
    )


    # Making prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )


    # Getting predicted class
    predicted_index = np.argmax(
        predictions[0]
    )

    predicted_class = class_names[predicted_index]

    confidence = predictions[0][predicted_index]


    # Displaying prediction
    st.subheader("Prediction")

    st.success(
        f"Predicted Class: {predicted_class}"
    )

    st.write(
        f"Confidence: {confidence * 100:.2f}%"
    )
