import streamlit as st
import tensorflow.lite as tflite
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import prepare_datasets

# ==========================
# Load model & class names
# ==========================

interpreter = tflite.Interpreter(model_path="models/tomato_model.tflite")
interpreter.allocate_tensors()

# Use data loader to get class names
_, _, class_names = prepare_datasets("dataset/train", "dataset/test")

# Disease info (expand this dict with all your classes!)
disease_info = {
    "bacterial spot": "❌ Bacterial Spot detected. Remove infected leaves and apply copper fungicide.",
    "early blight": "❌ Early Blight detected. Remove affected leaves, use fungicide spray.",
    "healthy": "✅ The plant is healthy. Keep monitoring regularly.",
    "late blight": "❌ Late Blight detected. Apply copper fungicide, improve air circulation.",
    "leaf mold": "❌ Leaf Mold detected. Provide better ventilation, use fungicide.",
    "mosaic virus": "❌ Mosaic Virus detected. Remove infected plants immediately.",
    "septoria leaf spot": "❌ Septoria Leaf Spot detected. Remove debris, use fungicide.",
    "spider mites two spotted spider mite": "❌ Spider Mite infestation detected. Use insecticidal soap or neem oil.",
    "target spot": "❌ Target Spot detected. Apply preventive fungicide sprays.",
    "yellowleaf curl virus": "❌ Yellow Leaf Curl Virus detected. Control whiteflies and remove infected plants."
}

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="🌿 Tomato Leaf Disease Detector", layout="centered")

st.title("🌿 Tomato Leaf Disease Detector")
st.write("Upload a tomato leaf image and let AI diagnose it!")

# File uploader
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Leaf", use_column_width=True)

    # Preprocess
    img = tf.keras.utils.load_img(uploaded_file, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    pred_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # Display result
    st.subheader(f"Prediction: *{pred_class}* ({confidence:.2f}%)")

    # Disease description
    st.info(disease_info.get(pred_class.lower(), "ℹ No treatment info available."))

    # Confidence bar chart
    st.subheader("🔎 Model Confidence by Class")
    fig, ax = plt.subplots()
    ax.barh(class_names, score, color="seagreen")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Classes")
    ax.set_xlim([0, 1])
    st.pyplot(fig)