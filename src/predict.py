import tensorflow as tf
import numpy as np
from data_loader import prepare_datasets  # optional, just to reuse class names

MODEL_PATH = "models/tomato_model.keras"
IMG_PATH = "test_images/diseased_leaf1.JPG"   # <-- change this to your image path

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load datasets just to get class names
_, _, class_names = prepare_datasets("dataset/train", "dataset/test")

# Load and preprocess image
img = tf.keras.utils.load_img(IMG_PATH, target_size=(224, 224))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # add batch dimension

# Predict
predictions = model.predict(img_array)
pred_class = np.argmax(predictions[0])
confidence = np.max(predictions[0])

print(f"Predicted class: {class_names[pred_class]} (confidence: {confidence:.2f})")