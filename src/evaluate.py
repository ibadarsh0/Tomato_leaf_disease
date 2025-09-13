import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from data_loader import prepare_datasets

# Paths
TEST_DIR = "dataset/test"
MODEL_PATH = "models/tomato_model.keras"

# Load dataset
_, test_ds, class_names = prepare_datasets("dataset/train", TEST_DIR)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Evaluate accuracy
loss, acc = model.evaluate(test_ds)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

# -------------------------
# Predictions
# -------------------------
y_true = np.concatenate([y for _, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# -------------------------
# Confusion Matrix
# -------------------------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
plt.title("Confusion Matrix - Tomato Leaf Disease")
plt.show()

# -------------------------
# Per-Class Accuracy
# -------------------------
print("\n🔎 Classification Report (per-class precision, recall, F1-score):\n")
print(classification_report(y_true, y_pred, target_names=class_names))