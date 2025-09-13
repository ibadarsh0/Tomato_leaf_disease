import tensorflow as tf
from data_loader import prepare_datasets
from model import build_model
import pickle
import os

# Dataset paths
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
MODEL_PATH = "models/tomato_model.keras"
HISTORY_PATH = "models/history.pkl"

# Load datasets
train_ds, test_ds, class_names = prepare_datasets(TRAIN_DIR, TEST_DIR)
num_classes = len(class_names)

# -----------------------
# Stage 1: Feature Extraction
# -----------------------
print("🔹 Stage 1: Training with frozen EfficientNetB0")
model = build_model(num_classes, fine_tune=False)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
]

history1 = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10,
    callbacks=callbacks
)

# -----------------------
# Stage 2: Fine-tuning
# -----------------------
print("🔹 Stage 2: Fine-tuning last 20 layers")
model = build_model(num_classes, fine_tune=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=5,
    callbacks=callbacks
)

# -----------------------
# Save final model & history
# -----------------------
if not os.path.exists("models"):
    os.makedirs("models")

model.save(MODEL_PATH)

# Combine histories
history = {}
for k in history1.history.keys():
    history[k] = history1.history[k] + history2.history[k]

with open(HISTORY_PATH, "wb") as f:
    pickle.dump(history, f)

print("✅ Training complete. Best model saved at:", MODEL_PATH)

# after model.fit(...)
import pickle
with open("models/history.pkl", "wb") as f:
    pickle.dump(history.history, f)