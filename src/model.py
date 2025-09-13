import tensorflow as tf

def build_model(num_classes, fine_tune=False):
    """
    Build an EfficientNetB0-based model for tomato disease classification.

    Args:
        num_classes (int): number of classes
        fine_tune (bool): whether to unfreeze last layers for fine-tuning
    """
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    if fine_tune:
        # Unfreeze last 20 layers
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
    else:
        # Keep frozen
        base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    return model