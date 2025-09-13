import tensorflow as tf

def prepare_datasets(train_dir, test_dir, img_size=(224, 224), batch_size=32):
    """
    Prepares training and test datasets with caching, shuffling, and prefetching.

    Args:
        train_dir (str): Path to training dataset directory
        test_dir (str): Path to testing dataset directory
        img_size (tuple): Target image size (default: (224,224))
        batch_size (int): Batch size (default: 32)

    Returns:
        train_ds (tf.data.Dataset): training dataset
        test_ds (tf.data.Dataset): testing dataset
        class_names (list): list of class names
    """

    # Training dataset
    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    # Extract class names before transformations
    class_names = raw_train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    # Optimize train dataset
    train_ds = (
        raw_train_ds
        .shuffle(1000)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )

    # Test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    test_ds = (
        test_ds
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )

    return train_ds, test_ds, class_names