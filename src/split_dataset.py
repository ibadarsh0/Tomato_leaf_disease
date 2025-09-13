import os
import shutil
import random

# Paths
DATASET_DIR = "raw_data"   # your unzipped PlantVillage folder
OUTPUT_DIR = "data"        # where train/ and test/ will be created

# Train-test split ratio
SPLIT_RATIO = 0.8  

# Make train and test directories
train_dir = os.path.join(OUTPUT_DIR, "train")
test_dir = os.path.join(OUTPUT_DIR, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Loop through only tomato folders
for folder in os.listdir(DATASET_DIR):
    if folder.startswith("Tomato"):
        class_dir = os.path.join(DATASET_DIR, folder)
        images = os.listdir(class_dir)
        random.shuffle(images)

        split_point = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split_point]
        test_imgs = images[split_point:]

        # Make class subfolders
        os.makedirs(os.path.join(train_dir, folder), exist_ok=True)
        os.makedirs(os.path.join(test_dir, folder), exist_ok=True)

        # Copy images
        for img in train_imgs:
            shutil.copy(os.path.join(class_dir, img), os.path.join(train_dir, folder, img))
        for img in test_imgs:
            shutil.copy(os.path.join(class_dir, img), os.path.join(test_dir, folder, img))

        print(f"Class {folder}: {len(train_imgs)} train, {len(test_imgs)} test")