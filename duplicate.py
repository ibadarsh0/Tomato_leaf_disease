import os
from PIL import Image

DATASET_DIR = "dataset/test"  # change to "dataset/test" and run again

bad_files = []
for root, _, files in os.walk(DATASET_DIR):
    for f in files:
        file_path = os.path.join(root, f)
        try:
            img = Image.open(file_path)
            img.verify()
        except Exception:
            bad_files.append(file_path)
            os.remove(file_path)  # delete bad file
            print("Deleted:", file_path)

print("\nTotal bad files deleted:", len(bad_files))