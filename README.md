🍅 Tomato Leaf Disease Detection

<p align="center">
  <img src="assets/banner.jpg" alt="Tomato Leaf Disease Detection Banner" width="100%" />
</p>
------------------------------------------
🌿 Overview

This project uses EfficientNetB0 transfer learning to classify tomato leaf diseases.
It detects 10 classes:

1.Bacterial Spot
2.Early Blight
3.Healthy
4.Late Blight
5.Leaf Mold
6.Mosaic Virus
7.Septoria Leaf Spot
8.Spider Mites (Two-Spotted)
9.Target Spot
10.Yellow Leaf Curl Virus

The model is trained on ~1550 images per class and integrated into a Streamlit UI for easy use.

-------------------------------------------------------------------------------------------------

✨ Features

✅ Upload leaf image → get instant prediction
✅ Confidence bar chart for all classes
✅ Disease description + treatment suggestion
✅ Interactive Streamlit web app
✅ Easy training & evaluation scripts


-------------------------------------------------------------------------------------------------

📂 Project Structure

TomatoLeafDisease/
│
├── models/                  # trained model (tomato_model.keras)
├── src/
│   ├── train.py             # training script
│   ├── evaluate.py          # evaluation script
│   ├── predict.py           # single image prediction
│   ├── app.py               # Streamlit UI
│   ├── model.py             # EfficientNetB0 architecture
│   └── data_loader.py       # dataset preparation
│
├── requirements.txt
├── README.md
└── .gitignore


---------------------------------------------------------------------------------------------------

⚙ Installation

# Clone repo
git clone https://github.com/ibadarsh0/Tomato_leaf_disease.git
cd TomatoLeafDisease

# Install dependencies
pip install -r requirements.txt


---------------------------------------------------------------------------------------------------

🚀 Usage

🔹 Train Model

python src/train.py

🔹 Evaluate Model

python src/evaluate.py

🔹 Predict Single Image

python src/predict.py --image sample_leaf.jpg

🔹 Run Streamlit App

streamlit run src/app.py


---------------------------------------------------------------------------------------------------

📊 Model Performance

Training Accuracy: ~83% (15 epochs)

Validation accuracy improves with fine-tuning


You can also visualize confusion matrix for better insights.


---------------------------------------------------------------------------------------------------

🌱 Example Streamlit App

(📸 Add a screenshot here once you run the app locally, e.g., screenshot.png)


---------------------------------------------------------------------------------------------------

🧩 Future Improvements

Deploy on Streamlit Cloud / HuggingFace Spaces

Add data augmentation for better generalization

Improve model with EfficientNetV2 or Vision Transformers



---------------------------------------------------------------------------------------------------

📜 License

This project is released under the MIT License.