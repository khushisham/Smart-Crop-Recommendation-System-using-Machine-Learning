# 🌾 Smart Crop Recommendation System using Machine Learning

This project is a machine learning-based crop recommendation system that suggests the most suitable crop to grow based on environmental factors like soil nutrients (N, P, K), temperature, humidity, pH, and rainfall. The system is trained on real-world agricultural data and uses a Random Forest Classifier for accurate predictions.

---

## 📌 Project Highlights

- Predicts the most suitable crop for cultivation using soil and climate data.
- Trained and tested multiple machine learning models, including SVM, KNN, Decision Tree, and Ensemble methods.
- Random Forest Classifier selected as the final model due to its superior accuracy.
- Includes a custom function for real-time crop prediction using user input.
- Available as both a Python script and Jupyter/Colab notebook for ease of use.

---

## 🛠 Tech Stack

- Programming Language: Python
- Libraries Used: pandas, numpy, scikit-learn, matplotlib, seaborn
- Tools: Jupyter Notebook, Google Colab, Python script

---

## 🔍 How It Works

1. Load the crop dataset containing features like N, P, K, temperature, humidity, pH, and rainfall.
2. Perform data cleaning, normalization, and label encoding.
3. Train multiple classification models and evaluate their performance.
4. Use the best-performing model (Random Forest) for prediction.
5. Implement a custom function that accepts environmental values and returns the most suitable crop.

---

## 📂 Files in the Repository

- `recommendation_model.py`: Python script containing the entire ML pipeline and prediction function.
- `crop_recommendation.ipynb`: Notebook version with step-by-step code and explanations.
- `Crop_recommendation.csv`: Dataset used for training and testing.
- `requirements.txt`: List of libraries required to run the project.
- `README.md`: Project documentation and instructions.

---

## ▶️ How to Run

### On Local Machine

1. Clone the repository and navigate into the folder.
2. Install the required libraries using the command:

