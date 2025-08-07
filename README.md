# Smart Crop Recommendation System using Machine Learning

This project is a machine learning-based crop recommendation system that predicts the most suitable crop to cultivate based on environmental features such as soil nutrients (N, P, K), temperature, humidity, pH, and rainfall. It uses real-world agricultural data and supervised learning models to deliver accurate predictions.

---

## Project Highlights

- Predicts the most suitable crop for cultivation using soil and climate data.
- Compares performance of 10+ machine learning models including SVM, KNN, Decision Tree, and Ensemble methods.
- Final model selected: Random Forest Classifier due to its superior accuracy.
- Custom prediction function implemented to generate real-time crop recommendations.
- Demonstrates practical application of AI in agriculture.

---

## Tech Stack

- **Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, seaborn, matplotlib  
- **Tools Used:** Jupyter Notebook, Python script

---

## How It Works

1. Load and preprocess the agricultural dataset (handle missing values, normalize features, encode labels).
2. Train multiple machine learning models and compare their accuracy.
3. Select the best-performing model (Random Forest) based on evaluation metrics.
4. Use the trained model to predict the most suitable crop based on input values like NPK levels, temperature, humidity, pH, and rainfall.



