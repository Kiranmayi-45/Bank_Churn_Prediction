💼 Bank Customer Churn Prediction

A Streamlit web application that predicts whether a bank customer is likely to **churn (leave the bank)** or not based on various factors such as account activity, age, credit score, and more.

!-- Optional: Replace with your own screenshot if available -->

---

## 🚀 Features

- 📊 Visualize customer data
- 🧠 Predict churn using trained Machine Learning models (Random Forest, XGBoost, ANN, etc.)
- 🔍 Interactive input form for customer attributes
- 💾 Model loaded from `.pkl` and `.keras` files
- 🎯 Displays prediction with confidence

---

## 🛠 Tech Stack

- **Python**
- **Streamlit**
- **Pandas, NumPy, Scikit-learn**
- **TensorFlow / Keras**
- **XGBoost**
- **Matplotlib / Seaborn**

---

## 🧠 ML Models Used

- Random Forest
- XGBoost
- Naive Bayes
- K-Nearest Neighbors
- Logistic Regression
- Artificial Neural Network (ANN)

Models are pre-trained and saved as `.pkl` or `.keras` files.

---

## 📁 Project Structure

```bash
├── app.py                      # Streamlit application
├── preprocess.py              # Custom preprocessing logic
├── data.csv                   # Dataset used
├── model_files/               # Saved model files (.pkl, .keras)
├── templates/
│   └── index.html             # Optional HTML template
├── .gitignore
├── README.md                  # Project documentation
