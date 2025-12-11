# Car_Evaluation_prediction
Here is your **FULL, FINAL, GITHUB-READY README.md**, updated with your actual Decision Tree accuracy (**97% train, 93% test**).
Just copy–paste into your GitHub repo.

---

# 🚗 Car Evaluation Prediction

*Machine Learning Classification Project*

![Car Evaluation](https://img.shields.io/badge/Machine%20Learning-Classification-blue)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Models-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📘 Overview

This repository contains a machine learning project for predicting **car acceptability** using the **Car Evaluation Dataset**.
The model classifies cars into four categories:

* **unacc** (Unacceptable)
* **acc** (Acceptable)
* **good**
* **vgood** (Very Good)

The dataset includes categorical attributes that describe different aspects of a car such as buying cost, safety rating, luggage boot size, and more.
This project covers data preprocessing, visualization, model training, hyperparameter tuning, and performance evaluation.

---

## 📁 Dataset Features

| Feature          | Description                 |
| ---------------- | --------------------------- |
| **Buying**       | Buying price of the car     |
| **Maintenance**  | Maintenance cost            |
| **Doors**        | Number of doors             |
| **Persons**      | Passenger capacity          |
| **Luggage Boot** | Size of luggage compartment |
| **Safety**       | Safety rating               |

**Target Variable:**
`Class` → `unacc`, `acc`, `good`, `vgood`

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

* ✔ Label Encoding of categorical variables
* ✔ Splitting data into training & testing sets
* ✔ Applying PCA (2 components) for visualization
* ✔ Handling large label values for PCA color-coding
* ✔ Optional scaling

---

## 📉 PCA Visualization

To understand the distribution of classes visually, PCA transforms the dataset into **2 principal components**.

Visualization highlights:

* Each point represents a car instance
* Colors represent encoded class labels
* Shows approximate cluster separation

PCA is used **only for visualization**, not for model training.

---

## 🤖 Machine Learning Model

### 🌳 **Decision Tree Classifier**

The Decision Tree model was trained and tuned using:

```python
parameters = {
    "max_depth": [2, 3, 4, 5, 6, 7],
    "min_samples_split": [25, 30, 35, 40, 45, 50],
    "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7]
}
```

GridSearchCV was used to find the best combination.

---

## 📊 Model Performance

### **Decision Tree Classifier Results**

| Metric             | Score   |
| ------------------ | ------- |
| **Train Accuracy** | **97%** |
| **Test Accuracy**  | **93%** |

📌 **Interpretation:**

* Only **4% gap**, meaning **very little overfitting**
* Model generalizes well
* High performance for a multiclass problem

This makes the Decision Tree a strong final model for this dataset.

---

## 📦 Tech Stack

* Python
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## 📁 Project Structure

```
├── data/
│   └── car_evaluation.csv
├── images/
│   └── PCA_visualization.png
├── model/
│   └── decision_tree_model.pkl
├── Car_Evaluation.ipynb
└── README.md
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/Car_Evaluation_Prediction.git
cd Car_Evaluation_Prediction
```

### 2️⃣ Install required packages

```bash
pip install -r requirements.txt
```

### 3️⃣ Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4️⃣ Train or load the model

```python
import pickle
model = pickle.load(open("model/decision_tree_model.pkl", "rb"))
```

### 5️⃣ Make predictions

```python
model.predict([[2, 1, 4, 4, 1, 2]])
```

---

## ⭐ Future Improvements

* Add Random Forest & XGBoost for comparison
* Build a Streamlit dashboard
* Deploy via Flask or FastAPI
* Add model explainability (SHAP / feature importance)

---

## ❤️ Contributions

Pull requests, suggestions, and improvements are welcome!



