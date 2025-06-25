# data_cleaning-of-titanic
🧹 Titanic Dataset cleaned and preprocessed with Pandas, NumPy, and Scikit-learn for ML model readiness.
# 🧹 Titanic Dataset - Data Cleaning & Preprocessing

This project involves preparing the Titanic dataset for machine learning by performing essential data cleaning and preprocessing tasks such as handling missing values, encoding categorical variables, scaling numerical features, detecting outliers, and visualizing the data.

---

## 🎯 Objective

The objective of this task is to clean and preprocess raw Titanic dataset to make it suitable for training ML models. Raw datasets often contain missing values, inconsistent data types, and outliers, all of which can negatively impact model performance. This notebook demonstrates step-by-step how to transform messy data into a clean, usable format for machine learning pipelines.

---

## ✅ Steps Performed

### 1. **Loading & Understanding the Dataset**
- Used `pandas` to load the dataset.
- Used `.info()`, `.head()`, `.describe()`, and `.isnull().sum()` to explore the dataset and understand the structure.

### 2. **Handling Missing Values**
- **Age**: Filled missing values with the **median** to preserve distribution and minimize outlier impact.
- **Embarked**: Filled missing values with the **mode** (most frequent category).
- **Cabin**: Dropped the column due to over 70% missing data, making it unreliable.

### 3. **Encoding Categorical Variables**
- **Sex**: Converted to binary using `map()` — `male` as `0`, `female` as `1`.
- **Embarked**: One-hot encoded using `pd.get_dummies()` with `drop_first=True` to avoid dummy variable trap.

### 4. **Feature Scaling**
- Applied `StandardScaler` to `Age` and `Fare` columns to standardize the values (mean = 0, std = 1).
- Standardization is important for models that are sensitive to feature magnitude.

### 5. **Outlier Detection & Removal**
- Visualized outliers using `Seaborn` boxplots.
- Removed extreme outliers in the `Fare` column using the **Z-score** method (removed values with z > 3).

### 6. **Data Visualization**
- Plotted missing values, boxplots, and distributions to better understand data behavior.
- Tools used: `Matplotlib`, `Seaborn`

---

## 🧠 What You'll Learn

- How to handle missing values in a real-world dataset
- Encoding strategies for categorical data
- Scaling techniques for numerical features
- Outlier detection using statistical methods
- Data visualization techniques to aid preprocessing
- End-to-end data preprocessing pipeline before model building

---

## 🧰 Tools & Libraries Used

- Python (Jupyter Notebook)
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn (StandardScaler)

---

---

## 📊 Dataset

- **Source**: [Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)
- Description: Contains passenger data used to predict survival on the Titanic.

---

## 📌 Final Output

A clean and preprocessed DataFrame ready to be used for building ML models like Logistic Regression, Decision Trees, or Random Forests.

---

## 🚀 Next Steps (Optional)

- Train a classification model to predict survival.
- Perform feature engineering (e.g., extract titles from names).
- Use cross-validation to evaluate model performance.
- Visualize feature importance.

---

## 🔹 One-liner description for repo:

> 🧹 Titanic dataset cleaned and preprocessed using Python, Pandas, and Scikit-learn for ML model readiness.

---

Feel free to explore, fork, and contribute!



