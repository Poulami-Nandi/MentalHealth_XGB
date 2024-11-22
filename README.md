
# **Mental Health data analysis**

This repository contains a solution for the Kaggle competition **Mental health data analysis**, which involves predicting outcomes using a Gradient Boosting model (XGBoost). The project follows a structured approach to preprocess data, explore patterns, train the model, and generate predictions.

---

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Solution Workflow](#solution-workflow)
    - [Step 1: Data Loading](#step-1-data-loading)
    - [Step 2: Exploratory Data Analysis (EDA)](#step-2-exploratory-data-analysis-eda)
    - [Step 3: Data Preprocessing](#step-3-data-preprocessing)
    - [Step 4: Model Training](#step-4-model-training)
    - [Step 5: Model Evaluation](#step-5-model-evaluation)
    - [Step 6: Generating Submission](#step-6-generating-submission)
4. [Usage](#usage)
5. [Contact](#contact)

---

## **Project Overview**

This solution tackles a supervised learning problem where the goal is to predict a target variable based on provided features. The steps include:

- Loading and inspecting the dataset.
- Conducting Exploratory Data Analysis (EDA) to uncover patterns and relationships.
- Preprocessing the data to ensure compatibility with machine learning algorithms.
- Training a Gradient Boosting model (XGBoost) for prediction.
- Evaluating the model's performance using appropriate metrics.
- Generating a submission file for the Kaggle competition.

---

## **Technologies Used**

- **Programming Language**: Python 3.10
- **Libraries**:
  - Data manipulation: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `xgboost`, `scikit-learn`

---

## **Solution Workflow**

### **Step 1: Data Loading**

Load the training and test datasets using `pandas` and inspect their structure. For example:

```python
import pandas as pd

# Load the data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Display the first few rows of the dataset
print(train_data.head())
```

---

### **Step 2: Exploratory Data Analysis (EDA)**

Perform EDA to understand the dataset and uncover insights. Key steps include:

- **Target Variable Distribution**:
  Visualize the distribution of the target variable to assess class balance.

  ```python
  import seaborn as sns
  import matplotlib.pyplot as plt

  sns.countplot(train_data['Depression'])
  plt.title("Target Variable Distribution")
  plt.xlabel("Depression")
  plt.ylabel("Count")
  plt.show()
  ```

- **Correlation Analysis**:
  Use a heatmap to analyze relationships between features.

  ```python
  numeric_data = train_data.select_dtypes(include=[np.number])
  correlation_matrix = numeric_data.corr()

  sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
  plt.title("Feature Correlation Heatmap")
  plt.show()
  ```

---

### **Step 3: Data Preprocessing**

Prepare the data for model training by:

1. **Handling Missing Values**:
   Fill missing values in numeric columns with the column mean, and categorical columns with `"Unknown"`.

   ```python
   train_data.fillna(train_data.mean(), inplace=True)
   test_data.fillna(test_data.mean(), inplace=True)
   ```

2. **Encoding Categorical Variables**:
   Convert categorical variables to numerical using one-hot encoding.

   ```python
   train_data = pd.get_dummies(train_data, drop_first=True)
   test_data = pd.get_dummies(test_data, drop_first=True)
   ```

3. **Align Columns in Train and Test Sets**:
   Ensure both datasets have matching columns.

   ```python
   test_data = test_data.reindex(columns=train_data.columns, fill_value=0)
   ```

---

### **Step 4: Model Training**

Train a Gradient Boosting model (XGBoost) on the processed dataset:

```python
from xgboost import XGBClassifier

# Initialize the model
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42)

# Train the model
model.fit(X_train, y_train)
```

---

### **Step 5: Model Evaluation**

Evaluate the model's performance on the validation set:

```python
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Predict on the validation set
y_pred = model.predict(X_val)

# Evaluate performance
print("Accuracy:", accuracy_score(y_val, y_pred))
print("F1 Score:", f1_score(y_val, y_pred, average="weighted"))
print("Classification Report:
", classification_report(y_val, y_pred))
```

---

### **Step 6: Generating Submission**

Predict on the test dataset and prepare a submission file:

```python
# Predict on the test set
test_predictions = model.predict(test_data)

# Create a submission file
submission = pd.DataFrame({
    "Id": test_data.index,
    "Depression": test_predictions
})

submission.to_csv("submission.csv", index=False)
```

---

## **Usage**

1. Clone this repository:
   ```bash
   git clone https://github.com/Poulami-Nandi/MentalHealth_XGB.git
   ```
2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn xgboost scikit-learn
   ```
3. Run the notebook file `solution.ipynb` in your Python environment like Jupyter Notebook or Google Colab.

---

## **Contact**

Feel free to reach out if you have any questions or suggestions:

- **Email**: [Dr. Poulami Nandi](mailto:nandi.poulami91@gmail.com)
- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/poulami-nandi-a8a12917b/)

---
