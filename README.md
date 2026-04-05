# titanic-logistic-regression
Predicting Titanic passenger survival using Logistic Regression with data preprocessing, feature engineering, and model evaluation.
---

## Objective

Predict whether a passenger survived using features such as:
- Passenger class (Pclass)
- Sex
- Age
- Fare
- Family information (SibSp, Parch)
- Port of embarkation (Embarked)

Target variable:
- `0` → did not survive  
- `1` → survived  

This is a binary classification problem.

---

## Workflow Completed

### 1. Data Loading
Loaded dataset from Kaggle:
- train.csv
- test.csv

---

### 2. Data Exploration
Identified:
- Target variable: Survived
- Useful features:
  - Pclass
  - Sex
  - Age
  - SibSp
  - Parch
  - Fare
  - Embarked

Dropped or avoided:
- PassengerId (identifier)
- Ticket (unstructured)
- Cabin (too many missing values)
- Name (raw text)

---

### 3. Handling Missing Values
- Dropped Cabin
- Filled Age with median
- Filled Embarked with mode

---

### 4. Encoding
- Sex: male → 0, female → 1
- Embarked: one-hot encoded into:
  - Embarked_C
  - Embarked_Q
  - Embarked_S

---

### 5. Feature Preparation
Separated:
- X → input features
- y → target (Survived)

---

### 6. Train-Test Split
- 80% training
- 20% testing
- Used train_test_split

---

### 7. Model Training
Used Logistic Regression as a baseline model.

---

### 8. Evaluation

Accuracy: ~0.81

Confusion Matrix:
[[90 15]
 [19 55]]

Interpretation:
- Model performs well overall
- Slight bias toward predicting non-survival
- Misses some actual survivors (false negatives)

Classification Report:
- Precision, Recall, F1-score evaluated
- Recall for survivors is lower → improvement area

---

## Project Structure

titanic-logistic-regression/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
│
├── notebooks/
│   └── titanic_exploration.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── main.py
│
├── README.md
└── .gitignore

---

## Modules

- data_loader.py → loads dataset  
- preprocessing.py → cleaning and encoding  
- train.py → split and model training  
- evaluate.py → metrics and evaluation  
- main.py → full pipeline execution  

---

## How to Run

From project root:

python src/main.py

---

## Key Learnings

- Difference between classification and regression  
- Importance of feature selection  
- Handling missing data  
- Encoding categorical variables  
- Train-test split and generalization  
- Logistic Regression fundamentals  
- Evaluation using confusion matrix, precision, recall, F1  
- Structuring ML projects into modular code  

---

## Current Status

Completed:
- Data preprocessing  
- Feature selection  
- Model training  
- Evaluation  
- Modular project structure  
- GitHub integration  

Next:
- Kaggle submission (baseline)  
- Model improvement (feature engineering, better models)  
- Optional model saving (.pkl)  

---

## Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- Jupyter Notebook  
- VS Code  
- Git  

---

## Author

Built as a hands-on machine learning project to understand end-to-end classification workflow and project structuring using the Titanic dataset.