# TitanicPrediction
A classification based model predicting which passengers survived from the Titanic ship
# Titanic Survival Prediction

## Overview
This project builds a machine learning classification model to predict 
whether a passenger survived the Titanic shipwreck based on demographic 
and ticket information. It was submitted as part of the Kaggle 
"Titanic - Machine Learning from Disaster" competition.

## Dataset
- **Source:** Kaggle — Titanic: Machine Learning from Disaster
- **Train set:** 891 passengers with survival labels
- **Test set:** 418 passengers without survival labels
- **Target variable:** Survived (0 = Died, 1 = Survived)

## Features Used
| Feature | Description |
|---|---|
| Pclass | Ticket class (1st, 2nd, 3rd) |
| Sex | Gender (encoded: male=0, female=1) |
| Age | Age in years (missing values filled with median) |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Fare | Ticket price |
| Embarked_Q | Embarked from Queenstown (one-hot encoded) |
| Embarked_S | Embarked from Southampton (one-hot encoded) |

## Project Structure
├── titanic_survival_prediction.ipynb  # Main notebook
├── train.csv                          # Training data
├── test.csv                           # Test data
├── submission.csv                     # Kaggle submission file
├── model_comparison.png               # Model comparison chart
├── feature_importance.png             # Feature importance chart
└── README.md                          # Project documentation

## Tools & Libraries
- Python, Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## Methodology

### 1. Exploratory Data Analysis
- Visualised survival rates by sex, passenger class and overall
- Key finding: females survived at 74% vs males at 19%
- Key finding: 1st class passengers survived at 63% vs 25% for 3rd class
- Identified class imbalance: 62% died vs 38% survived

### 2. Data Preprocessing
- Dropped columns with excessive missing values or low predictive 
  value: Cabin (77% missing), Name, Ticket
- Filled Age missing values with median (177 missing)
- Filled Embarked missing values with mode (2 missing)
- Filled Fare missing values in test set with median (1 missing)
- Encoded Sex using label encoding (male=0, female=1)
- Encoded Embarked using one-hot encoding with drop_first=True

### 3. Models Built
Two classification models were trained and evaluated:
- Logistic Regression (baseline)
- Random Forest Classifier

### 4. Model Evaluation
Both models were evaluated on a held-out test set (20%) and validated 
using 5-fold cross validation to ensure results were not dependent on 
a single train/test split. Overfitting was checked by comparing train 
and test accuracy — the gap was less than 0.01, confirming no 
overfitting.

## Results

### Test Set Performance
| Model | Accuracy | F1 Score (class 1) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.8101 | 0.76 | — |
| Random Forest | 0.7989 | 0.75 | — |

### 5-Fold Cross Validation
| Model | CV Accuracy | CV F1 | CV ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.7912 ± 0.019 | 0.7195 ± 0.028 | 0.8485 ± 0.015 |
| Random Forest | 0.8081 ± 0.030 | 0.7445 ± 0.047 | 0.8539 ± 0.038 |

## Key Findings
- Sex was the strongest predictor of survival — females were 
  nearly 4x more likely to survive than males
- Passenger class was the second most important feature — 
  1st class passengers had significantly higher survival rates
- Logistic Regression showed greater stability across CV folds 
  (std=0.019 vs 0.030) while Random Forest achieved slightly 
  higher CV accuracy and F1 score
- On small datasets like Titanic (891 rows), simpler models often 
  match or outperform complex ones — a key insight for model 
  selection in practice
- The gap between train and test accuracy was less than 0.01 
  confirming no overfitting in either model

## Confusion Matrix Analysis
The Logistic Regression confusion matrix showed:
- 90 True Negatives — correctly predicted deaths
- 55 True Positives — correctly predicted survivors
- 15 False Positives — predicted survived but actually died
- 19 False Negatives — predicted died but actually survived

The 19 False Negatives represent the model's biggest weakness — 
passengers who survived despite the model predicting otherwise. 
These are likely male passengers from 1st or 2nd class who beat 
the odds.

## Visualisations
### Model Comparison
![Model Comparison](model_comparison.png)

### Feature Importance
![Feature Importance](feature_importance.png)

## What I Would Do Next
- Extract title from passenger names (Mr, Mrs, Miss, Master) as 
  an additional feature — titles carry age and gender information
  that could improve predictions
- Create a Has_Cabin binary feature before dropping the Cabin 
  column — cabin presence strongly correlates with passenger class
- Create a FamilySize feature by combining SibSp and Parch
- Tune Random Forest hyperparameters using GridSearchCV
- Try XGBoost and compare against both baseline models
- Deploy the model as a Streamlit web app where users can input 
  passenger details and get a survival prediction

## Kaggle Submission
- **Competition:** Titanic - Machine Learning from Disaster
- **Model used:** Random Forest Classifier


## Author
Daniel Abifarin
Electrical Engineering Student | University of Lagos
Aspiring MLOps Engineer
GitHub: github.com/Daniel-Abifarin
