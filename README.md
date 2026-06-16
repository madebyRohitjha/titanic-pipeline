# Titanic Survival Prediction 

## Project Overview

This project predicts whether a passenger survived the Titanic disaster using Machine Learning.

The goal of this project was to practice:

* Data preprocessing
* Feature engineering
* Building ML pipelines
* Model comparison
* Model evaluation

---

## Dataset

Dataset: Titanic Dataset from Kaggle

Target Variable:

* `Survived`

  * 0 = Did not survive
  * 1 = Survived

---

## Features Used

* Pclass
* Sex
* Age
* SibSp
* Parch
* Fare
* Embarked

### Engineered Features

* FamilySize
* IsAlone
* Title extracted from passenger names

---

## Data Preprocessing

### Numerical Features

* Missing values filled using Median Imputation.

### Categorical Features

* Missing values filled using Most Frequent Imputation.
* One Hot Encoding applied.

### Pipeline Used

* Pipeline
* ColumnTransformer

---

## Models Trained

### Random Forest Classifier

* Accuracy: 83.80%
* Cross Validation Score: 79.78%

### Logistic Regression

* Accuracy: 81.56%
* Cross Validation Score: 82.72%

---

## Key Learnings

* Feature engineering can significantly improve model performance.
* Not every preprocessing technique improves every model.
* Random Forest generally does not require feature scaling.
* Cross-validation provides a more reliable estimate of model performance than a single train-test split.
* Different algorithms can behave very differently on the same dataset.

---

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* NumPy

---

## Future Improvements

* Hyperparameter tuning using GridSearchCV.
* Try additional algorithms such as XGBoost and SVM.
* Deploy the model as a web application.

---

## Author

Rohit Jha

Learning AI and Machine Learning by building projects and sharing the journey publicly.
